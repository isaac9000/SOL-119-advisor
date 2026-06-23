"""
MoE backward pass — padded-bmm with B aligned to multiple of 64, explicit contiguous
tensors, and streamlined padded_idx computation.

Key changes from Exp #2 baseline:
1. B rounded up to nearest multiple of 64 for cuBLAS GEMM tile alignment
2. All padded tensors explicitly .contiguous() before bmm calls
3. padded_idx computed without intermediate `ones` tensor — direct arange subtraction
4. gate_weights/up_weights transposed once and made contiguous before the bmm loop

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

Returns:
    grad_hidden_states, grad_topk_weights, grad_gate_weights,
    grad_up_weights, grad_down_weights
"""

import torch
import torch.nn.functional as F

HIDDEN_SIZE           = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS      = 256
NUM_EXPERTS_PER_TOK   = 8


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    T, K  = topk_indices.shape
    device = hidden_states.device
    dtype  = hidden_states.dtype
    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE           # 4096
    M = MOE_INTERMEDIATE_SIZE  # 2048
    N = T * K

    # -----------------------------------------------------------------------
    # Step 1: Sort tokens by expert
    # -----------------------------------------------------------------------
    flat_experts = topk_indices.reshape(-1)
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    slot_ids  = torch.arange(K, device=device).unsqueeze(0).expand(T, K).reshape(-1)

    sort_order       = torch.argsort(flat_experts, stable=True)
    sorted_experts   = flat_experts[sort_order]
    sorted_token_ids = token_ids[sort_order]
    sorted_slot_ids  = slot_ids[sort_order]

    expert_counts  = torch.bincount(sorted_experts, minlength=E)
    expert_offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    max_tokens_per_expert = int(expert_counts.max().item())
    if max_tokens_per_expert == 0:
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(topk_weights),
            torch.zeros_like(gate_weights),
            torch.zeros_like(up_weights),
            torch.zeros_like(down_weights),
        )

    # Round B up to nearest multiple of 64 for cuBLAS tile alignment
    B = ((max_tokens_per_expert + 63) // 64) * 64

    # -----------------------------------------------------------------------
    # Step 2: Compute padded indices — streamlined (no intermediate ones tensor)
    # expert_local_pos[i] = global_position[i] - expert_offset[sorted_expert[i]]
    # -----------------------------------------------------------------------
    global_pos       = torch.arange(N, device=device, dtype=torch.long)
    group_starts     = expert_offsets[:-1][sorted_experts]   # [N]
    expert_local_pos = global_pos - group_starts              # [N]
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Gather sorted inputs
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    # -----------------------------------------------------------------------
    # Step 4: Build padded tensors [E, B, H/M] — contiguous for cuBLAS
    # -----------------------------------------------------------------------
    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)  # already contiguous

    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)

    # -----------------------------------------------------------------------
    # Step 5: Pre-transpose weight matrices once for reuse
    # gate_weights: [E, M, H] → transpose to [E, H, M] for A @ W^T style
    # We need W^T for forward and W for backward; pre-compute both as contiguous.
    # -----------------------------------------------------------------------
    gate_wT = gate_weights.transpose(1, 2).contiguous()   # [E, H, M]
    up_wT   = up_weights.transpose(1, 2).contiguous()     # [E, H, M]
    # down_weights is [E, H, M] — already in the shape we need for scaled_grad @ dw
    # down_wT would be [E, M, H] for intermediate @ dw^T
    down_wT = down_weights.transpose(1, 2).contiguous()   # [E, M, H]

    # -----------------------------------------------------------------------
    # Step 6: Forward recomputation
    # padded_hidden [E, B, H] @ gate_wT [E, H, M] → [E, B, M]
    # -----------------------------------------------------------------------
    gate_pre_act   = torch.bmm(padded_hidden, gate_wT)   # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_wT)     # [E, B, M]
    gate_activated = F.silu(gate_pre_act)                 # [E, B, M]
    intermediate   = gate_activated * up_output           # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 7: grad_topk_weights
    # expert_output [E, B, H] = intermediate @ down_wT [E, M, H]
    # -----------------------------------------------------------------------
    expert_output    = torch.bmm(intermediate, down_wT)                      # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)          # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk    = grad_topk_w_flat.reshape(-1)[padded_idx]
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 8: Grad through down projection
    # scaled_grad_out [E, B, H] = padded_grad_out * routing_weights
    # grad_down_weights [E, H, M] = scaled_grad_out^T @ intermediate
    # grad_intermediate [E, B, M] = scaled_grad_out @ down_weights [E, H, M]
    # -----------------------------------------------------------------------
    scaled_grad_out = padded_grad_out * padded_weights.unsqueeze(2)          # [E, B, H]

    grad_down_weights = torch.bmm(scaled_grad_out.transpose(1, 2).contiguous(),
                                  intermediate)                               # [E, H, M]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)             # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 9: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated                  # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output                       # [E, B, M]
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                          # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 10: Weight gradients and grad_hidden
    # grad_gate_weights [E, M, H] = grad_gate_pre_act^T @ padded_hidden
    # grad_up_weights   [E, M, H] = grad_up_output^T    @ padded_hidden
    # grad_hidden_gate  [E, B, H] = grad_gate_pre_act   @ gate_weights [E, M, H]
    # grad_hidden_up    [E, B, H] = grad_up_output       @ up_weights   [E, M, H]
    # -----------------------------------------------------------------------
    grad_gate_pre_act_T = grad_gate_pre_act.transpose(1, 2).contiguous()     # [E, M, B]
    grad_up_output_T    = grad_up_output.transpose(1, 2).contiguous()        # [E, M, B]

    grad_gate_weights = torch.bmm(grad_gate_pre_act_T, padded_hidden)        # [E, M, H]
    grad_up_weights   = torch.bmm(grad_up_output_T,    padded_hidden)        # [E, M, H]

    grad_hidden_gate  = torch.bmm(grad_gate_pre_act, gate_weights)           # [E, B, H]
    grad_hidden_up    = torch.bmm(grad_up_output,    up_weights)             # [E, B, H]

    # -----------------------------------------------------------------------
    # Step 11: Scatter grad_hidden back
    # -----------------------------------------------------------------------
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).reshape(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_idx]                       # [N, H]

    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
