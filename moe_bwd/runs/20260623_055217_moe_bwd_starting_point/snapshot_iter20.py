"""
MoE backward pass — torch.vmap over the proven padded-bmm structure.

Replaces torch.bmm([E, B, H] @ [E, H, M]) with torch.vmap(torch.mm) which
maps to the same batched GEMM kernel but via a different dispatch path that
may utilize cuBLAS grouped GEMM more efficiently.

The padded [E, B, H] tensor layout (Exp #2) is retained as it is proven correct.
The element-wise ops (SiLU, SwiGLU grad, routing weight scaling) are unchanged.
Only the torch.bmm → torch.vmap(torch.mm) replacement is made.

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

# vmap-based batched matmul helpers (created once at module load)
# These map over the first (batch/expert) dimension
# A: [E, B, K_in], W: [E, K_out, K_in] → Out: [E, B, K_out]
# vmap(mm)(A, W^T) where W^T: [E, K_in, K_out]
_vmap_mm = torch.vmap(torch.mm, in_dims=(0, 0), out_dims=0)


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
    # Step 1: Sort tokens by expert (identical to Exp #2)
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

    B = max_tokens_per_expert

    # -----------------------------------------------------------------------
    # Step 2: Compute padded indices (identical to Exp #2)
    # -----------------------------------------------------------------------
    ones       = torch.ones(N, dtype=torch.long, device=device)
    cumsum_all = torch.cumsum(ones, dim=0) - 1
    group_starts     = expert_offsets[:-1][sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Gather sorted inputs (identical to Exp #2)
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    # -----------------------------------------------------------------------
    # Step 4: Build padded tensors [E, B, H] (identical to Exp #2)
    # -----------------------------------------------------------------------
    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)

    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)

    # -----------------------------------------------------------------------
    # Step 5: Forward recomputation via vmap(mm) instead of bmm
    # padded_hidden [E, B, H] @ gate_weights [E, M, H]^T → [E, B, M]
    # vmap maps mm over expert dim: mm([B, H], [H, M]) → [B, M] for each expert
    # gate_weights [E, M, H].transpose(1,2) = [E, H, M]
    # -----------------------------------------------------------------------
    gate_wT = gate_weights.transpose(1, 2).contiguous()  # [E, H, M]
    up_wT   = up_weights.transpose(1, 2).contiguous()    # [E, H, M]

    gate_pre_act   = _vmap_mm(padded_hidden, gate_wT)    # [E, B, M]
    up_output      = _vmap_mm(padded_hidden, up_wT)      # [E, B, M]
    gate_activated = F.silu(gate_pre_act)                 # [E, B, M]
    intermediate   = gate_activated * up_output           # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 6: grad_topk_weights
    # expert_output [E, B, H] = intermediate [E, B, M] @ down_weights [E, H, M]^T
    # down_weights [E, H, M].transpose(1,2) = [E, M, H]
    # -----------------------------------------------------------------------
    down_wT = down_weights.transpose(1, 2).contiguous()  # [E, M, H]
    expert_output    = _vmap_mm(intermediate, down_wT)               # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)  # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk    = grad_topk_w_flat.reshape(-1)[padded_idx]
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 7: Grad through down projection
    # scaled_grad_out [E, B, H] @ down_weights [E, H, M] → [E, B, M]
    # grad_down_weights [E, H, M] = scaled_grad_out^T [E, H, B] @ intermediate [E, B, M]
    # -----------------------------------------------------------------------
    scaled_grad_out = padded_grad_out * padded_weights.unsqueeze(2)    # [E, B, H]

    # grad_intermediate: [E, B, H] @ [E, H, M] → [E, B, M]
    grad_intermediate = _vmap_mm(scaled_grad_out, down_weights)        # [E, B, M]

    # grad_down_weights: [E, H, B] @ [E, B, M] → [E, H, M]
    scaled_grad_out_T = scaled_grad_out.transpose(1, 2).contiguous()  # [E, H, B]
    grad_down_weights = _vmap_mm(scaled_grad_out_T, intermediate)      # [E, H, M]

    # -----------------------------------------------------------------------
    # Step 8: Grad through SwiGLU (identical to Exp #2)
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated           # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output                # [E, B, M]
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                   # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 9: Weight gradients and grad_hidden via vmap(mm)
    # grad_gate_weights [E, M, H] = grad_gate_pre_act^T [E, M, B] @ padded_hidden [E, B, H]
    # grad_hidden_gate  [E, B, H] = grad_gate_pre_act [E, B, M] @ gate_weights [E, M, H]
    # -----------------------------------------------------------------------
    grad_gate_pre_act_T = grad_gate_pre_act.transpose(1, 2).contiguous()  # [E, M, B]
    grad_up_output_T    = grad_up_output.transpose(1, 2).contiguous()     # [E, M, B]

    grad_gate_weights = _vmap_mm(grad_gate_pre_act_T, padded_hidden)      # [E, M, H]
    grad_up_weights   = _vmap_mm(grad_up_output_T,    padded_hidden)      # [E, M, H]

    grad_hidden_gate  = _vmap_mm(grad_gate_pre_act, gate_weights)         # [E, B, H]
    grad_hidden_up    = _vmap_mm(grad_up_output,    up_weights)           # [E, B, H]

    # -----------------------------------------------------------------------
    # Step 10: Scatter grad_hidden (identical to Exp #2)
    # -----------------------------------------------------------------------
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).reshape(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_idx]
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
