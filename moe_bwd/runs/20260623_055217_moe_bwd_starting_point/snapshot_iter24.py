"""
MoE backward pass — padded-bmm with targeted optimizations over 18.51 ms baseline:
1. Pre-transpose all weight matrices once (contiguous) → avoids 8 per-call transposes
2. B padded to next power of 2 → better cuBLAS tile alignment
3. Combined [N, 2H+1] scatter for hidden + grad_out + weights → 3 scatters → 1

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

    # Pad B to next power of 2 for cuBLAS tile alignment
    # e.g. 65→128, 64→64, 100→128
    B_raw = max_tokens_per_expert
    B = 1
    while B < B_raw:
        B <<= 1

    # -----------------------------------------------------------------------
    # Step 2: Compute padded indices
    # -----------------------------------------------------------------------
    ones       = torch.ones(N, dtype=torch.long, device=device)
    cumsum_all = torch.cumsum(ones, dim=0) - 1
    group_starts     = expert_offsets[:-1][sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Gather sorted inputs
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    # -----------------------------------------------------------------------
    # Step 4: Combined single scatter for hidden + grad_out + weights
    # Build one [E*B, 2H+1] buffer, scatter [N, 2H+1] in one operation
    # -----------------------------------------------------------------------
    combined = torch.zeros(E * B, 2 * H + 1, dtype=dtype, device=device)
    # Pack [N, 2H+1] source
    combined_src = torch.cat([
        sorted_hidden,                    # [N, H]
        sorted_grad_out,                  # [N, H]
        sorted_weights.unsqueeze(1),      # [N, 1]
    ], dim=1)                             # [N, 2H+1]
    combined[padded_idx] = combined_src   # single scatter
    combined_3d = combined.view(E, B, 2 * H + 1)

    padded_hidden   = combined_3d[:, :, :H].contiguous()       # [E, B, H]
    padded_grad_out = combined_3d[:, :, H:2*H].contiguous()    # [E, B, H]
    padded_weights  = combined_3d[:, :, 2*H]                   # [E, B]

    # -----------------------------------------------------------------------
    # Step 5: Pre-transpose all weight matrices once (contiguous)
    # Avoids repeated non-contiguous transpose views inside bmm calls
    # gate_weights [E, M, H] → gate_wT [E, H, M]  (for A @ gate_wT)
    # up_weights   [E, M, H] → up_wT   [E, H, M]
    # down_weights [E, H, M] → down_wT [E, M, H]  (for intermediate @ down_wT → [E,B,H])
    # -----------------------------------------------------------------------
    gate_wT = gate_weights.transpose(1, 2).contiguous()   # [E, H, M]
    up_wT   = up_weights.transpose(1, 2).contiguous()     # [E, H, M]
    down_wT = down_weights.transpose(1, 2).contiguous()   # [E, M, H]

    # -----------------------------------------------------------------------
    # Step 6: Forward recomputation with pre-transposed weights
    # -----------------------------------------------------------------------
    gate_pre_act   = torch.bmm(padded_hidden, gate_wT)     # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_wT)       # [E, B, M]
    gate_activated = F.silu(gate_pre_act)
    intermediate   = gate_activated * up_output             # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 7: grad_topk_weights
    # -----------------------------------------------------------------------
    expert_output    = torch.bmm(intermediate, down_wT)                      # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)          # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk    = grad_topk_w_flat.view(-1)[padded_idx]
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 8: Grad through down projection
    # scaled_grad_out [E, B, H] @ down_weights [E, H, M] → [E, B, M]
    # grad_down_weights [E, H, M] = scaled_grad_out^T @ intermediate
    # -----------------------------------------------------------------------
    scaled_grad_out   = padded_grad_out * padded_weights.unsqueeze(2)        # [E, B, H]
    grad_down_weights = torch.bmm(scaled_grad_out.transpose(1, 2),
                                  intermediate)                               # [E, H, M]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)             # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 9: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated                  # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                          # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 10: Weight gradients and grad_hidden
    # Use pre-transposed gate_weights/up_weights for grad_hidden GEMMs
    # grad_hidden_gate [E, B, H] = grad_gate_pre_act [E, B, M] @ gate_weights [E, M, H]
    # -----------------------------------------------------------------------
    grad_gate_weights = torch.bmm(grad_gate_pre_act.transpose(1, 2),
                                  padded_hidden)                              # [E, M, H]
    grad_hidden_gate  = torch.bmm(grad_gate_pre_act, gate_weights)           # [E, B, H]

    grad_up_weights   = torch.bmm(grad_up_output.transpose(1, 2),
                                  padded_hidden)                              # [E, M, H]
    grad_hidden_up    = torch.bmm(grad_up_output, up_weights)                # [E, B, H]

    # -----------------------------------------------------------------------
    # Step 11: Scatter grad_hidden
    # -----------------------------------------------------------------------
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).view(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_idx]
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
