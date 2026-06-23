"""
MoE backward pass — Exp #2 baseline with targeted micro-optimizations:
1. Single combined [E*B, 2H+1] scatter replacing 3 separate scatter ops
2. torch.sort() instead of argsort (avoids stable-sort overhead)
3. Derived token_ids/slot_ids from sort_order without arange+expand+reshape
4. Streamlined expert_local_pos via direct arange subtraction (no ones tensor)

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
    # Use torch.sort (not argsort+stable) — faster for this use case.
    # Derive token_ids and slot_ids from sort_order to avoid arange+expand.
    # -----------------------------------------------------------------------
    flat_experts = topk_indices.reshape(-1)  # [N]

    # torch.sort returns (values, indices); stable=False is faster
    sorted_experts, sort_order = torch.sort(flat_experts, stable=False)  # [N] each

    # Recover token_ids = sort_order // K, slot_ids = sort_order % K
    # This avoids building the arange+expand token_ids tensor separately.
    sorted_token_ids = sort_order.div(K, rounding_mode='floor')  # [N]
    sorted_slot_ids  = sort_order.remainder(K)                   # [N]

    # Per-expert counts and offsets
    expert_counts  = torch.bincount(sorted_experts, minlength=E)   # [E]
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
    # Step 2: Compute padded indices
    # Direct arange subtraction — no intermediate ones tensor
    # -----------------------------------------------------------------------
    global_pos       = torch.arange(N, device=device, dtype=torch.long)
    group_starts     = expert_offsets[:-1][sorted_experts]
    expert_local_pos = global_pos - group_starts          # [N]
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Gather sorted inputs
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    # -----------------------------------------------------------------------
    # Step 4: Single combined scatter for hidden + grad_out + weights
    # Build one [E*B, 2H+1] buffer and scatter all three at once.
    # Slice back into views for bmm.
    # -----------------------------------------------------------------------
    combined_buf = torch.zeros(E * B, 2 * H + 1, dtype=dtype, device=device)
    # Pack source: [N, 2H+1]
    combined_src = torch.cat([
        sorted_hidden,                   # [N, H]
        sorted_grad_out,                 # [N, H]
        sorted_weights.unsqueeze(1),     # [N, 1]
    ], dim=1)                            # [N, 2H+1]
    combined_buf[padded_idx] = combined_src   # single scatter
    combined_3d = combined_buf.view(E, B, 2 * H + 1)

    padded_hidden   = combined_3d[:, :, :H].contiguous()     # [E, B, H]
    padded_grad_out = combined_3d[:, :, H:2*H].contiguous()  # [E, B, H]
    padded_weights  = combined_3d[:, :, 2*H]                 # [E, B]

    # -----------------------------------------------------------------------
    # Step 5: Forward recomputation
    # -----------------------------------------------------------------------
    gate_pre_act   = torch.bmm(padded_hidden, gate_weights.transpose(1, 2))  # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_weights.transpose(1, 2))    # [E, B, M]
    gate_activated = F.silu(gate_pre_act)                                     # [E, B, M]
    intermediate   = gate_activated * up_output                               # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 6: grad_topk_weights
    # -----------------------------------------------------------------------
    expert_output    = torch.bmm(intermediate, down_weights.transpose(1, 2)) # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)          # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk    = grad_topk_w_flat.reshape(-1)[padded_idx]
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 7: Grad through down projection
    # -----------------------------------------------------------------------
    scaled_grad_out   = padded_grad_out * padded_weights.unsqueeze(2)        # [E, B, H]
    grad_down_weights = torch.bmm(scaled_grad_out.transpose(1, 2),
                                  intermediate)                               # [E, H, M]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)             # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 8: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated                  # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output                       # [E, B, M]
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                          # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 9: Weight gradients and grad_hidden
    # -----------------------------------------------------------------------
    grad_gate_weights = torch.bmm(grad_gate_pre_act.transpose(1, 2),
                                  padded_hidden)                              # [E, M, H]
    grad_hidden_gate  = torch.bmm(grad_gate_pre_act, gate_weights)           # [E, B, H]

    grad_up_weights   = torch.bmm(grad_up_output.transpose(1, 2),
                                  padded_hidden)                              # [E, M, H]
    grad_hidden_up    = torch.bmm(grad_up_output, up_weights)                # [E, B, H]

    # -----------------------------------------------------------------------
    # Step 10: Scatter grad_hidden
    # -----------------------------------------------------------------------
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).reshape(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_idx]                       # [N, H]

    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
