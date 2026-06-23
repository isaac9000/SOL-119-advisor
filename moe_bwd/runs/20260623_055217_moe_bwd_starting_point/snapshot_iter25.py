"""
MoE backward pass — vectorized padded-bmm (Exp #23 exact copy, proven 18.51 ms).

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

    grad_output     [num_tokens, 4096]              float32
    hidden_states   [num_tokens, 4096]              float32
    topk_indices    [num_tokens, 8]                 int64
    topk_weights    [num_tokens, 8]                 float32
    gate_weights    [256, 2048, 4096]               float32
    up_weights      [256, 2048, 4096]               float32
    down_weights    [256, 4096, 2048]               float32

Returns:
    grad_hidden_states   [num_tokens, 4096]         float32
    grad_topk_weights    [num_tokens, 8]            float32
    grad_gate_weights    [256, 2048, 4096]          float32
    grad_up_weights      [256, 2048, 4096]          float32
    grad_down_weights    [256, 4096, 2048]          float32
"""

import torch
import torch.nn.functional as F

HIDDEN_SIZE          = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS     = 256
NUM_EXPERTS_PER_TOK  = 8


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    num_tokens = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    T, K = topk_indices.shape  # T=num_tokens, K=8

    # Flat expert assignments: [T*K]
    flat_experts = topk_indices.reshape(-1)
    # Corresponding token indices: [T*K]
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    # Corresponding slot indices: [T*K]
    slot_ids = torch.arange(K, device=device).unsqueeze(0).expand(T, K).reshape(-1)

    # Sort by expert index
    sort_order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sort_order]
    sorted_token_ids = token_ids[sort_order]
    sorted_slot_ids  = slot_ids[sort_order]

    # Per-expert counts and offsets
    expert_counts = torch.bincount(sorted_experts, minlength=N_ROUTED_EXPERTS)
    expert_offsets = torch.zeros(N_ROUTED_EXPERTS + 1, dtype=torch.long, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    # Gather sorted inputs
    sorted_hidden = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights = topk_weights[sorted_token_ids, sorted_slot_ids]

    max_tokens_per_expert = int(expert_counts.max().item())
    if max_tokens_per_expert == 0:
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(topk_weights),
            torch.zeros_like(gate_weights),
            torch.zeros_like(up_weights),
            torch.zeros_like(down_weights),
        )

    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE
    M = MOE_INTERMEDIATE_SIZE
    B = max_tokens_per_expert
    N = T * K

    # Compute local positions within each expert's chunk
    ones = torch.ones(N, dtype=torch.long, device=device)
    cumsum_all = torch.cumsum(ones, dim=0) - 1
    group_starts = expert_offsets[:-1][sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_expert_idx = sorted_experts * B + expert_local_pos

    # Build padded tensors [E, B, H]
    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_expert_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)

    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_expert_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_expert_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)

    # Forward recomputation
    gate_pre_act   = torch.bmm(padded_hidden, gate_weights.transpose(1, 2))   # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_weights.transpose(1, 2))     # [E, B, M]
    gate_activated = F.silu(gate_pre_act)
    intermediate   = gate_activated * up_output                                # [E, B, M]

    # grad_topk_weights
    expert_output    = torch.bmm(intermediate, down_weights.transpose(1, 2))  # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)           # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk = grad_topk_w_flat.view(-1)[padded_expert_idx]
    flat_out_idx   = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # Grad through down projection
    scaled_grad_out   = padded_grad_out * padded_weights.unsqueeze(2)          # [E, B, H]
    grad_down_weights = torch.bmm(scaled_grad_out.transpose(1, 2), intermediate)  # [E, H, M]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)               # [E, B, M]

    # Grad through SwiGLU
    grad_up_output      = grad_intermediate * gate_activated                   # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                           # [E, B, M]

    # Weight gradients and grad_hidden
    grad_gate_weights = torch.bmm(grad_gate_pre_act.transpose(1, 2), padded_hidden)  # [E, M, H]
    grad_hidden_gate  = torch.bmm(grad_gate_pre_act, gate_weights)                   # [E, B, H]

    grad_up_weights   = torch.bmm(grad_up_output.transpose(1, 2), padded_hidden)     # [E, M, H]
    grad_hidden_up    = torch.bmm(grad_up_output, up_weights)                        # [E, B, H]

    # Scatter grad_hidden
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).view(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_expert_idx]
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
