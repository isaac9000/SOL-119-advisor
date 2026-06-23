"""
Vectorized MoE backward pass — loop-free PyTorch implementation.

Eliminates the Python-level per-expert loop by:
1. Flattening topk_indices to [T*8] and sorting by expert to group tokens per expert
2. Using expert-sorted layout for batched GEMMs via torch.bmm with padding
3. Scattering results back with index_add_ in batch form

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

    grad_output     [num_tokens, 4096]              float32
    hidden_states   [num_tokens, 4096]              float32
    topk_indices    [num_tokens, 8]                 int64   (expert indices, unique per token)
    topk_weights    [num_tokens, 8]                 float32 (softmax-normalized routing weights)
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

HIDDEN_SIZE           = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS      = 256
NUM_EXPERTS_PER_TOK   = 8


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    num_tokens = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # -----------------------------------------------------------------------
    # Step 1: Build expert-sorted index layout
    # topk_indices: [T, 8] → flatten to [T*8]
    # Each element is an (token, slot) pair assigned to one expert.
    # -----------------------------------------------------------------------
    T, K = topk_indices.shape  # T=num_tokens, K=8

    # Flat expert assignments: [T*K]
    flat_experts = topk_indices.reshape(-1)          # [T*K]
    # Corresponding token indices: [T*K]
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)  # [T*K]
    # Corresponding slot indices: [T*K]
    slot_ids = torch.arange(K, device=device).unsqueeze(0).expand(T, K).reshape(-1)   # [T*K]

    # Sort by expert index → groups all (token, slot) pairs for the same expert contiguously
    sort_order = torch.argsort(flat_experts, stable=True)  # [T*K]
    sorted_experts = flat_experts[sort_order]              # [T*K], sorted
    sorted_token_ids = token_ids[sort_order]               # [T*K]
    sorted_slot_ids  = slot_ids[sort_order]                # [T*K]

    # Compute per-expert counts and offsets using bincount
    expert_counts = torch.bincount(sorted_experts, minlength=N_ROUTED_EXPERTS)  # [256]
    expert_offsets = torch.zeros(N_ROUTED_EXPERTS + 1, dtype=torch.long, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    # -----------------------------------------------------------------------
    # Step 2: Gather sorted hidden states, grad_output, and routing weights
    # -----------------------------------------------------------------------
    # Gather hidden states for all (token, slot) pairs in sorted order: [T*K, H]
    sorted_hidden = hidden_states[sorted_token_ids]       # [T*K, 4096]
    sorted_grad_out = grad_output[sorted_token_ids]       # [T*K, 4096]
    sorted_weights = topk_weights[sorted_token_ids, sorted_slot_ids]  # [T*K]

    # -----------------------------------------------------------------------
    # Step 3: Pad to uniform expert batch size and run batched GEMMs
    # -----------------------------------------------------------------------
    max_tokens_per_expert = int(expert_counts.max().item())
    if max_tokens_per_expert == 0:
        # Degenerate case: no tokens
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(topk_weights),
            torch.zeros_like(gate_weights),
            torch.zeros_like(up_weights),
            torch.zeros_like(down_weights),
        )

    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE           # 4096
    M = MOE_INTERMEDIATE_SIZE  # 2048
    B = max_tokens_per_expert

    # Build padded tensors: [E, B, H] for hidden and grad_out; [E, B] for weights
    # We'll scatter the flat sorted data into padded experts
    # Position within each expert's local batch
    # local_positions[i] = position of element i within its expert's chunk
    # We can compute this as: cumsum within expert groups
    # One approach: use arange trick
    expert_local_pos = torch.zeros(T * K, dtype=torch.long, device=device)
    # For each sorted element, its local position = global_index - expert_offset[expert]
    # We can compute via: cumcount per group
    # Trick: within each expert group, assign 0,1,2,...
    # Build using a counter reset at each expert boundary
    ones = torch.ones(T * K, dtype=torch.long, device=device)
    # cumsum then subtract cumsum at group starts
    cumsum_all = torch.cumsum(ones, dim=0) - 1  # [0, 1, 2, ..., T*K-1]
    # Offset for each element = expert_offsets[sorted_expert]
    group_starts = expert_offsets[:-1][sorted_experts]   # [T*K]
    expert_local_pos = cumsum_all - group_starts          # local index within expert

    # Build flat index into padded [E, B] layout
    padded_expert_idx = sorted_experts * B + expert_local_pos  # [T*K]

    # Padded hidden states: [E*B, H] → reshape to [E, B, H]
    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_expert_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)

    # Padded grad_out: [E*B, H]
    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_expert_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    # Padded routing weights: [E*B]
    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_expert_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)  # [E, B]

    # -----------------------------------------------------------------------
    # Step 4: Forward recomputation via batched matmul
    # gate_pre_act [E, B, M] = padded_hidden [E, B, H] @ gate_weights^T [E, H, M]^T
    # = padded_hidden @ gate_weights.transpose(1,2)  (since gate_weights=[E,M,H])
    # -----------------------------------------------------------------------
    # gate_weights: [E, M, H], so gate_weights.transpose(1,2): [E, H, M]
    gate_pre_act   = torch.bmm(padded_hidden, gate_weights.transpose(1, 2))   # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_weights.transpose(1, 2))     # [E, B, M]
    gate_activated = F.silu(gate_pre_act)                                      # [E, B, M]
    intermediate   = gate_activated * up_output                                # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 5: Compute grad_topk_weights
    # expert_output [E, B, H] = intermediate @ down_weights^T
    # down_weights: [E, H, M], so down_weights.transpose(1,2): [E, M, H]
    # expert_output = intermediate @ down_weights.transpose(1,2)  → [E, B, H]
    # grad_topk_weights_flat [E, B] = sum_h(padded_grad_out * expert_output)
    # -----------------------------------------------------------------------
    expert_output = torch.bmm(intermediate, down_weights.transpose(1, 2))     # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)           # [E, B]

    # Scatter grad_topk_weights back
    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    # flat grad_topk_w values in sorted order
    flat_grad_topk = grad_topk_w_flat.view(-1)[padded_expert_idx]             # [T*K]
    # Scatter into [T, K]
    flat_out_idx = sorted_token_ids * K + sorted_slot_ids                     # [T*K]
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 6: Grad through down projection
    # scaled_grad_out [E, B, H] = padded_grad_out * routing_weights
    # grad_down_weights [E, H, M] = scaled_grad_out^T @ intermediate
    #   = bmm(scaled_grad_out.transpose(1,2), intermediate)
    # -----------------------------------------------------------------------
    scaled_grad_out = padded_grad_out * padded_weights.unsqueeze(2)           # [E, B, H]
    grad_down_weights = torch.bmm(
        scaled_grad_out.transpose(1, 2), intermediate
    )                                                                          # [E, H, M]

    # grad_intermediate [E, B, M] = scaled_grad_out @ down_weights
    # down_weights: [E, H, M], need [E, M, H]^T = down_weights.transpose(1,2) is [E,M,H]
    # grad_intermediate = scaled_grad_out @ down_weights  → but down_weights is [E,H,M]
    # We want: [E,B,H] @ [E,H,M] = [E,B,M]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)              # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 7: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated                   # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output                        # [E, B, M]

    sigmoid_gate = torch.sigmoid(gate_pre_act)                                 # [E, B, M]
    grad_gate_pre_act = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                           # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 8: Grad through gate and up projections
    # grad_gate_weights [E, M, H] = grad_gate_pre_act^T @ padded_hidden
    #   = bmm(grad_gate_pre_act.transpose(1,2), padded_hidden)
    # grad_hidden_gate [E, B, H] = grad_gate_pre_act @ gate_weights
    # -----------------------------------------------------------------------
    grad_gate_weights = torch.bmm(
        grad_gate_pre_act.transpose(1, 2), padded_hidden
    )                                                                          # [E, M, H]
    grad_hidden_gate = torch.bmm(grad_gate_pre_act, gate_weights)             # [E, B, H]

    grad_up_weights = torch.bmm(
        grad_up_output.transpose(1, 2), padded_hidden
    )                                                                          # [E, M, H]
    grad_hidden_up = torch.bmm(grad_up_output, up_weights)                   # [E, B, H]

    # -----------------------------------------------------------------------
    # Step 9: Scatter grad_hidden back
    # grad_hidden_expert [E, B, H] = grad_hidden_gate + grad_hidden_up
    # Flatten to [E*B, H], pick valid entries, index_add_ into [T, H]
    # -----------------------------------------------------------------------
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).view(E * B, H)  # [E*B, H]

    # Only entries corresponding to actual (not padded) token-expert pairs
    valid_grad_hidden = grad_hidden_expert[padded_expert_idx]                 # [T*K, H]

    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
