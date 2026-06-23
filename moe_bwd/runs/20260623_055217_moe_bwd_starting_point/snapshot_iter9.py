"""
MoE backward pass — algebraically optimized padded-bmm with GEMM fusion.

Reduces bmm calls from 8 to 5 by:
1. Fusing gate+up forward projections into one bmm via concatenated weights [E, 2M, H]
2. Fusing grad_hidden (gate + up paths) into one bmm via concatenated operands [E, B, 2M]
3. Fusing grad_gate_weights + grad_up_weights into one bmm via 2E-batch trick

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

    B = max_tokens_per_expert

    # -----------------------------------------------------------------------
    # Step 2: Compute padded indices
    # -----------------------------------------------------------------------
    cumsum_all       = torch.arange(N, device=device, dtype=torch.long)
    group_starts     = expert_offsets[:-1][sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Gather sorted inputs and build padded tensors
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)

    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)  # [E, B]

    # -----------------------------------------------------------------------
    # FUSION 1: Combined gate+up forward projection
    # Instead of 2 bmms: hidden @ gate^T and hidden @ up^T,
    # concatenate weights along output dim: combined_wu [E, 2M, H]
    # One bmm: padded_hidden @ combined_wu^T → [E, B, 2M]
    # Then split to get gate_pre_act [E, B, M] and up_output [E, B, M]
    # -----------------------------------------------------------------------
    combined_wu = torch.cat([gate_weights, up_weights], dim=1)  # [E, 2M, H]
    # bmm 1/5: forward projection (replaces 2 bmms)
    gate_up_combined = torch.bmm(padded_hidden, combined_wu.transpose(1, 2))  # [E, B, 2M]
    gate_pre_act = gate_up_combined[:, :, :M]   # [E, B, M]  — view, no copy
    up_output    = gate_up_combined[:, :, M:]   # [E, B, M]  — view, no copy

    gate_activated = F.silu(gate_pre_act)        # [E, B, M]
    intermediate   = gate_activated * up_output  # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 4: grad_topk_weights
    # expert_output [E, B, H] = intermediate @ down_weights^T
    # bmm 2/5
    # -----------------------------------------------------------------------
    expert_output    = torch.bmm(intermediate, down_weights.transpose(1, 2))  # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)           # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk    = grad_topk_w_flat.view(-1)[padded_idx]
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 5: Grad through down projection
    # scaled_grad_out [E, B, H] = padded_grad_out * routing_weights
    # grad_down_weights [E, H, M] = scaled_grad_out^T @ intermediate  — bmm 3/5
    # grad_intermediate [E, B, M] = scaled_grad_out @ down_weights     — shares same bmm call
    #   (these two cannot be fused into one bmm, but use same scaled_grad_out)
    # -----------------------------------------------------------------------
    scaled_grad_out = padded_grad_out * padded_weights.unsqueeze(2)  # [E, B, H]

    # bmm 3/5: grad_down_weights
    grad_down_weights = torch.bmm(scaled_grad_out.transpose(1, 2), intermediate)  # [E, H, M]

    # bmm 4/5: grad_intermediate  (down_weights: [E, H, M])
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)    # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 6: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated         # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output              # [E, B, M]
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                 # [E, B, M]

    # -----------------------------------------------------------------------
    # FUSION 2: Fuse grad_hidden gate + up paths
    # grad_hidden_gate = grad_gate_pre_act @ gate_weights  [E, B, H]
    # grad_hidden_up   = grad_up_output    @ up_weights    [E, B, H]
    # Concatenate along M dim: [grad_gate_pre_act | grad_up_output] [E, B, 2M]
    # combined_wu is already [E, 2M, H]
    # bmm 5/5: [E, B, 2M] @ [E, 2M, H] → [E, B, H]  (replaces 2 bmms)
    # -----------------------------------------------------------------------
    grad_gate_up_concat = torch.cat([grad_gate_pre_act, grad_up_output], dim=2)  # [E, B, 2M]
    grad_hidden = torch.bmm(grad_gate_up_concat, combined_wu)                    # [E, B, H]

    # Scatter grad_hidden
    valid_grad_hidden  = grad_hidden.view(E * B, H)[padded_idx]   # [N, H]
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    # -----------------------------------------------------------------------
    # FUSION 3: Fuse grad_gate_weights + grad_up_weights weight grads
    # grad_gate_weights [E, M, H] = grad_gate_pre_act^T @ padded_hidden
    # grad_up_weights   [E, M, H] = grad_up_output^T    @ padded_hidden
    #
    # Stack along expert dim: treat as 2E experts each of size B
    # A_2E [2E, B, M]: [grad_gate_pre_act; grad_up_output] stacked on expert dim
    # B_2E [2E, B, H]: padded_hidden repeated twice
    # One bmm(A_2E.transpose(1,2), B_2E) → [2E, M, H]
    # Split → grad_gate_weights [:E], grad_up_weights [E:]
    # -----------------------------------------------------------------------
    A_2E = torch.cat([grad_gate_pre_act, grad_up_output], dim=0)  # [2E, B, M]
    B_2E = padded_hidden.expand(2, E, B, H).reshape(2 * E, B, H)  # [2E, B, H]
    grad_gate_up_weights = torch.bmm(A_2E.transpose(1, 2), B_2E)  # [2E, M, H]
    grad_gate_weights = grad_gate_up_weights[:E]   # [E, M, H]
    grad_up_weights   = grad_gate_up_weights[E:]   # [E, M, H]

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
