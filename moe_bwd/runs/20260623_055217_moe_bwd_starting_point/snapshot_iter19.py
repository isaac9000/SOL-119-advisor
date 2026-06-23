"""
MoE backward pass — per-expert Python loop with flat sorted slices + torch.mm
for forward GEMMs; padded-bmm for weight gradient outer products.

For forward-pass GEMMs (gate, up, down projections), iterate over active experts
with contiguous flat sorted slices and torch.mm. This avoids:
- Padding waste of the [E, B, H] bmm approach
- OOM from expanding W[expert_ids] to [N, K_out, K_in]
- Triton correctness issues

For weight gradient outer products (A^T @ B), use the proven padded-bmm.

CUDA kernels from the mm calls queue up asynchronously and can overlap.

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

    # Move to CPU for Python loop indexing (one blocking transfer, but small)
    expert_counts_cpu  = expert_counts.tolist()
    expert_offsets_cpu = expert_offsets.tolist()

    max_tokens_per_expert = max(expert_counts_cpu)
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
    # Step 2: Gather sorted inputs (contiguous for fast slicing)
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids].contiguous()  # [N, H]
    sorted_grad_out = grad_output[sorted_token_ids].contiguous()    # [N, H]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Per-expert forward GEMMs using flat sorted slices + torch.mm
    # Pre-allocate output buffers [N, M] for gate_pre_act and up_output
    # -----------------------------------------------------------------------
    gate_pre_act_flat = torch.empty(N, M, dtype=dtype, device=device)
    up_output_flat    = torch.empty(N, M, dtype=dtype, device=device)

    for e in range(E):
        cnt = expert_counts_cpu[e]
        if cnt == 0:
            continue
        s = expert_offsets_cpu[e]
        end = s + cnt

        h_slice = sorted_hidden[s:end]   # [cnt, H]  — contiguous slice
        gw = gate_weights[e]             # [M, H]    — view, no copy
        uw = up_weights[e]               # [M, H]    — view, no copy

        # torch.mm([cnt, H] @ [H, M]) = [cnt, M]
        # Both are stored in correct layout for cuBLAS: A row-major, B col-major via .t()
        torch.mm(h_slice, gw.t(), out=gate_pre_act_flat[s:end])
        torch.mm(h_slice, uw.t(), out=up_output_flat[s:end])

    # SwiGLU activations (flat layout, no padding)
    gate_activated_flat = F.silu(gate_pre_act_flat)         # [N, M]
    intermediate_flat   = gate_activated_flat * up_output_flat  # [N, M]

    # -----------------------------------------------------------------------
    # Step 4: grad_topk_weights via per-expert mm
    # expert_output[i] = intermediate[i] @ down_weights[e]^T → [N, H]
    # grad_topk_w[i] = sum_h(sorted_grad_out[i] * expert_output[i])
    # -----------------------------------------------------------------------
    expert_output_flat = torch.empty(N, H, dtype=dtype, device=device)

    for e in range(E):
        cnt = expert_counts_cpu[e]
        if cnt == 0:
            continue
        s = expert_offsets_cpu[e]
        end = s + cnt
        dw = down_weights[e]   # [H, M]
        # intermediate [cnt, M] @ [M, H] = [cnt, H]
        torch.mm(intermediate_flat[s:end], dw.t(), out=expert_output_flat[s:end])

    grad_topk_w_flat  = (sorted_grad_out * expert_output_flat).sum(dim=1)  # [N]
    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, grad_topk_w_flat)

    # -----------------------------------------------------------------------
    # Step 5: Grad through down projection
    # scaled_grad_out[i] = sorted_grad_out[i] * sorted_weights[i]
    # grad_intermediate[i] = scaled_grad_out[i] @ down_weights[e]  → [N, M]
    # -----------------------------------------------------------------------
    scaled_grad_out   = (sorted_grad_out * sorted_weights.unsqueeze(1)).contiguous()  # [N, H]
    grad_intermediate_flat = torch.empty(N, M, dtype=dtype, device=device)

    for e in range(E):
        cnt = expert_counts_cpu[e]
        if cnt == 0:
            continue
        s = expert_offsets_cpu[e]
        end = s + cnt
        dw = down_weights[e]   # [H, M]
        # scaled_grad [cnt, H] @ [H, M] = [cnt, M]
        torch.mm(scaled_grad_out[s:end], dw, out=grad_intermediate_flat[s:end])

    # -----------------------------------------------------------------------
    # Step 6: Grad through SwiGLU (flat layout)
    # -----------------------------------------------------------------------
    grad_up_output_flat      = (grad_intermediate_flat * gate_activated_flat).contiguous()   # [N, M]
    grad_gate_activated_flat = (grad_intermediate_flat * up_output_flat).contiguous()        # [N, M]
    sigmoid_gate_flat        = torch.sigmoid(gate_pre_act_flat)
    grad_gate_pre_act_flat   = (grad_gate_activated_flat * (
        gate_activated_flat + sigmoid_gate_flat * (1.0 - gate_activated_flat)
    )).contiguous()                                                                            # [N, M]

    # -----------------------------------------------------------------------
    # Step 7: grad_hidden via per-expert mm
    # grad_hidden_gate[i] = grad_gate_pre_act[i] @ gate_weights[e]  → [N, H]
    # grad_hidden_up[i]   = grad_up_output[i]    @ up_weights[e]    → [N, H]
    # -----------------------------------------------------------------------
    grad_hidden_flat = torch.zeros(N, H, dtype=dtype, device=device)

    for e in range(E):
        cnt = expert_counts_cpu[e]
        if cnt == 0:
            continue
        s = expert_offsets_cpu[e]
        end = s + cnt
        gw = gate_weights[e]   # [M, H]
        uw = up_weights[e]     # [M, H]

        # grad_gate_pre_act [cnt, M] @ [M, H] = [cnt, H]
        # grad_up_output    [cnt, M] @ [M, H] = [cnt, H]
        # Combine: (ggpa + guo) @ W via addmm for efficiency
        gh = torch.mm(grad_gate_pre_act_flat[s:end], gw) + \
             torch.mm(grad_up_output_flat[s:end], uw)    # [cnt, H]
        grad_hidden_flat[s:end] = gh

    # Scatter grad_hidden back (multiple flat entries per token → index_add_)
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, grad_hidden_flat)

    # -----------------------------------------------------------------------
    # Step 8: Weight gradient outer products — padded-bmm (proven correct)
    # grad_gate_weights[e] = grad_gate_pre_act[e_toks]^T @ sorted_hidden[e_toks]
    # grad_up_weights[e]   = grad_up_output[e_toks]^T   @ sorted_hidden[e_toks]
    # grad_down_weights[e] = scaled_grad_out[e_toks]^T  @ intermediate[e_toks]
    # -----------------------------------------------------------------------
    # Compute padded indices for the padded-bmm scatter
    cumsum_all       = torch.arange(N, device=device, dtype=torch.long)
    group_starts     = expert_offsets[:-1][sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_idx       = sorted_experts * B + expert_local_pos  # [N]

    def make_padded(flat_tensor, dim):
        """Scatter flat [N, dim] sorted tensor → padded [E, B, dim]."""
        buf = torch.zeros(E * B, dim, dtype=dtype, device=device)
        buf[padded_idx] = flat_tensor
        return buf.view(E, B, dim)

    p_hidden    = make_padded(sorted_hidden,         H)  # [E, B, H]
    p_ggpa      = make_padded(grad_gate_pre_act_flat, M)  # [E, B, M]
    p_guo       = make_padded(grad_up_output_flat,    M)  # [E, B, M]
    p_sgo       = make_padded(scaled_grad_out,        H)  # [E, B, H]
    p_inter     = make_padded(intermediate_flat,      M)  # [E, B, M]

    grad_gate_weights = torch.bmm(p_ggpa.transpose(1, 2), p_hidden)  # [E, M, H]
    grad_up_weights   = torch.bmm(p_guo.transpose(1, 2),  p_hidden)  # [E, M, H]
    grad_down_weights = torch.bmm(p_sgo.transpose(1, 2),  p_inter)   # [E, H, M]

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
