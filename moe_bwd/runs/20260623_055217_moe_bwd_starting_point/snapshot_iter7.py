"""
MoE backward pass — uses torch._grouped_mm for variable-length grouped GEMMs.

torch._grouped_mm accepts flat sorted token layout + per-expert offsets,
computing all expert projections simultaneously with no padding waste.
Falls back to padded-bmm if _grouped_mm is unavailable.

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

# Check if _grouped_mm is available
_has_grouped_mm = hasattr(torch, '_grouped_mm')


def _grouped_mm_wrapper(A, B, offs):
    """
    Compute grouped matmul: for each expert e,
      out[offs[e]:offs[e+1]] = A[offs[e]:offs[e+1]] @ B[e].T
    A: [N, K_in]  float32/bf16
    B: [E, K_out, K_in]  — each B[e] is [K_out, K_in], so A @ B[e].T → [n_e, K_out]
    offs: [E+1] int32 on CPU (required by _grouped_mm)
    Returns: [N, K_out]
    """
    return torch._grouped_mm(A, B, offs)


def _bmm_fallback(sorted_tokens, W, expert_offsets_cpu, E, B, padded_idx):
    """
    Fallback: padded bmm approach.
    sorted_tokens: [N, K_in]
    W: [E, K_out, K_in]
    Returns: [N, K_out]  in sorted expert order
    """
    N, K_in = sorted_tokens.shape
    K_out = W.shape[1]
    device = sorted_tokens.device
    dtype = sorted_tokens.dtype

    padded = torch.zeros(E * B, K_in, dtype=dtype, device=device)
    padded[padded_idx] = sorted_tokens
    padded = padded.view(E, B, K_in)

    out_padded = torch.bmm(padded, W.transpose(1, 2))  # [E, B, K_out]
    out_flat = out_padded.view(E * B, K_out)[padded_idx]  # [N, K_out]
    return out_flat


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    T, K  = topk_indices.shape
    device = hidden_states.device
    dtype  = hidden_states.dtype
    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE
    M = MOE_INTERMEDIATE_SIZE
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
    expert_offsets_gpu = torch.zeros(E + 1, dtype=torch.int32, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0).to(torch.int32)
    # _grouped_mm needs offsets on CPU
    expert_offsets_cpu = expert_offsets_gpu.cpu()

    max_tokens_per_expert = int(expert_counts.max().item())
    if max_tokens_per_expert == 0:
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(topk_weights),
            torch.zeros_like(gate_weights),
            torch.zeros_like(up_weights),
            torch.zeros_like(down_weights),
        )

    B_pad = max_tokens_per_expert  # padded batch size (for fallback)

    # -----------------------------------------------------------------------
    # Step 2: Gather sorted inputs
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids].contiguous()   # [N, H]
    sorted_grad_out = grad_output[sorted_token_ids].contiguous()     # [N, H]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Compute padded indices (needed for fallback + scatter ops)
    # -----------------------------------------------------------------------
    cumsum_all       = torch.arange(N, device=device, dtype=torch.long)
    group_starts     = expert_offsets_gpu[:-1].long()[sorted_experts]
    expert_local_pos = cumsum_all - group_starts
    padded_idx       = sorted_experts * B_pad + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 4: GEMM dispatch — try _grouped_mm, fall back to padded bmm
    # -----------------------------------------------------------------------
    if _has_grouped_mm:
        # _grouped_mm: A[N, K_in] @ B[E, K_out, K_in]^T → [N, K_out]
        # gate_weights: [E, M, H] → K_out=M, K_in=H
        gate_pre_act = _grouped_mm_wrapper(sorted_hidden, gate_weights, expert_offsets_cpu)   # [N, M]
        up_output    = _grouped_mm_wrapper(sorted_hidden, up_weights,   expert_offsets_cpu)   # [N, M]
    else:
        # Fallback: build padded [E, B, H] and use bmm
        padded_hidden = torch.zeros(E * B_pad, H, dtype=dtype, device=device)
        padded_hidden[padded_idx] = sorted_hidden
        padded_hidden = padded_hidden.view(E, B_pad, H)
        gate_pre_act_pad = torch.bmm(padded_hidden, gate_weights.transpose(1, 2))  # [E, B, M]
        up_output_pad    = torch.bmm(padded_hidden, up_weights.transpose(1, 2))    # [E, B, M]
        gate_pre_act = gate_pre_act_pad.view(E * B_pad, M)[padded_idx]             # [N, M]
        up_output    = up_output_pad.view(E * B_pad, M)[padded_idx]                # [N, M]

    # SiLU and SwiGLU in flat sorted layout
    gate_activated = F.silu(gate_pre_act)        # [N, M]
    intermediate   = gate_activated * up_output  # [N, M]

    # -----------------------------------------------------------------------
    # Step 5: grad_topk_weights
    # expert_output[i] = intermediate[i] @ down_weights[e]^T → [N, H]
    # down_weights: [E, H, M] → K_out=H, K_in=M
    # -----------------------------------------------------------------------
    if _has_grouped_mm:
        # down_weights is [E, H, M]: treat as W[E, K_out=H, K_in=M]
        expert_output = _grouped_mm_wrapper(intermediate, down_weights, expert_offsets_cpu)  # [N, H]
    else:
        padded_inter = torch.zeros(E * B_pad, M, dtype=dtype, device=device)
        padded_inter[padded_idx] = intermediate
        padded_inter = padded_inter.view(E, B_pad, M)
        expert_output_pad = torch.bmm(padded_inter, down_weights.transpose(1, 2))  # [E, B, H]
        expert_output = expert_output_pad.view(E * B_pad, H)[padded_idx]

    grad_topk_w_flat  = (sorted_grad_out * expert_output).sum(dim=1)  # [N]
    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_out_idx      = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, grad_topk_w_flat)

    # -----------------------------------------------------------------------
    # Step 6: Grad through down projection
    # scaled_grad_out[i] = sorted_grad_out[i] * sorted_weights[i]
    # grad_intermediate[i] = scaled_grad_out[i] @ down_weights[e]  → [N, M]
    # down_weights: [E, H, M] → for A[N,H] @ W[E,H,M] → need W^T shape [E,M,H]
    # Use down_weights.transpose(1,2): [E, M, H] with K_out=M, K_in=H
    # -----------------------------------------------------------------------
    scaled_grad_out = (sorted_grad_out * sorted_weights.unsqueeze(1)).contiguous()  # [N, H]

    down_weights_t = down_weights.transpose(1, 2).contiguous()  # [E, M, H]

    if _has_grouped_mm:
        grad_intermediate = _grouped_mm_wrapper(scaled_grad_out, down_weights_t, expert_offsets_cpu)  # [N, M]
    else:
        padded_sgo = torch.zeros(E * B_pad, H, dtype=dtype, device=device)
        padded_sgo[padded_idx] = scaled_grad_out
        padded_sgo = padded_sgo.view(E, B_pad, H)
        grad_inter_pad = torch.bmm(padded_sgo, down_weights)  # [E, B, M]
        grad_intermediate = grad_inter_pad.view(E * B_pad, M)[padded_idx]

    # -----------------------------------------------------------------------
    # Step 7: Grad through SwiGLU (element-wise, flat layout)
    # -----------------------------------------------------------------------
    grad_up_output      = (grad_intermediate * gate_activated).contiguous()   # [N, M]
    grad_gate_activated = (grad_intermediate * up_output).contiguous()        # [N, M]
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = (grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )).contiguous()                                                            # [N, M]

    # -----------------------------------------------------------------------
    # Step 8: grad_hidden
    # grad_hidden_gate[i] = grad_gate_pre_act[i] @ gate_weights[e]  → [N, H]
    # gate_weights: [E, M, H] → K_out=H, K_in=M  ✓
    # grad_hidden_up[i]   = grad_up_output[i]    @ up_weights[e]    → [N, H]
    # -----------------------------------------------------------------------
    if _has_grouped_mm:
        grad_hidden_gate = _grouped_mm_wrapper(grad_gate_pre_act, gate_weights, expert_offsets_cpu)  # [N, H]
        grad_hidden_up   = _grouped_mm_wrapper(grad_up_output,    up_weights,   expert_offsets_cpu)  # [N, H]
    else:
        padded_ggpa = torch.zeros(E * B_pad, M, dtype=dtype, device=device)
        padded_ggpa[padded_idx] = grad_gate_pre_act
        padded_ggpa = padded_ggpa.view(E, B_pad, M)
        ghg_pad = torch.bmm(padded_ggpa, gate_weights)  # [E, B, H]
        grad_hidden_gate = ghg_pad.view(E * B_pad, H)[padded_idx]

        padded_guo = torch.zeros(E * B_pad, M, dtype=dtype, device=device)
        padded_guo[padded_idx] = grad_up_output
        padded_guo = padded_guo.view(E, B_pad, M)
        ghu_pad = torch.bmm(padded_guo, up_weights)  # [E, B, H]
        grad_hidden_up = ghu_pad.view(E * B_pad, H)[padded_idx]

    grad_hidden_combined = grad_hidden_gate + grad_hidden_up  # [N, H]
    grad_hidden_states   = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, grad_hidden_combined)

    # -----------------------------------------------------------------------
    # Step 9: Weight gradients via grouped outer products
    # grad_down_weights[e] = scaled_grad_out[e_toks]^T @ intermediate[e_toks]  → [E, H, M]
    # grad_gate_weights[e] = grad_gate_pre_act[e_toks]^T @ sorted_hidden[e_toks] → [E, M, H]
    # grad_up_weights[e]   = grad_up_output[e_toks]^T   @ sorted_hidden[e_toks] → [E, M, H]
    #
    # _grouped_mm computes A @ B^T grouped by expert.
    # For A^T @ B we need to swap: _grouped_mm(B^T, A^T) ... not directly.
    # Instead use padded bmm for weight grads (these are already fast with bmm).
    # -----------------------------------------------------------------------
    # Build padded versions of the flat tensors for weight grad bmm
    def make_padded(flat_tensor, K_dim):
        """flat_tensor: [N, K_dim] → padded [E, B_pad, K_dim]"""
        padded = torch.zeros(E * B_pad, K_dim, dtype=dtype, device=device)
        padded[padded_idx] = flat_tensor
        return padded.view(E, B_pad, K_dim)

    p_scaled_grad_out  = make_padded(scaled_grad_out,    H)   # [E, B, H]
    p_intermediate     = make_padded(intermediate,       M)   # [E, B, M]
    p_grad_gate_pre    = make_padded(grad_gate_pre_act,  M)   # [E, B, M]
    p_grad_up_output   = make_padded(grad_up_output,     M)   # [E, B, M]
    p_sorted_hidden    = make_padded(sorted_hidden,      H)   # [E, B, H]

    grad_down_weights = torch.bmm(p_scaled_grad_out.transpose(1, 2), p_intermediate)  # [E, H, M]
    grad_gate_weights = torch.bmm(p_grad_gate_pre.transpose(1, 2),   p_sorted_hidden) # [E, M, H]
    grad_up_weights   = torch.bmm(p_grad_up_output.transpose(1, 2),  p_sorted_hidden) # [E, M, H]

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
