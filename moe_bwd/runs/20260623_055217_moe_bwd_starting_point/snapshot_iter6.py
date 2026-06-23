"""
MoE backward pass — padded bmm for GEMMs + Triton for weight gradient outer products.

The weight gradient kernels use the correct Triton pattern:
  - Loop bound is a tl.constexpr (max_tokens rounded up)
  - Runtime expert range enforced via mask inside the loop
  - No runtime values in range() calls

All GEMM operations use PyTorch bmm (proven correct at 85ms).
Triton is used only for the 3 outer-product accumulations (grad_gate, grad_up, grad_down).

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

Returns:
    grad_hidden_states, grad_topk_weights, grad_gate_weights,
    grad_up_weights, grad_down_weights
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HIDDEN_SIZE           = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS      = 256
NUM_EXPERTS_PER_TOK   = 8


# ---------------------------------------------------------------------------
# Triton kernel: Grouped outer-product weight gradient accumulation
#
# Computes: GradW[expert, k1_tile, k2_tile] += A[expert_tokens, K1]^T @ B[expert_tokens, K2]
#
# Grid: (E * num_K1_tiles, num_K2_tiles)
# Each program is responsible for exactly one (expert, k1_tile, k2_tile) output tile.
# The loop iterates over a fixed constexpr number of token chunks (MAX_LOOP_ITERS),
# with masking to handle the variable expert size safely.
# ---------------------------------------------------------------------------
@triton.jit
def grouped_outer_kernel(
    A_ptr,         # [N_total, K1]  contiguous
    B_ptr,         # [N_total, K2]  contiguous
    W_ptr,         # [E, K1, K2]   output (pre-zeroed)
    offsets_ptr,   # [E+1]  int32  per-expert start offsets in sorted layout
    N_total   : tl.constexpr,
    K1        : tl.constexpr,
    K2        : tl.constexpr,
    num_K1_tiles     : tl.constexpr,
    BLOCK_N   : tl.constexpr,
    BLOCK_K1  : tl.constexpr,
    BLOCK_K2  : tl.constexpr,
    MAX_LOOP_ITERS : tl.constexpr,  # ceil(max_tokens_per_expert / BLOCK_N), constexpr
):
    pid_0  = tl.program_id(0)
    pid_k2 = tl.program_id(1)

    expert_id = pid_0 // num_K1_tiles
    k1_tile   = pid_0 %  num_K1_tiles

    # K1 tile offsets
    k1_start = k1_tile * BLOCK_K1
    k1_offs  = k1_start + tl.arange(0, BLOCK_K1)
    mask_k1  = k1_offs < K1

    # K2 tile offsets
    k2_start = pid_k2 * BLOCK_K2
    k2_offs  = k2_start + tl.arange(0, BLOCK_K2)
    mask_k2  = k2_offs < K2

    # Expert token range (runtime values, only used for masking)
    e_start = tl.load(offsets_ptr + expert_id)
    e_end   = tl.load(offsets_ptr + expert_id + 1)

    acc = tl.zeros((BLOCK_K1, BLOCK_K2), dtype=tl.float32)

    # Loop over a FIXED constexpr number of iterations; mask handles variable expert size
    for i in range(MAX_LOOP_ITERS):
        n_base = e_start + i * BLOCK_N
        n_offs = n_base + tl.arange(0, BLOCK_N)
        # Mask: only process rows that belong to this expert and are in bounds
        mask_n = (n_offs >= e_start) & (n_offs < e_end) & (n_offs < N_total)

        # Load A: [BLOCK_N, BLOCK_K1]
        a_ptrs = A_ptr + n_offs[:, None] * K1 + k1_offs[None, :]
        a = tl.load(a_ptrs, mask=mask_n[:, None] & mask_k1[None, :], other=0.0)

        # Load B: [BLOCK_N, BLOCK_K2]
        b_ptrs = B_ptr + n_offs[:, None] * K2 + k2_offs[None, :]
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0.0)

        # Accumulate A^T @ B: [BLOCK_K1, BLOCK_K2]
        acc = tl.dot(tl.trans(a), b, acc)

    # Write output tile
    w_base = W_ptr + expert_id * K1 * K2
    w_ptrs = w_base + k1_offs[:, None] * K2 + k2_offs[None, :]
    tl.store(w_ptrs, acc, mask=mask_k1[:, None] & mask_k2[None, :])


def launch_grouped_outer(A, B, E, K1, K2, offsets, N_total, max_tokens):
    """
    Compute GradW[E, K1, K2] = A[N, K1]^T @ B[N, K2] grouped by expert.
    A and B must be contiguous float32 tensors of shape [N_total, K1] and [N_total, K2].
    offsets: [E+1] int32 tensor of per-expert start positions.
    """
    BLOCK_N  = 32
    BLOCK_K1 = 64
    BLOCK_K2 = 64

    MAX_LOOP_ITERS = triton.cdiv(max_tokens, BLOCK_N)
    num_K1_tiles   = triton.cdiv(K1, BLOCK_K1)
    num_K2_tiles   = triton.cdiv(K2, BLOCK_K2)

    GradW = torch.zeros(E, K1, K2, device=A.device, dtype=A.dtype)

    grid = (E * num_K1_tiles, num_K2_tiles)
    grouped_outer_kernel[grid](
        A, B, GradW, offsets,
        N_total, K1, K2,
        num_K1_tiles,
        BLOCK_N, BLOCK_K1, BLOCK_K2,
        MAX_LOOP_ITERS,
    )
    return GradW


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    T, K  = topk_indices.shape
    device = hidden_states.device
    dtype  = hidden_states.dtype
    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE
    M = MOE_INTERMEDIATE_SIZE

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
    expert_offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0).to(torch.int32)

    max_tokens_per_expert = int(expert_counts.max().item())
    if max_tokens_per_expert == 0:
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(topk_weights),
            torch.zeros_like(gate_weights),
            torch.zeros_like(up_weights),
            torch.zeros_like(down_weights),
        )

    N = T * K
    B = max_tokens_per_expert  # padded batch size

    # -----------------------------------------------------------------------
    # Step 2: Gather sorted inputs
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids]
    sorted_grad_out = grad_output[sorted_token_ids]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids]

    # -----------------------------------------------------------------------
    # Step 3: Compute local positions → padded indices
    # -----------------------------------------------------------------------
    cumsum_all    = torch.arange(N, device=device, dtype=torch.long)
    group_starts  = expert_offsets[:-1].long()[sorted_experts]
    expert_local_pos  = cumsum_all - group_starts
    padded_expert_idx = sorted_experts * B + expert_local_pos  # [N]

    # -----------------------------------------------------------------------
    # Step 4: Build padded tensors [E, B, *] — proven correct bmm approach
    # -----------------------------------------------------------------------
    padded_hidden = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_hidden[padded_expert_idx] = sorted_hidden
    padded_hidden = padded_hidden.view(E, B, H)

    padded_grad_out = torch.zeros(E * B, H, dtype=dtype, device=device)
    padded_grad_out[padded_expert_idx] = sorted_grad_out
    padded_grad_out = padded_grad_out.view(E, B, H)

    padded_weights = torch.zeros(E * B, dtype=dtype, device=device)
    padded_weights[padded_expert_idx] = sorted_weights
    padded_weights = padded_weights.view(E, B)

    # -----------------------------------------------------------------------
    # Step 5: Forward recomputation — PyTorch bmm (correct, ~85ms baseline)
    # -----------------------------------------------------------------------
    gate_pre_act   = torch.bmm(padded_hidden, gate_weights.transpose(1, 2))  # [E, B, M]
    up_output      = torch.bmm(padded_hidden, up_weights.transpose(1, 2))    # [E, B, M]
    gate_activated = F.silu(gate_pre_act)
    intermediate   = gate_activated * up_output

    # -----------------------------------------------------------------------
    # Step 6: grad_topk_weights
    # -----------------------------------------------------------------------
    expert_output    = torch.bmm(intermediate, down_weights.transpose(1, 2))  # [E, B, H]
    grad_topk_w_flat = (padded_grad_out * expert_output).sum(dim=2)           # [E, B]

    grad_topk_weights = torch.zeros(T, K, dtype=dtype, device=device)
    flat_grad_topk = grad_topk_w_flat.view(-1)[padded_expert_idx]
    flat_out_idx   = sorted_token_ids * K + sorted_slot_ids
    grad_topk_weights.view(-1).scatter_(0, flat_out_idx, flat_grad_topk)

    # -----------------------------------------------------------------------
    # Step 7: Grad through down projection
    # -----------------------------------------------------------------------
    scaled_grad_out  = padded_grad_out * padded_weights.unsqueeze(2)          # [E, B, H]
    grad_intermediate = torch.bmm(scaled_grad_out, down_weights)              # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 8: Grad through SwiGLU
    # -----------------------------------------------------------------------
    grad_up_output      = grad_intermediate * gate_activated                   # [E, B, M]
    grad_gate_activated = grad_intermediate * up_output
    sigmoid_gate        = torch.sigmoid(gate_pre_act)
    grad_gate_pre_act   = grad_gate_activated * (
        gate_activated + sigmoid_gate * (1.0 - gate_activated)
    )                                                                           # [E, B, M]

    # -----------------------------------------------------------------------
    # Step 9: grad_hidden — PyTorch bmm
    # -----------------------------------------------------------------------
    grad_hidden_gate = torch.bmm(grad_gate_pre_act, gate_weights)             # [E, B, H]
    grad_hidden_up   = torch.bmm(grad_up_output,    up_weights)               # [E, B, H]
    grad_hidden_expert = (grad_hidden_gate + grad_hidden_up).view(E * B, H)
    valid_grad_hidden  = grad_hidden_expert[padded_expert_idx]
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_hidden_states.index_add_(0, sorted_token_ids, valid_grad_hidden)

    # -----------------------------------------------------------------------
    # Step 10: Weight gradients — Triton outer-product kernels
    #
    # grad_down_weights[e] = scaled_grad_out[e_toks]^T @ intermediate[e_toks]
    #   → [E, H, M]  from A=[N,H], B=[N,M]
    #
    # grad_gate_weights[e] = grad_gate_pre_act[e_toks]^T @ sorted_hidden[e_toks]
    #   → [E, M, H]  from A=[N,M], B=[N,H]
    #
    # grad_up_weights[e]   = grad_up_output[e_toks]^T @ sorted_hidden[e_toks]
    #   → [E, M, H]  from A=[N,M], B=[N,H]
    #
    # All inputs are in the flat sorted layout (not padded), so we use
    # the expert_offsets to find each expert's token range.
    # -----------------------------------------------------------------------

    # Flatten padded tensors back to sorted layout for Triton kernels
    # We use the flat sorted tensors (already in sorted expert order)
    scaled_grad_out_flat  = scaled_grad_out.view(E * B, H)[padded_expert_idx]  # ← wrong order
    # Actually we need: flat sorted layout [N, H] where N=T*K, not padded
    # Reconstruct from padded: recover sorted values via padded_expert_idx reverse
    # Easier: we already have the flat sorted tensors, just need to compute
    # the flat versions of scaled_grad_out and grad_gate/up_pre_act

    # scaled_grad_out in flat sorted layout: sorted_grad_out * sorted_weights
    scaled_grad_out_sorted = (sorted_grad_out * sorted_weights.unsqueeze(1)).contiguous()  # [N, H]

    # grad_gate_pre_act and grad_up_output: extract from padded [E,B,M] using padded_expert_idx
    flat_grad_gate_pre_act = grad_gate_pre_act.view(E * B, M)[padded_expert_idx].contiguous()  # [N, M]
    flat_grad_up_output    = grad_up_output.view(E * B, M)[padded_expert_idx].contiguous()     # [N, M]
    sorted_hidden_cont     = sorted_hidden.contiguous()                                         # [N, H]

    grad_down_weights = launch_grouped_outer(
        scaled_grad_out_sorted, intermediate.view(E * B, M)[padded_expert_idx].contiguous(),
        E, H, M, expert_offsets, N, max_tokens_per_expert
    )

    grad_gate_weights = launch_grouped_outer(
        flat_grad_gate_pre_act, sorted_hidden_cont,
        E, M, H, expert_offsets, N, max_tokens_per_expert
    )

    grad_up_weights = launch_grouped_outer(
        flat_grad_up_output, sorted_hidden_cont,
        E, M, H, expert_offsets, N, max_tokens_per_expert
    )

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
