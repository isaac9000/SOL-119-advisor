"""
Optimized attention-backward kernel:
- Fused Triton kernel: computes dP = dO @ V^T tile-by-tile + softmax backward,
  eliminating the large intermediate dP_dropped tensor.
- 3D grid (bs*n_kv_heads, n_groups, seq_q) avoids integer division in kernel.
- cuBLAS for dV using group-reshape trick (no GQA expansion).
- All in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128


@triton.jit
def fused_dP_softmax_bwd_kernel(
    # dO: [bs*80, sq, HEAD_DIM] bfloat16, contiguous
    dO_ptr,
    # V: [bs*8, skv, HEAD_DIM] bfloat16, contiguous
    V_ptr,
    # P: [bs*80, sq, skv] bfloat16, contiguous
    P_ptr,
    # dm: [bs*80, sq, skv] bool, contiguous
    dm_ptr,
    # dS: [bs*80, sq, skv] bfloat16 (output), contiguous
    dS_ptr,
    # Shape params
    seq_q,
    seq_kv,
    n_groups: tl.constexpr,   # 10
    HEAD_DIM: tl.constexpr,   # 128
    scale: tl.constexpr,      # 1/(1-dropout)
    BLOCK_KV: tl.constexpr,   # tile size over kv dimension
):
    """
    Grid: (bs * n_kv_heads, n_groups, seq_q)
      pid0 = kv_bh_idx  in [0, bs*8)
      pid1 = group_idx  in [0, 10)
      pid2 = sq_idx     in [0, seq_q)

    bh_idx = kv_bh_idx * n_groups + group_idx  in [0, bs*80)

    For each (kv_bh, group, sq):
      dO_row = dO[bh_idx, sq_idx, :]          # [HEAD_DIM]
      For each kv tile:
        dp_tile = dO_row @ V[kv_bh, kv_tile, :].T  # dot product
        dp_tile = dp_tile * mask * scale
        row_sum += sum(dp_tile * P[bh_idx, sq_idx, kv_tile])
      For each kv tile:
        dS[bh_idx, sq_idx, kv_tile] = P[...] * (dp_tile - row_sum)
    """
    kv_bh_idx = tl.program_id(0)   # [0, bs*8)
    group_idx = tl.program_id(1)   # [0, 10)
    sq_idx    = tl.program_id(2)   # [0, seq_q)

    # Compute flat head index: [0, bs*80)
    bh_idx = kv_bh_idx * n_groups + group_idx

    # Strides (all tensors are contiguous):
    # dO: [bs*80, sq, HEAD_DIM] -> strides (sq*HEAD_DIM, HEAD_DIM, 1)
    # V:  [bs*8,  skv, HEAD_DIM] -> strides (skv*HEAD_DIM, HEAD_DIM, 1)
    # P:  [bs*80, sq, skv]       -> strides (sq*skv, skv, 1)
    # dm: [bs*80, sq, skv]       -> strides (sq*skv, skv, 1)
    # dS: [bs*80, sq, skv]       -> strides (sq*skv, skv, 1)

    # Load dO row: [HEAD_DIM] contiguous
    dO_row_offset = bh_idx * (seq_q * HEAD_DIM) + sq_idx * HEAD_DIM
    d_offsets = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_row_offset + d_offsets).to(tl.float32)
    # dO_row: [HEAD_DIM] float32

    # Row base for P, dm, dS
    row_base_p  = bh_idx * (seq_q * seq_kv) + sq_idx * seq_kv
    row_base_dm = bh_idx * (seq_q * seq_kv) + sq_idx * seq_kv
    row_base_ds = bh_idx * (seq_q * seq_kv) + sq_idx * seq_kv

    # V base for this KV head
    v_row_stride = HEAD_DIM  # stride between kv positions
    v_bh_offset  = kv_bh_idx * (seq_kv * HEAD_DIM)

    # =========================================================================
    # First pass: compute dP for each kv tile, accumulate row_sum
    # =========================================================================
    row_sum = tl.zeros([1], dtype=tl.float32)

    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < seq_kv

        # Load V tile: [BLOCK_KV, HEAD_DIM], contiguous
        # v_ptrs[i, j] = v_bh_offset + kv_offsets[i] * HEAD_DIM + j
        v_ptrs = v_bh_offset + kv_offsets[:, None] * v_row_stride + d_offsets[None, :]
        v_tile = tl.load(V_ptr + v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        # v_tile: [BLOCK_KV, HEAD_DIM]

        # dP tile: dot product dO_row with each row of v_tile -> [BLOCK_KV]
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_KV]

        # Load dropout mask and apply correction
        dm_tile = tl.load(dm_ptr + row_base_dm + kv_offsets, mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        # Load P tile and accumulate row_sum
        p_tile = tl.load(P_ptr + row_base_p + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dp_tile * p_tile, axis=0)

    # =========================================================================
    # Second pass: compute dS and write output
    # =========================================================================
    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < seq_kv

        # Recompute dP tile (V likely in L2)
        v_ptrs = v_bh_offset + kv_offsets[:, None] * v_row_stride + d_offsets[None, :]
        v_tile = tl.load(V_ptr + v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

        dm_tile = tl.load(dm_ptr + row_base_dm + kv_offsets, mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        p_tile = tl.load(P_ptr + row_base_p + kv_offsets, mask=kv_mask, other=0.0).to(tl.float32)

        ds_tile = p_tile * (dp_tile - row_sum)

        tl.store(dS_ptr + row_base_ds + kv_offsets, ds_tile.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def fused_softmax_bwd_single_pass(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SKV)
    mask_bounds = offsets < seq_kv
    base = row_id * seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
    dm = tl.load(dropout_mask_ptr + base + offsets, mask=mask_bounds, other=0).to(tl.int1)
    p = tl.load(P_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)

    dp = tl.where(dm, dp_dropped * scale, 0.0)
    row_sum = tl.sum(dp * p, axis=0)
    ds = p * (dp - row_sum)

    tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


@triton.jit
def fused_softmax_bwd_two_pass(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    row_id = tl.program_id(0)
    base = row_id * seq_kv

    row_sum = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv
        dp_dropped = tl.load(dP_dropped_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dm = tl.load(dropout_mask_ptr + base + offsets, mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(P_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dp = tl.where(dm, dp_dropped * scale, 0.0)
        row_sum += tl.sum(dp * p, axis=0)

    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv
        dp_dropped = tl.load(dP_dropped_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dm = tl.load(dropout_mask_ptr + base + offsets, mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(P_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dp = tl.where(dm, dp_dropped * scale, 0.0)
        ds = p * (dp - row_sum)
        tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    # =========================================================================
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16)
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Step 2: Fused dP + softmax backward via Triton
    # Grid: (bs*8, 10, seq_q) — no integer division in kernel
    # Each program: one (kv_batch_head, group, q_row)
    # Eliminates the large [bs, 80, sq, skv] intermediate dP_dropped tensor.
    # =========================================================================
    # All inputs reshaped for Triton:
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)              # [bs*80, sq, 128]
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM) # [bs*8, skv, 128]
    P_flat  = attn_weights.reshape(bs * n_heads, seq_q, seq_kv)       # [bs*80, sq, skv]
    dm_flat = dropout_mask.reshape(bs * n_heads, seq_q, seq_kv)       # [bs*80, sq, skv]

    # Ensure all inputs are contiguous (they should be from the inputs)
    if not vs_flat.is_contiguous():
        vs_flat = vs_flat.contiguous()
    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()
    # dO_flat is contiguous since dO is contiguous

    dS_flat = torch.empty((bs * n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_KV for the fused kernel
    if seq_kv <= 64:
        BLOCK_KV = 64
    elif seq_kv <= 128:
        BLOCK_KV = 128
    elif seq_kv <= 256:
        BLOCK_KV = 256
    else:
        BLOCK_KV = 256

    # 3D grid: (bs*n_kv_heads, n_groups, seq_q)
    grid_fused = (bs * n_kv_heads, n_groups, seq_q)

    fused_dP_softmax_bwd_kernel[grid_fused](
        dO_flat, vs_flat, P_flat, dm_flat, dS_flat,
        seq_q=seq_q,
        seq_kv=seq_kv,
        n_groups=n_groups,
        HEAD_DIM=HEAD_DIM,
        scale=scale,
        BLOCK_KV=BLOCK_KV,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 3: Compute dV without GQA expansion (bfloat16 matmul)
    # [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dO_groups_flat   = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    if not attn_groups_flat.is_contiguous():
        attn_groups_flat = attn_groups_flat.contiguous()

    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
