"""
Optimized attention-backward kernel:
- dP: torch.bmm (cuBLAS)
- dS: Triton row-batched softmax backward
- dV: Triton kernel using tl.dot for tile-level matrix multiplication,
  computing dV[kv_batch, s, d] = sum_{g,q} P_dropped[kv_batch, g, q, s] * dO[kv_batch, g, q, d]
  Tiles over (skv, HEAD_DIM) with inner loop over (groups*seq_q).
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
def dV_tiled_kernel(
    # attn_groups: [bs*8, n_groups*seq_q, seq_kv] bfloat16, contiguous
    attn_ptr,
    stride_attn_b, stride_attn_gq, stride_attn_s,
    # dO_groups: [bs*8, n_groups*seq_q, HEAD_DIM] bfloat16, contiguous
    dO_ptr,
    stride_dO_b, stride_dO_gq, stride_dO_d,
    # dV output: [bs*8, seq_kv, HEAD_DIM] bfloat16, contiguous
    dV_ptr,
    stride_dV_b, stride_dV_s, stride_dV_d,
    # Dims
    n_gq,          # n_groups * seq_q = 10 * seq_q
    seq_kv,
    HEAD_DIM: tl.constexpr,   # 128
    BLOCK_S: tl.constexpr,    # tile size over seq_kv
    BLOCK_D: tl.constexpr,    # tile size over HEAD_DIM (= 128)
    BLOCK_GQ: tl.constexpr,   # tile size over groups*seq_q (K dimension)
):
    """
    Grid: (bs*8, ceil(seq_kv/BLOCK_S))
    Each program computes a tile of dV[b, s_tile, :] for one (b, s_tile).

    dV[b, s, d] = sum_{gq} attn[b, gq, s] * dO[b, gq, d]

    This is: dV[b, :, :] = attn[b, :, :]^T @ dO[b, :, :]
           = [seq_kv, n_gq] @ [n_gq, HEAD_DIM]

    Each program handles one (b, s_tile) and accumulates over the gq dimension.
    Using tl.dot for tensor-core-accelerated tile multiplication.
    """
    b_idx   = tl.program_id(0)
    s_block = tl.program_id(1)

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)     # HEAD_DIM = 128 = BLOCK_D

    # Base pointers for this batch
    attn_b = attn_ptr + b_idx * stride_attn_b
    dO_b   = dO_ptr   + b_idx * stride_dO_b
    dV_b   = dV_ptr   + b_idx * stride_dV_b

    # Accumulator for dV tile: [BLOCK_S, BLOCK_D] in float32
    acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)

    # Inner loop over K = n_gq dimension
    for gq_start in range(0, n_gq, BLOCK_GQ):
        gq_offs = gq_start + tl.arange(0, BLOCK_GQ)
        gq_mask = gq_offs < n_gq

        # Load attn tile: [BLOCK_S, BLOCK_GQ] = attn[b, gq_offs, s_offs]^T
        # attn is [bs*8, n_gq, seq_kv] so attn[b, gq, s] = attn_b + gq*stride_gq + s*stride_s
        # We want [BLOCK_S, BLOCK_GQ] tile: rows=s, cols=gq
        attn_ptrs = attn_b + gq_offs[None, :] * stride_attn_gq + s_offs[:, None] * stride_attn_s
        s_mask = s_offs < seq_kv
        attn_tile = tl.load(attn_ptrs,
                            mask=(s_mask[:, None] & gq_mask[None, :]),
                            other=0.0)  # [BLOCK_S, BLOCK_GQ] bfloat16

        # Load dO tile: [BLOCK_GQ, BLOCK_D] = dO[b, gq_offs, :]
        dO_ptrs = dO_b + gq_offs[:, None] * stride_dO_gq + d_offs[None, :] * stride_dO_d
        dO_tile = tl.load(dO_ptrs,
                          mask=gq_mask[:, None],
                          other=0.0)  # [BLOCK_GQ, BLOCK_D] bfloat16

        # Accumulate: [BLOCK_S, BLOCK_GQ] @ [BLOCK_GQ, BLOCK_D] -> [BLOCK_S, BLOCK_D]
        acc += tl.dot(attn_tile, dO_tile, out_dtype=tl.float32)

    # Write output dV tile: [BLOCK_S, BLOCK_D]
    s_mask = s_offs < seq_kv
    dV_ptrs = dV_b + s_offs[:, None] * stride_dV_s + d_offs[None, :] * stride_dV_d
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=s_mask[:, None])


@triton.jit
def fused_softmax_bwd_batched(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    total_rows,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Batched softmax-backward kernel: each program handles ROWS_PER_BLOCK rows.
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
        if row_id < total_rows:
            base = row_id * seq_kv

            if BLOCK_SKV >= seq_kv:
                offsets = tl.arange(0, BLOCK_SKV)
                mask_bounds = offsets < seq_kv

                dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                     mask=mask_bounds, other=0.0).to(tl.float32)
                dm = tl.load(dropout_mask_ptr + base + offsets,
                             mask=mask_bounds, other=0).to(tl.int1)
                p = tl.load(P_ptr + base + offsets,
                            mask=mask_bounds, other=0.0).to(tl.float32)

                dp = tl.where(dm, dp_dropped * scale, 0.0)
                row_sum = tl.sum(dp * p, axis=0)
                ds = p * (dp - row_sum)

                tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)
            else:
                row_sum = tl.zeros([1], dtype=tl.float32)
                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
                    dp = tl.where(dm, dp_dropped * scale, 0.0)
                    row_sum += tl.sum(dp * p, axis=0)

                for block_start in range(0, seq_kv, BLOCK_SKV):
                    offsets = block_start + tl.arange(0, BLOCK_SKV)
                    mask_bounds = offsets < seq_kv
                    dp_dropped = tl.load(dP_dropped_ptr + base + offsets,
                                         mask=mask_bounds, other=0.0).to(tl.float32)
                    dm = tl.load(dropout_mask_ptr + base + offsets,
                                 mask=mask_bounds, other=0).to(tl.int1)
                    p = tl.load(P_ptr + base + offsets,
                                mask=mask_bounds, other=0.0).to(tl.float32)
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
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16).
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for dP matmul: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T without GQA expansion (cuBLAS bmm).
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_groups = torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1))

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    if seq_kv <= 128:
        BLOCK_SKV = 128
        ROWS_PER_BLOCK = 16
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        ROWS_PER_BLOCK = 8
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        ROWS_PER_BLOCK = 4
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        ROWS_PER_BLOCK = 2
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1
    else:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 1

    num_blocks = (total_rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    fused_softmax_bwd_batched[(num_blocks,)](
        dP_dropped_flat, P_flat, dm_flat, dS_flat,
        total_rows=total_rows,
        scale=scale,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 4: Compute dV via Triton kernel using tl.dot.
    # dV[b, s, d] = sum_{gq} attn_dropped[b, gq, s] * dO[b, gq, d]
    # = attn_dropped[b, :, :]^T @ dO[b, :, :]
    # = [seq_kv, n_gq] @ [n_gq, HEAD_DIM]
    #
    # attn_groups: [bs*8, n_gq, seq_kv] where n_gq = 10*seq_q
    # dO_groups:   [bs*8, n_gq, 128]
    # dV output:   [bs*8, seq_kv, 128]
    #
    # Grid: (bs*8, ceil(seq_kv / BLOCK_S))
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    n_gq = n_groups * seq_q

    dV_flat = torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=device)

    # Choose tile sizes for dV kernel
    # BLOCK_S: tile over seq_kv; BLOCK_D = HEAD_DIM = 128 (fixed)
    # BLOCK_GQ: tile over K = n_gq dimension
    if seq_kv <= 128:
        BLOCK_S = 64
    elif seq_kv <= 512:
        BLOCK_S = 64
    else:
        BLOCK_S = 64

    # BLOCK_GQ: K tile size. tl.dot requires power-of-2 and >= 16.
    if n_gq <= 32:
        BLOCK_GQ = 32
    elif n_gq <= 64:
        BLOCK_GQ = 64
    else:
        BLOCK_GQ = 128  # Larger K tiles for better tensor core utilization

    BLOCK_D = HEAD_DIM  # 128, always power of 2

    n_s_blocks = (seq_kv + BLOCK_S - 1) // BLOCK_S
    grid_dV = (bs * n_kv_heads, n_s_blocks)

    dV_tiled_kernel[grid_dV](
        attn_groups_flat,
        attn_groups_flat.stride(0), attn_groups_flat.stride(1), attn_groups_flat.stride(2),
        dO_groups_flat,
        dO_groups_flat.stride(0), dO_groups_flat.stride(1), dO_groups_flat.stride(2),
        dV_flat,
        dV_flat.stride(0), dV_flat.stride(1), dV_flat.stride(2),
        n_gq=n_gq,
        seq_kv=seq_kv,
        HEAD_DIM=HEAD_DIM,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_GQ=BLOCK_GQ,
    )

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
