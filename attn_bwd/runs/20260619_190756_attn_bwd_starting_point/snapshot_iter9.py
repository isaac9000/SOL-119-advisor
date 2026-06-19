"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- Triton softmax-backward kernel with row batching (ROWS_PER_BLOCK rows per program)
  for better occupancy on small seq_kv sizes.
- All in bfloat16. Sequential execution.

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
    For seq_kv <= BLOCK_SKV (single-pass), each row is loaded once.
    For seq_kv > BLOCK_SKV (two-pass), two passes per row.
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
        # Guard against out-of-bounds rows
        if row_id < total_rows:
            base = row_id * seq_kv

            if BLOCK_SKV >= seq_kv:
                # Single-pass: everything fits in one block
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
                # Two-pass for large seq_kv
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T without GQA expansion.
    # value_states: [bs, 8, skv, 128] -> [bs*8, skv, 128]  (free view)
    # matmul([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_groups = torch.matmul(dO_groups_flat, vs_flat.transpose(-2, -1))

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Row-batched kernel: ROWS_PER_BLOCK rows per program.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Flatten to [total_rows, seq_kv] — free views (all contiguous)
    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv.
    # Smaller seq_kv → more rows per block (kernel is latency-bound).
    # Larger seq_kv → fewer rows per block (kernel is memory-bound per row).
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
    grid = (num_blocks,)

    fused_softmax_bwd_batched[grid](
        dP_dropped_flat, P_flat, dm_flat, dS_flat,
        total_rows=total_rows,
        scale=scale,
        seq_kv=seq_kv,
        BLOCK_SKV=BLOCK_SKV,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 4: Compute dV without GQA expansion (bfloat16 matmul).
    # attn_weights_dropped [bs,80,sq,skv] -> [bs*8, 10*sq, skv]
    # matmul([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # attn_weights_dropped is expected to be contiguous (it's an input tensor)

    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
