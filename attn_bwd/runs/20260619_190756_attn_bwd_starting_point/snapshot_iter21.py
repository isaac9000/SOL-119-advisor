"""
Optimized attention-backward kernel:
- Triton transpose kernel: directly maps grad_attn_output [bs, sq, 80, 128]
  to dO_groups_flat [bs*8, 10*sq, 128] in one pass — eliminates intermediate
  [bs, 80, sq, 128] tensor and the separate contiguous() copy.
- Both bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching (ROWS_PER_BLOCK from #18).
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

# Module-level cached CUDA stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


@triton.jit
def transpose_to_groups_kernel(
    # Input: grad_attn_output [bs, seq_q, n_heads, HEAD_DIM] bfloat16
    src_ptr,
    stride_src_b,   # bs stride = seq_q * n_heads * HEAD_DIM
    stride_src_sq,  # seq_q stride = n_heads * HEAD_DIM
    stride_src_h,   # head stride = HEAD_DIM
    # Output: dO_groups_flat [bs*n_kv_heads, n_groups*seq_q, HEAD_DIM] bfloat16
    dst_ptr,
    stride_dst_bkv,   # bs*kv_head stride = n_groups * seq_q * HEAD_DIM
    stride_dst_gq,    # gq stride = HEAD_DIM
    # Dims
    seq_q,
    n_heads,           # 80
    n_kv_heads,        # 8
    n_groups: tl.constexpr,  # 10
    HEAD_DIM: tl.constexpr,  # 128
    BLOCK_GQ: tl.constexpr,  # tile size over gq dimension
):
    """
    Grid: (bs * n_kv_heads, ceil(n_groups * seq_q / BLOCK_GQ))

    Mapping: src[b, sq, h, d] -> dst[b*n_kv_heads + kv_h, group*seq_q + sq, d]
    where h = kv_h * n_groups + group

    Each program handles a tile of (kv_batch, gq) pairs.
    For each gq in the tile: gq = group * seq_q + sq_idx
    Reads 128-element contiguous row from src, writes 128-element contiguous row to dst.
    """
    kv_bh_idx = tl.program_id(0)  # [0, bs*n_kv_heads)
    gq_block  = tl.program_id(1)  # which tile of gq

    # Decompose kv_bh_idx into (b, kv_h)
    # kv_bh_idx = b * n_kv_heads + kv_h
    # We need b and kv_h separately for the src pointer
    # Use integer arithmetic: b = kv_bh_idx // n_kv_heads, kv_h = kv_bh_idx % n_kv_heads
    # n_kv_heads = 8 (power of 2!) — fast division
    b_idx    = kv_bh_idx >> 3    # kv_bh_idx // 8
    kv_h_idx = kv_bh_idx & 7    # kv_bh_idx % 8

    # gq tile start
    gq_start = gq_block * BLOCK_GQ
    d_offs = tl.arange(0, HEAD_DIM)

    for i in tl.static_range(BLOCK_GQ):
        gq = gq_start + i
        if gq < n_groups * seq_q:
            # Decompose gq = group * seq_q + sq_idx
            # n_groups = 10 (non-power-of-2) — but constexpr so compiler can optimize
            group_idx = gq // seq_q
            sq_idx    = gq - group_idx * seq_q   # = gq % seq_q

            # Compute head index in source: h = kv_h * n_groups + group
            h_idx = kv_h_idx * n_groups + group_idx

            # Source: src[b, sq_idx, h, :]
            src_offset = b_idx * stride_src_b + sq_idx * stride_src_sq + h_idx * stride_src_h
            row = tl.load(src_ptr + src_offset + d_offs)

            # Destination: dst[kv_bh_idx, gq, :]
            dst_offset = kv_bh_idx * stride_dst_bkv + gq * stride_dst_gq
            tl.store(dst_ptr + dst_offset + d_offs, row)


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
    # Step 1: Triton-based direct transpose from [bs, sq, 80, 128]
    # to dO_groups_flat [bs*8, 10*sq, 128] — eliminates intermediate tensor.
    #
    # Mapping: src[b, sq, h, d] -> dst[b*8 + h//10, (h%10)*seq_q + sq, d]
    # n_kv_heads=8 (power of 2) enables fast bit-ops in kernel.
    # =========================================================================
    n_gq = n_groups * seq_q
    dO_groups_flat = torch.empty(
        (bs * n_kv_heads, n_gq, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # Choose BLOCK_GQ: tile size over gq dimension
    # Each program does BLOCK_GQ scalar copies of HEAD_DIM=128 elements
    if n_gq <= 64:
        BLOCK_GQ = 4
    elif n_gq <= 256:
        BLOCK_GQ = 8
    else:
        BLOCK_GQ = 16

    n_gq_blocks = (n_gq + BLOCK_GQ - 1) // BLOCK_GQ
    grid_transpose = (bs * n_kv_heads, n_gq_blocks)

    transpose_to_groups_kernel[grid_transpose](
        grad_attn_output,
        grad_attn_output.stride(0), grad_attn_output.stride(1), grad_attn_output.stride(2),
        dO_groups_flat,
        dO_groups_flat.stride(0), dO_groups_flat.stride(1),
        seq_q=seq_q,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        HEAD_DIM=HEAD_DIM,
        BLOCK_GQ=BLOCK_GQ,
    )

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors for the bmms.
    # =========================================================================
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    dO_ready_event.record(main_stream)

    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # ROWS_PER_BLOCK from #18 (proven optimal)
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

    # Wait for side stream (dV) to complete
    main_stream.wait_stream(side_stream)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
