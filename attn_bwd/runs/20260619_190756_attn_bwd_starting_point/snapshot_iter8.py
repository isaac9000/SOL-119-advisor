"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV matmuls launched concurrently on separate CUDA streams with
  pre-allocated output tensors to avoid allocator data races.
- Triton single/two-pass fused softmax backward + dropout correction.
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
def fused_softmax_bwd_single_pass(
    dP_dropped_ptr,
    P_ptr,
    dropout_mask_ptr,
    dS_ptr,
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """Single-pass: load each row element once, compute row_sum, write dS."""
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
    """Two-pass: for seq_kv that doesn't fit in a single block."""
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
    # Step 1: Make dO contiguous in [bs, 80, sq, 128] layout (bfloat16).
    # One contiguous() call; all subsequent reshapes are free views.
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare operands for both matmuls (free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    if not attn_groups_flat.is_contiguous():
        attn_groups_flat = attn_groups_flat.contiguous()

    # =========================================================================
    # Step 2: Launch dP and dV matmuls concurrently on separate CUDA streams.
    # Pre-allocate output tensors on the current (default) stream BEFORE
    # launching either matmul, to avoid allocator interference between streams.
    #
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # Each matmul writes to its own pre-allocated output buffer.
    # =========================================================================

    # Pre-allocate outputs on current stream (before any stream switching)
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # Record event after all inputs are ready on current stream
    input_ready_event = torch.cuda.Event()
    input_ready_event.record()

    # Create stream1 for concurrent dV computation
    stream1 = torch.cuda.Stream(device)

    # stream1 waits until inputs are ready
    stream1.wait_event(input_ready_event)

    # Launch dP on the current (default) stream
    torch.matmul(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # Launch dV on stream1 concurrently
    with torch.cuda.stream(stream1):
        torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # This runs on the current stream and can overlap with dV on stream1.
    # Input: dP_groups (bf16), attn_weights (bf16), dropout_mask (bool)
    # Output: dS (bf16)
    # Note: dP_groups is written by the current-stream matmul which completes
    # before this Triton kernel launches (same stream, in-order).
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Reshape dP_groups [bs*8, 10*sq, skv] -> [total_rows, seq_kv] — free view
    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV — power of 2, >= seq_kv for single-pass when possible
    if seq_kv <= 128:
        BLOCK_SKV = 128
        single_pass = True
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        single_pass = True
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        single_pass = True
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        single_pass = True
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        single_pass = True
    else:
        BLOCK_SKV = 2048
        single_pass = False

    grid = (total_rows,)

    if single_pass:
        fused_softmax_bwd_single_pass[grid](
            dP_dropped_flat, P_flat, dm_flat, dS_flat,
            scale=scale,
            seq_kv=seq_kv,
            BLOCK_SKV=BLOCK_SKV,
        )
    else:
        fused_softmax_bwd_two_pass[grid](
            dP_dropped_flat, P_flat, dm_flat, dS_flat,
            scale=scale,
            seq_kv=seq_kv,
            BLOCK_SKV=BLOCK_SKV,
        )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # Wait for stream1 (dV) to complete before returning dV_flat
    torch.cuda.current_stream(device).wait_stream(stream1)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
