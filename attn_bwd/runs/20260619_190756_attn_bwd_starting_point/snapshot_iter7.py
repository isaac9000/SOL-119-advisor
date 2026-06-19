"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- Triton single/two-pass fused softmax backward + dropout correction.
- All in bfloat16. Sequential execution (no concurrent streams).

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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128]
    # This is a free view since dO is contiguous.
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T WITHOUT GQA expansion.
    # Same group-reshape trick as dV.
    #
    # value_states: [bs, 8, skv, 128] -> [bs*8, skv, 128]  (free view)
    # dO_groups_flat: [bs*8, 10*sq, 128]
    #
    # matmul([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # reshape -> [bs, 80, sq, skv]
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    # vs_flat: [bs*8, skv, 128], bfloat16

    # dP_groups: [bs*8, 10*sq, skv], bfloat16
    dP_groups = torch.matmul(dO_groups_flat, vs_flat.transpose(-2, -1))

    # Reshape to [bs, 80, sq, skv] for the Triton softmax kernel
    # (free view since dP_groups is a fresh contiguous matmul output)
    dP_dropped = dP_groups.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Input: dP_dropped (bf16), attn_weights (bf16), dropout_mask (bool)
    # Output: dS (bf16)
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Flatten to [total_rows, seq_kv] — free views (all contiguous)
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
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

    # =========================================================================
    # Step 4: Compute dV without GQA expansion (bfloat16 matmul).
    # attn_weights_dropped [bs,80,sq,skv] -> [bs*8, 10*sq, skv]
    # dO [bs,80,sq,128] -> [bs*8, 10*sq, 128]  (reuse dO_groups_flat)
    # matmul([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    if not attn_groups_flat.is_contiguous():
        attn_groups_flat = attn_groups_flat.contiguous()

    # dO_groups_flat is already computed above and shared between both matmuls
    # (sequential execution ensures no data race)
    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
