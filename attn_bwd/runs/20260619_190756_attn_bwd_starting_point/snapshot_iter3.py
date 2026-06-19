"""
Optimized attention-backward kernel using Triton for fused elementwise ops
and cuBLAS for heavy matmuls with eliminated GQA expansion for dV.
All matmuls use bfloat16 (faster tensor core utilization on B200).

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
    # Inputs
    dP_dropped_ptr,   # [total_rows, seq_kv] bfloat16
    P_ptr,            # [total_rows, seq_kv] bfloat16
    dropout_mask_ptr, # [total_rows, seq_kv] bool
    # Output
    dS_ptr,           # [total_rows, seq_kv] bfloat16
    # Scalar params
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Single-pass softmax backward when all of seq_kv fits in one block.
    Each program handles one row. Grid: [total_rows].
    """
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SKV)
    mask_bounds = offsets < seq_kv

    base = row_id * seq_kv

    dp_dropped = tl.load(dP_dropped_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
    dm = tl.load(dropout_mask_ptr + base + offsets, mask=mask_bounds, other=0).to(tl.int1)
    p = tl.load(P_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)

    # Apply dropout correction
    dp = tl.where(dm, dp_dropped * scale, 0.0)

    # Row sum of dP * P
    row_sum = tl.sum(dp * p, axis=0)

    # Softmax backward: dS = P * (dP - row_sum)
    ds = p * (dp - row_sum)

    tl.store(dS_ptr + base + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


@triton.jit
def fused_softmax_bwd_two_pass(
    # Inputs
    dP_dropped_ptr,   # [total_rows, seq_kv] bfloat16
    P_ptr,            # [total_rows, seq_kv] bfloat16
    dropout_mask_ptr, # [total_rows, seq_kv] bool
    # Output
    dS_ptr,           # [total_rows, seq_kv] bfloat16
    # Scalar params
    scale: tl.constexpr,
    seq_kv: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
):
    """
    Two-pass softmax backward for large seq_kv.
    Each program handles one row. Grid: [total_rows].
    """
    row_id = tl.program_id(0)
    base = row_id * seq_kv

    # First pass: accumulate row_sum
    row_sum = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv

        dp_dropped = tl.load(dP_dropped_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dm = tl.load(dropout_mask_ptr + base + offsets, mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(P_ptr + base + offsets, mask=mask_bounds, other=0.0).to(tl.float32)

        dp = tl.where(dm, dp_dropped * scale, 0.0)
        row_sum += tl.sum(dp * p, axis=0)

    # Second pass: write dS
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

    # =========================================================================
    # Step 1: Transpose grad_attn_output: [bs, sq, h, d] → [bs, h, sq, d]
    # Keep in bfloat16 for faster matmuls on B200.
    # Use contiguous() once here to ensure layout is matmul-compatible.
    # =========================================================================
    dO = grad_attn_output.transpose(1, 2).contiguous()
    # dO shape: [bs, 80, sq, 128], bfloat16

    # =========================================================================
    # Step 2: Compute dP_dropped = dO @ V^T using cuBLAS (bfloat16)
    # GQA expand via expand() — no copy, just stride tricks.
    # vs_exp: [bs, 8, skv, 128] -> [bs, 80, skv, 128] (expanded)
    # After expand+reshape, may trigger a copy; use contiguous to be safe.
    # =========================================================================
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)
    # vs_exp is bfloat16, shape [bs, 80, skv, 128]

    # dP_dropped: [bs, 80, sq, skv], bfloat16
    dP_dropped = torch.matmul(dO, vs_exp.transpose(-2, -1))

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton
    # Input: dP_dropped (bf16), attn_weights (bf16), dropout_mask (bool)
    # Output: dS (bf16)
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    batch_heads = bs * n_heads
    total_rows = batch_heads * seq_q

    # Flatten to [total_rows, seq_kv] — reshape is free if contiguous
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    P_flat = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # Ensure contiguous for Triton (attn_weights and dropout_mask should already be)
    if not dP_dropped_flat.is_contiguous():
        dP_dropped_flat = dP_dropped_flat.contiguous()
    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=dP_dropped.device)

    # Choose BLOCK_SKV — power of 2, >= seq_kv if possible for single-pass
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
    # Step 4: Compute dV without GQA expansion (bfloat16 matmul)
    # dV[b,kv,skv,d] = sum_g sum_sq P̃[b,kv*10+g,sq,skv] * dO[b,kv*10+g,sq,d]
    #
    # Reshape attn_weights_dropped [bs,80,sq,skv] -> [bs*8, 10*sq, skv]
    # Reshape dO [bs,80,sq,128] -> [bs*8, 10*sq, 128]
    # Single batched matmul: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    # attn: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # dO: [bs, 80, sq, 128] -> [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128]
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Ensure contiguous for matmul (dO is already contiguous, attn_weights_dropped should be)
    if not attn_groups_flat.is_contiguous():
        attn_groups_flat = attn_groups_flat.contiguous()

    # dV: [bs*8, skv, 128], bfloat16
    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV
