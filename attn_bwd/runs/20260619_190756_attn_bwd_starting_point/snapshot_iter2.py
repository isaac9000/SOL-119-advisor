"""
Optimized attention-backward kernel using Triton for fused elementwise ops
and cuBLAS for heavy matmuls with eliminated GQA expansion for dV.

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
def fused_softmax_backward_kernel(
    # Inputs
    dP_dropped_ptr,   # [batch_heads, seq_q, seq_kv] float32
    P_ptr,            # [batch_heads, seq_q, seq_kv] bfloat16
    dropout_mask_ptr, # [batch_heads, seq_q, seq_kv] bool
    # Output
    dS_ptr,           # [batch_heads, seq_q, seq_kv] bfloat16
    # Strides for dP_dropped (float32)
    stride_dp_bh, stride_dp_sq, stride_dp_skv,
    # Strides for P (bfloat16)
    stride_p_bh, stride_p_sq, stride_p_skv,
    # Strides for dropout_mask (bool)
    stride_dm_bh, stride_dm_sq, stride_dm_skv,
    # Strides for dS (bfloat16)
    stride_ds_bh, stride_ds_sq, stride_ds_skv,
    # Dimensions
    seq_q: tl.constexpr,
    seq_kv: tl.constexpr,
    scale: tl.constexpr,  # 1.0 / (1.0 - dropout)
    BLOCK_SKV: tl.constexpr,
):
    """
    Fused kernel: for each (batch_head, row) computes:
      dP = dP_dropped * mask * scale
      dS = P * (dP - sum(dP * P))
    """
    # Grid: [batch_heads, seq_q]
    bh_idx = tl.program_id(0)
    sq_idx = tl.program_id(1)

    # Pointers for this (bh, sq) row
    dp_row_ptr = dP_dropped_ptr + bh_idx * stride_dp_bh + sq_idx * stride_dp_sq
    p_row_ptr  = P_ptr          + bh_idx * stride_p_bh  + sq_idx * stride_p_sq
    dm_row_ptr = dropout_mask_ptr + bh_idx * stride_dm_bh + sq_idx * stride_dm_sq
    ds_row_ptr = dS_ptr         + bh_idx * stride_ds_bh + sq_idx * stride_ds_sq

    # Accumulate row sum of dP * P in float32
    row_sum = tl.zeros([1], dtype=tl.float32)

    # First pass: compute dP and accumulate row_sum
    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask = offsets < seq_kv

        dp_dropped = tl.load(dp_row_ptr + offsets * stride_dp_skv, mask=mask, other=0.0).to(tl.float32)
        dm = tl.load(dm_row_ptr + offsets * stride_dm_skv, mask=mask, other=0).to(tl.int1)
        p = tl.load(p_row_ptr + offsets * stride_p_skv, mask=mask, other=0.0).to(tl.float32)

        # Apply dropout correction
        dp = tl.where(dm, dp_dropped * scale, 0.0)

        # Accumulate dP * P
        row_sum += tl.sum(dp * p, axis=0)

    # Second pass: compute dS = P * (dP - row_sum)
    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask = offsets < seq_kv

        dp_dropped = tl.load(dp_row_ptr + offsets * stride_dp_skv, mask=mask, other=0.0).to(tl.float32)
        dm = tl.load(dm_row_ptr + offsets * stride_dm_skv, mask=mask, other=0).to(tl.int1)
        p = tl.load(p_row_ptr + offsets * stride_p_skv, mask=mask, other=0.0).to(tl.float32)

        # Apply dropout correction
        dp = tl.where(dm, dp_dropped * scale, 0.0)

        # Softmax backward
        ds = p * (dp - row_sum)

        tl.store(ds_row_ptr + offsets * stride_ds_skv, ds.to(tl.bfloat16), mask=mask)


@triton.jit
def fused_softmax_backward_kernel_full_row(
    # Inputs
    dP_dropped_ptr,   # [batch_heads, seq_q, seq_kv] float32
    P_ptr,            # [batch_heads, seq_q, seq_kv] bfloat16
    dropout_mask_ptr, # [batch_heads, seq_q, seq_kv] bool
    # Output
    dS_ptr,           # [batch_heads, seq_q, seq_kv] bfloat16
    # Strides for dP_dropped (float32)
    stride_dp_bh, stride_dp_sq, stride_dp_skv,
    # Strides for P (bfloat16)
    stride_p_bh, stride_p_sq, stride_p_skv,
    # Strides for dropout_mask (bool)
    stride_dm_bh, stride_dm_sq, stride_dm_skv,
    # Strides for dS (bfloat16)
    stride_ds_bh, stride_ds_sq, stride_ds_skv,
    # Dimensions
    seq_kv: tl.constexpr,
    scale: tl.constexpr,  # 1.0 / (1.0 - dropout)
    BLOCK_SKV: tl.constexpr,
):
    """
    Fused kernel where each thread block handles one full row.
    Grid: [batch_heads * seq_q]
    """
    row_id = tl.program_id(0)
    # row_id maps to (bh_idx, sq_idx) but we just use flat indexing
    
    dp_row_ptr = dP_dropped_ptr + row_id * stride_dp_sq
    p_row_ptr  = P_ptr          + row_id * stride_p_sq
    dm_row_ptr = dropout_mask_ptr + row_id * stride_dm_sq
    ds_row_ptr = dS_ptr         + row_id * stride_ds_sq

    # First pass: compute row_sum of dP * P
    row_sum = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv

        dp_dropped = tl.load(dp_row_ptr + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dm = tl.load(dm_row_ptr + offsets, mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(p_row_ptr + offsets, mask=mask_bounds, other=0.0).to(tl.float32)

        dp = tl.where(dm, dp_dropped * scale, 0.0)
        row_sum += tl.sum(dp * p, axis=0)

    # Second pass: write dS
    for block_start in range(0, seq_kv, BLOCK_SKV):
        offsets = block_start + tl.arange(0, BLOCK_SKV)
        mask_bounds = offsets < seq_kv

        dp_dropped = tl.load(dp_row_ptr + offsets, mask=mask_bounds, other=0.0).to(tl.float32)
        dm = tl.load(dm_row_ptr + offsets, mask=mask_bounds, other=0).to(tl.int1)
        p = tl.load(p_row_ptr + offsets, mask=mask_bounds, other=0.0).to(tl.float32)

        dp = tl.where(dm, dp_dropped * scale, 0.0)
        ds = p * (dp - row_sum)

        tl.store(ds_row_ptr + offsets, ds.to(tl.bfloat16), mask=mask_bounds)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)
    # dO shape: [bs, 80, sq, 128]

    # =========================================================================
    # Step 1: Compute dP_dropped = dO @ V^T using cuBLAS
    # Use expand (no copy) for GQA - cuBLAS handles strided views
    # vs_exp: [bs, 80, skv, 128] via expand (stride trick, no copy)
    # =========================================================================
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)
    # Note: reshape after expand may copy; use contiguous only if needed
    # For the matmul, we need contiguous or at least the last 2 dims to be contiguous
    vs_exp_f32 = vs_exp.to(torch.float32)

    # dP_dropped: [bs, 80, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp_f32.transpose(-2, -1))

    # =========================================================================
    # Step 2: Fused softmax backward + dropout correction via Triton
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Flatten batch and head dims for easier indexing: [bs*80, sq, skv]
    batch_heads = bs * n_heads
    dP_dropped_flat = dP_dropped.reshape(batch_heads, seq_q, seq_kv).contiguous()
    P_flat = attn_weights.reshape(batch_heads, seq_q, seq_kv).contiguous()
    dm_flat = dropout_mask.reshape(batch_heads, seq_q, seq_kv).contiguous()

    dS_flat = torch.empty((batch_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dP_dropped.device)

    # Choose BLOCK_SKV based on seq_kv
    if seq_kv <= 128:
        BLOCK_SKV = 128
    elif seq_kv <= 256:
        BLOCK_SKV = 256
    elif seq_kv <= 512:
        BLOCK_SKV = 512
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
    else:
        BLOCK_SKV = 2048

    # Launch flat row kernel: each thread block handles one (bh, sq) row
    total_rows = batch_heads * seq_q
    grid = (total_rows,)

    fused_softmax_backward_kernel_full_row[grid](
        dP_dropped_flat, P_flat, dm_flat, dS_flat,
        # strides for dP_dropped_flat (float32, contiguous: shape [bh, sq, skv])
        dP_dropped_flat.stride(0), dP_dropped_flat.stride(1), dP_dropped_flat.stride(2),
        # strides for P_flat (bfloat16)
        P_flat.stride(0), P_flat.stride(1), P_flat.stride(2),
        # strides for dm_flat (bool)
        dm_flat.stride(0), dm_flat.stride(1), dm_flat.stride(2),
        # strides for dS_flat (bfloat16)
        dS_flat.stride(0), dS_flat.stride(1), dS_flat.stride(2),
        seq_kv=seq_kv,
        scale=scale,
        BLOCK_SKV=BLOCK_SKV,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 3: Compute dV without GQA expansion
    # dV[b,kv,skv,d] = sum_g sum_sq P̃[b,kv*10+g,sq,skv] * dO[b,kv*10+g,sq,d]
    # 
    # Reshape: attn_weights_dropped [bs,80,sq,skv] -> [bs,8,10,sq,skv]
    # dO [bs,80,sq,128] -> [bs,8,10,sq,128]
    # Merge (bs*8) as batch, (10*sq) as inner dim:
    #   [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    aw_dropped_f32 = attn_weights_dropped.to(torch.float32)

    # Reshape to expose groups
    # attn: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    attn_groups = aw_dropped_f32.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    attn_groups_flat = attn_groups.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # dO: [bs, 80, sq, 128] -> [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128]
    dO_groups = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)
    dO_groups_flat = dO_groups.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # dV: [bs*8, skv, 128]
    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV
