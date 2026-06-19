# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-19 19:08:15 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3428.82 μs

**Kernel code:**
```python
"""
Reference attention-backward kernel — pure PyTorch baseline.

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

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Expand value_states for GQA: [bs, 8, skv, d] → [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)

    # 1. Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P ⊙ (dP − sum(dP ⊙ P))
    P = attn_weights.to(torch.float32)
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
    dV_exp = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
    )

    # 6. GQA aggregation: sum over groups  →  [bs, 8, skv, d]
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #2 — 2026-06-19 19:10:08 UTC ✅ KEEP

**Hypothesis:** Three-part optimization:

**Result:** 1934.82 μs

**Kernel code:**
```python
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

```

---

## Experiment #3 — 2026-06-19 19:11:42 UTC ✅ KEEP

**Hypothesis:** 1. **BF16 matmuls**: Removed all `.to(torch.float32)` casts before matmuls. `dO` stays bfloat16 after transpose, `vs_exp` stays bfloat16, `attn_weights_dropped` stays bfloat16, `dO_groups_flat` stays 

**Result:** 499.81 μs

**Kernel code:**
```python
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

```

---

## Experiment #4 — 2026-06-19 19:13:14 UTC 💥 CRASH

**Hypothesis:** 1. **First matmul without GQA expansion**: Instead of expanding value_states [bs,8,skv,128] to [bs,80,skv,128], we reshape dO from [bs,80,sq,128] to [bs*8, 10*sq, 128] and value_states to [bs*8, skv, 

**Result:** CRASH

**Error:**
```
Benchmark correctness failed. Benchmark correctness: grad_value_states mismatch: max=nan mean=nan
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel using Triton for fused elementwise ops
and cuBLAS for heavy matmuls with no GQA expansion anywhere.
All matmuls use bfloat16. Both matmuls use [bs*8, 10*sq, ...] form.
Two matmuls launched concurrently on separate CUDA streams.

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
    device = grad_attn_output.device

    # =========================================================================
    # Reshape dO from [bs, sq, 80, d] to [bs*8, 10*sq, d] directly
    # avoiding the transpose+contiguous copy.
    # grad_attn_output layout: [bs, sq, 80, d] — need [bs, 80, sq, d] logically
    # but we can achieve [bs*8, 10*sq, d] via permute+reshape if we're careful.
    #
    # Strategy: permute [bs, sq, 80, d] -> [bs, 80, sq, d] requires contiguous
    # for the subsequent reshape. We do ONE contiguous call here.
    # Then reshape [bs, 80, sq, d] -> [bs*8, 10*sq, d] is a free view.
    # =========================================================================
    # [bs, sq, 80, d] -> [bs, 80, sq, d] contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Reshape to group form: [bs*8, 10*sq, 128] — free view (contiguous)
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # value_states: [bs, 8, skv, 128] -> [bs*8, skv, 128] — free view
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)

    # =========================================================================
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs*8, 10*sq, skv]
    # This reshape is free if attn_weights_dropped is contiguous (it should be).
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Launch both matmuls concurrently on separate CUDA streams:
    #   Stream 0: dP_dropped = dO_groups_flat @ vs_flat^T -> [bs*8, 10*sq, skv]
    #   Stream 1: dV_flat = attn_groups_flat^T @ dO_groups_flat -> [bs*8, skv, 128]
    # These are independent computations and can overlap.
    # =========================================================================
    stream0 = torch.cuda.current_stream(device)
    stream1 = torch.cuda.Stream(device)

    # Launch dP on stream0 (current stream)
    with torch.cuda.stream(stream0):
        # dP_dropped: [bs*8, 10*sq, skv], bfloat16
        dP_dropped_groups = torch.matmul(dO_groups_flat, vs_flat.transpose(-2, -1))

    # Launch dV on stream1 concurrently
    with torch.cuda.stream(stream1):
        # dV_flat: [bs*8, skv, 128], bfloat16
        dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)

    # Sync stream1 back to stream0 before using dV_flat
    stream0.wait_stream(stream1)

    # Reshape dP back to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_groups.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Fused softmax backward + dropout correction via Triton
    # Input: dP_dropped (bf16), attn_weights (bf16), dropout_mask (bool)
    # Output: dS (bf16)
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Flatten to [total_rows, seq_kv] — free view since dP_dropped is contiguous
    dP_dropped_flat = dP_dropped.reshape(total_rows, seq_kv)
    P_flat = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # Ensure contiguous (attn_weights and dropout_mask should already be)
    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

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

    # dV: reshape [bs*8, skv, 128] -> [bs, 8, skv, 128]
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #5 — 2026-06-19 19:16:04 UTC 💥 CRASH

**Hypothesis:** A new `fused_dP_softmax_bwd_kernel` Triton kernel with grid `(bs*80, seq_q)` where each program handles one `(batch_head, q_row)`:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Fused Triton kernel: computes dP = dO @ V^T tile-by-tile + softmax backward,
  eliminating the large intermediate dP_dropped tensor.
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
    # dO: [bs*80, sq, HEAD_DIM] bfloat16
    dO_ptr,
    # V: [bs*8, skv, HEAD_DIM] bfloat16
    V_ptr,
    # P: [bs*80, sq, skv] bfloat16
    P_ptr,
    # dm: [bs*80, sq, skv] bool
    dm_ptr,
    # dS: [bs*80, sq, skv] bfloat16 (output)
    dS_ptr,
    # Strides for dO [bs*80, sq, HEAD_DIM]
    stride_do_bh, stride_do_sq, stride_do_d,
    # Strides for V [bs*8, skv, HEAD_DIM]
    stride_v_bkv, stride_v_skv, stride_v_d,
    # Strides for P [bs*80, sq, skv]
    stride_p_bh, stride_p_sq, stride_p_skv,
    # Strides for dm [bs*80, sq, skv]
    stride_dm_bh, stride_dm_sq, stride_dm_skv,
    # Strides for dS [bs*80, sq, skv]
    stride_ds_bh, stride_ds_sq, stride_ds_skv,
    # Dims
    seq_kv,
    n_groups: tl.constexpr,    # 10
    HEAD_DIM: tl.constexpr,    # 128
    scale: tl.constexpr,       # 1/(1-dropout)
    BLOCK_KV: tl.constexpr,    # tile size over kv dimension
):
    """
    Fused kernel: for each (batch_head bh, query row sq_idx):
      1. First pass: compute dP[kv] = dot(dO[sq,:], V[kv,:]) for all kv tiles,
         apply dropout mask+scale, accumulate row_sum = sum(dP * P)
      2. Second pass: write dS[kv] = P[kv] * (dP[kv] - row_sum)

    Grid: (bs*80, sq)
    """
    bh_idx = tl.program_id(0)   # which (batch, head) pair — range [0, bs*80)
    sq_idx = tl.program_id(1)   # which query row

    # Map bh_idx to KV head index: kv_head = bh_idx // n_groups
    kv_bh_idx = bh_idx // n_groups   # range [0, bs*8)

    # Load dO row: [HEAD_DIM] — fits entirely in registers
    dO_base = bh_idx * stride_do_bh + sq_idx * stride_do_sq
    d_offsets = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_base + d_offsets * stride_do_d).to(tl.float32)
    # dO_row: [HEAD_DIM] float32

    # Base pointers for P, dm, dS rows
    p_base  = bh_idx * stride_p_bh  + sq_idx * stride_p_sq
    dm_base = bh_idx * stride_dm_bh + sq_idx * stride_dm_sq
    ds_base = bh_idx * stride_ds_bh + sq_idx * stride_ds_sq

    # V base for this KV head
    v_base = kv_bh_idx * stride_v_bkv

    # =========================================================================
    # First pass: compute dP for each kv tile, accumulate row_sum
    # =========================================================================
    row_sum = tl.zeros([1], dtype=tl.float32)

    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < seq_kv

        # Load V tile: [BLOCK_KV, HEAD_DIM]
        v_ptrs = v_base + kv_offsets[:, None] * stride_v_skv + d_offsets[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Compute dP tile: dot(dO_row, v_tile.T) -> [BLOCK_KV]
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)  # [BLOCK_KV]

        # Load dropout mask and apply correction
        dm_tile = tl.load(dm_ptr + dm_base + kv_offsets * stride_dm_skv,
                          mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        # Load P tile and accumulate row_sum
        p_tile = tl.load(P_ptr + p_base + kv_offsets * stride_p_skv,
                         mask=kv_mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(dp_tile * p_tile, axis=0)

    # =========================================================================
    # Second pass: compute dS and write output
    # =========================================================================
    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < seq_kv

        # Recompute dP tile
        v_ptrs = v_base + kv_offsets[:, None] * stride_v_skv + d_offsets[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

        dm_tile = tl.load(dm_ptr + dm_base + kv_offsets * stride_dm_skv,
                          mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        p_tile = tl.load(P_ptr + p_base + kv_offsets * stride_p_skv,
                         mask=kv_mask, other=0.0).to(tl.float32)

        ds_tile = p_tile * (dp_tile - row_sum)

        tl.store(dS_ptr + ds_base + kv_offsets * stride_ds_skv,
                 ds_tile.to(tl.bfloat16), mask=kv_mask)


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
    # Eliminates the large [bs, 80, sq, skv] intermediate dP_dropped tensor.
    # Grid: (bs*80, seq_q) — each program handles one (batch_head, q_row)
    # =========================================================================
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)              # [bs*80, sq, 128]
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM) # [bs*8, skv, 128]
    P_flat  = attn_weights.reshape(bs * n_heads, seq_q, seq_kv)       # [bs*80, sq, skv]
    dm_flat = dropout_mask.reshape(bs * n_heads, seq_q, seq_kv)       # [bs*80, sq, skv]

    # Ensure contiguous (inputs should already be, but be safe)
    if not vs_flat.is_contiguous():
        vs_flat = vs_flat.contiguous()
    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    dS_flat = torch.empty((bs * n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_KV for the fused kernel (tile size over kv dimension)
    if seq_kv <= 64:
        BLOCK_KV = 64
    elif seq_kv <= 128:
        BLOCK_KV = 128
    elif seq_kv <= 256:
        BLOCK_KV = 256
    else:
        BLOCK_KV = 256

    grid_fused = (bs * n_heads, seq_q)

    fused_dP_softmax_bwd_kernel[grid_fused](
        dO_flat, vs_flat, P_flat, dm_flat, dS_flat,
        # dO strides
        dO_flat.stride(0), dO_flat.stride(1), dO_flat.stride(2),
        # V strides
        vs_flat.stride(0), vs_flat.stride(1), vs_flat.stride(2),
        # P strides
        P_flat.stride(0), P_flat.stride(1), P_flat.stride(2),
        # dm strides
        dm_flat.stride(0), dm_flat.stride(1), dm_flat.stride(2),
        # dS strides
        dS_flat.stride(0), dS_flat.stride(1), dS_flat.stride(2),
        # dims and params
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

```

---

## Experiment #6 — 2026-06-19 19:18:04 UTC ❌ DISCARD

**Hypothesis:** 1. **3D grid** `(bs*n_kv_heads, n_groups, seq_q)`: `pid0=kv_bh_idx`, `pid1=group_idx`, `pid2=sq_idx`. `bh_idx = kv_bh_idx * n_groups + group_idx` — clean multiplication, no division. This directly map

**Result:** 4329.97 μs

**Kernel code:**
```python
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

```

---

## Experiment #7 — 2026-06-19 19:19:55 UTC ✅ KEEP

**Hypothesis:** 1. **First matmul without GQA expansion**: `value_states [bs,8,skv,128]` → free reshape to `[bs*8, skv, 128]`. `dO_groups_flat [bs*8, 10*sq, 128]` @ `vs_flat.T [bs*8, 128, skv]` → `dP_groups [bs*8, 10

**Result:** 447.18 μs

**Kernel code:**
```python
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

```

---

## Experiment #8 — 2026-06-19 19:21:36 UTC ❌ DISCARD

**Hypothesis:** 1. **Pre-allocate both output tensors** (`dP_groups` and `dV_flat`) on the current default stream before any stream switching — this prevents the CUDA memory allocator from recycling in-use memory whe

**Result:** 501.36 μs

**Kernel code:**
```python
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

```

---

## Experiment #9 — 2026-06-19 19:23:16 UTC ✅ KEEP

**Hypothesis:** 1. **Row-batched Triton kernel** (`fused_softmax_bwd_batched`): A single unified kernel replacing the separate single-pass/two-pass kernels. Each program handles `ROWS_PER_BLOCK` consecutive rows usin

**Result:** 435.14 μs

**Kernel code:**
```python
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

```

---

## Experiment #10 — 2026-06-19 19:25:23 UTC ❌ DISCARD

**Hypothesis:** 1. **`_compute_matmuls` function**: Extracted the dO permute+contiguous, both matmuls (dP and dV), and all reshape operations into a standalone function. This computes: `dO = permute(0,2,1,3).contiguo

**Result:** 820.27 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- Triton softmax-backward kernel with row batching (ROWS_PER_BLOCK rows per program).
- torch.compile applied to the matmul-heavy inner function for graph-level fusion.
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
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
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


def _compute_matmuls(grad_attn_output, value_states, attn_weights_dropped,
                     bs, n_kv_heads, n_groups, seq_q, seq_kv):
    """
    Compiled inner function for the two matmuls.
    torch.compile can fuse permute+contiguous with matmul setup and
    optimize dispatch/allocation overhead.
    """
    # Transpose and make contiguous: [bs, sq, 80, d] -> [bs, 80, sq, d]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    # Group reshape for both matmuls: [bs*8, 10*sq, 128]
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # dP: [bs*8, skv, 128] -> compute [bs*8, 10*sq, skv]
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_groups = torch.matmul(dO_groups_flat, vs_flat.transpose(-2, -1))

    # dV: [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)

    return dP_groups, dV_flat


# Compile the matmul-heavy inner function with max-autotune
_compute_matmuls_compiled = torch.compile(
    _compute_matmuls,
    mode='max-autotune',
    fullgraph=True,
)


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
    # Run the compiled matmul function (handles dO permute, dP, and dV)
    # =========================================================================
    dP_groups, dV_flat = _compute_matmuls_compiled(
        grad_attn_output, value_states, attn_weights_dropped,
        bs, n_kv_heads, n_groups, seq_q, seq_kv
    )

    # =========================================================================
    # Fused softmax backward + dropout correction via Triton.
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

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #11 — 2026-06-19 19:27:34 UTC ❌ DISCARD

**Hypothesis:** 1. **Pre-transposed value_states**: Added `vs_T = vs_flat.transpose(-2, -1).contiguous()` to create a contiguous `[bs*8, 128, skv]` tensor. Then `dP_groups = torch.matmul(dO_groups_flat, vs_T)` is a s

**Result:** 451.27 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP matmul: pre-transposes value_states to contiguous [bs*8, 128, skv] layout
  so cuBLAS gets a fully contiguous B matrix (no "B transposed" GEMM).
- Triton softmax-backward kernel with row batching (ROWS_PER_BLOCK rows per program).
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
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
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
    #
    # Pre-transpose value_states to contiguous [bs*8, 128, skv] so cuBLAS
    # gets a fully contiguous B matrix (avoids "B transposed" GEMM path).
    # Cost: one small transpose of [bs*8, skv, 128] -> [bs*8, 128, skv].
    # Benefit: cuBLAS NN (no-transpose) GEMM which may be faster than NT.
    #
    # matmul([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    # Pre-transpose: [bs*8, skv, 128] -> [bs*8, 128, skv] contiguous
    vs_T = vs_flat.transpose(-2, -1).contiguous()  # [bs*8, 128, skv]

    # dP: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dP_groups = torch.matmul(dO_groups_flat, vs_T)

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

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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

    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #12 — 2026-06-19 19:30:12 UTC ❌ DISCARD

**Hypothesis:** New `fused_dP_softmax_bwd_3d` Triton kernel with 3D grid `(bs, n_heads, seq_q)`:

**Result:** 5209.49 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Fused Triton kernel reads grad_attn_output in native [bs, sq, 80, d] layout,
  computes dP = dO @ V^T tile-by-tile, and immediately applies softmax backward.
  Eliminates the separate permute+contiguous copy of dO AND the dP intermediate.
- cuBLAS for dV still needs the transposed dO copy.
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
def fused_dP_softmax_bwd_strided(
    # grad_attn_output: [bs, sq, n_heads, HEAD_DIM] bfloat16
    # strides: (sq*n_heads*HEAD_DIM, n_heads*HEAD_DIM, HEAD_DIM, 1)
    dO_ptr,
    stride_dO_b,    # bs stride
    stride_dO_sq,   # seq_q stride
    stride_dO_h,    # head stride  (= HEAD_DIM for contiguous last dim)
    # value_states: [bs, n_kv_heads, seq_kv, HEAD_DIM] bfloat16 -> use flat [bs*n_kv_heads, seq_kv, HEAD_DIM]
    V_ptr,
    stride_V_bkv,   # bs*kv_head stride
    stride_V_skv,   # seq_kv stride
    # P (attn_weights): [bs*n_heads, sq, seq_kv] bfloat16, contiguous
    P_ptr,
    stride_P_bh,
    stride_P_sq,
    # dm (dropout_mask): [bs*n_heads, sq, seq_kv] bool, contiguous
    dm_ptr,
    stride_dm_bh,
    stride_dm_sq,
    # dS output: [bs*n_heads, sq, seq_kv] bfloat16, contiguous
    dS_ptr,
    stride_dS_bh,
    stride_dS_sq,
    # Dimensions
    seq_kv,
    n_groups: tl.constexpr,   # 10
    HEAD_DIM: tl.constexpr,   # 128
    scale: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Grid: (bs * n_heads, seq_q)
    Each program: one (batch_head, q_row) pair.

    Reads dO from grad_attn_output in its NATIVE [bs, sq, n_heads, d] layout
    (no prior transpose needed). Computes dP tile-by-tile via dot products
    with V, then applies softmax backward — no intermediate dP tensor written.

    Layout of grad_attn_output: [bs, sq, 80, 128]
    - stride along bs:     sq * 80 * 128
    - stride along sq:     80 * 128
    - stride along head:   128
    - stride along dim:    1
    """
    bh_idx = tl.program_id(0)   # [0, bs*n_heads)
    sq_idx = tl.program_id(1)   # [0, seq_q)

    # Decompose bh_idx into (batch, head)
    # n_heads = 80, but we avoid division: pass n_heads as a param?
    # Instead, use n_groups to get kv_head: kv_bh_idx = bh_idx // n_groups
    # We can compute this as: kv_bh_idx = bh_idx // n_groups
    # But to avoid non-power-of-2 division, note n_groups=10.
    # Use the 3D grid trick: grid = (bs*n_kv_heads, n_groups, seq_q)
    # Then bh_idx = kv_bh_idx * n_groups + group_idx
    # But we're using a 2D grid here...
    # Actually tl.cdiv and // work fine in Triton for constants.
    # n_groups=10 is constexpr so this is compile-time.
    kv_bh_idx = bh_idx // n_groups   # [0, bs*n_kv_heads)

    # Load dO row [HEAD_DIM] from native [bs, sq, n_heads, d] layout
    # bh_idx = b * n_heads + h, so b = bh_idx // n_heads, h = bh_idx % n_heads
    # But we pass strides directly and compute offset:
    # offset = b * stride_dO_b + sq_idx * stride_dO_sq + h * stride_dO_h
    # bh_idx = b * n_heads + h -> but n_heads isn't constexpr here.
    # Use the passed strides: stride_dO_h is HEAD_DIM, stride_dO_sq is n_heads*HEAD_DIM.
    # Offset for this (bh_idx, sq_idx):
    # = b * stride_dO_b + sq_idx * stride_dO_sq + h * HEAD_DIM
    # where b * n_heads + h = bh_idx
    # = (b * stride_dO_b + h * HEAD_DIM) + sq_idx * stride_dO_sq
    # = bh_idx_offset + sq_idx * stride_dO_sq
    # bh_idx_offset = b * stride_dO_b + h * stride_dO_h
    # We precompute this from bh_idx and passed strides.
    # But we need b and h separately... use:
    # b = bh_idx // n_heads_val, h = bh_idx % n_heads_val
    # n_heads = n_groups * n_kv_heads = 80. We know n_groups constexpr.
    # So: b = kv_bh_idx // n_kv_heads, but n_kv_heads varies...
    # Simplest: pass bh_idx_dO_offset as computed on Python side.
    # INSTEAD: note grad_attn_output [bs,sq,80,128] — the (bh_idx, sq_idx) row
    # is at: bh_idx * HEAD_DIM + sq_idx * stride_dO_sq   (treating [bs*80] as batch)
    # wait — the layout is [bs, sq, 80, 128], not [bs, 80, sq, 128].
    # So for (b, h, sq) we need: b*sq*80*128 + sq_idx*80*128 + h*128
    # = (b*80 + h)*128 + sq_idx*80*128... no:
    # = b * (sq*80*128) + sq_idx * (80*128) + h * 128
    # With bh_idx = b*80 + h:
    #   b = bh_idx // 80 (non-power-of-2 division — OK for Triton constexpr)
    # n_heads is not constexpr. We need to be careful.
    # CLEANEST SOLUTION: pass dO_bh_offset and dO_sq_stride explicitly.
    # dO_bh_offset = b * stride_dO_b + h * stride_dO_h  (computed in Python)
    # But these vary per program... use:
    # offset = bh_idx * HEAD_DIM  (treating [bs*80, 128] for the head×dim part)
    #        + sq_idx * stride_dO_sq  (= sq_idx * 80 * 128)
    # This works IF we view grad_attn_output as [bs*sq, 80, 128] and then
    # the (bh_idx, sq_idx) element is at sq_idx*(80*128) + h*128... no.
    # 
    # Actually the simplest correct formula:
    # grad_attn_output[b, sq_idx, h, :] where b*80+h = bh_idx
    # offset = b*stride_b + sq_idx*stride_sq + h*stride_h
    # = b*(sq*80*128) + sq_idx*(80*128) + h*128
    # 
    # Given bh_idx = b*80+h, and stride_dO_b = sq*80*128, stride_dO_sq = 80*128:
    # b*stride_b + h*stride_h = b*(sq*80*128) + h*128
    # This is NOT simply bh_idx * 128 because stride_b >> stride_h.
    #
    # The passed stride_dO_b and stride_dO_h handle this:
    # base_dO = bh_idx_dO_base (passed as extra arg) + sq_idx * stride_dO_sq
    # We need to pass bh_idx_dO_base as a per-program value... can't do that.
    #
    # FINAL SOLUTION: Use 3D grid (bs, n_heads, seq_q) so we have b and h separately.

    # NOTE: This kernel uses a 3D grid (bs, n_heads, seq_q) for clean indexing.
    # Programs are identified differently; see launch site.
    pass


@triton.jit
def fused_dP_softmax_bwd_3d(
    # grad_attn_output: [bs, sq, n_heads, HEAD_DIM] bfloat16
    dO_ptr,
    stride_dO_b,    # = seq_q * n_heads * HEAD_DIM
    stride_dO_sq,   # = n_heads * HEAD_DIM
    stride_dO_h,    # = HEAD_DIM
    # value_states flat: [bs*n_kv_heads, seq_kv, HEAD_DIM] bfloat16, contiguous
    V_ptr,
    stride_V_bkv,   # = seq_kv * HEAD_DIM
    stride_V_skv,   # = HEAD_DIM (contiguous)
    # P (attn_weights) flat: [bs*n_heads, seq_q, seq_kv] bfloat16, contiguous
    P_ptr,
    stride_P_bh,    # = seq_q * seq_kv
    stride_P_sq,    # = seq_kv
    # dm flat: [bs*n_heads, seq_q, seq_kv] bool, contiguous
    dm_ptr,
    stride_dm_bh,
    stride_dm_sq,
    # dS flat: [bs*n_heads, seq_q, seq_kv] bfloat16, contiguous
    dS_ptr,
    stride_dS_bh,
    stride_dS_sq,
    # Dims
    n_heads,        # 80
    n_kv_heads,     # 8
    n_groups: tl.constexpr,  # 10
    seq_kv,
    HEAD_DIM: tl.constexpr,  # 128
    scale: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Grid: (bs, n_heads, seq_q)
    Each program handles one (batch b, head h, query row sq_idx).
    Reads dO from native [bs, sq, n_heads, HEAD_DIM] layout — NO COPY.
    Computes dP tile-by-tile, applies softmax backward, writes dS.
    """
    b_idx    = tl.program_id(0)
    h_idx    = tl.program_id(1)
    sq_idx   = tl.program_id(2)

    kv_h_idx = h_idx // n_groups  # [0, n_kv_heads)

    # Flat head index for P, dm, dS: bh = b * n_heads + h
    bh_idx = b_idx * n_heads + h_idx

    # Load dO row [HEAD_DIM] from native layout
    # grad_attn_output[b, sq_idx, h, :] offset:
    dO_base = b_idx * stride_dO_b + sq_idx * stride_dO_sq + h_idx * stride_dO_h
    d_offs = tl.arange(0, HEAD_DIM)
    dO_row = tl.load(dO_ptr + dO_base + d_offs).to(tl.float32)

    # P and dm row bases
    p_base  = bh_idx * stride_P_bh  + sq_idx * stride_P_sq
    dm_base = bh_idx * stride_dm_bh + sq_idx * stride_dm_sq
    ds_base = bh_idx * stride_dS_bh + sq_idx * stride_dS_sq

    # V base for this KV head: flat index kv_bh = b * n_kv_heads + kv_h
    kv_bh = b_idx * n_kv_heads + kv_h_idx
    v_base = kv_bh * stride_V_bkv

    # =========================================================================
    # First pass: compute row_sum = sum_kv(dP[kv] * P[kv])
    # dP[kv] = dot(dO_row, V[kv, :]) * mask * scale
    # =========================================================================
    row_sum = tl.zeros([1], dtype=tl.float32)

    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        # V tile: [BLOCK_KV, HEAD_DIM]
        v_ptrs = v_base + kv_offs[:, None] * stride_V_skv + d_offs[None, :]
        v_tile = tl.load(V_ptr + v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # dP tile: dot product -> [BLOCK_KV]
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

        # Dropout mask
        dm_tile = tl.load(dm_ptr + dm_base + kv_offs, mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        # P tile
        p_tile = tl.load(P_ptr + p_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)

        row_sum += tl.sum(dp_tile * p_tile, axis=0)

    # =========================================================================
    # Second pass: compute dS and write output
    # =========================================================================
    for kv_start in range(0, seq_kv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < seq_kv

        v_ptrs = v_base + kv_offs[:, None] * stride_V_skv + d_offs[None, :]
        v_tile = tl.load(V_ptr + v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        dp_tile = tl.sum(dO_row[None, :] * v_tile, axis=1)

        dm_tile = tl.load(dm_ptr + dm_base + kv_offs, mask=kv_mask, other=0).to(tl.int1)
        dp_tile = tl.where(dm_tile, dp_tile * scale, 0.0)

        p_tile = tl.load(P_ptr + p_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)

        ds_tile = p_tile * (dp_tile - row_sum)
        tl.store(dS_ptr + ds_base + kv_offs, ds_tile.to(tl.bfloat16), mask=kv_mask)


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

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    device = grad_attn_output.device

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Step 1: Fused dP + softmax backward via Triton.
    # Reads grad_attn_output in NATIVE [bs, sq, 80, 128] layout — no copy!
    # Computes dP = dO @ V^T tile-by-tile and applies softmax backward.
    # Grid: (bs, n_heads, seq_q) = (bs, 80, seq_q)
    # =========================================================================
    # Flatten P and dm for Triton: [bs*80, seq_q, seq_kv]
    P_flat  = attn_weights.reshape(bs * n_heads, seq_q, seq_kv)
    dm_flat = dropout_mask.reshape(bs * n_heads, seq_q, seq_kv)
    if not P_flat.is_contiguous():
        P_flat = P_flat.contiguous()
    if not dm_flat.is_contiguous():
        dm_flat = dm_flat.contiguous()

    # V flat: [bs*8, seq_kv, 128]
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    if not vs_flat.is_contiguous():
        vs_flat = vs_flat.contiguous()

    dS_flat = torch.empty((bs * n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_KV
    if seq_kv <= 64:
        BLOCK_KV = 64
    elif seq_kv <= 128:
        BLOCK_KV = 128
    elif seq_kv <= 256:
        BLOCK_KV = 256
    else:
        BLOCK_KV = 256

    grid_3d = (bs, n_heads, seq_q)

    fused_dP_softmax_bwd_3d[grid_3d](
        grad_attn_output,
        grad_attn_output.stride(0), grad_attn_output.stride(1), grad_attn_output.stride(2),
        vs_flat,
        vs_flat.stride(0), vs_flat.stride(1),
        P_flat,
        P_flat.stride(0), P_flat.stride(1),
        dm_flat,
        dm_flat.stride(0), dm_flat.stride(1),
        dS_flat,
        dS_flat.stride(0), dS_flat.stride(1),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        seq_kv=seq_kv,
        HEAD_DIM=HEAD_DIM,
        scale=scale,
        BLOCK_KV=BLOCK_KV,
    )

    dS = dS_flat.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 2: Make dO contiguous for the dV cuBLAS matmul.
    # This copy is unavoidable for cuBLAS correctness.
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 3: Compute dV without GQA expansion (bfloat16 matmul).
    # [bs*8, 10*sq, skv]^T @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.matmul(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #13 — 2026-06-19 19:31:53 UTC ✅ KEEP

**Hypothesis:** 1. **torch.bmm for dP**: `torch.bmm(dO_groups_flat, vs_flat.transpose(-2,-1))` — strict batched GEMM, no fallback paths from torch.matmul's overloaded dispatch.

**Result:** 429.16 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dV matmul: swapped operand order to dO_groups_flat.T @ attn_groups_flat
  changing which tensor cuBLAS treats as M vs N.
- torch.bmm used instead of torch.matmul for stricter batched GEMM path.
- Triton softmax-backward kernel with row batching.
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T without GQA expansion.
    # value_states: [bs*8, skv, 128]
    # dP: [bs*8, 10*sq, skv]
    # Use torch.bmm for strict batched GEMM path.
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    # bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    dP_groups = torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1))

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

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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
    #
    # Swapped operand order vs previous:
    #   Previous: attn_groups_flat.T @ dO_groups_flat
    #     = [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    #   New: dO_groups_flat.T @ attn_groups_flat
    #     = [bs*8, 128, 10*sq] @ [bs*8, 10*sq, skv] -> [bs*8, 128, skv]
    #   Then transpose result -> [bs*8, skv, 128]
    #
    # This changes which dimension cuBLAS treats as M vs N.
    # For typical cases where skv > 128, this makes M=128 and N=skv
    # (instead of M=skv and N=128), which may better match cuBLAS tile sizes.
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # bmm([bs*8, 128, 10*sq], [bs*8, 10*sq, skv]) -> [bs*8, 128, skv]
    dV_T = torch.bmm(dO_groups_flat.transpose(-2, -1), attn_groups_flat)
    # Transpose to get [bs*8, skv, 128]
    dV_flat = dV_T.transpose(-2, -1).contiguous()

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #14 — 2026-06-19 19:33:46 UTC 💥 CRASH

**Hypothesis:** The dV computation now uses a single `torch.mm` instead of a batched `torch.bmm`:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape — no GQA expansion anywhere.
- dV: uses single 2D torch.mm by collapsing all batch dims into one large GEMM.
  [bs*8*skv, 10*sq] @ [bs*8*10*sq, 128] -> [bs*8*skv, 128]
  This replaces the batched GEMM with one large matrix multiply.
- dP: torch.bmm as before.
- Triton softmax-backward kernel with row batching.
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T without GQA expansion.
    # value_states: [bs*8, skv, 128]
    # dP: [bs*8, 10*sq, skv]
    # Use torch.bmm for strict batched GEMM path.
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    # bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    dP_groups = torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1))

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

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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
    # Step 4: Compute dV using a single 2D torch.mm (no batched GEMM).
    #
    # Math: dV[b, kv_h, s, d] = sum_{g,q} P_dropped[b, kv_h*10+g, q, s] * dO[b, kv_h*10+g, q, d]
    #
    # attn_weights_dropped [bs, 80, sq, skv] layout:
    #   Reshape to [bs*8, 10*sq, skv] then transpose(-2,-1) -> [bs*8, skv, 10*sq]
    #   Further reshape to [bs*8*skv, 10*sq]
    #
    # dO_groups_flat [bs*8, 10*sq, 128]:
    #   Reshape to [bs*8*10*sq, 128]
    #
    # Both are contiguous, so the 2D reshape is a free view.
    # Then: torch.mm([bs*8*skv, 10*sq], [bs*8*10*sq, 128]) -> [bs*8*skv, 128]
    #
    # CRITICAL: the row order of attn_T_2d must align with the row order of dO_2d.
    # attn_T_2d row i corresponds to (b, kv_h, s) triple at position i = b*8*skv + kv_h*skv + s
    # dO_2d row j corresponds to (b, kv_h, g, q) at position j = b*8*10*sq + kv_h*10*sq + g*sq + q
    # The contraction is over the K=10*sq dimension: sum_{g,q} attn[b,kv_h,g,q,s] * dO[b,kv_h,g,q,d]
    # This EXACTLY matches when we use:
    #   attn_T_2d = attn_groups_flat.transpose(-2,-1).reshape(bs*8*skv, 10*sq) — NOT contiguous!
    #   We need contiguous for this to work as a 2D mm.
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # attn_groups_flat: [bs*8, 10*sq, skv], contiguous

    # Transpose: [bs*8, skv, 10*sq] — NOT contiguous, need .contiguous()
    attn_T = attn_groups_flat.transpose(-2, -1).contiguous()  # [bs*8, skv, 10*sq]

    # Reshape to 2D: [bs*8*skv, 10*sq]
    attn_T_2d = attn_T.reshape(bs * n_kv_heads * seq_kv, n_groups * seq_q)

    # dO_groups_flat is already [bs*8, 10*sq, 128] contiguous
    # Reshape to 2D: [bs*8*10*sq, 128]
    dO_2d = dO_groups_flat.reshape(bs * n_kv_heads * n_groups * seq_q, HEAD_DIM)

    # Single large 2D matmul: [bs*8*skv, 10*sq] @ [bs*8*10*sq, 128] -> [bs*8*skv, 128]
    dV_2d = torch.mm(attn_T_2d, dO_2d)

    dV = dV_2d.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #15 — 2026-06-19 19:35:50 UTC ❌ DISCARD

**Hypothesis:** New `dV_tiled_kernel` Triton kernel:

**Result:** 431.42 μs

**Kernel code:**
```python
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

```

---

## Experiment #16 — 2026-06-19 19:37:22 UTC ❌ DISCARD

**Hypothesis:** 1. **Clean dV matmul**: `torch.bmm(attn_groups_flat.transpose(-2,-1), dO_groups_flat)` — the `.transpose(-2,-1)` creates a non-contiguous view which cuBLAS handles via its NT GEMM path. The output `dV

**Result:** 441.18 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- torch.bmm for both dP and dV.
- dV: clean attn_groups_flat.T @ dO_groups_flat (no extra .contiguous() on result).
- Triton softmax-backward with increased ROWS_PER_BLOCK for mid-range seq_kv.
- Removed unnecessary is_contiguous() checks for input tensors.
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # =========================================================================
    # Step 2: Compute dP = dO @ V^T without GQA expansion.
    # torch.bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_groups = torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1))

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    # Flatten to [total_rows, seq_kv] — free views (all contiguous)
    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    # Input tensors are guaranteed contiguous — skip is_contiguous() checks
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv.
    # Increased ROWS_PER_BLOCK for mid-range seq_kv to improve SM occupancy.
    if seq_kv <= 128:
        BLOCK_SKV = 128
        ROWS_PER_BLOCK = 16
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        ROWS_PER_BLOCK = 16   # increased from 8
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        ROWS_PER_BLOCK = 8    # increased from 4
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        ROWS_PER_BLOCK = 4    # increased from 2
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 2    # increased from 1
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
    # Step 4: Compute dV without GQA expansion.
    # Clean formulation: attn.T @ dO — cuBLAS handles NT GEMM efficiently.
    # attn_groups_flat: [bs*8, 10*sq, skv]
    # attn_groups_flat.T: [bs*8, skv, 10*sq] (non-contiguous, cuBLAS NT path)
    # dO_groups_flat: [bs*8, 10*sq, 128]
    # Result: [bs*8, skv, 128] — contiguous output, no extra .contiguous() needed.
    # =========================================================================
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # torch.bmm with non-contiguous first arg (transposed view) — cuBLAS NT GEMM
    dV_flat = torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #17 — 2026-06-19 19:39:01 UTC ✅ KEEP

**Hypothesis:** 1. **Module-level cached stream**: `_side_stream` and `_dO_ready_event` are created once at module level and reused every call, eliminating `torch.cuda.Stream()` creation overhead from the hot path.

**Result:** 417.10 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- Triton softmax-backward with row batching overlaps with dV on stream1.
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # This prevents allocator interference when we switch to the side stream.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV intermediate: [bs*8, 128, skv] (from swapped dV formulation in #13)
    dV_T = torch.empty(
        (bs * n_kv_heads, HEAD_DIM, seq_kv),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Current stream: dP bmm → Triton softmax
    # - Side stream: dV bmm
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, 128, 10*sq], [bs*8, 10*sq, skv]) -> [bs*8, 128, skv]
        # Using proven #13 formulation: dO.T @ attn (M=128, N=skv, K=10*sq)
        torch.bmm(dO_groups_flat.transpose(-2, -1), attn_groups_flat, out=dV_T)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — can overlap with dV on side stream.
    # dP_groups is ready (main stream is in-order).
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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

    # Finalize dV: transpose [bs*8, 128, skv] -> [bs*8, skv, 128] then reshape
    dV_flat = dV_T.transpose(-2, -1).contiguous()
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #18 — 2026-06-19 19:40:40 UTC ✅ KEEP

**Hypothesis:** 1. **Direct dV output layout**: Pre-allocate `dV_flat = torch.empty((bs*n_kv_heads, seq_kv, HEAD_DIM), ...)` as `[bs*8, skv, 128]` — the final target layout.

**Result:** 394.30 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching overlaps with dV on side stream.
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV output: [bs*8, skv, 128] — direct final layout, no post-transpose needed.
    # attn.T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # attn_groups_flat.transpose(-2,-1) is a non-contiguous [bs*8, skv, 10*sq] view,
    # cuBLAS handles this as TN GEMM (transpose first arg). Output is contiguous.
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → Triton softmax
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
        # attn_groups_flat.T is non-contiguous: cuBLAS TN (transpose-N) GEMM
        # Output dV_flat is directly contiguous [bs*8, skv, 128] — no post-copy.
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #19 — 2026-06-19 19:42:13 UTC ❌ DISCARD

**Hypothesis:** 1. **`_output_cache` dict**: Module-level dict mapping `(bs, seq_q, seq_kv, device_index)` → `(dP_groups, dV_flat, dS_flat)` tuple of pre-allocated tensors.

**Result:** 397.27 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream, event, AND output tensors to eliminate allocator calls.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching overlaps with dV on side stream.
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

# Module-level output tensor cache: maps (bs, seq_q, seq_kv, device_index) -> (dP_groups, dV_flat, dS_flat)
# All tensors are completely overwritten before being read, so reuse is safe.
_output_cache = {}

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


def _get_output_tensors(bs, seq_q, seq_kv, n_heads, n_kv_heads, n_groups, device):
    """
    Returns pre-allocated output tensors for the given shape.
    Creates and caches them on first call; reuses on subsequent calls.
    Safe to reuse since all tensors are fully overwritten before being read.
    """
    global _output_cache
    key = (bs, seq_q, seq_kv, device.index if hasattr(device, 'index') else 0)
    if key not in _output_cache:
        total_rows = bs * n_heads * seq_q
        _output_cache[key] = (
            # dP_groups: [bs*8, 10*sq, skv]
            torch.empty((bs * n_kv_heads, n_groups * seq_q, seq_kv),
                        dtype=torch.bfloat16, device=device),
            # dV_flat: [bs*8, skv, 128]
            torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                        dtype=torch.bfloat16, device=device),
            # dS_flat: [total_rows, skv]
            torch.empty((total_rows, seq_kv),
                        dtype=torch.bfloat16, device=device),
        )
    return _output_cache[key]


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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Get pre-allocated output tensors from cache (no allocator calls on repeat).
    # All tensors are fully overwritten before being read — safe to reuse.
    # =========================================================================
    dP_groups, dV_flat, dS_flat = _get_output_tensors(
        bs, seq_q, seq_kv, n_heads, n_kv_heads, n_groups, device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → Triton softmax
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: bmm([bs*8, skv, 10*sq], [bs*8, 10*sq, 128]) -> [bs*8, skv, 128]
        # attn_groups_flat.T is non-contiguous: cuBLAS TN GEMM
        # Output dV_flat is directly contiguous [bs*8, skv, 128] — no post-copy.
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # Choose BLOCK_SKV and ROWS_PER_BLOCK based on seq_kv
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

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape (free view)
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #20 — 2026-06-19 19:43:54 UTC ❌ DISCARD

**Hypothesis:** - seq_kv ≤ 128: ROWS_PER_BLOCK = **32** (was 16)

**Result:** 458.08 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream to avoid creation overhead in hot path.
- Pre-allocated output tensors before any stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward: doubled ROWS_PER_BLOCK (32,16,8,4,2) vs prior (16,8,4,2,1).
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
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
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Doubled ROWS_PER_BLOCK values: (32, 16, 8, 4, 2) instead of (16, 8, 4, 2, 1)
    # Reduces grid size by 2x to better amortize kernel launch overhead on B200.
    if seq_kv <= 128:
        BLOCK_SKV = 128
        ROWS_PER_BLOCK = 32
    elif seq_kv <= 256:
        BLOCK_SKV = 256
        ROWS_PER_BLOCK = 16
    elif seq_kv <= 512:
        BLOCK_SKV = 512
        ROWS_PER_BLOCK = 8
    elif seq_kv <= 1024:
        BLOCK_SKV = 1024
        ROWS_PER_BLOCK = 4
    elif seq_kv <= 2048:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 2
    else:
        BLOCK_SKV = 2048
        ROWS_PER_BLOCK = 2

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

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #21 — 2026-06-19 19:45:47 UTC ❌ DISCARD

**Hypothesis:** New `transpose_to_groups_kernel` Triton kernel:

**Result:** 466.80 μs

**Kernel code:**
```python
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

```

---

## Experiment #22 — 2026-06-19 19:47:38 UTC ❌ DISCARD

**Hypothesis:** 1. **Full tensor cache**: `_tensor_cache` dict maps `(bs, seq_q, seq_kv, device_idx)` → dict of pre-allocated tensors: `dO [bs,80,sq,128]`, `dP_groups [bs*8,10*sq,skv]`, `dV_flat [bs*8,skv,128]`, `dS_

**Result:** 394.36 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel (#18 baseline + cached dO buffer):
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached stream, event, AND dO_groups_flat buffer.
- dO copy: writes into pre-allocated cached buffer via permute+copy_ to avoid allocation.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching (proven #18 ROWS_PER_BLOCK values).
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

# Module-level cached CUDA stream and event
_side_stream = None
_dO_ready_event = None

# Module-level cached tensors: maps (bs, seq_q, seq_kv, device_idx) -> tensors
# Caches: dO_groups_flat, dP_groups, dV_flat, dS_flat
_tensor_cache = {}

def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


def _get_cached_tensors(bs, seq_q, seq_kv, n_heads, n_kv_heads, n_groups, device):
    global _tensor_cache
    dev_idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    key = (bs, seq_q, seq_kv, dev_idx)
    if key not in _tensor_cache:
        n_gq = n_groups * seq_q
        total_rows = bs * n_heads * seq_q
        _tensor_cache[key] = {
            # dO in [bs, 80, sq, 128] layout — written by permute+copy_
            'dO': torch.empty((bs, n_heads, seq_q, HEAD_DIM),
                              dtype=torch.bfloat16, device=device),
            # dP output: [bs*8, 10*sq, skv]
            'dP_groups': torch.empty((bs * n_kv_heads, n_gq, seq_kv),
                                     dtype=torch.bfloat16, device=device),
            # dV output: [bs*8, skv, 128]
            'dV_flat': torch.empty((bs * n_kv_heads, seq_kv, HEAD_DIM),
                                   dtype=torch.bfloat16, device=device),
            # dS output: [total_rows, skv]
            'dS_flat': torch.empty((total_rows, seq_kv),
                                   dtype=torch.bfloat16, device=device),
        }
    return _tensor_cache[key]


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

    # Get cached tensors (no allocation on repeated calls with same shape)
    cache = _get_cached_tensors(bs, seq_q, seq_kv, n_heads, n_kv_heads, n_groups, device)
    dO_buf      = cache['dO']         # [bs, 80, sq, 128] pre-allocated
    dP_groups   = cache['dP_groups']  # [bs*8, 10*sq, skv]
    dV_flat     = cache['dV_flat']    # [bs*8, skv, 128]
    dS_flat     = cache['dS_flat']    # [total_rows, skv]

    # =========================================================================
    # Step 1: Transpose grad_attn_output [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    # Write into pre-allocated buffer using copy_ to avoid allocation.
    # =========================================================================
    dO_buf.copy_(grad_attn_output.permute(0, 2, 1, 3))
    # dO_buf: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO_buf.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both bmms read from dO_groups_flat (concurrent reads are safe).
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    dO_ready_event.record(main_stream)

    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # dP: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    # Proven optimal ROWS_PER_BLOCK from #18
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

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #23 — 2026-06-19 19:49:07 UTC 💥 CRASH

**Hypothesis:** 1. **Cached main_stream**: `_main_stream` stored at module level alongside `_side_stream` and `_dO_ready_event`. `_init_streams()` called once on first invocation. Eliminates repeated `torch.cuda.curr

**Result:** CRASH

**Error:**
```
Benchmark correctness failed. Benchmark correctness: grad_attn_scores mismatch: max=2.5353e+30 mean=6.2763e+27
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel (#18 + cached main stream + dO copy on side stream):
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached main_stream, side_stream, and event.
- dO copy launched on side stream while CPU sets up reshape/alloc on main thread.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching (proven #18 ROWS_PER_BLOCK values).
- All in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

# Module-level cached streams and event
_main_stream = None
_side_stream = None
_dO_ready_event = None


def _init_streams(device):
    global _main_stream, _side_stream, _dO_ready_event
    if _side_stream is None:
        _main_stream = torch.cuda.current_stream(device)
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()


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

    # Initialize streams once
    _init_streams(device)
    main_stream = _main_stream
    side_stream = _side_stream
    dO_ready_event = _dO_ready_event

    # =========================================================================
    # Step 1: Allocate dO output buffer and launch the transpose copy on the
    # SIDE STREAM. While it runs on GPU, CPU sets up reshape/alloc below.
    # =========================================================================
    dO = torch.empty((bs, n_heads, seq_q, HEAD_DIM), dtype=torch.bfloat16, device=device)

    # Launch dO copy on side stream: [bs,sq,80,128] -> [bs,80,sq,128]
    with torch.cuda.stream(side_stream):
        dO.copy_(grad_attn_output.permute(0, 2, 1, 3))

    # =========================================================================
    # CPU-side setup runs while dO copy executes on GPU side stream:
    # - Prepare free-view reshapes of input tensors (no GPU work)
    # - Allocate output buffers on main stream
    # =========================================================================
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Allocate outputs on main stream
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Wait for dO copy, then launch both bmms concurrently.
    # Record event on side_stream after dO is done; main_stream waits for it.
    # =========================================================================
    # Record dO-ready event on the side stream
    dO_ready_event.record(side_stream)

    # Both bmms need dO — main stream waits for side stream's dO copy to finish
    main_stream.wait_event(dO_ready_event)

    # dO_groups_flat: free reshape of dO (no copy)
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Launch dV on side stream (concurrent with dP on main stream)
    side_stream.wait_event(dO_ready_event)  # side stream already past this point, but safe
    with torch.cuda.stream(side_stream):
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Triton softmax backward (main stream, overlaps with dV on side).
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

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

    # Wait for side stream (dV) to complete
    main_stream.wait_stream(side_stream)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #24 — 2026-06-19 19:50:43 UTC ❌ DISCARD

**Hypothesis:** 1. **`torch.cuda.current_stream()` without device argument**: Avoids a Python-level device lookup. Safe because we're always operating on the current CUDA device.

**Result:** 395.29 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel (clean #18 + minor safe micro-optimizations):
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached side_stream and event.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching (proven optimal ROWS_PER_BLOCK).
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

# Module-level cached side stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None

def _get_side_stream():
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


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
    # Step 1: Transpose dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128] (bfloat16).
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    # All reshape views up front (free, no GPU work)
    n_kv_batch = bs * n_kv_heads
    n_gq       = n_groups * seq_q
    dO_groups_flat   = dO.reshape(n_kv_batch, n_gq, HEAD_DIM)
    vs_flat          = value_states.reshape(n_kv_batch, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(n_kv_batch, n_gq, seq_kv)
    total_rows       = bs * n_heads * seq_q
    P_flat           = attn_weights.reshape(total_rows, seq_kv)
    dm_flat          = dropout_mask.reshape(total_rows, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on current stream before stream switching.
    # =========================================================================
    dP_groups = torch.empty((n_kv_batch, n_gq, seq_kv), dtype=torch.bfloat16, device=device)
    dV_flat   = torch.empty((n_kv_batch, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=device)
    dS_flat   = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both bmms read dO_groups_flat (safe concurrent reads).
    # =========================================================================
    main_stream = torch.cuda.current_stream()   # no device arg — slightly faster
    side_stream, dO_ready_event = _get_side_stream()

    dO_ready_event.record(main_stream)
    side_stream.wait_event(dO_ready_event)

    with torch.cuda.stream(side_stream):
        # dV: [n_kv_batch, skv, n_gq] @ [n_kv_batch, n_gq, 128] -> [n_kv_batch, skv, 128]
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # dP on main stream: [n_kv_batch, n_gq, 128] @ [n_kv_batch, 128, skv] -> [n_kv_batch, n_gq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Triton softmax backward (main stream, overlaps with dV on side).
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)

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

    # Sync: wait for dV on side stream
    main_stream.wait_stream(side_stream)

    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #25 — 2026-06-19 19:52:16 UTC ❌ DISCARD

**Hypothesis:** Verbatim #18 logic:

**Result:** 460.57 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — final clean version of #18:
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached side stream and event.
- Pre-allocated output tensors before stream switching.
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy).
- Triton softmax-backward with row batching (proven optimal ROWS_PER_BLOCK).
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

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands (all free views, no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # =========================================================================
    # dP output: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV output: [bs*8, skv, 128] — direct final layout, no post-transpose needed.
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Concurrent stream execution.
    # Both matmuls read from dO_groups_flat (concurrent reads are safe).
    # - Side stream: dV bmm (attn.T @ dO → directly contiguous [bs*8, skv, 128])
    # - Main stream: dP bmm → Triton softmax
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Record event: dO is ready on the main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO to be ready, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # dV: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
        # attn_groups_flat.T is non-contiguous: cuBLAS TN GEMM
        # Output dV_flat is directly contiguous [bs*8, skv, 128] — no post-copy.
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (concurrent with dV on side stream)
    # dP: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Runs on main stream — overlaps with dV on side stream.
    # dP_groups is ready (main stream is in-order).
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Proven optimal ROWS_PER_BLOCK from experiments
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

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is already [bs*8, skv, 128] contiguous — just reshape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

---

## Experiment #26 — 2026-06-19 19:53:34 UTC ❌ DISCARD

**Hypothesis:** Exact #18 algorithm with clean comments:

**Result:** 457.55 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — proven best configuration (#18):
- Both matmuls use group-reshape [bs*8, 10*sq, ...] — no GQA expansion anywhere.
- dP and dV bmms launched concurrently on separate CUDA streams.
- Module-level cached side stream and event.
- Pre-allocated output tensors before stream switching (using out= parameter).
- dV: direct attn.T @ dO -> [bs*8, skv, 128] (no post-transpose copy needed).
- Triton softmax-backward with row batching (proven optimal ROWS_PER_BLOCK).
- All matmuls in bfloat16 (tensor cores).

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

# Module-level cached CUDA side stream and event (created once, reused every call)
_side_stream = None
_dO_ready_event = None


def _get_side_stream(device):
    global _side_stream, _dO_ready_event
    if _side_stream is None:
        _side_stream = torch.cuda.Stream(device)
        _dO_ready_event = torch.cuda.Event()
    return _side_stream, _dO_ready_event


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
    For seq_kv <= BLOCK_SKV: single-pass (load once, compute row_sum, write).
    For seq_kv > BLOCK_SKV: two-pass (accumulate row_sum, then write).
    Grid: ceil(total_rows / ROWS_PER_BLOCK)
    """
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK

    for r in tl.static_range(ROWS_PER_BLOCK):
        row_id = row_start + r
        if row_id < total_rows:
            base = row_id * seq_kv

            if BLOCK_SKV >= seq_kv:
                # Single-pass: all elements fit in one block
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
    # Step 1: Transpose dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128] (bfloat16).
    # One contiguous() call; all subsequent reshapes are free views.
    # =========================================================================
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()
    # dO: [bs, 80, sq, 128], bfloat16, contiguous

    # Shared group-reshape for both matmuls: [bs*8, 10*sq, 128] — free view
    dO_groups_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)

    # Prepare matmul operands — all free views (no copies)
    vs_flat = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    attn_groups_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # =========================================================================
    # Pre-allocate output tensors on the CURRENT stream before any switching.
    # Using out= parameter in bmm prevents allocator interference between streams.
    # =========================================================================
    # dP: [bs*8, 10*sq, skv]
    dP_groups = torch.empty(
        (bs * n_kv_heads, n_groups * seq_q, seq_kv),
        dtype=torch.bfloat16, device=device
    )
    # dV: [bs*8, skv, 128] — final layout directly, no post-transpose needed
    dV_flat = torch.empty(
        (bs * n_kv_heads, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=device
    )

    # =========================================================================
    # Step 2: Launch dP and dV bmms concurrently on separate CUDA streams.
    # Both read from dO_groups_flat (safe concurrent reads).
    # dV on side stream, dP on main stream.
    # =========================================================================
    main_stream = torch.cuda.current_stream(device)
    side_stream, dO_ready_event = _get_side_stream(device)

    # Signal that dO is ready on main stream
    dO_ready_event.record(main_stream)

    # Side stream waits for dO, then launches dV
    side_stream.wait_event(dO_ready_event)
    with torch.cuda.stream(side_stream):
        # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
        # attn_groups_flat.transpose(-2,-1) is non-contiguous: cuBLAS TN GEMM
        # Result dV_flat is contiguous [bs*8, skv, 128] — no post-copy needed
        torch.bmm(attn_groups_flat.transpose(-2, -1), dO_groups_flat, out=dV_flat)

    # Launch dP on main stream (runs concurrently with dV on side stream)
    # [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    torch.bmm(dO_groups_flat, vs_flat.transpose(-2, -1), out=dP_groups)

    # =========================================================================
    # Step 3: Fused softmax backward + dropout correction via Triton.
    # Main stream only — dP is ready (in-order), overlaps with dV on side stream.
    # =========================================================================
    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    total_rows = bs * n_heads * seq_q

    dP_dropped_flat = dP_groups.reshape(total_rows, seq_kv)
    P_flat  = attn_weights.reshape(total_rows, seq_kv)
    dm_flat = dropout_mask.reshape(total_rows, seq_kv)

    dS_flat = torch.empty((total_rows, seq_kv), dtype=torch.bfloat16, device=device)

    # Proven optimal ROWS_PER_BLOCK: balance between parallelism and register pressure
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

    # Wait for side stream (dV) to complete — dV_flat is already in final layout
    main_stream.wait_stream(side_stream)

    # dV_flat is [bs*8, skv, 128] contiguous — free reshape to final shape
    dV = dV_flat.reshape(bs, n_kv_heads, seq_kv, HEAD_DIM)

    return dS, dV

```

