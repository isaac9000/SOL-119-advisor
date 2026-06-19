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
