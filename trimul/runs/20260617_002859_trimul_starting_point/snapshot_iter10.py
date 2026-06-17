"""
TriMul submission — low-overhead functional PyTorch baseline.

Eliminates per-call module construction and parameter re-wrapping. Operates
directly on the provided weight tensors with functional ops, avoids redundant
dtype casts, and expresses the contraction as a single batched matmul.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _contract_configs():
    cfgs = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (2, 3, 4):
                        cfgs.append(triton.Config(
                            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_contract_configs(), key=['N'])
@triton.jit
def _trimul_contract_kernel(
    left_ptr, right_ptr, out_ptr,
    BH, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Contraction per (b,d) batch: out[bh, i, j] = sum_k left[bh,i,k]*right[bh,j,k]
    # left/right are (BH, N, N) contiguous fp16 -> K (last dim) is unit-stride
    # (same contiguous-K layout cuBLAS bmm enjoys). out is (BH, N, N) fp32.
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    base = pid_bh * (N * N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, N, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < N
        # left tile (BLOCK_M, BLOCK_K): offset = base + i*N + k  (k unit-stride)
        l_off = base + offs_i[:, None] * N + k[None, :]
        l_mask = (offs_i[:, None] < N) & k_mask[None, :]
        l = tl.load(left_ptr + l_off, mask=l_mask, other=0.0)
        # right tile (BLOCK_N, BLOCK_K): offset = base + j*N + k
        r_off = base + offs_j[:, None] * N + k[None, :]
        r_mask = (offs_j[:, None] < N) & k_mask[None, :]
        r = tl.load(right_ptr + r_off, mask=r_mask, other=0.0)
        acc += tl.dot(l, r.T)

    # out[bh, i, j]: offset = base + i*N + j  (contiguous, coalesced store)
    o_off = base + offs_i[:, None] * N + offs_j[None, :]
    o_mask = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(out_ptr + o_off, acc, mask=o_mask)


def _trimul_contract(left, right, bs, n, h):
    # left, right: (BH, N, N) fp16 contiguous (gated/masked already).
    BH, N, _ = left.shape
    out = torch.empty((BH, N, N), dtype=torch.float32, device=left.device)
    grid = lambda meta: (BH, triton.cdiv(N, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _trimul_contract_kernel[grid](
        left, right, out,
        BH, N,
    )
    return out


def _pre_stage(x, mask, dim,
               norm_w, norm_b, lp_w, rp_w, lg_w, rg_w, og_w):
    x = F.layer_norm(x, (dim,), norm_w, norm_b)
    left = F.linear(x, lp_w)
    right = F.linear(x, rp_w)
    left_gate = torch.sigmoid(F.linear(x, lg_w))
    right_gate = torch.sigmoid(F.linear(x, rg_w))
    out_gate = torch.sigmoid(F.linear(x, og_w))
    m = mask.unsqueeze(-1)
    bs, n, _, h = left.shape
    # Produce contiguous-K (B*H, N, N) fp16 layout for the contraction kernel:
    # permute (b,n,n,h)->(b,h,n,n) so the contraction axis k becomes innermost.
    left = (left * m * left_gate).permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16).contiguous()
    right = (right * m * right_gate).permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16).contiguous()
    return left, right, out_gate, bs, n, h


def _post_stage(out, out_gate, hidden_dim, ton_w, ton_b, out_w):
    out = F.layer_norm(out, (hidden_dim,), ton_w, ton_b)
    out = out * out_gate
    out = F.linear(out, out_w)
    return out


# Keep healthy elementwise/projection stages in torch.compile; the Triton kernel
# owns the contraction and consumes/produces (B,N,N,H) layout (no permute copies).
_compiled = {}


def _get_compiled(dim, hidden_dim):
    key = (dim, hidden_dim)
    fns = _compiled.get(key)
    if fns is None:
        pre = torch.compile(_pre_stage, dynamic=True)
        post = torch.compile(_post_stage, dynamic=True)
        fns = (pre, post)
        _compiled[key] = fns
    return fns


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    pre, post = _get_compiled(dim, hidden_dim)

    left, right, out_gate, bs, n, h = pre(
        input_tensor, mask, dim,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
    )

    out = _trimul_contract(left, right, bs, n, h)  # (B*H, N, N) fp32
    out = out.reshape(bs, h, n, n).permute(0, 2, 3, 1)  # (B, N, N, H) strided view

    return post(
        out, out_gate, hidden_dim,
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )
