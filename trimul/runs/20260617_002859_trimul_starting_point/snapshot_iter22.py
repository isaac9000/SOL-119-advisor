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

    # out[bh, i, j]: offset = base + i*N + j
    o_off = base + offs_i[:, None] * N + offs_j[None, :]
    o_mask = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(out_ptr + o_off, acc, mask=o_mask)


# Pinned (deterministic) launch configs per N-regime. Replaces @triton.autotune,
# whose single-shot timing search proved nondeterministic (#13-#16 all regressed
# to ~6200us even when the good config was provably in the candidate pool). These
# fixed tilings are the ones textbook + surviving evidence most favor for a
# contiguous-K fp16 GEMM on H100 at each problem size, giving a deterministic
# launch every run with no per-run autotune timing tax.
def _pick_contract_config(N):
    # returns (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    if N <= 64:
        return (64, 64, 32, 4, 3)
    if N <= 128:
        return (64, 64, 64, 4, 3)
    if N <= 256:
        return (128, 128, 64, 8, 3)
    if N <= 512:
        return (128, 128, 64, 8, 3)
    # 768 / 1024 (the dim=384/768 large cases that dominate geomean)
    return (128, 128, 64, 8, 4)


def _trimul_contract(left, right):
    # left, right: (BH, N, N) fp16 contiguous (gated/masked already).
    # out[bh,i,j] = sum_k left[bh,i,k]*right[bh,j,k] == bmm(left, right^T).
    # cuBLAS fp16 batched GEMM on the identical contiguous-K layout #18 produces;
    # A/B vs the hand-written Triton kernel on the FLOP-bound large dim=384 cases.
    return torch.bmm(left, right.transpose(1, 2))


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

    out = _trimul_contract(left, right)  # (B*H, N, N) fp16
    out = out.reshape(bs, h, n, n).permute(0, 2, 3, 1)  # (B, N, N, H) strided view

    return post(
        out, out_gate, hidden_dim,
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )
