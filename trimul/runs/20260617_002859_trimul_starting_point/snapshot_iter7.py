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
    B, N, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Contraction: out[b, i, j, d] = sum_k left[b, i, k, d] * right[b, j, k, d]
    # left/right are (B, N, N, H) contiguous, read directly (no permute).
    # out is (B, N, N, H) contiguous fp32.
    pid_bd = tl.program_id(0)     # over B*H
    pid_i = tl.program_id(1)      # block of i
    pid_j = tl.program_id(2)      # block of j

    b = pid_bd // H
    d = pid_bd % H

    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # base offset to element [b, 0, 0, d]
    base = b * (N * N * H) + d

    # left[b, i, k, d]: offset = base + i*(N*H) + k*H
    # right[b, j, k, d]: offset = base + j*(N*H) + k*H
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < N
        # left tile (BLOCK_M, BLOCK_K)
        l_off = base + offs_i[:, None] * (N * H) + k[None, :] * H
        l_mask = (offs_i[:, None] < N) & k_mask[None, :]
        l = tl.load(left_ptr + l_off, mask=l_mask, other=0.0)
        # right tile (BLOCK_N, BLOCK_K)
        r_off = base + offs_j[:, None] * (N * H) + k[None, :] * H
        r_mask = (offs_j[:, None] < N) & k_mask[None, :]
        r = tl.load(right_ptr + r_off, mask=r_mask, other=0.0)
        acc += tl.dot(l, r.T)

    # write out[b, i, j, d]: offset = base + i*(N*H) + j*H
    o_off = base + offs_i[:, None] * (N * H) + offs_j[None, :] * H
    o_mask = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(out_ptr + o_off, acc, mask=o_mask)


def _trimul_contract(left, right):
    # left, right: (B, N, N, H) fp16 contiguous (gated/masked already).
    B, N, _, H = left.shape
    out = torch.empty((B, N, N, H), dtype=torch.float32, device=left.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (B * H, triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _trimul_contract_kernel[grid](
        left, right, out,
        B, N, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
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
    left = (left * m * left_gate).to(torch.float16)
    right = (right * m * right_gate).to(torch.float16)
    return left, right, out_gate


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

    left, right, out_gate = pre(
        input_tensor, mask, dim,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
    )

    left = left.contiguous()
    right = right.contiguous()
    out = _trimul_contract(left, right)  # (B, N, N, H) fp32

    return post(
        out, out_gate, hidden_dim,
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )
