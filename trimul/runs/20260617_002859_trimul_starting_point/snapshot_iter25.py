"""
TriMul submission — low-overhead functional PyTorch baseline.

Eliminates per-call module construction and parameter re-wrapping. Operates
directly on the provided weight tensors with functional ops, avoids redundant
dtype casts, and expresses the contraction as a single batched matmul.
"""

import torch
import torch.nn.functional as F


def _trimul_contract(left, right):
    # left, right: (BH, N, N) fp16 contiguous (gated/masked already).
    # out[bh,i,j] = sum_k left[bh,i,k]*right[bh,j,k] == bmm(left, right^T).
    # cuBLAS's autotuned fp16 batched GEMM on this contiguous-K layout is at the
    # achievable compute/bandwidth ceiling (beat the hand-written Triton kernel in
    # #19; strided/hybrid feeds in #20/#22 and epilogue fusion in #21 all regressed).
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
    # Produce contiguous-K (B*H, N, N) fp16 layout for the cuBLAS bmm contraction:
    # permute (b,n,n,h)->(b,h,n,n) so the contraction axis k becomes innermost
    # (load-bearing: the explicit .contiguous() enables cuBLAS's fast tensor-core
    # path; strided operands fall back to a slow path, see #20/#22).
    left = (left * m * left_gate).permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16).contiguous()
    right = (right * m * right_gate).permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16).contiguous()
    return left, right, out_gate, bs, n, h


def _post_stage(out, out_gate, hidden_dim, ton_w, ton_b, out_w):
    out = F.layer_norm(out, (hidden_dim,), ton_w, ton_b)
    out = out * out_gate
    out = F.linear(out, out_w)
    return out


# Elementwise/projection stages (LayerNorm + 5 projections + sigmoid gates + mask
# in pre; LayerNorm + out-gate + out-projection in post) run under torch.compile;
# cuBLAS owns the contraction in between via the contiguous-K fp16 layout.
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
