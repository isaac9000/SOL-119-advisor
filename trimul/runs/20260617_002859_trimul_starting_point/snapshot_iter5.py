"""
TriMul submission — low-overhead functional PyTorch baseline.

Eliminates per-call module construction and parameter re-wrapping. Operates
directly on the provided weight tensors with functional ops, avoids redundant
dtype casts, and expresses the contraction as a single batched matmul.
"""

import torch
import torch.nn.functional as F


def _pre_stage(x, mask, dim,
               norm_w, norm_b, lp_w, rp_w, lg_w, rg_w, og_w):
    # LayerNorm over last dim
    x = F.layer_norm(x, (dim,), norm_w, norm_b)

    # Projections + gates (F.linear: x @ W.T)
    left = F.linear(x, lp_w)
    right = F.linear(x, rp_w)
    left_gate = torch.sigmoid(F.linear(x, lg_w))
    right_gate = torch.sigmoid(F.linear(x, rg_w))
    out_gate = torch.sigmoid(F.linear(x, og_w))

    m = mask.unsqueeze(-1)
    left = left * m * left_gate
    right = right * m * right_gate

    # Produce fp16 tensors already in (b*h, n, n) bmm layout.
    bs, n, _, h = left.shape
    lt = left.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)   # (b*h, i, k)
    rt = right.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)  # (b*h, j, k)
    return lt, rt, out_gate, bs, n, h


def _post_stage(out, out_gate, hidden_dim, ton_w, ton_b, out_w):
    out = F.layer_norm(out, (hidden_dim,), ton_w, ton_b)
    out = out * out_gate
    out = F.linear(out, out_w)
    return out


# Cache compiled wings per (dim, hidden_dim). The contraction GEMM stays as an
# explicit eager cuBLAS fp16 bmm (which beat inductor's path on dim=384).
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

    lt, rt, out_gate, bs, n, h = pre(
        input_tensor, mask, dim,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
    )

    # Explicit cuBLAS fp16 batched matmul for the contraction.
    out = torch.bmm(lt, rt.transpose(1, 2))  # (b*h, i, j) fp16
    out = out.to(torch.float32).reshape(bs, h, n, n).permute(0, 2, 3, 1)  # (b, i, j, h)

    return post(
        out, out_gate, hidden_dim,
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )
