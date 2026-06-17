"""
TriMul submission — low-overhead functional PyTorch baseline.

Eliminates per-call module construction and parameter re-wrapping. Operates
directly on the provided weight tensors with functional ops, avoids redundant
dtype casts, and expresses the contraction as a single batched matmul.
"""

import torch
import torch.nn.functional as F


def _trimul_core(x, mask, dim, hidden_dim,
                 norm_w, norm_b,
                 lp_w, rp_w, lg_w, rg_w, og_w,
                 ton_w, ton_b, out_w):
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

    # Contraction: "b i k d, b j k d -> b i j d"
    bs, n, _, h = left.shape
    lt = left.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)   # (b*d, i, k)
    rt = right.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)  # (b*d, j, k)
    out = torch.bmm(lt, rt.transpose(1, 2))  # (b*d, i, j) fp16
    out = out.to(torch.float32).reshape(bs, h, n, n).permute(0, 2, 3, 1)  # (b, i, j, d)

    # Post layernorm + gate + out proj
    out = F.layer_norm(out, (hidden_dim,), ton_w, ton_b)
    out = out * out_gate
    out = F.linear(out, out_w)
    return out


# Single monolithic compiled region (#4 best), now with max-autotune so inductor
# searches GEMM tilings for the contraction inside the fused graph.
_compiled = {}


def _get_compiled(dim, hidden_dim):
    key = (dim, hidden_dim)
    fn = _compiled.get(key)
    if fn is None:
        fn = torch.compile(_trimul_core, dynamic=True, mode="max-autotune")
        _compiled[key] = fn
    return fn


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    fn = _get_compiled(dim, hidden_dim)
    return fn(
        input_tensor, mask, dim, hidden_dim,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight'],
    )
