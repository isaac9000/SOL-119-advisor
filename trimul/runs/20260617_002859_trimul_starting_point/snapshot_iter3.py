"""
TriMul submission — low-overhead functional PyTorch baseline.

Eliminates per-call module construction and parameter re-wrapping. Operates
directly on the provided weight tensors with functional ops, avoids redundant
dtype casts, and expresses the contraction as a single batched matmul.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]

    x = input_tensor  # (bs, n, n, dim) float32

    # LayerNorm over last dim
    x = F.layer_norm(x, (dim,), weights['norm.weight'], weights['norm.bias'])

    # Projections + gates (F.linear: x @ W.T)
    left = F.linear(x, weights['left_proj.weight'])
    right = F.linear(x, weights['right_proj.weight'])
    left_gate = torch.sigmoid(F.linear(x, weights['left_gate.weight']))
    right_gate = torch.sigmoid(F.linear(x, weights['right_gate.weight']))
    out_gate = torch.sigmoid(F.linear(x, weights['out_gate.weight']))

    m = mask.unsqueeze(-1)
    left = left * m * left_gate
    right = right * m * right_gate

    # Contraction: "b i k d, b j k d -> b i j d"
    # For each (b, d): out[i,j] = sum_k left[i,k] * right[j,k] = left @ right.T
    # Cast to fp16 once during the projection/gate stage to fuse the cast into
    # the elementwise ops and feed H100 tensor cores directly.
    bs, n, _, h = left.shape
    lt = left.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)   # (b*d, i, k)
    rt = right.permute(0, 3, 1, 2).reshape(bs * h, n, n).to(torch.float16)  # (b*d, j, k)
    out = torch.bmm(lt, rt.transpose(1, 2))  # (b*d, i, j) fp16
    out = out.to(torch.float32).reshape(bs, h, n, n).permute(0, 2, 3, 1)  # (b, i, j, d) strided view

    # Post layernorm + gate + out proj
    out = F.layer_norm(out, (hidden_dim,), weights['to_out_norm.weight'], weights['to_out_norm.bias'])
    out = out * out_gate
    out = F.linear(out, weights['to_out.weight'])

    return out
