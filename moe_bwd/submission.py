"""
Reference MoE backward pass kernel — pure PyTorch baseline.

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

    grad_output     [num_tokens, 4096]              float32
    hidden_states   [num_tokens, 4096]              float32
    topk_indices    [num_tokens, 8]                 int64   (expert indices, unique per token)
    topk_weights    [num_tokens, 8]                 float32 (softmax-normalized routing weights)
    gate_weights    [256, 2048, 4096]               float32
    up_weights      [256, 2048, 4096]               float32
    down_weights    [256, 4096, 2048]               float32

Returns:
    grad_hidden_states   [num_tokens, 4096]         float32
    grad_topk_weights    [num_tokens, 8]            float32
    grad_gate_weights    [256, 2048, 4096]          float32
    grad_up_weights      [256, 2048, 4096]          float32
    grad_down_weights    [256, 4096, 2048]          float32
"""

import torch
import torch.nn.functional as F

HIDDEN_SIZE          = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS     = 256
NUM_EXPERTS_PER_TOK  = 8


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    num_tokens = hidden_states.shape[0]

    grad_hidden_states  = torch.zeros_like(hidden_states)
    grad_topk_weights   = torch.zeros_like(topk_weights)
    grad_gate_weights   = torch.zeros_like(gate_weights)
    grad_up_weights     = torch.zeros_like(up_weights)
    grad_down_weights   = torch.zeros_like(down_weights)

    for expert_idx in range(N_ROUTED_EXPERTS):
        # Find which tokens were routed to this expert and which slot they used
        # topk_indices: [num_tokens, 8]
        mask = (topk_indices == expert_idx)          # [num_tokens, 8]
        token_positions = mask.any(dim=1).nonzero(as_tuple=True)[0]  # token ids

        if token_positions.numel() == 0:
            continue

        # Gather token hidden states for this expert
        expert_hidden = hidden_states[token_positions]   # [E, 4096]

        # Recompute forward activations
        # gate_pre_act = expert_hidden @ gate_weights[expert_idx].T  → [E, 2048]
        gate_pre_act = expert_hidden @ gate_weights[expert_idx].t()  # [E, 2048]
        # up_output    = expert_hidden @ up_weights[expert_idx].T    → [E, 2048]
        up_output    = expert_hidden @ up_weights[expert_idx].t()    # [E, 2048]
        # SwiGLU: intermediate = silu(gate_pre_act) * up_output
        gate_activated = F.silu(gate_pre_act)                         # [E, 2048]
        intermediate   = gate_activated * up_output                   # [E, 2048]

        # For each token, find the routing weight for this expert
        # slot_indices: which column in topk_indices corresponds to expert_idx
        slot_idx = mask[token_positions].float().argmax(dim=1)        # [E]
        routing_weights = topk_weights[token_positions, slot_idx]     # [E]

        # Gradient through output projection:
        # out = intermediate @ down_weights[expert_idx].T → [E, 4096]
        # loss contribution: routing_weight * out is added to grad_output[token]
        # grad w.r.t. (routing_weight * out):
        grad_out_tokens = grad_output[token_positions]                # [E, 4096]

        # grad_topk_weights: d(loss)/d(routing_weight) = sum over hidden of
        #   grad_output[token] * out[token]
        #   = grad_output[token] @ down_weights[expert_idx] @ intermediate^T
        expert_output = intermediate @ down_weights[expert_idx].t()   # [E, 4096]
        grad_topk_weights_expert = (grad_out_tokens * expert_output).sum(dim=1)  # [E]
        # Scatter back to correct slot
        for i, (tok, slot) in enumerate(zip(token_positions.tolist(),
                                            slot_idx.tolist())):
            grad_topk_weights[tok, slot] = grad_topk_weights_expert[i]

        # Grad through down projection:
        # out = routing_weight * (intermediate @ down_weights^T)
        # grad_down_weights[expert_idx] += intermediate^T @ (routing_weight * grad_out)
        scaled_grad_out = grad_out_tokens * routing_weights.unsqueeze(1)  # [E, 4096]
        grad_down_weights[expert_idx] += scaled_grad_out.t() @ intermediate  # [4096, 2048]

        # grad_intermediate = scaled_grad_out @ down_weights[expert_idx]   → [E, 2048]
        grad_intermediate = scaled_grad_out @ down_weights[expert_idx]    # [E, 2048]

        # Grad through SwiGLU: intermediate = silu(gate_pre_act) * up_output
        # grad_up_output    = grad_intermediate * gate_activated
        # grad_gate_activated = grad_intermediate * up_output
        grad_up_output      = grad_intermediate * gate_activated           # [E, 2048]
        grad_gate_activated = grad_intermediate * up_output                # [E, 2048]

        # Grad through silu: d/dx silu(x) = silu(x) + sigmoid(x)*(1 - silu(x))
        sigmoid_gate = torch.sigmoid(gate_pre_act)                        # [E, 2048]
        grad_gate_pre_act = grad_gate_activated * (
            gate_activated + sigmoid_gate * (1.0 - gate_activated)
        )                                                                   # [E, 2048]

        # Grad through gate projection: gate_pre_act = expert_hidden @ gate_weights^T
        grad_gate_weights[expert_idx] += grad_gate_pre_act.t() @ expert_hidden  # [2048,4096]
        grad_hidden_gate = grad_gate_pre_act @ gate_weights[expert_idx]          # [E, 4096]

        # Grad through up projection: up_output = expert_hidden @ up_weights^T
        grad_up_weights[expert_idx] += grad_up_output.t() @ expert_hidden       # [2048,4096]
        grad_hidden_up = grad_up_output @ up_weights[expert_idx]                 # [E, 4096]

        # Total grad_hidden for this expert's tokens
        grad_hidden_expert = grad_hidden_gate + grad_hidden_up                   # [E, 4096]
        grad_hidden_states.index_add_(0, token_positions, grad_hidden_expert)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
