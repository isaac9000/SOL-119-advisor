"""
MoE backward pass — multi-stream parallel expert execution with contiguous per-expert views.

Strategy:
- Sort tokens by expert to get contiguous per-expert slices (zero-copy views, no padding)
- Create a pool of CUDA streams and dispatch expert GEMMs round-robin
- Multiple experts execute in parallel on the GPU via stream concurrency
- Per-expert GEMMs are [~65, 2048] × [2048, 4096] — small enough that parallelism helps

custom_kernel(data) receives:
    data = (grad_output, hidden_states, topk_indices, topk_weights,
            gate_weights, up_weights, down_weights)

Returns:
    grad_hidden_states, grad_topk_weights, grad_gate_weights,
    grad_up_weights, grad_down_weights
"""

import torch
import torch.nn.functional as F

HIDDEN_SIZE           = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS      = 256
NUM_EXPERTS_PER_TOK   = 8

# Create a pool of CUDA streams at module load time (reused across calls)
_NUM_STREAMS = 16
_stream_pool = None


def _get_stream_pool(device):
    global _stream_pool
    if _stream_pool is None:
        _stream_pool = [torch.cuda.Stream(device=device) for _ in range(_NUM_STREAMS)]
    return _stream_pool


def custom_kernel(data):
    (grad_output, hidden_states, topk_indices, topk_weights,
     gate_weights, up_weights, down_weights) = data

    T, K  = topk_indices.shape
    device = hidden_states.device
    dtype  = hidden_states.dtype
    E = N_ROUTED_EXPERTS
    H = HIDDEN_SIZE           # 4096
    M = MOE_INTERMEDIATE_SIZE  # 2048
    N = T * K

    # -----------------------------------------------------------------------
    # Step 1: Sort tokens by expert to get contiguous per-expert views
    # -----------------------------------------------------------------------
    flat_experts = topk_indices.reshape(-1)
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
    slot_ids  = torch.arange(K, device=device).unsqueeze(0).expand(T, K).reshape(-1)

    sort_order       = torch.argsort(flat_experts, stable=True)
    sorted_experts   = flat_experts[sort_order]
    sorted_token_ids = token_ids[sort_order]
    sorted_slot_ids  = slot_ids[sort_order]

    expert_counts  = torch.bincount(sorted_experts, minlength=E)
    expert_offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    # Move offsets to CPU for Python-side slicing (one transfer)
    expert_offsets_cpu = expert_offsets.cpu().tolist()
    expert_counts_cpu  = expert_counts.cpu().tolist()

    # -----------------------------------------------------------------------
    # Step 2: Gather sorted inputs (contiguous)
    # -----------------------------------------------------------------------
    sorted_hidden   = hidden_states[sorted_token_ids].contiguous()   # [N, H]
    sorted_grad_out = grad_output[sorted_token_ids].contiguous()     # [N, H]
    sorted_weights  = topk_weights[sorted_token_ids, sorted_slot_ids].contiguous()  # [N]

    # -----------------------------------------------------------------------
    # Step 3: Pre-allocate output tensors
    # -----------------------------------------------------------------------
    # All grads pre-zeroed; experts accumulate into them
    grad_hidden_states = torch.zeros(T, H, dtype=dtype, device=device)
    grad_topk_weights  = torch.zeros(T, K, dtype=dtype, device=device)
    grad_gate_weights  = torch.zeros(E, M, H, dtype=dtype, device=device)
    grad_up_weights    = torch.zeros(E, M, H, dtype=dtype, device=device)
    grad_down_weights  = torch.zeros(E, H, M, dtype=dtype, device=device)

    # -----------------------------------------------------------------------
    # Step 4: Per-expert backward pass dispatched across CUDA streams
    # -----------------------------------------------------------------------
    streams = _get_stream_pool(device)
    main_stream = torch.cuda.current_stream(device)

    # We'll store per-expert result tensors to scatter after all streams finish
    # (index_add_ requires main stream sync)
    expert_grad_hidden_list    = []
    expert_token_ids_list      = []
    expert_grad_topk_list      = []
    expert_flat_out_idx_list   = []

    for expert_idx in range(E):
        count = expert_counts_cpu[expert_idx]
        if count == 0:
            continue

        start = expert_offsets_cpu[expert_idx]
        end   = expert_offsets_cpu[expert_idx + 1]

        # Select stream for this expert
        stream = streams[expert_idx % _NUM_STREAMS]

        # Make stream wait for main stream's gather ops to finish
        stream.wait_stream(main_stream)

        with torch.cuda.stream(stream):
            # Zero-copy views into sorted contiguous tensors
            expert_hidden   = sorted_hidden[start:end]    # [count, H]
            expert_grad_out = sorted_grad_out[start:end]  # [count, H]
            expert_weights  = sorted_weights[start:end]   # [count]

            # Expert weight matrices (views into pre-existing tensors)
            gw = gate_weights[expert_idx]   # [M, H]
            uw = up_weights[expert_idx]     # [M, H]
            dw = down_weights[expert_idx]   # [H, M]

            # Forward recomputation
            gate_pre_act   = expert_hidden @ gw.t()   # [count, M]
            up_out         = expert_hidden @ uw.t()   # [count, M]
            gate_activated = F.silu(gate_pre_act)     # [count, M]
            intermediate   = gate_activated * up_out  # [count, M]

            # grad_topk_weights: dot(grad_out, expert_output) per token
            expert_output = intermediate @ dw.t()     # [count, H]
            grad_topk_w   = (expert_grad_out * expert_output).sum(dim=1)  # [count]

            # Grad through down projection
            scaled_grad = expert_grad_out * expert_weights.unsqueeze(1)  # [count, H]

            # grad_down_weights[e] += scaled_grad^T @ intermediate
            grad_down_weights[expert_idx].add_(scaled_grad.t() @ intermediate)  # [H, M]

            # grad_intermediate
            grad_inter = scaled_grad @ dw   # [count, M]

            # Grad through SwiGLU
            grad_up_out      = grad_inter * gate_activated   # [count, M]
            grad_gate_act    = grad_inter * up_out           # [count, M]
            sigmoid_gate     = torch.sigmoid(gate_pre_act)
            grad_gate_pre    = grad_gate_act * (
                gate_activated + sigmoid_gate * (1.0 - gate_activated)
            )                                                # [count, M]

            # grad_gate_weights[e] += grad_gate_pre^T @ expert_hidden
            grad_gate_weights[expert_idx].add_(grad_gate_pre.t() @ expert_hidden)  # [M, H]

            # grad_up_weights[e] += grad_up_out^T @ expert_hidden
            grad_up_weights[expert_idx].add_(grad_up_out.t() @ expert_hidden)      # [M, H]

            # grad_hidden for this expert
            grad_hid = grad_gate_pre @ gw + grad_up_out @ uw  # [count, H]

        # Store for later scatter (these are futures on 'stream')
        expert_grad_hidden_list.append((stream, grad_hid, sorted_token_ids[start:end]))
        expert_grad_topk_list.append((stream, grad_topk_w,
                                      sorted_token_ids[start:end],
                                      sorted_slot_ids[start:end]))

    # -----------------------------------------------------------------------
    # Step 5: Sync all streams back to main stream and scatter results
    # -----------------------------------------------------------------------
    for stream, grad_hid, tok_ids in expert_grad_hidden_list:
        main_stream.wait_stream(stream)
        grad_hidden_states.index_add_(0, tok_ids, grad_hid)

    for stream, grad_topk_w, tok_ids, sl_ids in expert_grad_topk_list:
        # stream already waited above (or will be waited before use)
        flat_idx = tok_ids * K + sl_ids
        grad_topk_weights.view(-1).scatter_add_(0, flat_idx, grad_topk_w)

    return (grad_hidden_states, grad_topk_weights,
            grad_gate_weights, grad_up_weights, grad_down_weights)
