# MoE-Bwd Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the **MoE (Mixture of Experts) backward pass** on NVIDIA B200. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements exactly one change to `submission.py`, evaluates it on a B200 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Implement the fastest possible **MoE backward pass** (NVIDIA SOL-ExecBench):

```
For expert_idx in 0..255:
  token_positions = tokens routed to this expert
  expert_hidden   = hidden_states[token_positions]              # [E, 4096]
  gate_pre_act    = expert_hidden @ gate_weights[expert_idx]^T  # [E, 2048]
  up_output       = expert_hidden @ up_weights[expert_idx]^T    # [E, 2048]
  intermediate    = silu(gate_pre_act) * up_output              # SwiGLU [E, 2048]
  # backward through down proj, SwiGLU, gate/up projs
  # accumulate grad_hidden_states, grad_{gate,up,down}_weights, grad_topk_weights
```

`custom_kernel` receives a tuple `(grad_output, hidden_states, topk_indices, topk_weights, gate_weights, up_weights, down_weights)` and returns `(grad_hidden_states, grad_topk_weights, grad_gate_weights, grad_up_weights, grad_down_weights)`:

| Name | Shape | Dtype |
|---|---|---|
| `grad_output` | `[num_tokens, 4096]` | float32 |
| `hidden_states` | `[num_tokens, 4096]` | float32 |
| `topk_indices` | `[num_tokens, 8]` | int64 |
| `topk_weights` | `[num_tokens, 8]` | float32 |
| `gate_weights` | `[256, 2048, 4096]` | float32 |
| `up_weights` | `[256, 2048, 4096]` | float32 |
| `down_weights` | `[256, 4096, 2048]` | float32 |
| return `grad_hidden_states` | `[num_tokens, 4096]` | float32 |
| return `grad_topk_weights` | `[num_tokens, 8]` | float32 |
| return `grad_gate_weights` | `[256, 2048, 4096]` | float32 |
| return `grad_up_weights` | `[256, 2048, 4096]` | float32 |
| return `grad_down_weights` | `[256, 4096, 2048]` | float32 |

Fixed architecture: 256 experts, 8 experts/token, hidden_size=4096, moe_intermediate_size=2048.

**Benchmark cases (16 total) — from NVIDIA SOL-ExecBench:**

| # | num_tokens | SOL (ms) |
|---|-----------|---------|
| 1 | 2080 | 1.71 |
| 2 | 2112 | 1.71 |
| 3 | 4096 | 2.73 |
| 4 | 2048 | 1.71 |
| 5 | 2144 | 1.71 |
| 6 | 2176 | 1.71 |
| 7 | 2208 | 1.71 |
| 8 | 2560 | 1.72 |
| 9 | 6144 | 4.10 |
| 10 | 2240 | 1.71 |
| 11 | 2272 | 1.71 |
| 12 | 2304 | 1.71 |
| 13 | 2336 | 1.71 |
| 14 | 2368 | 1.72 |
| 15 | 2400 | 1.72 |
| 16 | 2432 | 1.72 |

All 16 cases used for correctness testing and benchmarking. Correctness tolerance: `rtol=1e-2, atol=1e-2` (hidden/topk), `atol=1e-1` (weight grads). Score = `14.23 / geomean_ms` (≈7.9 at SOL).

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the B200 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_moe_bwd.py
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

## Running the agent

```bash
bash run_agent_moe.sh
```

Or directly:

```bash
uv run moe_bwd/agent.py --baseline moe_bwd/starting_point.py --iterations 25
```

Quick correctness check without a full benchmark:

```bash
cd moe_bwd
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
eval_modal_moe_bwd.py    — deployable Modal B200 evaluator
moe_bwd/
├── agent.py             — advisor-worker agentic loop (direct Anthropic SDK)
├── advisor_prompt.md    — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md     — worker system prompt: task spec, mandatory sequence, rules
├── submission.py        — the kernel file the worker edits each iteration
├── starting_point.py    — baseline PyTorch kernel to seed each run
├── run_eval.py          — submits submission.py to the deployed Modal evaluator
├── tools.py             — logging, plotting, and get_experiment_history tool
└── runs/                — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it
