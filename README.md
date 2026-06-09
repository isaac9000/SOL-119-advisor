# TriMul Autoresearch

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the Triangle Multiplicative Update (TriMul) operator on NVIDIA H100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements exactly one change to `submission.py`, evaluates it on an H100 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Implement the **outgoing** TriMul operator from AlphaFold3 — a core operation in protein structure prediction models (AlphaFold3, Chai, Protenix):

```
x    = LayerNorm(input)
left = left_proj(x) * sigmoid(left_gate(x))
right= right_proj(x) * sigmoid(right_gate(x))
left, right = left * mask, right * mask
out  = einsum("... i k d, ... j k d -> ... i j d", left, right)
out  = LayerNorm(out) * sigmoid(out_gate(x))
return to_out(out)
```

`custom_kernel` receives a tuple `(input_tensor, mask, weights, config)` and returns the output tensor:

| Argument | Shape | Dtype |
|---|---|---|
| `input_tensor` | `[bs, seqlen, seqlen, dim]` | `float32` |
| `mask` | `[bs, seqlen, seqlen]` | `float32` |
| `weights` | dict of named tensors | `float32` |
| `config` | `{"dim": int, "hidden_dim": int}` | — |
| return | `[bs, seqlen, seqlen, dim]` | `float32` |

**Test cases (correctness) — 18 total:**

| seqlen | bs | dim | nomask | distribution |
|---|---|---|---|---|
| 32 | 1 | 128 | ✓ | normal |
| 32 | 1 | 128 | ✗ | normal |
| 64 | 2 | 256 | ✓ | normal |
| 64 | 2 | 256 | ✗ | normal |
| 128 | 1 | 768 | ✓ | normal |
| 256 | 1 | 128 | ✓ | normal |
| 256 | 1 | 128 | ✗ | normal |
| 768 | 2 | 128 | ✓ | normal |
| 1024 | 1 | 384 | ✗ | normal |
| 1024 | 1 | 768 | ✓ | normal |
| 1024 | 1 | 768 | ✗ | normal |
| 32–1024 | 1–2 | 128–768 | ✓/✗ | cauchy (×7) |

**Benchmark cases (timing) — 7 total:**

| seqlen | bs | dim | nomask | distribution |
|---|---|---|---|---|
| 256 | 2 | 128 | ✓ | normal |
| 768 | 1 | 128 | ✓ | cauchy |
| 256 | 2 | 384 | ✗ | normal |
| 512 | 1 | 128 | ✓ | normal |
| 1024 | 1 | 128 | ✓ | cauchy |
| 768 | 1 | 384 | ✗ | normal |
| 1024 | 1 | 384 | ✓ | normal |

Ranked by geometric mean latency across all seven benchmark cases (lower is better). Score = `3000 / geomean_us` (higher is better). Timing uses adaptive iteration: stops when `stderr/mean < 0.1%`, after 10 s per case, or 120 s wall time. Correctness tolerance: `rtol=2%, atol=2%`.

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the H100 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_trimul.py
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
bash run_agent.sh
```

Or directly:

```bash
uv run trimul/agent.py --baseline trimul/starting_point.py --iterations 25
```

Start from a specific baseline file instead of the current `submission.py`:

```bash
uv run trimul/agent.py --baseline path/to/baseline.py --iterations 20
```

Quick correctness check without a full benchmark:

```bash
cd trimul
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
eval_modal_trimul.py   — deployable Modal H100 evaluator
trimul/
├── agent.py           — advisor-worker agentic loop (LangChain + DeepAgents)
├── advisor_prompt.md  — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md   — worker system prompt: task spec, mandatory sequence, rules
├── submission.py      — the kernel file the worker edits each iteration
├── starting_point.py  — baseline PyTorch TriMul kernel to seed each run
├── run_eval.py        — submits submission.py to the deployed Modal evaluator
├── tools.py           — log_experiment and get_experiment_history tools
└── runs/              — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
