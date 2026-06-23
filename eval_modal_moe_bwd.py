"""
Deployable Modal B200 evaluator for the MoE backward pass kernel task.

Deploy once:
    uv run modal deploy eval_modal_moe_bwd.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# ── 16 benchmark workloads — only num_tokens varies ──────────────────────────
# Fixed: hidden_size=4096, moe_intermediate_size=2048,
#        n_routed_experts=256, num_experts_per_tok=8

CASES = [
    {"num_tokens": 2080, "seed": 2001},
    {"num_tokens": 2112, "seed": 2002},
    {"num_tokens": 4096, "seed": 2003},
    {"num_tokens": 2048, "seed": 2004},
    {"num_tokens": 2144, "seed": 2005},
    {"num_tokens": 2176, "seed": 2006},
    {"num_tokens": 2208, "seed": 2007},
    {"num_tokens": 2560, "seed": 2008},
    {"num_tokens": 6144, "seed": 2009},
    {"num_tokens": 2240, "seed": 2010},
    {"num_tokens": 2272, "seed": 2011},
    {"num_tokens": 2304, "seed": 2012},
    {"num_tokens": 2336, "seed": 2013},
    {"num_tokens": 2368, "seed": 2014},
    {"num_tokens": 2400, "seed": 2015},
    {"num_tokens": 2432, "seed": 2016},
]

TEST_CASES      = CASES
BENCHMARK_CASES = CASES

# Scoring: score = SCORE_SCALE / geomean_ms (higher is better); SOL ≈ 1.80 ms → score ≈ 7.9
SCORE_SCALE = 14.23

HIDDEN_SIZE           = 4096
MOE_INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS      = 256
NUM_EXPERTS_PER_TOK   = 8

BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR       = 0.001      # stop when stderr/mean < 0.1%
BENCH_WALL_TIMEOUT_NS = 30e9       # 30s per case (was 120s; 120s×16=1920s exceeded timeout)
BENCH_NO_GRAD         = True
BENCH_MAX_REPEATS     = 100
BENCH_MAX_TIME_NS     = 10e9

# ── Modal image ───────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("moe-bwd-kernel-eval")


@app.function(gpu="B200", image=image, timeout=1200)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import contextlib
    import copy
    import dataclasses
    import gc
    import importlib.util
    import json as _json
    import math
    import os as _os
    import tempfile
    import time
    import traceback

    import torch
    import torch.nn.functional as F

    # ── Reference implementation ─────────────────────────────────────────────

    def ref_kernel(data):
        (grad_output, hidden_states, topk_indices, topk_weights,
         gate_weights, up_weights, down_weights) = data

        num_tokens = hidden_states.shape[0]

        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_topk_weights  = torch.zeros_like(topk_weights)
        grad_gate_weights  = torch.zeros_like(gate_weights)
        grad_up_weights    = torch.zeros_like(up_weights)
        grad_down_weights  = torch.zeros_like(down_weights)

        for expert_idx in range(N_ROUTED_EXPERTS):
            mask = (topk_indices == expert_idx)
            token_positions = mask.any(dim=1).nonzero(as_tuple=True)[0]

            if token_positions.numel() == 0:
                continue

            expert_hidden = hidden_states[token_positions]

            gate_pre_act   = expert_hidden @ gate_weights[expert_idx].t()
            up_output      = expert_hidden @ up_weights[expert_idx].t()
            gate_activated = F.silu(gate_pre_act)
            intermediate   = gate_activated * up_output

            slot_idx        = mask[token_positions].float().argmax(dim=1)
            routing_weights = topk_weights[token_positions, slot_idx]

            grad_out_tokens    = grad_output[token_positions]
            expert_output      = intermediate @ down_weights[expert_idx].t()
            grad_topk_w_expert = (grad_out_tokens * expert_output).sum(dim=1)
            for i, (tok, slot) in enumerate(zip(token_positions.tolist(),
                                                slot_idx.tolist())):
                grad_topk_weights[tok, slot] = grad_topk_w_expert[i]

            scaled_grad_out = grad_out_tokens * routing_weights.unsqueeze(1)
            grad_down_weights[expert_idx] += scaled_grad_out.t() @ intermediate

            grad_intermediate   = scaled_grad_out @ down_weights[expert_idx]
            grad_up_output      = grad_intermediate * gate_activated
            grad_gate_activated = grad_intermediate * up_output

            sigmoid_gate      = torch.sigmoid(gate_pre_act)
            grad_gate_pre_act = grad_gate_activated * (
                gate_activated + sigmoid_gate * (1.0 - gate_activated)
            )

            grad_gate_weights[expert_idx] += grad_gate_pre_act.t() @ expert_hidden
            grad_hidden_gate               = grad_gate_pre_act @ gate_weights[expert_idx]

            grad_up_weights[expert_idx]   += grad_up_output.t() @ expert_hidden
            grad_hidden_up                 = grad_up_output @ up_weights[expert_idx]

            grad_hidden_expert = grad_hidden_gate + grad_hidden_up
            grad_hidden_states.index_add_(0, token_positions, grad_hidden_expert)

        return (grad_hidden_states, grad_topk_weights,
                grad_gate_weights, grad_up_weights, grad_down_weights)

    # ── Input generation ─────────────────────────────────────────────────────

    def generate_input(num_tokens, seed):
        # Generate everything on GPU to avoid CPU RNG + PCIe transfer overhead.
        # Weights are ~8.6 GB each; token tensors are small but also moved to GPU
        # to eliminate the CPU argsort on (num_tokens, 256) which serialises on CPU.
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)

        hidden_states = torch.randn(
            num_tokens, HIDDEN_SIZE, generator=g, dtype=torch.float32, device="cuda"
        )
        grad_output = torch.randn(
            num_tokens, HIDDEN_SIZE, generator=g, dtype=torch.float32, device="cuda"
        )

        # Each token selects NUM_EXPERTS_PER_TOK unique experts — argsort on GPU.
        rand_scores  = torch.rand(num_tokens, N_ROUTED_EXPERTS, generator=g, device="cuda")
        topk_indices = rand_scores.argsort(dim=-1)[:, :NUM_EXPERTS_PER_TOK]

        topk_weights_raw = torch.randn(
            num_tokens, NUM_EXPERTS_PER_TOK, generator=g, dtype=torch.float32, device="cuda"
        )
        topk_weights = F.softmax(topk_weights_raw, dim=-1)

        gate_weights = torch.randn(
            N_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
            generator=g, dtype=torch.float32, device="cuda"
        ) * 0.02

        up_weights = torch.randn(
            N_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
            generator=g, dtype=torch.float32, device="cuda"
        ) * 0.02

        down_weights = torch.randn(
            N_ROUTED_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE,
            generator=g, dtype=torch.float32, device="cuda"
        ) * 0.02

        torch.cuda.synchronize()
        return (grad_output, hidden_states, topk_indices, topk_weights,
                gate_weights, up_weights, down_weights)

    # ── Correctness check ────────────────────────────────────────────────────

    def check_implementation(data, submission_output):
        try:
            ref_out = ref_kernel(data)
            (ref_grad_hidden, ref_grad_topk_w,
             ref_grad_gate, ref_grad_up, ref_grad_down) = ref_out
            (sub_grad_hidden, sub_grad_topk_w,
             sub_grad_gate, sub_grad_up, sub_grad_down) = submission_output
        except Exception as e:
            return False, f"unpack failed: {e}"

        # Shape checks
        for name, ref, sub in [
            ("grad_hidden_states",   ref_grad_hidden,  sub_grad_hidden),
            ("grad_topk_weights",    ref_grad_topk_w,  sub_grad_topk_w),
            ("grad_gate_weights",    ref_grad_gate,     sub_grad_gate),
            ("grad_up_weights",      ref_grad_up,       sub_grad_up),
            ("grad_down_weights",    ref_grad_down,     sub_grad_down),
        ]:
            if ref.shape != sub.shape:
                return False, f"{name} shape mismatch: {ref.shape} vs {sub.shape}"

        # Tight tolerances for hidden states and topk_weights
        for name, ref, sub, rtol, atol in [
            ("grad_hidden_states", ref_grad_hidden, sub_grad_hidden, 1e-2, 1e-2),
            ("grad_topk_weights",  ref_grad_topk_w, sub_grad_topk_w, 1e-2, 1e-2),
            # Weight gradients accumulate over many tokens → looser
            ("grad_gate_weights",  ref_grad_gate,   sub_grad_gate,   1e-2, 1e-1),
            ("grad_up_weights",    ref_grad_up,      sub_grad_up,    1e-2, 1e-1),
            ("grad_down_weights",  ref_grad_down,    sub_grad_down,  1e-2, 1e-1),
        ]:
            ok = torch.allclose(ref.float(), sub.float(), rtol=rtol, atol=atol)
            if not ok:
                d = torch.abs(ref.float() - sub.float())
                return False, (
                    f"{name} mismatch: max={d.max().item():.4e} mean={d.mean().item():.4e}"
                )

        return True, "Match"

    # ── Helpers ───────────────────────────────────────────────────────────────

    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _clone(data):
        if isinstance(data, tuple):
            return tuple(_clone(x) for x in data)
        if isinstance(data, list):
            return [_clone(x) for x in data]
        if isinstance(data, dict):
            return {k: _clone(v) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            return data.clone()
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            fields = {f.name: _clone(getattr(data, f.name)) for f in dataclasses.fields(data)}
            return type(data)(**fields)
        return data

    def _stats(durations):
        n = len(durations)
        avg = sum(durations) / n
        if n > 1:
            var = sum((x - avg) ** 2 for x in durations) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": avg, "std": std, "err": err}

    def clear_l2_cache():
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        dummy.fill_(42)
        del dummy

    # ── Timing ───────────────────────────────────────────────────────────────
    _t0 = time.time()
    def _log(msg):
        print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)

    # ── Load submission ───────────────────────────────────────────────────────

    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__
    _log(f"GPU: {gpu_name}  torch: {torch_ver}")

    tmp_dir  = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")
    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom_kernel = mod.custom_kernel
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "import",
        })

    # ── Correctness tests ─────────────────────────────────────────────────────

    _log("starting correctness tests")
    test_details = []
    tests_passed = 0

    for tc in TEST_CASES:
        label = f"num_tokens={tc['num_tokens']}"
        try:
            data      = generate_input(**tc)
            data_copy = _clone(data)
            torch.cuda.synchronize()
            with torch.no_grad():
                output = custom_kernel(data)
            torch.cuda.synchronize()
            del data
            gc.collect()
            torch.cuda.empty_cache()

            passed, msg = check_implementation(data_copy, output)
            del data_copy, output
            gc.collect()
            torch.cuda.empty_cache()

            test_details.append({
                "num_tokens": tc["num_tokens"],
                "seed": tc["seed"],
                "passed": passed,
                "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "num_tokens": tc["num_tokens"],
                "seed": tc["seed"],
                "passed": False,
                "error": traceback.format_exc()[:600],
            })

    if tests_passed < len(TEST_CASES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
            "failure_stage": "correctness",
        })

    if mode == "test":
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-b200",
        })

    _log(f"correctness passed {tests_passed}/{len(TEST_CASES)}")

    # ── Benchmarks ────────────────────────────────────────────────────────────

    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    benchmark_details = []
    bench_means_ns    = []

    for bench_args in BENCHMARK_CASES:
        _log(f"benchmark num_tokens={bench_args['num_tokens']}")
        data = generate_input(**bench_args)

        # Warmup
        for _ in range(3):
            custom_kernel(data)
            torch.cuda.synchronize()

        durations_ns = []
        bm_start     = time.perf_counter_ns()

        with ctx:
            for t in range(BENCH_MAX_REPEATS):
                clear_l2_cache()
                torch.cuda.synchronize()

                if BENCH_USE_CUDA_EVENTS:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    output = custom_kernel(data)
                    e.record()
                    torch.cuda.synchronize()
                    duration_ns = s.elapsed_time(e) * 1e6  # ms → ns
                else:
                    t0 = time.perf_counter_ns()
                    output = custom_kernel(data)
                    torch.cuda.synchronize()
                    duration_ns = time.perf_counter_ns() - t0

                del output
                durations_ns.append(duration_ns)

                if t > 1:
                    st = _stats(durations_ns)
                    if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                        break
                    if st["mean"] * st["runs"] > BENCH_MAX_TIME_NS:
                        break
                    if (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                        break

        st      = _stats(durations_ns)
        mean_ms = st["mean"] / 1e6
        err_ms  = st["err"]  / 1e6
        _log(f"  → {mean_ms:.3f} ms  ({st['runs']} reps)")
        benchmark_details.append({
            "num_tokens": bench_args["num_tokens"],
            "seed":       bench_args["seed"],
            "mean_ms":    round(mean_ms, 3),
            "err_ms":     round(err_ms, 3),
            "runs":       st["runs"],
        })
        bench_means_ns.append(st["mean"])

    means_s     = [ns / 1e9 for ns in bench_means_ns]
    geomean_s   = math.pow(math.prod(means_s), 1.0 / len(means_s))
    geomean_ms  = geomean_s * 1e3
    score       = SCORE_SCALE / geomean_ms

    _log(f"done  geomean={geomean_ms:.3f} ms  score={score:.3f}")
    return _json.dumps({
        "success": True,
        "tests_passed": tests_passed,
        "tests_total": len(TEST_CASES),
        "test_details": test_details,
        "benchmark": {
            "geomean_ms": round(geomean_ms, 3),
            "score":      round(score, 3),
        },
        "benchmark_details": benchmark_details,
        "gpu_name":    gpu_name,
        "torch_version": torch_ver,
        "platform":    "modal-b200",
    })


@app.local_entrypoint()
def run_baseline():
    """Quick test: modal run eval_modal_moe_bwd.py"""
    import json, os
    baseline = os.path.join(os.path.dirname(__file__), "moe_bwd", "starting_point.py")
    with open(baseline) as f:
        code = f.read()
    print("Submitting baseline to B200...")
    raw = evaluate_kernel.remote(code, mode="leaderboard")
    data = json.loads(raw)
    bm = data.get("benchmark", {})
    print(f"passed={data['tests_passed']}/{data['tests_total']}  "
          f"geomean={bm.get('geomean_ms','?')} ms  score={bm.get('score','?')}")
