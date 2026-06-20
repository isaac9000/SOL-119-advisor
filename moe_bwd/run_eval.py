#!/usr/bin/env python3
"""
CLI wrapper that submits a kernel to the deployed Modal B200 evaluator
and writes results.json in markdown format the agent can parse.

Deploy the evaluator once before running:
    uv run modal deploy eval_modal_moe_bwd.py

Usage:
    python run_eval.py submission.py -o results.json
    python run_eval.py submission.py -o results.json --mode test
"""

import argparse
import json
import sys
import threading

import modal

TEST_CASES = [
    {"num_tokens": 2080},
    {"num_tokens": 2112},
    {"num_tokens": 4096},
    {"num_tokens": 2048},
    {"num_tokens": 2144},
    {"num_tokens": 2176},
    {"num_tokens": 2208},
    {"num_tokens": 2560},
    {"num_tokens": 6144},
    {"num_tokens": 2240},
    {"num_tokens": 2272},
    {"num_tokens": 2304},
    {"num_tokens": 2336},
    {"num_tokens": 2368},
    {"num_tokens": 2400},
    {"num_tokens": 2432},
]

BASELINE_GEOMEAN_MS = 14.23
SOL_GEOMEAN_MS      = 1.80


def _case_label(tc: dict) -> str:
    return f"num_tokens={tc['num_tokens']}"


def format_results_markdown(res: dict, mode: str = "leaderboard") -> str:
    gpu       = res.get("gpu_name", "NVIDIA B200")
    torch_ver = res.get("torch_version", "unknown")
    plat      = res.get("platform", "modal-b200")

    if res["success"]:
        status_line = "**B200 on Modal ✅ success**"
    else:
        status_line = "**B200 on Modal ❌ failure**"

    lines = [status_line]

    if res["success"]:
        lines.append("> ✅ Testing successful")
        if mode == "leaderboard":
            lines.append("> ✅ Benchmarking successful")
    elif res.get("tests_passed", 0) == res.get("tests_total", 1):
        lines.append("> ✅ Testing successful")
        lines.append("> ❌ Benchmarking failed")
    else:
        lines.append("> ❌ Testing failed")

    lines += [
        "",
        "Running on:",
        f"* GPU: `{gpu}`",
        f"* Runtime: `CUDA`",
        f"* Platform: `{plat}`",
        f"* Torch: `{torch_ver}`",
        "",
    ]

    passed = res.get("tests_passed", 0)
    total  = res.get("tests_total", 0)
    lines.append(f"## {'✅' if passed == total else '❌'} Passed {passed}/{total} tests:")
    lines.append("```")
    for td in res.get("test_details", []):
        icon  = "✅" if td["passed"] else "❌"
        label = _case_label(td)
        lines.append(f"{icon} {label}")
        if td.get("error"):
            lines.append(f"   ERROR: {td['error']}")
    lines.append("```")

    if res.get("error") and not res["success"]:
        lines += ["", "## Error:", "```", res["error"], "```"]

    bm = res.get("benchmark")
    if bm and mode == "leaderboard":
        geomean = bm["geomean_ms"]
        score   = bm.get("score", "")
        lines += ["", "## Benchmarks:", "```",
                  f"Geometric mean: ⏱ {geomean} ms", ""]
        if score:
            lines.append(f"Score: {score}")
            lines.append(
                f"(baseline ≈ {BASELINE_GEOMEAN_MS} ms, SOL ≈ {SOL_GEOMEAN_MS} ms)"
            )
            lines.append("")
        for bd in res.get("benchmark_details", []):
            label = _case_label(bd)
            lines.append(
                f"  {label}: ⏱ {bd['mean_ms']} ± {bd['err_ms']} ms"
                f"  (runs={bd.get('runs', '?')})"
            )
        lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a moe_bwd kernel on Modal B200"
    )
    parser.add_argument("submission", help="Path to submission.py")
    parser.add_argument("-o", "--output", default="results.json")
    parser.add_argument(
        "--mode",
        choices=["test", "leaderboard"],
        default="leaderboard",
        help="'test' for correctness only, 'leaderboard' for correctness + benchmark",
    )
    args = parser.parse_args()

    try:
        with open(args.submission) as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.submission} not found")
        sys.exit(1)

    print(f"Submitting {args.submission} to Modal B200 ({args.mode} mode)...")

    evaluate_kernel = modal.Function.from_name("moe-bwd-kernel-eval", "evaluate_kernel")

    MODAL_TIMEOUT = 600
    result_holder = [None]
    error_holder  = [None]

    def _call():
        try:
            result_holder[0] = evaluate_kernel.remote(kernel_code, mode=args.mode)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=MODAL_TIMEOUT)

    if t.is_alive():
        print(f"Error: Modal call timed out after {MODAL_TIMEOUT}s", file=sys.stderr)
        sys.exit(2)
    if error_holder[0] is not None:
        print(f"Error: Modal call failed: {error_holder[0]}", file=sys.stderr)
        sys.exit(1)

    raw = result_holder[0]
    res = json.loads(raw)
    md  = format_results_markdown(res, mode=args.mode)

    with open(args.output, "w") as f:
        json.dump(md, f)

    print(md)
    sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
    main()
