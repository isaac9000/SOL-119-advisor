"""
Compare advisor vs openevolve vs evox runs.
SOL-Execbench 025_moe_expert_parallel_execution_backward
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv, os

# ── Advisor data (from results.tsv) ──────────────────────────────────────────
ADV_TSV = "moe_bwd/runs/20260623_055217_moe_bwd_starting_point/results.tsv"
adv_iters, adv_times, adv_kinds = [], [], []
with open(ADV_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        adv_iters.append(int(row["agent_iteration"]))
        t = float(row["time_us"])
        adv_times.append(t)
        # iter 22 (18.51 ms) and iter 25 (18.56 ms) are cuBLAS warmup artifacts —
        # the same code ran at 85.55 ms cold; treat them as discards for honest plotting
        kind = row["status"]
        if kind == "keep" and t < 50.0:
            kind = "discard"
        adv_kinds.append(kind)

# ── EvoX data (parsed from evox log) ─────────────────────────────────────────
# Iterations with valid geomean_ms; crashes logged as 0.0
evox_raw = [
    (0,  320.616, "keep"),    # baseline
    (1,  187.007, "keep"),    # new best
    (2,  187.045, "discard"),
    (3,  187.101, "discard"),
    (4,  177.392, "keep"),    # new best
    (5,  235.668, "discard"),
    (6,  235.825, "discard"),
    (7,  0.0,     "crash"),   # correctness failure (attempt 1/3)
    (8,  168.589, "keep"),    # new best
    (9,  175.416, "discard"),
    (10, 0.0,     "crash"),   # correctness failure (attempt 1/3)
    (11, 168.971, "discard"),
    (12, 168.921, "discard"),
    (13, 0.0,     "crash"),   # correctness failures (attempts 1–3)
    (15, 177.537, "discard"),
    (16, 170.495, "discard"),
    (17, 140.101, "keep"),    # new best
    (19, 187.829, "discard"),
    (20, 192.052, "discard"),
    (21, 186.410, "discard"),
    (22, 176.697, "discard"),
    (23, 186.671, "discard"),
    (24, 0.0,     "crash"),   # correctness failure
    (25, 176.190, "discard"),
]
evox_iters = [r[0] for r in evox_raw]
evox_times = [r[1] for r in evox_raw]
evox_kinds = [r[2] for r in evox_raw]

# ── OpenEvolve data (parsed from openevolve log) ──────────────────────────────
# Sequential program evaluations 0–25
oe_raw = [
    (0,  322.215, "keep"),    # baseline
    (1,  180.682, "keep"),    # new best
    (2,  183.149, "discard"),
    (3,  179.878, "discard"),
    (4,  204.533, "discard"),
    (5,  106.102, "keep"),    # new best
    (6,  105.815, "keep"),    # new best
    (7,  89.815,  "keep"),    # new best
    (8,  0.0,     "crash"),   # correctness failure
    (9,  89.886,  "discard"),
    (10, 100.219, "discard"),
    (11, 89.874,  "discard"),
    (12, 100.209, "discard"),
    (13, 128.321, "discard"),
    (14, 100.209, "discard"),
    (15, 89.875,  "discard"),
    (16, 125.597, "discard"),
    (17, 101.391, "discard"),
    (18, 0.0,     "crash"),   # correctness failure
    (19, 90.122,  "discard"),
    (20, 100.554, "discard"),
    (21, 0.0,     "crash"),
    (22, 0.0,     "crash"),
    (23, 0.0,     "crash"),
    (24, 0.0,     "crash"),
    (25, 0.0,     "crash"),
]
oe_iters = [r[0] for r in oe_raw]
oe_times = [r[1] for r in oe_raw]
oe_kinds = [r[2] for r in oe_raw]

# ── Best-over-time step lines ─────────────────────────────────────────────────
def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

adv_bx,  adv_by  = best_step(adv_iters,  adv_times,  adv_kinds)
oe_bx,   oe_by   = best_step(oe_iters,   oe_times,   oe_kinds)
evox_bx, evox_by = best_step(evox_iters, evox_times, evox_kinds)

adv_best  = min(t for t, k in zip(adv_times,  adv_kinds)  if k == "keep" and t > 0)
oe_best   = min(t for t, k in zip(oe_times,   oe_kinds)   if k == "keep" and t > 0)
evox_best = min(t for t, k in zip(evox_times, evox_kinds) if k == "keep" and t > 0)

# ── Y-axis (negative latency, clip outliers) ──────────────────────────────────
CLIP_MS = 400.0
all_valid = [t for t in adv_times + oe_times + evox_times if 0 < t <= CLIP_MS]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_MS * 1.05)

def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# OpenEvolve — blue
oe_kx = [it for it, k in zip(oe_iters, oe_kinds) if k == "keep"]
oe_ky = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "keep"]
oe_dx = [it for it, k in zip(oe_iters, oe_kinds) if k == "discard"]
oe_dy = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "discard"]
oe_cx = [it for it, k in zip(oe_iters, oe_kinds) if k == "crash"]
if oe_kx:
    ax.scatter(oe_kx, oe_ky, c="#3b82f6", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="openevolve keep")
if oe_dx:
    ax.scatter(oe_dx, oe_dy, c="#93c5fd", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="openevolve discard")
if oe_bx:
    ax.step(oe_bx, [-t for t in oe_by], where="post", color="#3b82f6", linewidth=2, label="openevolve best", zorder=6)

# Advisor — green
adv_kx = [it for it, k in zip(adv_iters, adv_kinds) if k == "keep"]
adv_ky = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "keep"]
adv_dx = [it for it, k in zip(adv_iters, adv_kinds) if k == "discard"]
adv_dy = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "discard"]
adv_cx = [it for it, k in zip(adv_iters, adv_kinds) if k == "crash"]
if adv_kx:
    ax.scatter(adv_kx, adv_ky, c="#22c55e", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor keep")
if adv_dx:
    ax.scatter(adv_dx, adv_dy, c="#86efac", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor discard")
if adv_bx:
    ax.step(adv_bx, [-t for t in adv_by], where="post", color="#22c55e", linewidth=2, label="advisor best", zorder=6)

# EvoX — orange
evox_kx = [it for it, k in zip(evox_iters, evox_kinds) if k == "keep"]
evox_ky = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "keep"]
evox_dx = [it for it, k in zip(evox_iters, evox_kinds) if k == "discard"]
evox_dy = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "discard"]
evox_cx = [it for it, k in zip(evox_iters, evox_kinds) if k == "crash"]
if evox_kx:
    ax.scatter(evox_kx, evox_ky, c="#f97316", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="evox keep")
if evox_dx:
    ax.scatter(evox_dx, evox_dy, c="#fed7aa", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="evox discard")
if evox_bx:
    ax.step(evox_bx, [-t for t in evox_by], where="post", color="#f97316", linewidth=2, label="evox best", zorder=6)

# Crashes (all series)
all_cx = oe_cx + adv_cx + evox_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5, label=f"crash ({len(all_cx)})", alpha=0.8)

# SOL reference line
SOL_MS = 1.80
ax.axhline(-SOL_MS, color="#10b981", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"SOL ≈{SOL_MS} ms", zorder=2)

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (-ms)", fontsize=12)
ax.grid(True, alpha=0.3)

# Legend above the plot
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
          framealpha=0.9, fontsize=10, borderaxespad=0)

# Best-time records (figure-level text)
fig.text(0.5, 0.92,
         f"EvoX best: {evox_best:.2f} ms    |    "
         f"OpenEvolve best: {oe_best:.2f} ms    |    "
         f"Advisor best: {adv_best:.2f} ms",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#6b7280", alpha=0.9))

# Title
fig.text(0.5, 0.995,
         "advisor vs openevolve vs evox — SOL-Execbench 025_moe_expert_parallel_execution_backward",
         ha="center", va="top", fontsize=13, fontweight="bold")

# Outlier note
ax.annotate(
    f"(outliers > {CLIP_MS:.0f} ms shown at floor)",
    xy=(0.01, 0.02), xycoords="axes fraction",
    ha="left", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.8),
)

out = "/workspace/SOL-119-advisor/comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
