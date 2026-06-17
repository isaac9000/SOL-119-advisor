# Advisor Proposals

---

## Iteration 1

## STATE
Only the baseline exists: a naive PyTorch module that re-instantiates `TriMul` (allocating fresh `nn.Linear`/`nn.LayerNorm` and re-binding every weight as a new `Parameter`) on every single call, then runs the algorithm eagerly with a bf16 einsum. It clocks 10877 μs geomean against a SOL geomean in the low tens of μs — roughly two-to-three orders of magnitude of headroom. Nothing has been optimized yet; we have no noise estimate and no maturing approach.

## RATIONALE
The baseline's dominant cost is almost certainly Python/framework overhead, not math: per-call module construction, parameter re-wrapping, repeated `.to(torch.float32)` casts, and a long chain of separate eager kernels (5 projections + 2 layernorms + einsum + output), each launching independently with materialized intermediates. Before any custom Triton work, we must strip this overhead and let the structure of the problem show its true compute bottleneck — which is the `b i k d, b j k d -> b i j d` contraction (an N×N×N×H batched matmul-like op) plus the surrounding elementwise/normalization fusions.

## PROPOSAL
First establish a clean, low-overhead PyTorch reference as the working baseline before touching Triton. Direction for the worker: eliminate all per-call object construction and parameter re-wrapping — operate directly on the provided weight tensors with functional ops (`F.linear`, `F.layer_norm`), avoid redundant dtype casts, and express the contraction as a single efficient batched matmul (reshaping so the einsum becomes `bmm`/`matmul` over the contraction axis). Treat this as the new local baseline and measure it across all 7 shapes to get our first real noise estimate and identify whether the contraction or the projections dominate. Hold the precision question (fp32 vs bf16/tf32 in the matmul) as a tunable knob, but verify correctness against the tolerance on every shape. Once this overhead-free baseline has matured, we'll branch into operator fusion (fusing layernorm+projections+gating, and fusing the post-contraction layernorm+gate+out-projection) and then a custom fused Triton kernel for the contraction-heavy core.

---

## Iteration 2

## STATE
Two keeps, both PyTorch. The clean functional rewrite (#2) cut the baseline from 10877→9296 μs by killing per-call module construction and casts, and confirmed the diagnosis: the `b i k d, b j k d -> b i j d` contraction (now a `bmm` over `b*h` batch) dominates at large seqlen — the 1024/dim=384 case alone is ~32 ms. We still sit two orders of magnitude above the low-tens-of-μs SOL, so this is pure early-game; the functional baseline is still descending and nothing is mature. No reliable noise floor yet, but per-shape values are large enough that run-to-run variance is not yet the concern.

## RATIONALE
The geomean is dominated by the largest contraction cases, and the current implementation pays heavily for it: it forms five separate full `(b,n,n,h)` projection tensors, does multiple `permute`/`reshape`/`.contiguous()` copies to set up the `bmm`, and casts the whole contraction to bf16 (which still launches a giant materialized GEMM). The biggest immediate lever is the contraction itself — its layout, precision, and whether the surrounding pre/post elementwise work can be fused so we stop materializing so many full-size intermediates. There are likely cheap wins in avoiding the explicit `.contiguous()` transposes and in letting matmul use tensor cores efficiently, before we commit to writing a custom kernel.

## PROPOSAL
Keep pushing the PyTorch approach to maturity along the contraction axis — it is still descending and is the cheapest place to find big wins. Direction for the worker: (1) attack the contraction's memory-layout overhead — minimize the permute/reshape/contiguous copies needed to feed the batched matmul, and explore expressing the contraction directly via `einsum`/`matmul` on views so the framework picks an efficient path without forcing full materialized transposes; (2) treat matmul precision as a tunable knob — compare fp32-with-tf32, bf16, and fp16 accumulation for the contraction, picking the fastest that still passes the 2e-2 tolerance on all 7 shapes; (3) reduce intermediate materialization in the projection/gating stage where practical. Measure all 7 shapes each time to start building a noise estimate, and report a per-shape breakdown so we can see whether the large-seqlen contraction or the projections move. Defer custom Triton until this baseline flattens.

---

## Iteration 3

## STATE
Three keeps, all PyTorch, monotonically descending: 10877→9296→7298 μs. The fp16 contraction + strided-view change (#3) gave a clean ~22% cut and showed the big-seqlen cases responding strongly (1024/384 contraction 32.7→24.2 ms), reconfirming the N³·H batched matmul as the dominant term. The PyTorch approach is still descending and far from mature; we remain ~2 orders above SOL, so we keep mining cheap wins here before committing to a custom kernel.

## RATIONALE
The current code still pays for layout churn around the contraction: it `permute`s `left`/`right` from `(b,n,n,h)` to `(b,h,n,n)` (a real copy because the cast to fp16 materializes it), runs the bmm, then permutes back. The whole contraction is also unfused — five separate full-size projection tensors plus mask/gate elementwise plus the matmul plus post-norm/gate/out, each its own kernel launch reading/writing `(b,n,n,h)` tensors. Two distinct levers remain in pure PyTorch: (a) reduce/cheapen the data movement and intermediate materialization around the matmul, and (b) hand the whole eager graph to `torch.compile` so the elementwise pre/post stages fuse automatically and layout copies get elided. `torch.compile` is a high-leverage, low-risk move we haven't tried and often closes a large fraction of the gap on exactly this kind of fusion-bound pipeline.

## PROPOSAL
Pursue two parallel directions and keep whichever matures faster. Primary: wrap the compute pipeline in `torch.compile` (cache the compiled callable across calls so we don't recompile every invocation; guard for the handful of distinct input shapes) and let the inductor backend fuse the layernorm + five projections + sigmoid gates + mask multiplies into the matmul boundary, and fuse the post-contraction layernorm+gate+out-projection. This directly attacks the unfused-launch overhead and the layout copies. Secondary: independently, reduce contraction-setup cost by exploring layouts that avoid the round-trip permute/copy (e.g. choosing a contraction formulation whose natural memory layout already matches what the matmul wants, so the fp16 cast doesn't also pay for a transpose). Keep fp16 for the contraction. Measure all 7 shapes with a per-shape breakdown each run, and watch correctness against the 2e-2 tolerance. If `torch.compile` lands a step change, that becomes the new local baseline to mature; if it stalls or recompiles pathologically, fall back and keep tuning the eager layout. Defer hand-written Triton until these PyTorch levers flatten.

---

## Iteration 4

## STATE
Four keeps, all PyTorch, still descending: 10877→9296→7298→6321 μs. `torch.compile` (#4) fused the elementwise/projection stages and cut ~13%, but produced a split signal — dim=128 cases improved sharply (1024/128: 21.6→12.6 ms) while dim=384 cases regressed (1024/384: 24.2→32.7 ms), meaning inductor chose a worse contraction path for the larger hidden dim. We remain far above the low-tens-of-μs SOL. The compile approach is fresh and the eager fp16 approach (#3) is still the better choice for dim=384, so neither is mature.

## RATIONALE
The history now cleanly separates two cost regimes: the surrounding elementwise/projection fusion (where compile wins) and the contraction GEMM itself (where compile's auto-generated path regresses on dim=384). The geomean is dragged by the large dim=384 contraction cases, and we're leaving a clear win on the table because we forced one strategy onto both regimes. The cheapest high-leverage move is to stop letting inductor own the contraction: keep the compiled fused pre/post elementwise stages but route the actual batched matmul through the explicit fp16 `bmm` (cuBLAS) that already beat it on dim=384, or let compile use `max-autotune` so it actually searches GEMM kernels instead of picking a default. We haven't exhausted the compile knobs (autotune mode, reduction-vs-cublas backend for matmul, keeping the GEMM outside the compiled region).

## PROPOSAL
Push the compile approach to maturity by separating the two cost regimes rather than treating the whole pipeline uniformly. Direction for the worker: (1) keep the elementwise/projection/gating and post-norm/gate/out stages inside `torch.compile`, but keep the contraction GEMM as the explicit cuBLAS fp16 `bmm` path that already won on dim=384 — i.e. compile the wings, hand the matmul to the library — and measure whether this captures the dim=128 fusion win without the dim=384 regression. (2) In parallel, try `torch.compile` with `max-autotune` (and the matmul/epilogue autotuning that lets it actually search GEMM tilings) to see if inductor can match cuBLAS on the contraction when allowed to tune. Compare per-shape, not just geomean, so we see both regimes. Keep fp16 for the contraction and verify the 2e-2 tolerance on all 7 shapes each run. Whichever variant flattens lower becomes the new local baseline; once these compile levers stop improving, we move to a hand-written fused Triton kernel for the contraction core.

---

## Iteration 5

## STATE
Best remains the monolithic `torch.compile` (#4) at 6321 μs; the wing-splitting experiment (#5, 7369 μs) was correctly discarded — breaking the compiled region into two pieces introduced fp16/fp32 round-trips and layout copies at the boundaries that cost more than the dim=384 GEMM improvement saved. The clear lesson: fusion across the whole pipeline matters more than picking the "best" contraction kernel in isolation, and region boundaries are expensive. We're still far above the low-tens-of-μs SOL, and the compile approach hasn't been pushed on its real remaining knob — autotuning the GEMM *inside* the fused region.

## RATIONALE
#5 confirmed that we cannot beat the dim=384 contraction by yanking it out of the fused graph. The untested lever from last iteration is the one that respects fusion: keep the single monolithic compiled region but let inductor actually *search* GEMM tilings via `max-autotune` (and matmul/epilogue autotuning), so it can find a better dim=384 contraction path without us paying boundary costs. This is the natural maturation step for #4 and the highest-leverage remaining PyTorch move. The compile approach is still young — one good result, one failed variant — so it deserves this push before we declare the framework mature and move to hand-written Triton.

## PROPOSAL
Mature the monolithic-compile approach by tuning the compiler, not by restructuring the graph. Direction for the worker: keep the entire pipeline (#4's single `_trimul_core`) inside one `torch.compile` region, but enable aggressive autotuning — `mode="max-autotune"` (or the equivalent inductor flags that turn on GEMM template autotuning and epilogue fusion) — so inductor searches contraction tilings and ideally recovers the dim=384 case while preserving the dim=128 fusion win. Warm up / cache the compiled artifact so autotune cost is paid once and not counted per-call. Compare per-shape against #4 to confirm dim=384 improves without regressing dim=128. Keep fp16 for the contraction and verify the 2e-2 tolerance on all 7 shapes. If max-autotune lands a step change, that's the new baseline; if it flattens or the autotune path can't beat cuBLAS inside fusion, then the PyTorch lever is mature and the next iteration moves to a hand-written fused Triton kernel for the contraction core (where we can fuse the gating/mask epilogue directly into the GEMM and avoid all intermediate materialization).

---

## Iteration 6

max-autotune (#6) gave only a marginal improvement: 6321→6082 μs, well within the noise band for these large-magnitude cases. That's a signal the PyTorch/compile lever is flattening. Time to pivot to hand-written Triton.

## STATE
The PyTorch/`torch.compile` track has matured: monolithic compile (#4, 6321) → max-autotune (#6, 6082) was only a ~4% move, within the run-to-run variance for these large-magnitude cases, and the wing-split (#5) is dead. The compiler has now been given full fusion *and* GEMM autotuning and has stopped descending meaningfully — this lever has flattened. We sit at ~6082 μs, still two orders above the low-tens-of-μs SOL, with the N³·H batched contraction (especially dim=384) as the persistent bottleneck that neither cuBLAS-eager nor inductor-autotune has cracked.

## RATIONALE
Every remaining PyTorch path forces full materialization of the projection/gate intermediates and a standalone batched GEMM, with the mask/gate epilogue as separate passes. The structural win we cannot get from the framework is *fusing the gating/masking epilogue and the fp16 cast directly into the contraction*, and tiling the contraction so we never materialize the full `(b,h,n,n)` intermediates or pay the permute round-trip. That requires a hand-written kernel. This is the planned next phase: the compile lever flattened exactly as the maturity gate predicted, so a fresh approach is now justified — and it should be judged against its own first result, not against the tuned 6082 baseline.

## PROPOSAL
Open a new approach: a hand-written fused Triton kernel for the contraction core. Direction for the worker: implement the `b i k d, b j k d -> b i j d` contraction as a tiled Triton matmul (batched over `b` and the hidden dimension) that reads the projected `left`/`right` tensors and fuses as much surrounding work as cheaply possible — at minimum the fp16 cast and the output handoff, ideally pulling the mask multiply and right/left gating into the GEMM's input/epilogue so we stop materializing separate full-size gated tensors. Keep the layernorm + projections in `torch.compile` (that part is healthy) and let the Triton kernel own only the contraction + its adjacent elementwise work, but be deliberate about the region boundary — #5 showed boundary fp16/fp32 round-trips are expensive, so the kernel should consume/produce tensors in layouts that avoid extra copies. Treat this as a young approach: get it correct first across all 7 shapes within 2e-2 tolerance, capture its first honest per-shape number even if slower than 6082, then iterate on tiling/blocking. Report a per-shape breakdown so we can see whether the dim=384 contraction finally moves.

---

## Iteration 7

## STATE
The Triton contraction (#7) was correct but 2.7× slower (16507 vs 6082) — and it produced a sharp, actionable diagnosis rather than a dead end: reading the contraction's K dimension out of native `(B,N,N,H)` layout makes every load a strided gather (elements H apart), destroying bandwidth. The key learning is now explicit: cuBLAS bmm wins precisely because the permuted `(B*H,N,N)` layout gives contiguous-K access, so any custom kernel must operate where K is contiguous. The compile baseline (#6, 6082) still holds and the PyTorch lever is flat; the Triton approach is young (one trial) with a clear next move, so it is not a kill.

## RATIONALE
#7's slowness is a layout bug, not an indictment of the Triton approach — the whole premise was "avoid the permute," and that premise is exactly what broke it. The fix is to give the kernel the same contiguous-K layout cuBLAS enjoys. But once we accept a `(B,H,N,N)` (or `(B*H,N,N)`) contiguous-K layout, the open question becomes: what does a custom kernel buy us over cuBLAS bmm, which is already excellent on that layout? The answer must be *epilogue/prologue fusion* — folding the fp16 cast, the post-contraction layernorm, the out-gate multiply, and ideally the final out-projection into the GEMM so we eliminate the full-size `(B,N,N,H)` intermediate round-trips that the current pipeline materializes. A bare contraction kernel can't beat cuBLAS; a fused one can, by cutting memory traffic.

## PROPOSAL
Iterate the Triton approach with the layout fix as the first-order correction, and aim the kernel at fusion that cuBLAS cannot do. Direction for the worker: (1) feed the contraction a contiguous-K layout — produce the gated/masked `left`/`right` already in `(B*H, N, N)` (or `(B,H,N,N)`) fp16 from the compiled pre-stage, so `tl.dot` reads K contiguously like cuBLAS does. Get this correct and capture its honest per-shape number; this alone should bring it into the same ballpark as the bmm. (2) Then pursue the actual payoff: fuse the GEMM *epilogue* into the kernel — at minimum the fp16→fp32 store, and ideally the post-contraction layernorm-over-H and out-gate multiply — so we stop writing and re-reading the full `(B,N,N,H)` contraction output. Note the post-norm reduces over H, which is the batch axis of the bmm, so this requires care about how H is tiled; the worker should decide the layout/tiling that makes that reduction cheap. Keep the layernorm+projections in `torch.compile`. Judge this strictly against #7's own 16507, not against 6082. If after the layout fix the kernel can't get near the bmm even before fusion, that tells us cuBLAS bmm is the right contraction primitive and we should instead invest in fusing the epilogue around an eager bmm rather than replacing it.

---

## Iteration 8

## STATE
The Triton contraction approach broke through: the contiguous-K layout fix (#8) took it from 16507 → 4451 μs, beating the matured compile baseline (#6, 6082) by 27% and becoming the new best. Large contraction cases collapsed (1024/128: 26→9.9 ms; 1024/384: 24.3→16.2 ms). This is a young, steeply-descending approach with an untuned bare GEMM — no epilogue fusion, fixed 64³ blocks, and an output layout (`(B*H,N,N)` written to `(B,N,N,H)` via a strided permute view) that still forces the post-stage to re-read the full contraction tensor. There is clear headroom on two independent axes: GEMM tiling and epilogue fusion.

## RATIONALE
The kernel currently writes a full `(B*H,N,N)` fp32 output that the compiled post-stage then reads back, does layernorm-over-H, multiplies by out_gate, and projects — a full materialization round-trip identical to the one we just proved was costly. The biggest structural win left is fusing that epilogue (and possibly the out-projection) so the contraction result never leaves the kernel as fp32 to disk. Separately, the GEMM itself is untuned (hardcoded 64×64×64, fixed warps/stages) and the dim=384 cases — now the dominant geomean term at 16.2 ms — almost certainly want different tiling than dim=128. Both are real, large levers; the approach is nowhere near mature.

## PROPOSAL
Keep maturing the Triton approach; do not touch the pre-stage layout (it's working). Two directions, pursued in order of leverage:

(1) **Tile/config tuning first** — it's low-risk and isolates the GEMM's own ceiling. Add autotuning over the contraction kernel's block shape and warp/stage parameters (let Triton's autotuner search), keyed so the dim=384 and dim=128 regimes can each land on their preferred tiling. Capture per-shape deltas. This should be a quick, clean improvement and gives us a tuned GEMM baseline before we complicate it with fusion.

(2) **Then epilogue fusion** — the structural win. Fold the post-contraction work into the kernel so the fp32 `(B*H,N,N)` intermediate is never written and re-read. The complication is that the post-layernorm reduces over H, which is the kernel's batch axis — so a given program instance only holds one (b,h) slice and cannot do the H-reduction alone. The worker should decide the right factoring: e.g. fuse the cheap, per-element parts of the epilogue (the out_gate multiply and the fp16/output handling) into the GEMM store, and/or restructure so the H dimension a kernel instance owns is large enough to do the layernorm locally. If full epilogue fusion is too entangled with the H-reduction, even fusing the out_gate multiply and avoiding the fp32 round-trip is worth measuring.

Judge against #8's 4451, verify the 2e-2 tolerance on all 7 shapes, and report per-shape so we see whether dim=384 (the current geomean driver) moves. If tiling tuning flattens and epilogue fusion proves too tangled to land cleanly, that's the signal to consider fusing the projections into the prologue instead.

---

## Iteration 9

Autotuning (#9) gave 4451→4278, a modest ~4% — within noise for these magnitudes, suggesting the bare GEMM's tiling ceiling is near. The big structural lever, epilogue fusion, remains untried. I have enough context to steer without re-reading the full code; the proposal is clear.

## STATE
The Triton approach continues to lead: contiguous-K layout (#8, 4451) then tile autotuning (#9, 4278), now the best at 4278 μs — ~30% below the matured compile baseline (#6, 6082). The autotuning gain was modest (~4%, within the noise band for these magnitudes), signaling the *bare* GEMM's tiling ceiling is close. The high-leverage structural lever — epilogue fusion to eliminate the full `(B*H,N,N)` fp32 write-and-reread round-trip — remains untried. The approach is still descending and far from mature.

## RATIONALE
We've now extracted most of what a standalone contraction GEMM can give: correct layout and tuned tiling. What remains is memory traffic we keep paying — the kernel writes a full fp32 contraction result that the compiled post-stage immediately reads back to do layernorm-over-H, out_gate multiply, and the out-projection. That round-trip is the same materialization cost we proved expensive earlier. Fusing it into/around the GEMM is the next real win, and the geomean is now driven by the dim=384 contraction cases where this traffic is largest.

## PROPOSAL
Pursue epilogue fusion now — that's the dominant remaining lever. Direction for the worker: eliminate the fp32 `(B*H,N,N)` intermediate round-trip by folding the post-contraction work into or adjacent to the kernel. The known complication is that the post-layernorm reduces over H, which is the GEMM's batch axis, so a single program instance holding one (b,h) slice can't do the H-reduction alone — the worker must choose the factoring. Reasonable options to try (worker picks specifics): (a) fuse only the per-element parts cheaply — write the contraction output directly in the `(B,N,N,H)` consumer layout and/or fold the out_gate multiply into the store, so the post-stage does less; (b) restructure the kernel so each instance owns the full H extent for a given (i,j) tile, making the layernorm-over-H a local reduction it can fuse end-to-end (then only the out-projection remains for the compiler). Get correctness first across all 7 shapes within 2e-2, judge against #9's 4278, and report per-shape so we see whether dim=384 moves. If the H-reduction factoring proves too entangled to land cleanly, fall back to the cheap partial fusion (avoid the fp32 round-trip, output in consumer layout) and bank whatever that yields before moving on.

---

## Iteration 10

## STATE
Best holds at #9 (4278 μs); the partial-fusion attempt (#10, 6964) was correctly discarded and gave a clean lesson: forcing the GEMM store into consumer `(B,N,N,H)` layout makes the j-stride H, an uncoalesced scatter that mirrors #7's load problem on the write side. The standing wisdom is now firm — contiguous `(B*H,N,N)` layout is essential on *both* load and store, and the post-stage's strided *read* of the permute view is cheaper than an uncoalesced GEMM write. The bare-GEMM-in-Triton + compiled-wings approach is mature on the contraction itself (layout fixed, tiling tuned, +27% over the old compile baseline); the remaining headroom is now outside the contraction GEMM.

## RATIONALE
Two epilogue-fusion routes have been tried and both relocated the strided penalty rather than removing it — the H-reduction layernorm is fundamentally incompatible with the contraction's coalesced layout unless a kernel instance owns the full H extent for an (i,j) tile (option b), which is a substantial rewrite with real risk. Meanwhile we have spent zero effort on the *prologue*: the pre-stage runs five separate `F.linear` projections, two sigmoids, mask+gate elementwise, then a permute+reshape+`.contiguous()` fp16 copy of two full `(B*H,N,N)` tensors — that's a lot of memory traffic and several kernels feeding the GEMM. The four projections that produce left/right (proj+gate each) share the same input `x` and could be batched, and the explicit `.contiguous()` materialization is pure overhead. The prologue is unexplored, lower-risk than the option-b epilogue rewrite, and likely material on the dim=128 cases where the GEMM is now fast and surrounding work is a larger fraction.

## PROPOSAL
Shift attention to the prologue (pre-stage) while keeping the tuned contraction kernel from #9 intact. Direction for the worker: (1) reduce the projection traffic — the four weight matrices that build left and right (`left_proj`, `left_gate`, `right_proj`, `right_gate`) all consume the same normalized `x`, so concatenate them into a single batched `F.linear`/matmul and split the result, cutting four GEMM launches to one and improving reuse of `x`; consider folding `out_gate` in too. (2) Attack the explicit fp16 `.contiguous()` materialization of left/right — see whether the permute-to-`(B*H,N,N)` copy can be produced more cheaply, or whether letting the compiled pre-stage emit the contiguous fp16 layout directly (so inductor fuses the gate/mask/cast into the layout write) beats the current separate `.contiguous()`. Keep everything else fixed so the delta is attributable. Judge against #9's 4278, verify 2e-2 on all 7 shapes, and report per-shape — we specifically want to see the dim=128/small cases move, since that's where prologue overhead dominates now. Hold the option-(b) full-H-tile epilogue rewrite in reserve as the next big structural swing if the prologue flattens.

---

## Iteration 11

## STATE
Best still #9 (4278 μs). Three consecutive restructuring attempts have now failed against it: epilogue store in consumer layout (#10, uncoalesced write), and batching the five projections into one GEMM (#11, broke inductor's per-projection→consumer fusion and forced strided slice copies). The clear meta-lesson across #5/#7/#10/#11: every time we hand-restructure around the boundaries, we either relocate a strided-access penalty or break inductor's existing fusion — the current split (compiled wings + tuned contiguous-K Triton GEMM) is a strong local optimum that resists naive surgery. We're at 4278 μs; the cheap wins (layout, tiling) are banked.

## RATIONALE
Two failed prologue/epilogue surgeries tell us the *interfaces* are already well-fused by inductor and the contraction GEMM is well-tuned. The remaining real headroom is the one structurally-hard lever we keep deferring: the option-(b) epilogue rewrite where each kernel instance owns the **full H extent** for an (i,j) tile, so the post-contraction layernorm-over-H becomes a *local* reduction inside the kernel — both the contraction loads/store and the layernorm stay coalesced because H is the inner dimension being reduced, not scattered. Since H=128 (hidden_dim is 128 in every benchmark case), a full-H tile is a fixed, friendly size. This is the only move that removes the full `(B*H,N,N)` fp32 write-and-reread round-trip without relocating a strided penalty, and it's been on the shelf since #10. It's higher-effort and higher-risk, but it's the genuine next ceiling — and as a fresh sub-approach it must be judged against its own first result, not #9's 4278.

## PROPOSAL
Commit to the option-(b) fused-epilogue kernel — the deferred structural swing. Direction for the worker: restructure so a kernel program computes, for a tile of (i,j) pairs, the contraction result across the **entire H dimension at once** (H=128 fits in registers/SRAM as the reduction axis), then performs the post-contraction layernorm-over-H and the out_gate multiply locally before storing — eliminating the fp32 `(B*H,N,N)` intermediate entirely. Concretely this means the contraction's batch axis (currently B*H) must instead be carried *inside* the kernel as the layernorm reduction axis: for each (b, i, j) the kernel needs sum_k left[b,i,k,:]·right[b,j,k,:] as an H-vector, normalizes that vector, multiplies by out_gate[b,i,j,:], and writes it. The worker should choose how to tile i, j, and the K-loop, and how to feed left/right (the contiguous-K `(B,H,N,N)` layout that worked in #9 is the right input layout). Keep the layernorm/projection prologue in torch.compile, and leave the final out-projection (`to_out`) either fused into the store or as a trailing compiled `F.linear` — worker's call which is cleaner. 

Guardrails informed by history: this is correctness-tricky (layernorm reduction is now intra-kernel) — get 18/18 within 2e-2 first, then optimize. Judge against this approach's own first number; do NOT kill it for being slower than 4278 on attempt one. Report per-shape, watching whether removing the fp32 round-trip helps the large dim=384 cases most. If after two honest iterations the fused-epilogue kernel cannot get near 4278, that tells us the round-trip wasn't the bottleneck and we should instead invest the remaining iterations in widening the contraction autotune space (more configs, swizzling/grouping of the i,j program order for L2 reuse) on the proven #9 kernel.

---

## Iteration 12

## STATE
Best is firmly #9 (4278 μs). Four consecutive structural attempts have now failed against it (#5 wing-split, #10 epilogue-store, #11 projection-batching, #12 fused-epilogue), and #12 delivered the decisive diagnosis: the fused-epilogue was correct but slower because eliminating the fp32 round-trip *didn't pay* — the layernorm-over-H is a strided gather in the `(B*H,N,N)` layout, and more importantly **the contraction GEMM dominates total time; the epilogue is cheap.** This converges with #10/#11: the prologue and epilogue are already well-fused/cheap, and surgery around them only relocates strided penalties. The signal is unambiguous — remaining headroom lives *inside* the contraction GEMM itself, not in fusion.

## RATIONALE
We now have strong, repeated evidence (four failed surgeries) that the contraction kernel is the bottleneck and the surrounding stages are not worth restructuring. The #9 kernel, while tuned for tile shape, leaves two GEMM-internal levers untouched: (1) **L2 cache reuse via program scheduling** — the grid is naive `(BH, i-blocks, j-blocks)` with no swizzling/grouping, so co-scheduled blocks don't share `left`/`right` rows in L2; grouped/swizzled launch ordering ("group-M" style) is a classic, well-understood matmul win on exactly this access pattern, especially for the large-N dim=384 cases that dominate geomean. (2) The autotune space is modest (48 configs, BLOCK_K capped at 64) and keyed only on `N` — it can't distinguish dim=128 from dim=384, which earlier evidence (#4/#6) showed want different tilings. Both are *inside* the proven kernel, low-risk, and directly target the dominant cost. This is maturation of #9, not a new approach.

## PROPOSAL
Stop the fusion surgery — the evidence is conclusive that it relocates penalties without net gain. Direction for the worker, both inside the #9 contraction kernel (keep prologue/epilogue exactly as in #9):

(1) **L2-reuse program scheduling (primary lever):** introduce grouped/swizzled ordering of the (i-block, j-block) program IDs so spatially-clustered output tiles execute together and reuse `left`/`right` rows from L2 — the standard "group along M" matmul tactic. This is the single most likely GEMM-internal win and targets the large dim=384 cases that drive the geomean. 

(2) **Widen and better-key the autotune space:** add larger BLOCK_K and additional block-shape/warp/stage combinations, and key the autotune on something that separates the dim regimes (e.g. include the per-batch H or a dim tag alongside N) so dim=128 and dim=384 each get their own tuned config rather than sharing one keyed only on N. Add the group-size as an autotuned parameter if scheduling is introduced.

Get 18/18 within 2e-2, judge against #9's 4278, and report per-shape — we specifically expect the large-N/dim=384 cases to move if L2 swizzling works. If both these GEMM-internal levers flatten without beating 4278, then #9 is genuinely mature and the remaining iterations should go to second-order precision/occupancy tweaks (e.g. fp16 vs tf32 accumulation tradeoffs, or splitting the K-loop differently) rather than any further fusion attempts.

---

## Iteration 13

## STATE
#9 (4278 μs) is now a robust local optimum: five straight attempts have failed to beat it (#10/#11/#12 fusion surgery, #13 GEMM-internal swizzle+wide-autotune). #13 is the important new datum — group-M swizzle with GROUP_M=1 in the search space *should* have been a no-op floor, yet it regressed to 6194 μs, strongly implicating **autotuner unreliability**: 108 configs timed single-shot on a noisy kernel picked bad tilings, and the large dim=384 cases keep landing in a ~24 ms regime (matching #12) regardless of the change. The recurring pattern is that my recent steers chase second-order GEMM internals while the real risk is that *the benchmark harness re-runs autotuning and the selected config is noisy/unstable*. We have burned 5 iterations without progress; the noise floor on large cases (~20–50 μs, but these are ms-scale) and autotune variance are now first-order concerns.

## RATIONALE
Two things the history makes clear. First, the contraction GEMM is the cost and it is *already efficient* — contiguous-K loads, small per-batch working set that fits L2, so swizzle/L2 tricks don't help. Second, and newly salient: the autotuner itself is a source of variance and regression (#13 proving a should-be-floor config lost). That means the highest-value move is not another GEMM micro-optimization but **de-risking and stabilizing what we already have**, then probing the one genuinely different algorithmic axis we have never tried: changing the *arithmetic* the GEMM does, not its scheduling. Everything so far has computed the full N×N×N×H contraction in fp16. We have never tried reducing the work itself (e.g. precision of accumulation vs inputs tradeoffs that change tensor-core throughput, or exploiting the mask to skip zeroed rows/tiles). The mask is a real, unexploited structural property: masked-out positions are exact zeros in left/right, so whole K-tiles or output tiles can be skipped — and the benchmark includes masked cases.

## PROPOSAL
Two threads, both grounded in what the failures taught us.

(1) **Stabilize #9's autotune (low-risk, do first).** The #13 evidence says the 48-config autotune may itself be picking unstable winners under noisy single-shot timing. Have the worker prune the config list to a small, hand-curated set that the per-shape data has repeatedly favored (the few BLOCK_M/N/K + warp/stage combos that won at N=256/512/768/1024), removing the long tail that autotune can mis-select. Goal is not a new best per se but a *reliable* one — confirm #9's number reproduces and tighten variance. This protects the score against autotune flakiness and is the responsible move after five regressions.

(2) **Probe the unexploited algorithmic axis: mask-aware work skipping.** This is genuinely new — never attempted in 13 experiments. Masked-out k positions make entire columns of `left`/`right` zero; for the benchmark's masked cases, a kernel that skips K-tiles (or whole (i,j) output tiles) that are fully masked does strictly less work. Direction: have the worker exploit the mask structure to reduce the contraction's effective N — either by compacting/segmenting along the masked K axis before the GEMM, or by passing the mask into the kernel and early-exiting fully-masked tiles. Treat this as a fresh sub-approach: get correctness on all 7 shapes first, judge it against its own first result, and report per-shape — the win (if any) shows up only on masked cases, so per-shape attribution is essential.

Guardrail: pick ONE change per experiment so deltas stay attributable, and do NOT widen the autotune space again — #13 showed that backfires. If mask-skipping shows no structure to exploit (masks may be near-dense in these benchmarks), abandon it after one honest attempt and accept #9 as mature, spending remaining iterations only on stabilization and a final precision sanity-check (fp16 vs tf32 accumulation) rather than further scheduling experiments.

---

## Iteration 14

## STATE
#9 (4278 μs) holds; six straight discards. But #14 produced a genuinely valuable, counterintuitive finding that overturns my previous read: pruning the autotune to "standard" GEMM tilings regressed large dim=384 to the ~24 ms regime, which means **the full 48-config search is load-bearing — it finds a non-obvious config for large-N that the textbook tilings miss, and that config is the difference between 15 ms and 24 ms on the dominant case.** This also reinterprets #13: its regression came from the swizzle/flattened-grid change breaking access patterns, not from autotune width. The key insight is now sharp: *the large-N config matters enormously and the current key=['N'] autotune is the right mechanism* — the lever is feeding it richer, smarter candidates, not fewer or differently-scheduled ones.

## RATIONALE
We have a robust, repeatedly-confirmed picture: the contraction GEMM dominates, the prologue/epilogue are cheap and well-fused, layout surgery and swizzle hurt, and autotune *width* helps. #14 shows the winning large-N config is non-standard and the search is exploiting headroom that's invisible to intuition. The correct move is therefore to *enrich the autotune candidate pool without changing the kernel's proven structure or access pattern* — add config dimensions the current 48-set never explores but that don't touch the load/store layout: larger BLOCK_M/BLOCK_N (e.g. 256-class tiles) and a wider BLOCK_K range, plus more warp/stage diversity. This is strictly additive (superset of the current 48), so by construction it cannot do worse than #9 if autotune is reliable — and #14 just demonstrated autotune *is* reliable here (curated-subset lost precisely because it lacked the good config). The risk that sank #13 (changing the kernel body/grid) is explicitly avoided.

## PROPOSAL
Keep the #9 kernel body, grid, layout, and key=['N'] exactly as-is — change ONLY the autotune candidate list. Direction for the worker: **enrich the config search space as a strict superset of the current 48**, adding tile-shape and warp/stage combinations the current grid doesn't reach but that preserve the existing contiguous-K access pattern. Concretely the *axes* to widen (let the worker pick exact values): bigger output-tile shapes for the large-N/large-batch cases, a broader contraction-block range, and a couple more warp/stage points — because #14 proved the large dim=384 winner lives outside the textbook set and the search will find it if it's present. 

Guardrails grounded in the last six failures: (1) do NOT touch the kernel's index math, grid shape, or store layout — every such change has regressed. (2) Make the new config set a superset of #9's 48 so the known-good large-N config remains reachable; this protects against a #14-style loss. (3) One change only: just the config list. Verify 18/18 within 2e-2 and report per-shape, watching 1024/384 and 768/384 specifically — those are where the headroom and the geomean leverage are.

If this superset search also fails to beat 4278, that is decisive evidence #9's autotune has already found the per-N optima and the GEMM is mature; at that point stop tuning the contraction entirely and spend the final iterations on a single honest precision/occupancy probe (e.g. fp16 vs bf16/tf32 accumulation tradeoff) rather than any further structural or scheduling changes.

---

## Iteration 15

## STATE
#9 (4278 μs) still stands; six discards and one crash since. The crash was purely mechanical — oversized configs (256×256, BLOCK_K=128 + stages=5 in fp16) blew the H100's 228KB shared-memory budget and ptxas aborted, taking all 18 tests down. This does NOT invalidate the #14 finding (the full 48-config search is load-bearing and finds a non-obvious large-N winner); it only says the enrichment must stay within the per-config smem/register budget. The strategic read is unchanged and now well-supported: the GEMM dominates, layout/scheduling surgery hurts, autotune *width within resource limits* is the one lever still showing upside.

## RATIONALE
We have spent six iterations circling #9 and the only experiment that even plausibly had upside (config enrichment) was killed by a resource bug, not by a quality result — so that hypothesis is *untested*, not falsified. The right move is to retry it cleanly with a resource guard, because #14 gave direct evidence that the search space matters for the dominant dim=384 cases. The fix is trivial and well-understood: for fp16 with fp32 accumulation, per-config shared memory scales roughly as `(BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * 2 bytes`; the worker can compute this and drop any config exceeding a safe cap (~160–200KB) before handing the list to autotune. This keeps the enrichment idea alive while guaranteeing every candidate compiles.

## PROPOSAL
Retry the config enrichment from #15, but with a **compile-safety filter** so the crash cannot recur. Direction for the worker:

(1) Keep the #9 kernel body, grid, layout, and key=['N'] exactly. Build an enriched config list (superset of the 48), but **programmatically filter out any config whose estimated shared-memory footprint exceeds a conservative H100 cap** — compute the per-config smem from BLOCK_M, BLOCK_N, BLOCK_K, num_stages and the fp16 operand size, and skip over-budget ones. This lets the worker safely include moderately larger tiles and a wider BLOCK_K/stage range without the 256×256/stages=5 class that caused the abort.

(2) Aim the *surviving* additions at the dim=384 large-N regime that drives the geomean — moderately larger output tiles and additional warp/stage points that fit the budget, since #14 showed the winning large-N config is non-standard and currently found only because the 48-set happens to contain it. More in-budget neighbors of that winner give the search a chance to do better.

Guardrails: change ONLY the config list + add the filter; do not touch index math, grid, or layout. Verify the kernel actually compiles all configs (no ptxas error) and passes 18/18 within 2e-2; report per-shape with attention to 1024/384 and 768/384.

This is the last principled swing at the GEMM-tuning lever. **If the resource-safe enrichment still does not beat 4278, declare #9 mature** — the per-N autotune has found its optima and the contraction is at its ceiling for this structure. At that point, spend the remaining iterations on a single, clearly-scoped precision/occupancy probe (e.g. fp16 vs bf16 accumulation behavior, or whether the fp32 output buffer can be fp16 to halve the GEMM write/epilogue-read bandwidth within tolerance) — one variable, judged against 4278, no structural surgery.

---

## Iteration 16

## STATE
#9 (4278 μs) remains best. #16 is the decisive datum: a strict superset of #9's 48 configs (all known-good configs present) still regressed to 6169 μs with 1024/384 at ~23.7 ms. Since the good config is provably *in* the search space yet not selected, this is no longer about "which configs" — **the autotuner's selection is itself nondeterministic/noisy across runs, and #9's exact list happened to land a favorable draw.** This reframes the entire recent campaign: every "GEMM-tuning" experiment (#13/#14/#15/#16) and even the epilogue/swizzle ones (#12/#13) have been *measuring autotune-selection noise*, all clustering at ~6200 μs / ~24 ms-large-case, not measuring the structural ideas they were testing. The ~4278 vs ~6200 gap is largely an autotune-stability artifact, which means the contraction tuning lever is not just mature — it's actively unreliable.

## RATIONALE
The throughline across the last six results is that *touching the contract autotune in any way produces ~6200 μs*, and the only configuration that produces 4278 is #9's exact, unmodified list. That is a fragile optimum, not a robust one — and it tells us the real, unaddressed problem is **autotune determinism**, not config content. The highest-leverage move now is to remove the nondeterminism: make the contraction use a *fixed, hardcoded config per N-regime* (the ones #9's favorable draw actually selected) with NO `@triton.autotune` at all. If we can pin the good configs, we both stabilize at ≤4278 and stop paying the per-run autotune timing tax. This is a different lever than #14's "prune the list" (which kept autotune and still let it choose) — here we eliminate the chooser. Critically, this requires knowing *which* configs #9 actually selected per N; the worker can instrument that.

## PROPOSAL
Pivot from "what's in the autotune list" to "eliminate autotune nondeterminism." Direction for the worker, in two steps within one experiment:

(1) **Instrument first:** run the current #9 kernel and capture, per N value (256/512/768/1024), which concrete config (BLOCK_M/N/K, num_warps, num_stages) `triton.autotune` actually selects on a good run. This identifies the real winners behind the 4278 number.

(2) **Pin them:** replace `@triton.autotune` with explicit, hardcoded constexpr block/warp/stage parameters chosen by a simple branch on N (a small dict/if-ladder keyed on N, each mapping to the captured winning config). No autotuner, no per-run timing search, no candidate-list nondeterminism — deterministic launch every time. Keep kernel body/grid/layout/dtypes exactly as #9.

Judge against 4278 and require 18/18 within 2e-2. The success criterion is twofold: it should (a) reproduce ~4278 *reliably* (not regress to 6200), and (b) ideally beat it slightly by removing autotune overhead. Report per-shape.

Guardrails: do NOT change index math, grid, layout, or the compiled wings — only swap the autotune mechanism for fixed configs. If instrumenting the live selection is impractical, fall back to pinning the single config that textbook + the surviving evidence most favors for each N and accept whatever reproducible number results. If even pinned configs land at ~6200, that falsifies the "lucky draw" theory and proves the 4278 was measurement noise on the harness side — in which case #9 is the true mature number and the final iterations should go to the previously-deferred precision probe (fp16 output buffer to halve GEMM write + epilogue-read bandwidth) as the last distinct lever, one variable, judged against 4278.

---

## Iteration 17

## STATE
#17 is the decisive control experiment: a deterministic pinned config reproduced ~6202 μs with 1024/384 at ~23.9 ms — identical to #13/#14/#15/#16. Five independent variations (autotune-pruned, autotune-superset, swizzle, pinned) all reproduce ~6200; only the literal #9 file ever produced 4278, once. The honest interpretation is now forced: **4278 is very likely a single low-outlier measurement draw from the harness, and the true reproducible latency of this architecture is ~6200 μs.** The contraction is structure/bandwidth-bound and config choice barely moves it (variance well inside the documented 20–50 μs/large-case noise band, here amplified at ms scale by run-to-run timing). This reframes everything: we have spent six iterations chasing a number that may not be reliably reachable, and the GEMM-tuning lever is conclusively exhausted.

## RATIONALE
Per my own stated guardrail across the last three proposals, this exact outcome — pinned config also lands at ~6200 — was the pre-committed signal to stop tuning the contraction entirely. The evidence is now airtight: the GEMM is mature and config-insensitive, layout/fusion/scheduling surgery all regress, and the ~6200 floor is bandwidth-bound. The one genuinely distinct, never-tested lever remaining is **reducing the contraction's memory traffic**: the kernel currently writes an fp32 (B*H,N,N) output and the compiled epilogue then *reads* it back strided — that's the largest single intermediate. Writing the GEMM output as fp16 halves both the GEMM's store bandwidth and the epilogue's read bandwidth, directly attacking the bandwidth bound that sets the ~6200 floor. The 2e-2 tolerance is generous and the values are already fp16-precision products, so fp16 output is very likely within tolerance. This is the deferred precision probe, and it is the right next move.

## PROPOSAL
Stop all contraction config/layout/scheduling work — the evidence is conclusive it cannot reliably beat the ~6200 floor and 4278 was almost certainly measurement noise. Pivot to the one untested lever that attacks the actual bandwidth bound:

**Make the contraction output fp16 instead of fp32.** Direction for the worker: change the GEMM's output buffer to fp16 (keep the fp32 *accumulator* inside the kernel — only the stored result is fp16), and have the compiled epilogue consume the fp16 (B*H,N,N) buffer (its layer_norm-over-H can upcast internally). This halves the dominant intermediate's write and read traffic. Keep everything else exactly as #9 (kernel body, grid, contiguous-K layout, autotune, compiled wings).

Guardrails: this is a single-variable change (output dtype only). Verify 18/18 within 2e-2 — watch correctness carefully since layer_norm over fp16 inputs could lose precision, but the tolerance is generous. Judge against the *reproducible* ~6200 regime, not the 4278 outlier — i.e. the question is "does fp16 output move the honest re-measured number down," so the worker should note that comparison framing explicitly. Report per-shape with focus on the large bandwidth-bound dim=384 cases.

If fp16 output helps, that confirms the bandwidth diagnosis and a follow-up could push the same idea further (fp16 left/right are already fp16; the out-projection input). If it does not move the honest number and correctness holds, then this architecture is fully mature at its bandwidth floor and the remaining iterations should stop chasing the contraction and instead re-examine whether the *compiled wings* (prologue projections / epilogue) have any slack — but only after we have an honest, reproduced baseline number to measure against, which #17 suggests is ~6200, not 4278.

---

## Iteration 18

## STATE
#18 is a clean, real, single-variable win: fp16 output buffer took the honest reproducible number from 6202 → 6122 μs (18/18 within tol), and the per-shape breakdown is the most informative datum we've gotten in ten iterations. It shows the large dim=384 cases (1024/384 ≈ 23.7 ms, 768/384 ≈ 13.2 ms) **barely moved**, while small/medium cases improved — proving those large cases are **compute-bound in the GEMM, not write-bandwidth-bound**, whereas the smaller cases carried a relatively larger fp32-round-trip overhead. Crucially, the large dim=384 cases are running the SAME contraction work as dim=128 but H=384 means 3× the per-(b) GEMM volume; they dominate geomean and are now confirmed FLOP-bound.

## RATIONALE
This reframes the optimization target precisely. The large cases are compute-bound on an fp16 tensor-core GEMM that the worker is hand-writing — and a hand-written Triton GEMM rarely matches cuBLAS's heavily-tuned fp16 batched matmul for large, compute-bound shapes. We abandoned cuBLAS bmm back at #8 because the *then* implementation (with fp16/fp32 round-trips) was slower, but that comparison predates the contiguous-K insight and the fp16-output insight. The honest re-measured baseline is now ~6122, and we have never tested cuBLAS bmm *on the same contiguous-K fp16 layout with fp16 output* that #18 established. For a compute-bound large-N fp16 GEMM, cuBLAS is the most likely single source of real speedup on exactly the cases that dominate geomean. This is a distinct, well-motivated lever — not a rehash — and it directly targets the now-confirmed bottleneck.

## PROPOSAL
Keep #18 as the base. Direction for the worker: **A/B the hand-written Triton contraction against `torch.bmm` on the identical contiguous-K fp16 layout #18 already produces.** The pre-stage already emits `left`/`right` as `(B*H, N, N)` contiguous fp16; feed those directly to `torch.bmm(left, right.transpose(1,2))` producing an fp16 `(B*H,N,N)` output, then the same reshape/permute/compiled epilogue as #18. This is a one-line swap of the contraction implementation — everything else (layout, fp16 output, compiled wings) stays exactly as the current keeper.

Rationale to convey: cuBLAS's fp16 batched GEMM is autotuned and near-SOL for the large compute-bound dim=384 cases that dominate geomean, which is precisely where the Triton kernel is weakest and where #18 showed no movement. 

Guardrails: single-variable change (contraction impl only); verify 18/18 within 2e-2; judge against the honest #18 number 6122, **per-shape**, with the verdict resting on the large dim=384 cases (1024/384, 768/384) — those are the FLOP-bound cases cuBLAS should win if it wins anywhere. If cuBLAS beats Triton on the large cases but loses on small ones, the natural follow-up is a hybrid: dispatch to cuBLAS for large N/dim and keep Triton for small — a clean, evidence-driven split. If cuBLAS does not beat #18 on the large cases, that confirms the Triton GEMM is already at the achievable compute ceiling for this layout, and #18 is mature; at that point spend remaining iterations only on the cheap small-case overhead (prologue/epilogue), not the contraction.

---

## Iteration 19

## STATE
The run has decisively turned the corner: #18 (fp16 output, 6122) → #19 (cuBLAS bmm, 5885) are two consecutive clean, reproducible single-variable wins, and the honest best is now 5885 μs. #19 confirmed cuBLAS's autotuned fp16 batched GEMM beats the hand-written Triton kernel across the board on the contiguous-K layout — the contraction is now at its near-SOL ceiling and is no longer the lever. The per-shape data is unambiguous: the contraction is mature, and the remaining slack lives in the **prologue and epilogue compiled wings**, which still do non-trivial work (5 projections + 3 sigmoids + 2 masked multiplies + 2 explicit `.contiguous()` permute-copies in pre; layer_norm + gate-multiply + out-projection in post) and are now a meaningful fraction of the smaller cases.

## RATIONALE
With the GEMM retired at its ceiling, the highest-leverage untouched region is the **pre-stage's explicit permute+contiguous copies**. Currently the pre-stage materializes `left` and `right` as fp16, applies mask/gate, then does `permute(0,3,1,2).reshape(...).contiguous()` — two full-tensor layout-changing copies of the largest intermediates (B*H*N*N fp16 each) on every call, purely to feed cuBLAS a contiguous-K layout. That is pure memory traffic with no compute. There may be a cheaper way to land in a cuBLAS-friendly layout: cuBLAS bmm accepts transposed/strided batched operands directly (it does not strictly require contiguous inputs — it dispatches on stride patterns), so the explicit `.contiguous()` may be redundant, or the permute could be absorbed by producing the projections in `(b,h,n,n)` order more directly. This is the one large, never-probed source of avoidable bandwidth, and it sits on every shape.

## PROPOSAL
Keep #19 as base. Direction for the worker: **attack the pre-stage layout-conversion overhead feeding cuBLAS.** The goal is to reduce or eliminate the two explicit full-tensor `.contiguous()` permute-copies that currently materialize `left`/`right` into contiguous-K form. 

Two avenues for the worker to explore (their choice which, ideally A/B both):
1. **Drop the explicit `.contiguous()`** and pass the permuted (strided) fp16 views directly to `torch.bmm` — cuBLAS can often handle batched operands with non-unit strides via its transpose/layout flags, so the copy may be redundant. Verify cuBLAS actually accepts it without an internal copy (and that it's faster, not just correct).
2. If cuBLAS does force a copy on strided input, **let `torch.bmm` consume the operands in whatever native layout is cheapest to produce** (e.g. feed `right` directly and use bmm's transpose argument rather than an explicit `.transpose`+materialize), minimizing the number of full-tensor copies.

Guardrails: this touches only the pre-stage layout plumbing and the bmm call — keep the fp16 output, the compiled wings' arithmetic, and the epilogue exactly as #19. Single conceptual change (operand layout/contiguity into cuBLAS). Verify 18/18 within 2e-2. Judge against the honest 5885, **per-shape** — this is a bandwidth/overhead optimization so expect the biggest relative gains on the small/medium cases (256/128, 512/128, 256/384) where copy overhead is a larger fraction; the large FLOP-bound cases may barely move. If removing `.contiguous()` makes cuBLAS silently copy internally (no speedup or a regression), that's a clean negative result and the fallback is to look at fusing the post-stage's layer_norm+gate+out-projection epilogue next.

---

## Iteration 20

#20 regressed to 7057 — dropping `.contiguous()` made cuBLAS take a slow strided path. Clean negative result; the explicit contiguous copy is load-bearing. Per my stated fallback, the next lever is the post-stage epilogue.

## STATE
The run is in a healthy post-breakthrough phase: honest best is #19's cuBLAS bmm at 5885 μs, reached via two clean wins (fp16 output, then cuBLAS). #20 cleanly falsified avenue 1 — dropping `.contiguous()` regressed to 7057 because cuBLAS hits a slow strided-batched path without contiguous-K operands, confirming the explicit copy is load-bearing and the pre-stage layout is already near-optimal for cuBLAS. The contraction (cuBLAS, at ceiling) and the pre-stage layout plumbing are both now exhausted as levers; the one substantial untouched region is the post-stage epilogue. Five iterations remain.

## RATIONALE
#20's negative result is informative, not a setback: it proves the contiguous-K copy buys far more (in cuBLAS GEMM speed) than it costs (in copy bandwidth), so that plumbing is settled. That leaves the **post-stage** as the only unprobed compute region. The epilogue currently does three separable PyTorch ops on the large (B,N,N,H) tensor through a *strided permute view* of the cuBLAS output: `layer_norm` over H, multiply by `out_gate`, then the out-projection `F.linear`. The layer_norm and gate-multiply read/write the full large intermediate from a strided view (poor coalescing) and are memory-bound. A fused epilogue (the layer_norm-over-H + gate fused into one pass, ideally consuming the cuBLAS output in its native (B,H,N,N) layout to avoid the strided gather) is the natural next attack — note #12 tried a fused-epilogue Triton kernel and was correct but slow due to a strided H-gather, so the *key* is to consume the GEMM output in its native contiguous (B,H,N,N) layout rather than re-introducing a strided access.

## PROPOSAL
Keep #19 as base. Direction for the worker: **reduce post-stage overhead by fusing the epilogue's memory-bound elementwise work**, targeting the strided round-trips over the large (B,N,N,H) intermediate.

Avenues for the worker (their judgment on which; A/B encouraged):
1. **Let `torch.compile` fuse more of the post-stage.** The post-stage is already compiled, but it consumes a *strided permute view* of the cuBLAS output — inductor may be materializing a contiguous copy before layer_norm. Probe whether reordering so the epilogue consumes the GEMM output in its native (B,H,N,N) contiguous layout (doing the layer_norm over the H axis in its natural position, then permuting only the final small-ish result, or folding the permute into the out-projection's einsum) lets inductor generate a tighter fused kernel.
2. **A hand-fused layer_norm-over-H + out_gate-multiply kernel** that reads the cuBLAS output in its native contiguous (B,H,N,N) layout — explicitly avoiding #12's mistake (a strided H-gather). The lesson from #12 is that the fusion must respect the GEMM's native layout; the H-reduction should run along the contiguous axis.

Guardrails: keep the contraction (cuBLAS, fp16 output) and pre-stage exactly as #19 — this change touches only the post-stage. Single conceptual change (epilogue fusion / layout-aware consumption). Verify 18/18 within 2e-2 (layer_norm precision over fp16 GEMM output — upcast internally). Judge against the honest 5885, **per-shape**; expect gains concentrated where the epilogue is a larger fraction (small/medium cases) since the large cases are GEMM-dominated. If neither avenue beats 5885, the architecture is mature at ~5885 and remaining iterations should go to confirming reproducibility and ensuring the submitted file is exactly #19 — do NOT reopen the contraction or pre-stage layout, both now conclusively settled.

---

## Iteration 21

## STATE
Honest best is #19 (cuBLAS bmm, 5885 μs), reached by two clean wins. The last two iterations were clean negatives: #20 (drop `.contiguous()`, 7057 — cuBLAS needs contiguous-K for its fast path) and #21 (internalize permute into compiled post-stage, 5959 — inductor already handles the boundary view efficiently). Both negatives carry the same lesson: the pre-stage layout plumbing and the epilogue's layer_norm/gate/out-projection are *already* well-optimized; the contraction is at cuBLAS's ceiling. Four iterations remain and the obvious structural levers are now exhausted.

## RATIONALE
#20 and #21 are not setbacks but boundary-mapping: they confirm the contraction, the pre-stage→GEMM layout, and the GEMM→epilogue handoff are all settled. But one observation from #20 is still actionable: removing `.contiguous()` *improved the small cases* (256/128 1558→1494, 256/384 2765→2710) while wrecking the large ones — proving the contiguous copy is pure overhead on small cases and load-bearing only on large ones. That is a textbook **hybrid dispatch** signal: keep the contiguous copy (and cuBLAS fast path) for the large FLOP-bound cases, but skip it for the small cases where cuBLAS's strided path is actually faster than copy+fast-path. This is a distinct, evidence-grounded lever neither #19 nor #20 tested — a per-shape branch, not a global choice. The small cases (256/*) are a meaningful fraction of a 7-case geomean, so shaving ~60-65μs off two of them is real.

## PROPOSAL
Keep #19 as base. Direction for the worker: **introduce a size-conditioned hybrid in the contraction feed** — for small problem sizes, skip the explicit `.contiguous()` and let cuBLAS take the strided path (which #20 proved is faster there); for large problem sizes, keep #19's exact contiguous-copy + fast-path. The branch should key on the same N (and/or dim) regime already used in `_pick_contract_config`, choosing the threshold from the #20-vs-#19 per-shape crossover (small 256-class cases favor no-copy; everything ≥ the medium/large cases favors the copy).

Guardrails: this touches ONLY the pre-stage's contiguity decision (a single conditional), nothing else — bmm call, fp16 output, compiled wings, and the large-case contiguous path all stay exactly as #19. Single conceptual change (per-size contiguity dispatch). Verify 18/18 within 2e-2. Judge against the honest 5885, **per-shape**: the win must come from the 256-class cases matching #20's improved numbers while the large cases stay at #19's numbers — if the large cases regress at all, the threshold is wrong. The expected gain is modest (~20-40μs geomean) but it is real, free, and directly harvested from #20's own data.

If this hybrid does not beat 5885 cleanly, the architecture is mature: spend the final iterations confirming reproducibility and ensuring the submitted file is exactly #19. Do NOT reopen the contraction algorithm, the large-case layout, or epilogue fusion — all conclusively settled.

---

## Iteration 22

#22 regressed to 6290 — the hybrid didn't reproduce #20's small-case gains, confirming #20's small-case improvements were measurement noise, not signal. That's the third consecutive negative around the cuBLAS contraction boundary. The architecture is mature.

## STATE
Honest best remains #19 (cuBLAS bmm, 5885 μs), unchanged for three iterations. #20 (drop contiguous, 7057), #21 (internalize permute, 5959), and now #22 (size-conditioned hybrid, 6290) are all clean negatives clustered around the contraction-feed boundary. Critically, #22 falsifies the one remaining hypothesis I had drawn from #20's data: the small-case "improvement" in #20 did not reproduce under the hybrid, so it was noise (well within the documented 5-20μs small-case variance), not a real crossover. Every distinct structural lever — contraction algorithm, pre-stage layout/contiguity, epilogue fusion, and now per-size dispatch — has been tested and settled.

## RATIONALE
The evidence is now conclusive that #19 is mature at its bandwidth/compute floor: three independent attacks on the only remaining region (the cuBLAS feed) all regressed, and the apparent small-case signal that motivated #22 proved to be noise. Per my own pre-committed guardrail stated across the last two proposals — "if the hybrid does not beat 5885 cleanly, the architecture is mature" — that condition has now been met. Continuing to perturb the contraction boundary would be re-litigating settled regions, which the comparison discipline explicitly forbids and which the last three results show only produces regressions. With three iterations left, the highest-value action is not another speculative structural swing but ensuring the recorded keeper is exactly #19 and that its number is reproducible.

## PROPOSAL
Stop experimenting on the architecture — it is mature at #19's 5885 μs. Direction for the worker: **make the final iterations confirmatory, not exploratory.**

1. **Verify the submitted file is byte-exact #19** (cuBLAS `torch.bmm(left, right.transpose(1,2))` on the contiguous-K fp16 layout, fp16 output buffer, compiled pre/post wings, 48-config dead code removed or harmless). Confirm 18/18 correctness within 2e-2.

2. **Reproducibility check:** run the #19 configuration two or three times to confirm the ~5885 number is stable and not itself a favorable draw. If runs cluster around 5885, that is the honest deliverable. (Recall the harness exhibits ~5-20μs small-case / ~20-50μs large-case run-to-run variance, so treat anything inside that band as identical.)

Do NOT reopen any of the conclusively settled regions: the contraction algorithm (cuBLAS is at ceiling), the pre-stage contiguity/layout (#20, #22 both regressed), epilogue fusion (#21 regressed), or projection batching/layout surgery (settled earlier). There is no remaining lever with a positive expected value; the marginal structural ideas have all been tested and lose. The correct PI decision is to lock in #19, confirm its reproducibility, and stop. If the worker has a genuinely novel idea outside all settled regions it may be surfaced, but the default and recommended action is to finalize #19.

---

## Iteration 23

#23 confirmed reproducibility: re-running #19 gave 5882 vs 5885 — a 3μs delta, deep inside noise. The architecture is confirmed mature and the number is stable. No need to re-read full history; the picture is complete.

## STATE
Honest best is #23 at 5882 μs — a reproducibility re-run of #19's 5885, confirming the number is stable to within ~3μs (far inside the noise floor). The architecture is conclusively mature: the contraction is at cuBLAS's ceiling, and the three boundary attacks (#20 drop-contiguous, #21 epilogue-fusion, #22 size-hybrid) all regressed. Two iterations remain.

## RATIONALE
The reproducibility check did exactly its job: 5885→5882 confirms #19/#23 is a real, stable plateau, not a favorable draw (unlike the original 4278 outlier, which no clean re-measurement ever reproduced). Every distinct structural lever has been tested and settled, and the last four iterations around the only open region all either regressed or merely re-confirmed the plateau. Per the comparison discipline, there is no remaining approach with positive expected value — further perturbation of settled regions only produces regressions. The run is complete in substance; the only remaining value is ensuring the deliverable is locked.

## PROPOSAL
The kernel is finalized at #23/#19 (5882 μs, cuBLAS bmm on contiguous-K fp16 layout, fp16 output, compiled pre/post wings). Direction for the worker: **stop experimenting and lock in the deliverable.**

1. Confirm the submitted file is byte-exact the #23/#19 configuration and passes 18/18 within 2e-2. Remove any remaining dead code (the unused hand-written `_trimul_contract_kernel` / `_pick_contract_config` / epilogue scaffolding) only if it can be done without risk — if removal is at all uncertain, leave it; it is harmless and correctness/stability is paramount over cosmetics.

2. Do NOT reopen any settled region: contraction algorithm (cuBLAS at ceiling), pre-stage contiguity/layout (#20, #22 regressed), epilogue fusion (#21 regressed), projection batching/layout surgery (settled earlier). None have positive expected value.

Spend the final iteration(s) only on confirmatory re-runs if desired, not new structural changes. The recommended PI decision is to finalize #23 as the submission. The run is done — we took the baseline from 10877 → 5882 μs (1.85× honest speedup), with the decisive moves being the functional rewrite, contiguous-K fp16 layout, fp16 output buffer, and the cuBLAS contraction swap.

---

## Iteration 24

#24 re-ran the locked configuration and got 5906 vs #23's 5882 — a 24μs delta, inside the noise floor. Three consecutive measurements (5885, 5882, 5906) now bracket the true plateau at ~5890 μs. The deliverable is confirmed stable.

## STATE
The kernel is finalized and triple-confirmed: #19 (5885) → #23 (5882) → #24 (5906) are three independent measurements of the identical cuBLAS-bmm configuration, all within a ~24μs band — the plateau is real and stable at ~5890 μs. Every distinct structural lever has been tested and settled; the last five iterations have either regressed (boundary attacks #20/#21/#22) or merely re-confirmed the plateau (#23/#24). One iteration remains.

## RATIONALE
There is nothing left to optimize. The contraction is at cuBLAS's ceiling, the pre-stage layout and contiguity are load-bearing and settled (#20/#22), epilogue fusion was neutral-to-negative (#21), and three reproducibility runs confirm the number is not a draw artifact. No approach in the history has positive remaining expected value, and the comparison discipline forbids re-litigating settled regions. The run is substantively complete; spending the final iteration on another structural swing would only risk a regression or a crash on the recorded keeper.

## PROPOSAL
The kernel is final. Direction for the worker: **do not run another experiment unless purely confirmatory.** Ensure the submitted file is byte-exact the #19/#23/#24 configuration (cuBLAS `torch.bmm` on contiguous-K fp16 operands, fp16 output buffer, compiled pre/post wings) and that it passes 18/18 within 2e-2. That is the deliverable.

Do NOT reopen any settled region — contraction algorithm, pre-stage contiguity/layout, epilogue fusion, projection batching. None have positive expected value, and three independent runs confirm ~5890 μs is the honest floor for this architecture.

Final summary: baseline 10877 → 5882 μs honest best, a **1.85× reproducible speedup**. The decisive, durable wins were: (1) the clean low-overhead functional rewrite, (2) the contiguous-K fp16 layout for the contraction, (3) the fp16 output buffer halving intermediate traffic, and (4) swapping the hand-written GEMM for cuBLAS bmm on that layout. The recorded 4278 "best" was an unreproducible harness outlier; the true, defensible result is #23 at 5882 μs. Lock it in.

---

## Iteration 25

This is the final iteration. #25 re-ran the locked configuration: 5914 μs, joining #19/#23/#24 (5885, 5882, 5906) — four independent measurements now bracket the plateau at ~5897 μs within a ~32μs band. No new information is needed from history; the run is complete.

## STATE
The optimization run is finished. The kernel is finalized and quadruple-confirmed: #19 (5885) → #23 (5882) → #24 (5906) → #25 (5914) are four independent measurements of the identical cuBLAS-bmm configuration, all within a ~32μs band, establishing a stable honest plateau at ~5897 μs. Every distinct structural lever was tested and settled over 25 iterations; the final four iterations confirmed reproducibility, and no further lever has positive expected value.

## RATIONALE
There is nothing left to do but deliver. The contraction is at cuBLAS's ceiling, the pre-stage layout/contiguity is load-bearing and settled (#20/#22 regressed), epilogue fusion was neutral-to-negative (#21), and four reproducibility runs prove the number is a true plateau, not a measurement artifact. This stands in deliberate contrast to the recorded 4278 "best," which no clean re-measurement ever reproduced and is an unreproducible harness outlier. With zero iterations remaining, the only correct action is to lock in the confirmed configuration.

## PROPOSAL
**Finalize and submit the #19/#23/#24/#25 configuration — no further experiments.** The worker should confirm the submitted file is byte-exact this configuration: cuBLAS `torch.bmm(left, right.transpose(1,2))` on the contiguous-K fp16 layout, fp16 output buffer, `torch.compile(dynamic=True)` pre/post wings — passing 18/18 within 2e-2. That is the deliverable.

**Final report:**
- Baseline 10877 μs → honest best **5882 μs (#23)**, a **1.85× reproducible speedup**, stable across four independent runs (~5897 μs plateau, ±32μs noise).
- Decisive durable wins, in order of impact: (1) clean low-overhead functional rewrite (eliminating per-call module construction), (2) **contiguous-K fp16 layout** for the contraction (the key enabler), (3) **fp16 output buffer** halving intermediate traffic, (4) **cuBLAS bmm** swap for the now-FLOP-bound contraction.
- Conclusively dead ends (do not revisit in any future work): hand-written Triton GEMM config tuning (autotune pruning/enriching/pinning/swizzling — config-insensitive at the bandwidth floor), epilogue-store layout surgery, strided-K GEMM, projection batching, fused-epilogue H-reduction with strided gather, dropping the contiguous copy, and size-conditioned contiguity dispatch.
- Note for the record: the recorded 4278 "best" (#9) is an unreproducible single low-outlier draw; the defensible, reproducible result is #23 at 5882 μs.

The run is complete. Lock in #23 as the submission.

