# COMMANDMENT — GLM-5.1-FP8 MoE stage-1 g1u1 (`moe_g1u1_fp8`)

IMMUTABLE measurement contract. Every agent in this run MUST use these exact commands and this
exact parse rule. Do not modify this file, `test_harness.py`, or `config.yaml`.

Variables used below (expand them literally):
- `WORKSPACE = /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace`
- `SKILL_DIR = /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow`
- `GPU_ID = 0`

## HARNESS PROVENANCE
The workspace already ships a task runner (`config.yaml` + a self-contained `test_harness.py`).
It is REUSED VERBATIM — no new harness was authored. It is NOT a workload-weighted oracle
(it prints no `GEAK_WEIGHTED_SPEEDUP`, and there is no `meta.json`/`WORKLOAD_SPEC`), so the
PRIMARY metric is the unweighted geomean over its 4 built-in cases.

Correctness oracle = `run_ref()` inside `test_harness.py`: a pure-torch fp32 dequant + grouped
GEMM + silu*mul, fully independent of the file under optimization. Tolerance rtol=atol=5e-2 with
`err_ratio <= 0.05`. This is a TRUSTED reference, not a naive-baseline perf strawman.

Perf baseline (speedup denominator) = the pristine original
`/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/baseline/moe_fp8_blockscale_g1u1.py`
(the seed Triton implementation as it entered the run), measured below. It is REAL workload code,
not an LLM reimplementation.

## SETUP
```
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace
```
Nothing to clear (the workspace is a fresh artifact-free copy). If you ever suspect a stale torch
extension build, MOVE it aside, never delete:
`mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`

## CORRECTNESS  (always run BEFORE any benchmark; all 4 cases must PASS)
```
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace && bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 python3 test_harness.py --correctness
```
PASS iff the final line is `ALL CORRECTNESS CHECKS PASSED` (exit 0). Any `FAIL`/`ERROR` line, or
`CORRECTNESS FAILED`, is a hard reject — no perf number from that candidate counts.

## BENCHMARK  (quick screen; 3 of the 4 cases: indices 0, 2, 3)
```
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace && bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 python3 test_harness.py --benchmark
```

## FULL_BENCHMARK  (AUTHORITATIVE — all 4 cases; this is the number of record)
```
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace && bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 python3 test_harness.py --full-benchmark
```
20 warmup + 50 timed iterations per case, CUDA events, MEDIAN reported. Whole command takes ~7 s.
Do NOT change `GEAK_WARMUP_ITERS` / `GEAK_BENCHMARK_ITERATIONS` — they must stay 20/50 for every
candidate or the numbers are not comparable.

## PROFILE
```
WARMUP_RUNS=0 bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/profile_kernel.sh 0 "cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp/team_seed_op_20260813_105508_1474_15052/seed_op/workspace && python3 test_harness.py --profile" <out_dir>
```
`WARMUP_RUNS=0` is REQUIRED on this box: the script's default 3 warmup runs leave the GPU above
`gpu_lock.sh`'s idleness threshold for a few seconds, so the profiled run is refused with
`ERROR: GPU 0 has foreign work running` and `profile_report.txt` contains a `!!! PROFILER FAILED`
block. Verified: with `WARMUP_RUNS=0` rocprofv3 succeeds and writes
`<out_dir>/rocprofv3/*/{kernel_stats,kernel_trace,domain_stats,agent_info}.csv` containing
`_moe_g1u1_fp8_kernel`. If you need warm clocks, run FULL_BENCHMARK once yourself, sleep ~10 s,
then run the profile command. If a `!!! PROFILER FAILED` block still appears, follow the
fault-tolerance ladder in `knowledge/profiling_guide.md` and state the degradation explicitly.

Note for profiling: `--profile` executes each case exactly ONCE (no warmup loop), so the CSV
per-kernel durations include first-call JIT/autotune effects. Use `kernel_trace.csv` and drop the
first dispatch per shape, or profile `--benchmark` instead when you need steady-state kernel times.

## PARSE
Per-case latency comes ONLY from the benchmark stdout lines of the form:

```
  T=16 topk=8 E=256 N=2048 K=6144  10.3603ms  (case=0 GEAK_RESULT_LATENCY_MS=10.3603)
```

Regex: `case=(?P<case>\d+)\s+GEAK_RESULT_LATENCY_MS=(?P<ms>[0-9.]+)` — the value is milliseconds,
the median of 50 CUDA-event-timed iterations. The FINAL standalone line
`GEAK_RESULT_LATENCY_MS=<geo>` (no `case=` prefix, last line of output) is the geomean over the
cases that ran; take it as the aggregate but ALWAYS also record the per-case values.

Case-id mapping (fixed by `ALL_CONFIGS` in `test_harness.py`, `(num_tokens, top_k, E, N, K)`):
- `case=0` → T=16  topk=8 E=256 N=2048 K=6144 — decode, primary trace hot shape
- `case=1` → T=32  topk=8 E=256 N=2048 K=6144 — decode
- `case=2` → T=64  topk=8 E=256 N=2048 K=6144 — decode, heavier batch
- `case=3` → T=128 topk=8 E=256 N=2048 K=6144 — small chunked-prefill

`--benchmark` runs cases 0, 2, 3 only (`GEAK_SHAPES_USED=[0, 2, 3]`); `--full-benchmark` runs all
four. A case that raised prints `ERROR:` and emits NO latency line — treat a missing case as a
FAILURE of that candidate, never as "skip and geomean the rest".

## METRIC (PRIMARY)
No WORKLOAD_SPEC and no weighted oracle → PRIMARY = **unweighted geomean of per-case speedups over
all 4 FULL_BENCHMARK cases**:

```
speedup_i = baseline_ms_i / candidate_ms_i          (per case i in {0,1,2,3})
PRIMARY   = exp( (1/4) * Σ_i ln(speedup_i) )
```

Equivalently `PRIMARY = baseline_geomean_ms / candidate_geomean_ms` using the trailing
`GEAK_RESULT_LATENCY_MS` of each full-benchmark run. This is the number the round-winner gate and
the final result use.

Frozen baseline (median of 3 full-benchmark runs, spread <= 0.36% per case):

| case | shape | baseline_ms |
|---|---|---|
| 0 | T=16  topk=8 E=256 N=2048 K=6144 | 10.3734 |
| 1 | T=32  topk=8 E=256 N=2048 K=6144 | 14.3788 |
| 2 | T=64  topk=8 E=256 N=2048 K=6144 | 17.6388 |
| 3 | T=128 topk=8 E=256 N=2048 K=6144 | 19.4824 |

`baseline_geomean_ms = 15.0466`; sum over cases = 61.8734 ms.
Secondary diagnostic (report it, do not optimize against it): the total-time ratio
`61.8734 / Σ_i candidate_ms_i`.

## MODIFIABLE FILES
- `moe_fp8_blockscale_g1u1.py` — the ONLY editable file. Both `_moe_g1u1_fp8_kernel`,
  `moe_align_block_size`, and the `moe_g1u1_fp8` launcher live in it and are all fair game.

NEVER modify: `test_harness.py`, `config.yaml`, this COMMANDMENT, `baseline/`, or anything outside
the workspace. Do not add new top-level files that the harness would pick up in place of the kernel
(`test_harness.py` locates `moe_fp8_blockscale_g1u1.py` by name next to itself — keep that filename).

## RULES
1. Run CORRECTNESS before every BENCHMARK. A perf win with a correctness failure scores zero.
2. Every GPU command goes through `gpu_lock.sh 0`, invoked from inside the workspace dir.
3. FULL_BENCHMARK stdout is the source of truth. Never report a hand-rolled timing loop, a
   `time python3 ...` wall clock, or a number measured outside `gpu_lock`.
4. The public signature of `moe_g1u1_fp8(a_fp8, a_scale, w1_fp8, w1_scale, topk_ids, N, group_n,
   group_k, ...)` and its return (`[num_tokens*top_k, N]` bf16) are FIXED — the harness calls it
   with keyword args `N=`, `group_n=`, `group_k=`.
5. No caching keyed on input tensor identity / no memoizing the result across the 50 timed
   iterations. Every timed call must do the full real work (align + GEMM + epilogue). Caching a
   *shape-derived* artifact (a compiled config, a preallocated scratch buffer) is fine; caching the
   *output* for identical inputs is cheating and will be rejected on review.
6. Correctness compares against the harness's own fp32 oracle, so any numerics change (different
   accumulation order, split-K) must still land inside rtol=atol=5e-2, err_ratio<=0.05.
