# COMMANDMENT — DSA indexer `fp8_mqa_logits` (Triton, gfx942/MI300X)

IMMUTABLE MEASUREMENT CONTRACT. Every agent in this workflow MUST use exactly these commands.
Do not modify this file, `test_harness.py`, or anything outside the workspace.

- WORKSPACE: `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace`
- EVAL_DIR:  `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op`
- SKILL_DIR: `/shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow`
- GPU_ID: `0`  (AMD MI300X, gfx942)

## SETUP

```bash
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace
```

Nothing else. The workspace is a fresh artifact-free copy; `gpu_lock.sh` sets
`TORCH_EXTENSIONS_DIR=$PWD/.torch_ext` and `PYTORCH_ROCM_ARCH=gfx942` for you.
There is no compile step (pure Triton, JIT-compiled at first launch).
If you ever suspect a stale Triton/torch cache, MOVE it aside, never delete:
`mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`

## CORRECTNESS  (run BEFORE every benchmark; a failing kernel's timing is void)

```bash
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace && \
bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 \
python3 test_harness.py --correctness
```

Runs all 4 configs against `run_ref`, a pure-torch fp32 oracle re-implementation of the kernel math
that lives in `test_harness.py` and is INDEPENDENT of the file under optimization.
Gate: `rtol=atol=5e-2`, `err_ratio <= 0.02` per case. Must print `ALL CORRECTNESS CHECKS PASSED`
(exit 0). Baseline measures `err_ratio=0.0000` on all four cases — a candidate is allowed to be
looser but must stay under 0.02. Wall time ~11 s.

## BENCHMARK  (quick, 3 of 4 cases: idx 0, 2, 3)

```bash
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace && \
bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 \
python3 test_harness.py --benchmark
```

## FULL_BENCHMARK  (AUTHORITATIVE — all 4 cases; this is the number that decides winners)

```bash
cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace && \
bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/gpu_lock.sh 0 \
python3 test_harness.py --full-benchmark
```

20 warmup + 50 timed iterations per case, CUDA events, **median** reported. Wall time ~4 min at
baseline speed. Do NOT change `GEAK_WARMUP_ITERS` / `GEAK_BENCHMARK_ITERATIONS` — the recorded
baseline used the defaults (20/50) and altering them makes results incomparable.

## PROFILE

```bash
bash /shared_inference/mdeopuja/Hyperloom/.cache/GEAK@5107c7e4b2878b0e36a1a9cf9abd4aea37615953/kernel_workflow/scripts/profile_kernel.sh 0 \
  "cd /shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/workspace && python3 test_harness.py --profile" \
  <out_dir>
```

Verified working on this box: `rocprof-compute`/`omniperf` are NOT installed, so it degrades to
**rocprofv3** (`--kernel-trace --stats --output-format csv`), no `!!! PROFILER FAILED` block.
That gives durations + dispatch counts + `LDS_Block_Size / Scratch_Size / VGPR_Count /
Accum_VGPR_Count / SGPR_Count / Workgroup_Size / Grid_Size` per dispatch, but **no SoL / cache /
wavefront sections** — classify from durations, occupancy fields and the per-case latency table,
and say so (see `knowledge/profiling_guide.md` → per-profiler extraction, rocprofv3 bullet).
Artifacts: `<out_dir>/profile_report.txt` plus
`<out_dir>/rocprofv3/<host>/<pid>_kernel_stats.csv` and `..._kernel_trace.csv`.
Baseline profile: `_fp8_mqa_logits_kernel` = 4 dispatches, 94.12% of all GPU time, one dispatch per
case — i.e. exactly **1 kernel launch per call**, so there is no dispatch-count/overhead lever here.

## PARSE

Per-case latency lines in benchmark stdout look exactly like:

```
  M=8000 N=8000 H=64 D=128 torch.float8_e4m3fn  14.7343ms  (case=2 GEAK_RESULT_LATENCY_MS=14.7343)
```

Extract with `case=(\d+)\s+GEAK_RESULT_LATENCY_MS=([\d.]+)` — group 1 is the case id (index into
`ALL_CONFIGS`), group 2 is the median latency in ms. The FINAL standalone line
`GEAK_RESULT_LATENCY_MS=<x>` (no `case=` prefix, printed after `GEAK_SHAPES_USED=[...]`) is the
unweighted geomean over the cases that ran — use the LAST match in the output for the geomean and
the `case=`-tagged matches for per-case values. Case-id map (`M=N`, all H=64 D=128 fp8_e4m3fn):
`0 -> 2048`, `1 -> 4096`, `2 -> 8000`, `3 -> 8192`. `--full-benchmark` emits all of 0,1,2,3;
`--benchmark` emits only 0,2,3 (so its geomean 6.1867 ms is NOT comparable to the full one).
A case that errors prints `ERROR:` instead of a latency line and is silently dropped from the
geomean — treat any missing case id in a full-benchmark run as a FAILURE, not a speedup.

## METRIC

No `WORKLOAD_SPEC` was supplied and `test_harness.py` is not workload-weighted, so the PRIMARY
metric is the **unweighted geomean of per-case speedups over all 4 full-benchmark cases**:

```
speedup = geomean_i( baseline_ms[i] / candidate_ms[i] )   , i in {0,1,2,3}
        = 5.5324 / candidate_geomean_ms                    (equivalent, since baselines are fixed)
```

Baseline denominators (median of 3 full-benchmark runs, this box, this GPU):

| case | shape (M=N, H=64, D=128, fp8_e4m3fn) | baseline_ms |
|------|--------------------------------------|-------------|
| 0 | 2048 | 1.0605 |
| 1 | 4096 | 3.9661 |
| 2 | 8000 | 14.7343 |
| 3 | 8192 | 15.1161 |
| **geomean** | | **5.5324** |

The baseline is the **pristine original** `triton_fp8_mqa_logits.py` (the real vLLM/aiter gfx942
vendored kernel), frozen at `EVAL_DIR/baseline/` and byte-identical to the workspace copy at
round 0. It is NOT a naive LLM re-implementation — the denominator is representative.

Secondary diagnostics (report, do not optimize against): arithmetic-mean sum of the 4 cases
(baseline 34.877 ms) and the per-case speedups. The op is prefill/long-context, so cases 2 and 3
dominate any wall-clock view; a candidate that wins only at M=2048 is not a win.

## MODIFIABLE FILES

- `workspace/triton_fp8_mqa_logits.py` — the ONLY file any engineer may edit
  (both `_fp8_mqa_logits_kernel` and the `fp8_mqa_logits_gfx942` launcher, incl. host-side prep).

## RULES

1. NEVER modify `test_harness.py`, `config.yaml`, this COMMANDMENT, `EVAL_DIR/baseline/`, or
   anything outside the workspace.
2. NEVER weaken the correctness gate (`rtol=atol=5e-2`, `err_ratio<=0.02`) or special-case the
   oracle's inputs. `setup_inputs` uses `torch.manual_seed(42)` — do NOT exploit the fixed values
   (no caching keyed on data, no precomputed outputs).
3. ALWAYS run CORRECTNESS before BENCHMARK. A benchmark of an incorrect kernel is void.
4. ALWAYS run GPU commands through `gpu_lock.sh 0` from inside the workspace dir.
5. The kernel must remain a drop-in for `fp8_mqa_logits_gfx942(q, k_fp8, kv_scales, weights,
   cu_starts, cu_ends) -> [M, N] fp32 logits`, with `-inf` outside `[cu_starts[i], cu_ends[i])`.
   Host-side work inside the launcher (scrubs, allocations, autotune) IS counted in the timing —
   it runs inside the timed region.
6. FULL_BENCHMARK output is the single source of truth. Report the number the harness printed;
   never extrapolate from a microbenchmark or a partial `--benchmark` run.
