# TechLead Final Report — `fp8_mqa_logits` (Triton, gfx942 / MI300X)

## Summary

| | |
|---|---|
| Kernel | `_fp8_mqa_logits_kernel` / `fp8_mqa_logits_gfx942` in `triton_fp8_mqa_logits.py` (real vLLM/aiter gfx942 path, not a toy reimplementation) |
| Kernel type | `triton` |
| Device | MI300X-class, gfx942 / CDNA3, 304 CU, 192 GB HBM (~5.3 TB/s peak) |
| Baseline geomean | **5.5324 ms** (median of 3 full-benchmark runs, cross-run spread ≤ 0.27%) |
| Final geomean | **0.7014 ms** |
| **Headline speedup (geomean, unweighted — run is NOT workload-aligned)** | **7.8554×** |
| Arithmetic mean of per-case speedups | 7.9804× |
| Ratio-of-sums (34.877 ms → 3.940 ms) | 8.85× |
| Rounds run | 2 |
| Budget used | 3 directions (2 in round 1, 1 in round 2) |
| Correctness | ALL PASSED on all 4 cases, `err_ratio = 0.0000`, `cos_diff` 3.45e-09 … 3.55e-09 |
| Final patch | `final_patch.diff` (291 lines, single file, 175 insertions / 28 deletions) |
| Optimized source | `optimized/triton_fp8_mqa_logits.py` |

The op is a rank-128 fp8 dot per (row, KV-tile) plus a 64-way cross-lane head reduction, ReLU, a
per-KV scale and a windowed `-inf` fill. It is **compute/instruction-mix bound throughout** — never
memory bound (HBM peaked at 3.8–4.7% of peak and *fell* as the kernel got faster), never
latency/occupancy bound (more resident waves measured slower in all three rounds), and there is no
launch-overhead floor (exactly 1 dispatch per call, latency scales linearly with work).

---

## Round-by-round

### Round 1 — two orthogonal lanes (kernel body vs launcher)

| id | specialty | strategy | verified | verdict |
|---|---|---|---|---|
| `r1_d0` | compute | Bitcast both `tl.dot` operands to `tl.float8e4b8` (fnuz, the gfx942-native fp8) so the dot emits `v_mfma_f32_16x16x32_fp8_fp8` instead of a software fp8→fp16 upconvert + fp16 MFMA; fold the ×4 exponent-bias correction into the loop-invariant `w_block`; scrub the 0x80 codepoint | **2.7928×** (target 3.5×) | success — partial vs target |
| `r1_d1` | host_runtime | Repair the `_gfx942_default_tile_fits_lds` LDS model (it charged the KV tile 2 B/elem, an fp16-expanded tile, wrongly rejecting BLOCK_KV=128), key `matrix_instr_nonkdim` off the dot shape (16) instead of `seq_len`, raise `waves_per_eu` 2→3 | **2.0512×** (target 1.9×) | success — beat target |

**Integrate:** the two lanes are strictly disjoint (`@triton.jit` body vs launcher), so they merged
with a plain `git apply`, no hand-merge. Result **6.7551×** — *super-multiplicative* vs the 5.73×
product, because the narrower fp8 operand is exactly what makes the wider 128-wide tile pay.

**Round winner:** the integrated patch, 6.7551× (0.8169 ms).

**Bottleneck shift:** baseline was "fp8 math executed in fp16" (inner loop 2570 instrs, 16
`v_mfma_f32_32x32x8_f16`). After round 1: still compute-bound but on a *new* critical path — inner
loop 884 instrs with only 7.2% MFMA, dominated by two named VALU taxes: **258 instrs (29%) for the
in-loop K 0x80 scrub** and 128 (14%) for the head reduction, plus a `torch.full(-inf)` FillFunctor
dispatch of 8.7/20.4/55.9/52.0 µs *inside* the timed region.

### Round 2 — single direction, aimed at the round-1 shift

| id | specialty | strategy | verified | verdict |
|---|---|---|---|---|
| `r2_d0` | host_runtime | **Step 1:** hoist the K 0x80 fnuz scrub out of the inner loop into a one-shot host-side `torch.clamp_min(k_fp8.view(int8), -127)` (out-of-place, contiguity-guarded, strides re-read), deleting both in-kernel scrubs. **Step 2:** replace `torch.full(-inf)` with `torch.empty` + a kernel-side per-row `-inf` epilogue | **7.8554× cumulative** (1.163× this round vs a 1.2× target) | success |

No integrate step (single direction). **Round winner:** `r2_d0`, 7.8554× (0.7014 ms).

Step 1 delivered essentially all of it and is verifiable in the ISA, not just the clock:
`v_max_i16_sdwa` went to 0 and module-wide SDWA 194 → 2. The host clamp costs 3.7–6.5 µs against
138–1933 µs of kernel (0.4–1.2%) and captured ~98% of the no-scrub theoretical ceiling. Step 2 was
kept on a strict alternating A/B where every B run beat every A run on all four cases (~1.5%) — a
host-side fixed cost that was noise at 1× became real once the kernel was 7× faster. Bit-for-bit
identical output to the round-1 kernel (`err_ratio = 0.0000`), no `.item()` sync introduced, so the
launcher stays HIP-graph-capture safe.

**Bottleneck shift:** inner loop 884 → **605 instrs**, MFMA share 7.2% → 10.6%, arch VGPR 36 → 20
(total 168 → 152), 542–657 TFLOPS = 21–25% of fp8 MFMA peak. The mix is now **symmetric** — head
reduction 128 instrs (21.2%), post-dot per-element scale+ReLU 128 instrs (21.2%), 106
`v_mov_b32_e32` of accumulator shuffling (17.5%), MFMA only 10.6%. Removing 40% of the per-tile
instruction count also exposed a brand-new **shape** effect: M=8192 does 4.7% *more* work than
M=8000 in 13% *less* kernel time (1673.6 vs 1933.1 µs; 24.9 vs 30.2 ns per output Melem), because
N=8000 = 62.5 × BLOCK_KV=128 so every program pays a full masked tail block at 50% useful lanes,
and 8000 CTAs leaves the last of 26.3 CU-waves a third empty.

---

## Final per-test-case table

Run is not workload-aligned (no `WORKLOAD_SPEC`); all weights = 1.0.

| case | shape | baseline ms | optimized ms | speedup |
|---|---|---|---|---|
| 0 | M=2048 N=2048 H=64 D=128 fp8_e4m3fn | 1.0605 | 0.1815 | **5.843×** |
| 1 | M=4096 N=4096 H=64 D=128 fp8_e4m3fn | 3.9661 | 0.5065 | **7.830×** |
| 2 | M=8000 N=8000 H=64 D=128 fp8_e4m3fn | 14.7343 | 1.5883 | **9.277×** |
| 3 | M=8192 N=8192 H=64 D=128 fp8_e4m3fn | 15.1161 | 1.6849 | **8.972×** |

- **Geomean: 5.5324 ms → 0.7014 ms = 7.8554×** (headline)
- Arithmetic mean of per-case speedups: **7.9804×**
- Ratio of sums: 34.877 ms → 3.940 ms = **8.85×**
- Time-weighted speedup: not applicable (unweighted run); equals the geomean, 7.8554×.

---

## Key optimizations applied

1. **Native fnuz fp8 MFMA (the single biggest lever, 2.79× alone).** The baseline did fp8 math *in
   fp16*: the dot compiled to `v_mfma_f32_32x32x8_f16` behind a software upconvert. Bitcasting both
   operands to `tl.float8e4b8` (fnuz, the gfx942-native fp8 format) emits
   `v_mfma_f32_16x16x32_fp8_fp8` and deleted ~97% of the inner loop (2570 → 634 instrs). The ×4
   exponent-bias correction (bias 7 vs 8 on two operands) is folded into the loop-invariant
   `w_block`, so it costs zero in-loop instructions.
2. **The 0x80 NaN scrub — a correctness prerequisite, made nearly free.** In fnuz, byte 0x80 is NaN
   (it is OCP −0.0), and one NaN byte poisons its whole output column: without a scrub `err_ratio`
   is exactly 0.1831 = the fraction of K columns containing ≥1 0x80 byte (375/2048). Viewed as
   int8, 0x80 == INT8_MIN, so a single saturating `maximum(x, −127)` neutralises it and is the
   identity on every other byte (0x80 → 0x81, max rel err 1.9e-3, ~26× under the 5e-2 gate). In
   round 2 this was hoisted entirely out of the loop into one host-side `torch.clamp_min` on the
   int8 view — non-mutating (the harness reuses inputs across 70 timed iterations), stride-guarded,
   with strides re-read from the post-clamp tensor.
3. **Repaired launcher tile heuristic.** The LDS-fit gate had a units bug (`NUM_STAGES` used in the
   bytes-per-element slot) that wrongly rejected BLOCK_KV=128; measured `kernel.metadata.shared` is
   16896 B = one 128×128 fp8 tile + 512 B pipeline slack, far under budget. Fixing it was the
   single biggest host-side win (14.66 → 7.99 ms at M=8000). Plus `matrix_instr_nonkdim`=16 keyed
   off the dot shape rather than `seq_len`, and `waves_per_eu` 2→3 (a knob outside the brief, worth
   ~6% uniformly).
4. **Retired the `torch.full(-inf)` prefill** in favour of `torch.empty` + a kernel-side per-row
   `-inf` epilogue that writes `[0, start_ind)` and `[end_ind, seq_len_kv)` of each program's own
   row. FillFunctor (4 × 34.3 µs) left the timed region entirely; ~1.5%, sync-free, graph-safe.

---

## What didn't work (dead-ends — do not re-open)

| direction | expected | actual | lesson |
|---|---|---|---|
| Transpose the dot to `[BLOCK_KV, NUM_HEADS]` to make heads the contiguous reduction axis | 1.10× | **0.42×** | `tl.trans` on fp8 operands forces a relayout costing far more than the reduction it saves (3.923 vs 1.647 ms). |
| Cheaper cross-lane reduction tree for the 64-head sum | 1.10× | 0.91× | The DPP chain is already near the compiler's floor for this layout (2.168 vs 1.975 ms). |
| Host-side "is the window full?" test to skip the prefill | 1.02× | 0.97× | Needs a `.item()` sync costing 60–67 µs flat, more than the 8–56 µs fill. Superseded by the kernel-side epilogue, which claimed the same ~1.5% with no sync. |
| `input_precision="ieee"` on the dot | 1.05× | 1.006× | With native fp8 operands the accumulate is already fp32; a measured no-op (inside noise). |
| Hoist the `kv_scale` multiply past ReLU and past the head-sum | 1.05× | 0.989× | Mathematically valid (scales ~ U[0.5,1.0] > 0) but slower (0.4374 vs 0.4328 ms): it trades 64 broadcast multiplies for 1 on the reduced vector, but pulls the scale load onto a critical path the scheduler had already hidden. |
| BLOCK_KV / `matrix_instr_nonkdim` / `num_stages` / `num_warps` / `waves_per_eu` sweeps | 1.10× | 1.00× | All strict interior optima with both neighbours measurably worse, re-verified twice at M=4096. |
| Ablation probes as upper bounds (no-head-sum 0.2617 ms, MFMA-only 0.2316 ms) | — | invalid | **Methodological dead-end:** removing the head-sum let the compiler DCE the MFMA from 64 → 16 per loop, so those numbers are not bounds. Always verify the MFMA count is unchanged before quoting an ablation ceiling. Only the post-dot scale+ReLU ablation (0.3710 ms, MFMA verified still 64) is a clean bound = a real 1.17×. |
| Other scrub formulations | — | — | `tl.where` on uint8/int8 (761 instrs), SWAR magnitude-carry (811), OR-fold nonzero detect (1324), `-minimum(-x,127)` (optimized away → incorrect). `tl.inline_asm_elementwise` with `pack=4` fails to compile here. |

## Headroom left on the table (for a future run)

1. **M=8000 tail + grid quantization** — the #1 remaining item, worth ~1.2× on the heaviest case and
   ~1.05× on the geomean, entirely numerics-free. Route: trim the masked tail to the true remainder,
   or a **divisibility-conditional** BLOCK_KV when `seq_len_kv % 128 != 0` (this is *not* the
   exhausted sweep — that ran at M=4096 where the tail never fires), or persistent-CTA / grid-stride.
2. **Pack the post-dot ReLU + `kv_scale` into `v_pk_*` f32x2 form** — clean measured 1.17× ceiling;
   the compiler already packs the `w_block` fold but leaves 64 scalar `v_max_f32` + 64 `v_mul_f32`.
3. The 64-way head reduction (21.2%) ranks last: same size, contaminated evidence, and every cheap
   rewrite has already regressed.

## Profiling caveat

`rocprof-compute` / `omniperf` are **not installed** on this box. `rocprofv3` runs clean but emits no
SoL/cache/wavefront sections, so `valu_pct` / `vmem_pct` / `lds_pct` / `l2_hit_pct` are recorded as
`-1` = **unmeasured, not zero**. Instruction mix was read from the `.amdgcn` of the actually-launched
config; HBM and TFLOPS were hand-computed from bytes/FLOPs over the measured duration.
