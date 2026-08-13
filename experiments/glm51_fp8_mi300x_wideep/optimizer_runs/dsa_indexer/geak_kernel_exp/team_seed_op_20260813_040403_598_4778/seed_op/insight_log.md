# Insight Log — `fp8_mqa_logits` / `_fp8_mqa_logits_kernel` (Triton, gfx942 / MI300X-class)

Baseline geomean 5.5324 ms.
- After round 1: 0.8169 ms = **6.7551x** (integrated `r1_d0 + r1_d1`).
- After round 2: 0.7014 ms = **7.8554x** (winner `r2_d0`, single direction; +16.3% over round 1).

Correctness at the current winner: `ALL PASSED`, `err_ratio=0.0000`, `cos_diff` 3.45e-09 … 3.55e-09
on all four cases — bit-for-bit identical to the round-1 kernel.

---

## Round table

| round | id | specialty | title | expected | verified | verdict |
|---|---|---|---|---|---|---|
| 1 | r1_d0 | compute | Native fnuz fp8 MFMA in the kernel dot | 3.5x | 2.7928x | partial (mechanism confirmed) |
| 1 | r1_d1 | host_runtime | Repair launcher tile heuristic + host-side waste | 1.9x | 2.0512x | confirmed |
| 1 | integrate | — | stacked r1_d0 + r1_d1 (clean `git apply`) | 5.73x | **6.7551x** | super-multiplicative |
| 2 | r2_d0 | host_runtime | Hoist K 0x80 fnuz scrub host-side + kernel `-inf` epilogue | 1.2x (cum ~8.0x) | **7.8554x** cum | confirmed |

Round-2 per-case: 0.1819 / 0.5023 / 1.5927 / 1.6630 ms = 5.83 / 7.90 / 9.25 / 9.09x at
M=2048/4096/8000/8192. Arithmetic 34.877 -> 3.940 ms = 8.85x.

Bottleneck shift r1 -> r2: **compute(VALU tax around a native MFMA) -> compute(two equal VALU
blocks) + a newly exposed SHAPE effect**. Loop 884 -> 605 instrs, MFMA share 7.2% -> 10.6%,
VGPR 168 -> 152 (arch 36 -> 20), 542-657 TFLOPS = 21-25% of fp8 peak.

---

## Insight blackboard (durable)

1. **The baseline was doing fp8 math in fp16.** Bitcasting the operands to `tl.float8e4b8` (fnuz,
   the gfx942-native format) emits `v_mfma_f32_16x16x32_fp8_fp8` instead of
   `v_mfma_f32_32x32x8_f16` + a software upconvert, deleting ~97% of the inner loop. Worth 2.79x
   alone — THE root cause.
2. **The fnuz 0x80 NaN scrub is a correctness PREREQUISITE, not an optimization.** Without it
   `err_ratio` is exactly 0.1831 = the fraction of K columns containing >=1 `0x80` byte (375/2048);
   one NaN byte poisons its entire output column.
3. **The cheapest scrub is an int8 saturate, not compare+select:** viewed as int8, `0x80` is
   `INT8_MIN`, so one `tl.maximum(x,-127)` neutralises it and is the identity on every other byte.
   `0x80 -> 0x81` turns a would-be zero into +-2^-9; max rel err 1.9e-3, ~26x under the 5e-2 gate.
4. **Scrub PLACEMENT beat scrub formulation and is now DONE — topic closed.** `k_fp8` is `[N,D]`,
   so the in-loop scrub re-scrubbed the same bytes once per output row (M times). One
   `torch.clamp_min(k_fp8.view(torch.int8),-127).view(fp8)` in the launcher removed 258 instrs
   (29% of the loop). Verified in the ISA, not just the clock: `v_max_i16_sdwa` 0, module-wide
   sdwa 194 -> 2. Host clamp costs 3.7-6.5 us against 138-1933 us of kernel (0.4-1.2%) and
   captured ~98% of the no-scrub INVALID ceiling (0.4908-0.5130 measured vs 0.4758 ceiling).
5. **Host-side rewrites of a caller's tensor must be non-mutating and stride-guarded.** The
   winning clamp works because `torch.clamp_min` returns a NEW tensor (the harness reuses the same
   inputs across 70 iterations, so in-place would corrupt), it is guarded on
   `k_fp8.stride(-1)!=1` (densify before the int8 dtype-view), and `stride_kv_*` are re-read from
   the post-clamp tensor. This is the reusable pattern for any host-side input canonicalization.
6. **The `torch.full(-inf)` prefill is RETIRED via a kernel-side epilogue — and it did pay.**
   `FillFunctor` is gone from the timed region (was 4 x 34.3 us). The earlier "not worth it"
   estimate was wrong at 7x: alternating A/B full-benchmark geomeans were `torch.full`
   0.7149/0.7095/0.7162 vs `torch.empty`+epilogue 0.7064/0.7056/0.6998/0.7042 — every B run beats
   every A run, on all four cases individually (~1.5%). LESSON: **a fixed-cost host item that was
   "noise" at 1x becomes a real percentage once the kernel is 7x faster — re-evaluate retired
   host-side items after every big kernel win.** The epilogue is only cheap because the benchmark
   windows are near-full, so the out-of-window store traffic is almost nothing. No `.item()` sync
   was introduced, so the launcher stays sync-free and HIP-graph-capture safe.
7. **The LDS gate bug was a UNITS error, not a budget problem.** `kv_bytes = head_size*BLOCK_KV*
   NUM_STAGES` put `NUM_STAGES` in the bytes-per-element slot and wrongly rejected `BLOCK_KV=128`.
   Measured `kernel.metadata.shared` is 16896 B, not 32768 — `num_stages=2` does NOT double LDS
   here because the pipeliner keeps stage 2 in registers for a single-dot loop.
8. **Disjoint lanes COMPOUND.** 2.79 x 2.05 = 5.73 predicted, 6.76 measured: the native fp8 MFMA
   halves the KV operand width, which is exactly what makes the corrected LDS model admit the
   128-wide tile. Keeping one lane strictly inside `@triton.jit` and the other strictly in the
   launcher gave a zero-conflict `git apply` stack. Reuse this lane split.
9. **The kernel is NOT latency-bound — confirmed for the THIRD round running.** Post-round-2 at
   M=4096: `waves_per_eu` 2/3/4 = 0.5275/**0.5048**/0.5131; `num_warps` 4/8 = **0.5054**/0.8580.
   More resident waves are *slower* (the anti-signature of issue-wait C2). VGPR 168 -> 152 does
   not cross a waves/SIMD boundary (floor(512/152) still 3 = 37.5%), so there is no
   register-pressure lever.
10. **The tuning search space is EXHAUSTED at power-of-two shapes — do not re-sweep.** All five
    axes are strict interior optima: `BLOCK_KV` 64/128/256 = 0.873/**0.557**/1.137;
    `matrix_instr_nonkdim` 16/32 = **0.557**/0.706; `num_stages` 1/2/3/4 =
    1.620/**1.583**/2.292/2.313; `num_warps` 1/2/4/8 = 10.75/2.89/**1.60**/3.01; `waves_per_eu`
    2/3/4 = 0.586/**0.557**/0.579. ONE exception, see #12.
11. **No launch/overhead floor exists here, and it is not memory-bound.** Exactly 1 dispatch per
    call; latency scales linearly with work and the SMALL cases are cheaper per element — the
    opposite of an overhead-bound signature. HBM 168-249 GB/s = 3.2-4.7% of 5.3 TB/s, and it went
    DOWN as the kernel got faster. HIP-graph capture / dispatch collapse is not a lever for this op.
12. **NEW #1 (round 2): TAIL + GRID QUANTIZATION at the non-power-of-two shape.** M=8192 does 4.7%
    MORE work than M=8000 in 13% LESS kernel time (1673.6 vs 1933.1 us; 24.9 vs 30.2 ns per output
    Melem; 657 vs 542 TFLOPS). Causes: (a) N=8000 = 62.5 x BLOCK_KV=128, so *every* program falls
    into the masked tail — a full duplicate of the 605-instr body at 50% useful lanes; (b) 8000
    CTAs / 304 CU = 26.3 CU-waves with the last ~32% empty. **This penalty was always there; it was
    ~2% of a slow kernel and is ~21% of a fast one.** Generalizable: a fixed shape inefficiency's
    SHARE grows as everything else shrinks, so re-check shape effects after every big win. A
    *divisibility-conditional* BLOCK_KV (`seq_len_kv % 128 != 0`) has NEVER been measured — the
    exhausted sweep in #10 ran at M=4096 where the tail never fires. Numerics-free; ~1.2x on the
    worst case, ~1.05x on the geomean.
13. **The 605-instr loop is now SYMMETRIC — no single dominant tax.** head reduction (96
    `v_mov_b32_dpp` + 32 `v_pk_add_f32`) = 128 (21.2%); post-dot per-element 64 `v_mul_f32`
    (kv_scale) + 64 `v_max_f32` (relu) = 128 (21.2%); 106 `v_mov_b32_e32` accumulator shuffling
    (17.5%); the MFMA itself only 64 (10.6%); `v_pk_fma/mul_f32` w_block fold 32 (5.3%);
    loads/LDS/barriers ~147 (24%).
14. **Post-dot VALU has a CLEAN measured ceiling of 1.17x** (0.4328 -> 0.3710 ms at M=4096 with the
    MFMA count verified still 64, so not a DCE artifact). The untried route is PACKING: the
    compiler already emits `v_pk_*` for the w_block fold but leaves relu and the scale as 64
    scalar-per-lane ops each; presenting the accumulator as f32x2 pairs would halve them (~64
    instrs, 10%).
15. **`v_mov_b32_e32` (106, the single largest opcode) is a symptom, not a target.** It comes from
    the 20:132 arch:accum VGPR split — the post-dot VALU chain is fed almost entirely out of
    accumulator registers. Arch VGPRs FELL 36 -> 20 this round while accum stayed pinned at 132.
    No source-level knob names it; cut the consumers (#14) and re-read the count.
16. **Ablation probes that let the compiler DCE the MFMA are worthless as bounds.** Removing the
    head-sum measured 0.2617 ms but dropped MFMA 64 -> 16; MFMA-only measured 0.2316 with the same
    contamination. Always verify the MFMA count is unchanged before quoting an ablation ceiling.
17. **Profiler caveat (unchanged):** rocprof-compute/omniperf are not installed; rocprofv3 emits no
    SoL/cache/wavefront sections, so `valu_pct`/`vmem_pct`/`lds_pct`/`l2_hit_pct` are `-1` =
    UNMEASURED. Instruction mix comes from the `.amdgcn` of the actually-launched config; HBM and
    TFLOPS are hand-computed.

## Confirmed dead-ends (do not re-explore)

- **Transposing the dot** to `[BLOCK_KV, NUM_HEADS]` to make heads the contiguous reduction axis:
  **2.4x REGRESSION** (3.923 vs 1.647 ms); `tl.trans` on fp8 operands forces a costly relayout.
- **Cheaper cross-lane reduction tree** (probed via `tl.max`): slower (2.168 vs 1.975 ms) — the DPP
  chain is already near the compiler's floor for this layout.
- **Hoisting `kv_scale` past relu + head-sum** (mathematically VALID — scales are U[0.5,1.0] > 0,
  and `sum_h relu(a*s)*w == s * sum_h relu(a)*w` exactly): measured **SLOWER**, 0.4374 vs 0.4328 ms.
  It trades 64 broadcast `v_mul_f32` for 1 on the reduced vector but pulls the scale load onto a
  critical path the scheduler had already hidden. CLOSED.
- **Host-side "window already full" test** to skip the prefill: needs a `.item()` sync costing
  60-67 us flat, more than the fill. (Superseded anyway by the kernel-side epilogue, #6.)
- `matrix_instr_nonkdim=32`, `waves_per_eu=4`, `num_warps=8`: all re-probed twice, all regress.
- `input_precision="ieee"`: a measured no-op with native fp8 operands (0.5651 vs 0.5683 ms).
- Re-sweeping BLOCK_KV / num_stages / num_warps / waves_per_eu **at power-of-two shapes**: zero
  return (#10). The divisibility-conditional variant (#12) is a DIFFERENT question and is OPEN.
- Dispatch collapse / HIP-graph capture: 1 dispatch per call, no floor to collapse.
- Spill: `Scratch_Size = 0`. Grid underfill: 6.7-27x oversubscribed (quantization in #12 is a
  different failure mode).

## Remaining headroom (ranked for round 3)

1. **M=8000 tail / grid quantization** (#12) — ~1.2x on the heaviest case, ~1.05x geomean,
   numerics-free, and the only BLOCK_KV question still open. Cheapest probes in order: make the
   masked tail cover only the true remainder instead of re-running a full BLOCK_KV block; pick
   BLOCK_KV = 100/125/64 when `seq_len_kv % 128 != 0`; persistent-CTA / grid-stride over rows to
   absorb the ragged last wave.
2. **Pack post-dot relu + scale into `v_pk_*`** (#14) — clean measured ceiling 1.17x, ~10%
   instruction cut. One engineer, kernel-body lane (orthogonal to #1's launcher lane).
3. **Head reduction LAST** (#13) — same 21% size as #2 but its ablation probes are contaminated
   (#16), so round 1's ~1.10x is still the only honest bound, and every cheap rewrite already
   regressed. Only route left is folding `w_block` into the MFMA so the head-sum becomes MFMA
   accumulation — blocked by the `relu` between dot and sum, so it needs real algebra. Higher risk.
4. Register shuffling (#15) — report-only; re-read after #2 lands.
- Structural ceiling: ~25% of fp8 MFMA peak already reached (657 TFLOPS at M=8192); the op is a
  rank-128 dot plus a 64-way cross-lane reduction per 128 outputs.
