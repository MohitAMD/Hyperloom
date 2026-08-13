# Round 2 — Bottleneck Shift Analysis (`fp8_mqa_logits`, gfx942)

Device: MI300X-class / gfx942 / CDNA3, 304 CU, 192 GB HBM, ~5.3 TB/s peak (rocm-smi GFX Version gfx942, Card Model 0x74a1, SKU M3000108)
Profiler: **rocprofv3** (rocprof-compute / omniperf are NOT installed on this box; rocprofv3
ran clean, no `!!! PROFILER FAILED` block, but emits **no SoL / cache / wavefront sections**, so
`valu_pct` / `vmem_pct` / `lds_pct` / `l2_hit_pct` are recorded as `-1` = UNMEASURED, not zero.
HBM and TFLOPS are computed by hand from bytes/FLOPs over the measured kernel duration; the
instruction mix comes from the `.amdgcn` of the actually-launched config.)

Artifacts: `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/profile_output_r2/` · metrics `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/round_2_metrics.json`
Profiled + benchmarked at `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/round_2/profile_ws/` (winner kernel + **untouched** `test_harness.py`
/ `config.yaml`, md5-verified against the canonical workspace copies).
Winner: `/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp/team_seed_op_20260813_040403_598_4778/seed_op/round_2/engineer_0/best_patch.diff` — `git apply --check` against the canonical
WORKSPACE: **clean**.

```
BEFORE (round 1): compute-bound — 884-instr loop, 64 MFMA (7.2%), 258-instr in-loop K scrub (29%),
                  128-instr head reduction (14%), torch.full(-inf) FillFunctor inside the timed
                  region (8.7/20.4/55.9/52.0 us), 168 VGPR, 0.8169 ms geomean = 6.77x
AFTER  (round 2): compute-bound — 605-instr loop, 64 MFMA (10.6%), scrub GONE, prefill GONE,
                  head reduction 128 (21.2%) TIED WITH post-dot scale+relu 128 (21.2%),
                  106 v_mov_b32 (17.5%), 152 VGPR, 0.7014 ms geomean = 7.89x
SHIFT: compute(VALU-tax-around-native-MFMA) -> compute(two equal VALU blocks) + a newly exposed
       SHAPE effect: M=8000 runs 21% below M=8192 per output element
NEXT:  (1) M=8000 tail/grid quantization  (2) pack relu+scale into v_pk_*  (3) head reduction last
```

## 1. What changed

Both round-1 recommendations landed and are verifiable in the ISA / trace, not just in the timings.

| | round 1 | round 2 |
|---|---|---|
| inner-loop instructions | 884 | **605** |
| `v_mfma_f32_16x16x32_fp8_fp8` per loop | 64 (7.2%) | 64 (**10.6%**) |
| K 0x80 scrub in loop | 258 instrs (29%) | **0** (`v_max_i16_sdwa` 0; module sdwa 194 -> 2) |
| `FillFunctor` in timed region | 4 x 34.3 us avg | **0** (only setup_inputs fills remain, outside) |
| new host `clamp` in timed region | — | 4.61 / 3.73 / 6.09 / 6.50 us (0.4–1.2%) |
| Arch VGPR / Accum / total | 36 / 132 / 168 | **20** / 132 / **152** |
| LDS (`metadata.shared`) | 16896 B | 16896 B |
| Scratch | 0 | 0 |
| occupancy (waves/SIMD) | 3 (37.5%) | 3 (37.5%) — unchanged, 512/152 still floors to 3 |
| geomean | 0.8169 ms (6.77x) | **0.7014 ms (7.89x)** |
| TFLOPS (max case) | 564.9 (21.7% fp8 peak) | **657.0 (25.3%)** |

Authoritative FULL_BENCHMARK (all 4 cases, harness defaults 20/50, median):

| case | M=N | ms | speedup | kernel us | ns / output Melem | TFLOPS | HBM |
|---|---|---|---|---|---|---|---|
| 0 | 2048 | 0.1819 | 5.83x | 138.0 | 32.9 | 498 | 249 GB/s (4.7%) |
| 1 | 4096 | 0.5023 | 7.90x | 502.1 | 29.9 | 548 | 204 GB/s (3.8%) |
| 2 | 8000 | 1.5927 | 9.25x | 1933.1 | **30.2** | 542 | 168 GB/s (3.2%) |
| 3 | 8192 | 1.6630 | 9.09x | 1673.6 | **24.9** | 657 | 202 GB/s (3.8%) |
| **geomean** | | **0.7014** | **7.888x** | | | | |

Correctness re-verified independently of the engineer: `ALL CORRECTNESS CHECKS PASSED`,
`err_ratio=0.0000`, `cos_diff` 3.45e-09 … 3.55e-09 on all four cases. Arithmetic-mean sum
34.877 -> **3.940 ms**. This reproduces the engineer's claimed 7.8765x to within 0.1%.

## 2. The new bottleneck

Still **compute-bound**, third distinct critical path in three rounds. The instruction mix is now
symmetric — no single dominant tax, two equal ones:

| block in the 605-instr loop | instrs | share |
|---|---|---|
| cross-lane head reduction (`tl.sum` over H=64): 96 `v_mov_b32_dpp` + 32 `v_pk_add_f32` | 128 | 21.2% |
| post-dot per-element math: 64 `v_mul_f32` (kv_scale) + 64 `v_max_f32` (relu) | 128 | 21.2% |
| accumulator/operand shuffling: `v_mov_b32_e32` | 106 | 17.5% |
| **the actual MFMA** | 64 | 10.6% |
| `v_pk_fma_f32` 24 + `v_pk_mul_f32` 8 (w_block fold — already packed) | 32 | 5.3% |
| loads / LDS / barriers / control | ~147 | 24% |

`v_mov_b32_e32` being the single largest opcode is a direct consequence of the 20:132 arch:accum
VGPR split — the post-dot VALU chain is fed almost entirely out of accumulator registers.

## 3. The effect that only became visible now

**M=8192 does 4.7% MORE work than M=8000 in 13% LESS kernel time** (1673.6 vs 1933.1 us; 24.9 vs
30.2 ns per output Melem; 657 vs 542 TFLOPS). Two compounding causes:

1. **Tail**: N=8000 = 62.5 x BLOCK_KV=128, so *every one* of the 8000 programs drops out of the
   unmasked loop and executes the masked tail — a full duplicate of the 605-instr body — at 50%
   useful lanes.
2. **Grid quantization**: 8000 CTAs / 304 CU = 26.3 CU-waves, last one ~32% empty (8192 -> 26.9).

This penalty was always present; it was ~2% of a slow kernel and is ~21% of a fast one. It is the
clearest remaining win and it touches no numerics.

## 4. Latency sub-case — re-read after the change (required by the role)

Not latency-bound, for the third round running. At M=4096: `waves_per_eu` 2/3/4 =
0.5275 / **0.5048** / 0.5131 ms; `num_warps` 4/8 = **0.5054** / 0.8580 ms. More resident waves are
*slower* — the anti-signature of issue-wait (C2). The VGPR drop 168 -> 152 does not cross a
waves/SIMD boundary (floor(512/152) = 3, same 37.5%), so the register saving bought nothing and
there is no register-pressure lever here.

## 5. Probes run this round

| probe (M=4096) | ms | note |
|---|---|---|
| current winner | 0.4328 / 0.4236 / 0.5048 (harness path) | |
| drop post-dot scale+relu (INVALID) | 0.3710 | MFMA count verified still 64 -> **clean 1.17x ceiling** |
| hoist kv_scale past relu+head-sum (**VALID**) | 0.4374 | numerically exact (scales > 0) but a **regression** — closed |
| drop head-sum (INVALID) | 0.2617 | **contaminated** — compiler DCE'd MFMA 64 -> 16, not a valid bound |
| MFMA-only (INVALID) | 0.2316 | same contamination |

## 6. Ranked next targets

1. M=8000 TAIL/QUANTIZATION IS THE NEW #1 AND IT IS BRAND NEW THIS ROUND. M=8192 is 4.7% MORE work than M=8000 yet its kernel is 13% FASTER (1673.6 us vs 1933.1 us). Per output element: 24.9 ns/Melem at 8192 vs 30.2 at 8000, i.e. the non-power-of-two case runs 21% below its own sibling's efficiency, and 657 vs 542 TFLOPS. Two compounding causes, both addressable in the launcher/kernel and neither previously visible (at baseline every case was buried under the software upconvert): (a) N=8000 = 62.5 x BLOCK_KV=128, so every one of the 8000 programs falls out of the fast unmasked loop into the masked tail block for its last half-tile - the tail is a full duplicate of the loop body (605 instrs) executed for 50% useful lanes; (b) grid quantization, 8000 CTAs / 304 CU = 26.3 waves, so the last wave is 32% empty, while 8192/304 = 26.9. Case 2 is the heaviest case and holds the WORST per-case speedup among the large shapes' efficiency, so this is worth ~1.2x on case 2 alone and ~1.05x on the geomean. Concrete probes, in order of cheapness: pad the KV extent handling so the masked tail only covers the true remainder rather than re-running a full BLOCK_KV block; or select BLOCK_KV=100/125/64 when seq_len_kv % 128 != 0 (NOTE: this is NOT the exhausted BLOCK_KV sweep - that sweep was run at M=4096, a power of two where the tail never fires; a divisibility-conditional tile choice has never been measured); or persistent-CTA / grid-stride over rows so the ragged last wave is absorbed. Nothing here touches numerics.

2. POST-DOT PER-ELEMENT VALU: 64 v_mul_f32 (kv_scale) + 64 v_max_f32 (relu) = 128 of 605 loop instrs (21.2%), MEASURED CEILING 1.17x (0.4328 -> 0.3710 ms at M=4096 with the MFMA count verified unchanged at 64, so this bound is clean and not a DCE artifact). This is now tied with the head reduction as the largest non-MFMA block and it is the one with a trustworthy number. One reformulation is already SPENT: hoisting the kv_scale multiply past relu and past the head-sum (VALID - kv_scales are drawn from U[0.5,1.0] so strictly positive, and relu(a*s)*w summed over h == s * sum_h(relu(a)*w) exactly) measured 0.4374 vs 0.4328, i.e. a small REGRESSION, because it trades 64 broadcast v_mul_f32 for 1 v_mul on the reduced vector but forces the scale load off the critical path the scheduler had already hidden. Do not re-run that. The untried route is packing: the compiler already emits v_pk_fma_f32/v_pk_mul_f32 for the w_block fold but leaves relu and the scale as 64 SCALAR-per-lane v_max_f32/v_mul_f32 - if the accumulator can be presented as f32x2 pairs the same work becomes 32+32 packed ops, a ~64-instr (10%) cut. Worth one engineer.

3. CROSS-LANE HEAD REDUCTION (tl.sum over NUM_HEADS=64): 96 v_mov_b32_dpp + 32 v_pk_add_f32 = 128 instrs, 21.2% of the loop - unchanged in absolute terms since round 1 but its SHARE grew from 14% to 21% because everything around it shrank. WARNING ON THE NUMBERS: my two ablation probes for this (no-head-sum 0.2617, mfma-only 0.2316) are CONTAMINATED - removing the sum let the compiler DCE the MFMA from 64 down to 16 per loop, so those are not valid upper bounds, only proof that the DPP chain and the dot are entangled. The round-1 estimate of ~1.10x still stands as the honest figure. Both cheap rewrites remain CLOSED with evidence (transposing the dot to [BLOCK_KV,NUM_HEADS] regressed 2.4x; tl.max instead of tl.sum was slower). Only route left is folding w_block into the MFMA so the head-sum becomes MFMA accumulation - relu sits between the dot and the sum so it needs real algebra. Rank it BELOW the two items above: same size, worse evidence, higher risk.

4. REGISTER SHUFFLING: 106 v_mov_b32_e32 in the 605-instr loop (17.5%) - the single largest opcode by count, larger than the MFMA block itself. These are the compiler moving accumulator/operand registers between the MFMA's AGPR-adjacent allocation (Accum_VGPR=132 vs Arch_VGPR only 20 - a 6.6:1 split) and the VALU that consumes them. Arch VGPRs actually FELL 36 -> 20 this round while accum stayed pinned at 132, so the post-dot VALU chain is being fed almost entirely out of accumulator registers, which is exactly what generates the moves. Speculative and compiler-mediated (no source-level knob names it), but if either of the two VALU items above lands, re-read this count: cutting the consumers should cut the moves with them. Report-only for now; do not assign a whole round.

5. TILE / OCCUPANCY AXES: STILL EXHAUSTED AT THE POWER-OF-TWO SHAPES - re-verified this round as the role requires after the source changed. waves_per_eu 2/3/4 = 0.5275/0.5048/0.5131 ms and num_warps 4/8 = 0.5054/0.8580 ms at M=4096: 3 and 4 remain strict interior optima, and more resident waves are still slower, so the kernel is confirmed NOT latency-bound C2 for the third round running. VGPR total fell 168 -> 152 (arch 36 -> 20) but floor(512/152) is still 3 waves/SIMD = 37.5% occupancy, so the register saving bought nothing and there is no register-pressure lever here. The ONE exception to 'do not re-sweep' is the divisibility-conditional BLOCK_KV named in opportunity #1 - that is a different question from the M=4096 sweep and has never been measured.

6. CLOSED / RULED OUT WITH EVIDENCE, do not re-open: (a) the K 0x80 scrub - DONE and verifiable in the ISA, v_max_i16_sdwa is 0 and module-wide sdwa fell 194 -> 2, with only 2 sdwa left in the main loop; the host clamp costs 3.7-6.5 us inside the timed region against 138-1933 us of kernel, i.e. 0.4-1.2%, and captured 98% of the predicted headroom. (b) the torch.full(-inf) prefill - RETIRED; FillFunctor no longer appears in the timed region at all (was 4 x 34.3 us avg, now only 8 setup_inputs fills of 1.6-5.7 us outside it), replaced by the per-row OOW_FILL kernel epilogue with no host sync, so it stays HIP-graph-capture safe. (c) memory bandwidth - HBM 168-249 GB/s = 3.2-4.7% of 5.3 TB/s, and it went DOWN not up as the kernel sped up, so bandwidth is not closing in. (d) grid fill - CTAs = M = 2048..8192 vs 304 CU, 6.7-27x oversubscribed (but see the QUANTIZATION half of opportunity #1, which is a different failure mode from underfill). (e) dispatch collapse / HIP-graph - still exactly 1 dispatch of the target kernel per call. (f) spill - Scratch_Size = 0. (g) input_precision='ieee' - a measured no-op with native fp8 operands. (h) host-side fullness test with .item() - proven a net loss (60-67 us sync).

