# Round 1 re-profile — `_fp8_mqa_logits_kernel` (gfx942 / MI300X-class)

Profiler: **rocprofv3** (`--kernel-trace --stats --output-format csv`), no `!!! PROFILER FAILED`
block. rocprof-compute / omniperf are still absent on this box, so there are **no SoL / cache /
wavefront sections**: `valu_pct`, `vmem_pct`, `lds_pct`, `l2_hit_pct` are reported as `-1`
(= *unmeasured*, not zero). Everything below is derived from durations, the per-dispatch occupancy
fields, the disassembled inner loop, and hand-computed roofline / A-B probes.

Device (`rocminfo`): `gfx942` / CDNA3, **304 CU**, 192 GB HBM, ~5.3 TB/s nameplate peak,
`rocm-smi` model `0x74a1`, SKU `M3000108` — MI300X-class.

## What was measured

The canonical `WORKSPACE` still contains the **pristine baseline** (md5 identical to
`EVAL_DIR/baseline/`); the round-1 winner is the integrated patch at
`round_1/integrate/ws_1786598413_10555/triton_fp8_mqa_logits.py`. I copied that file plus the
untouched `test_harness.py` / `config.yaml` into `round_1/profile_ws/` and profiled **that**.
The integrated patch also applies cleanly to the canonical workspace (`git apply --check` OK).

Correctness re-verified before timing: `err_ratio=0.0000`, `cos_diff≈3.5e-09` on all 4 cases,
`ALL CORRECTNESS CHECKS PASSED`.

FULL_BENCHMARK (authoritative, this re-profile run):

| case | M=N | baseline ms | round-1 ms | speedup |
|---|---|---|---|---|
| 0 | 2048 | 1.0605 | 0.1965 | 5.40x |
| 1 | 4096 | 3.9661 | 0.5659 | 7.01x |
| 2 | 8000 | 14.7343 | 1.9571 | 7.53x |
| 3 | 8192 | 15.1161 | 2.0464 | 7.39x |
| **geomean** | | **5.5324** | **0.8169** | **6.77x** |

Arithmetic sum 34.877 -> 4.766 ms. This is materially better than the two engineers' individual
numbers (2.79x for r1_d0 kernel-body, 2.05x for r1_d1 host/tile) — the two fixes **compound**:
the native fp8 MFMA makes the 128-wide KV tile affordable, and the corrected LDS model is what
admits it. 2.79 x 2.05 = 5.7 predicted, 6.77 measured, i.e. slightly super-multiplicative.

## BEFORE -> AFTER

```
BEFORE: compute  — VALU-throughput-bound on a SOFTWARE fp8(OCP)->fp16 upconvert.
        Inner KV loop = 2570 instrs containing only 16 v_mfma_f32_32x32x8_f16 (fp16!).
        Config: BLOCK_KV=64, num_stages=1, nonkdim=32, waves_per_eu=2, shared=8192 B.
        VGPR 108 + AccVGPR 132 = 240 -> 2 waves/SIMD (25% occ). 72.7 TFLOPS = 2.8% of fp8 peak.
        HBM 40 GB/s = 0.8% of 5.3 TB/s. geomean 5.5324 ms.

AFTER:  compute  — still compute-bound, but now MFMA-adjacent and on a DIFFERENT critical path.
        Inner KV loop = 884 instrs containing 64 v_mfma_f32_16x16x32_fp8_fp8 (NATIVE fp8).
        Config: BLOCK_KV=128, num_stages=2, nonkdim=16, waves_per_eu=3, shared=16896 B.
        VGPR 36 + AccVGPR 132 = 168 -> 3 waves/SIMD (37.5% occ). Scratch 0. LDS_Block_Size 0
        (rocprofv3 field is unpopulated for Triton here; kernel.metadata.shared = 16896 B).
        429-565 TFLOPS = 16.5-21.7% of fp8 peak. HBM 148-215 GB/s = 2.8-4.1% of 5.3 TB/s.
        geomean 0.8169 ms.

SHIFT:  compute(software-dequant) -> compute(VALU-tax-around-a-native-MFMA)
        because the fp8->fp16 conversion chain that was 97% of the loop is gone. The loop is
        now 884 instrs for 64 MFMAs, and the residual is NOT the MFMA: it is 128 v_max_i16_sdwa
        (the in-loop K 0x80 scrub), 96 v_or_b32_sdwa + 34 v_lshrrev (byte packing for the
        SDWA scrub), and 96 v_mov_b32_dpp + 32 v_pk_add_f32 (the cross-lane tl.sum over
        NUM_HEADS=64). MFMA is 64/884 = 7.2% of the loop's instruction slots.

NEXT:   Target the two named VALU taxes, in this order:
        (1) move the K 0x80 scrub OUT of the loop (host-side, once over K) — MEASURED 1.19x;
        (2) restructure the NUM_HEADS=64 cross-lane reduction — bounded at ~1.10x.
        Do NOT re-tune tiles/registers: every axis is already at its measured optimum.
```

## Evidence for the new classification

**Still compute, not memory / not latency / not overhead / not LDS.**
- HBM is 148-215 GB/s, **2.8-4.1% of 5.3 TB/s**. Nowhere near memory-bound; ruled out by a
  4-order-of-magnitude margin (the guide's "small AI alone is not memory-bound" note applies).
- Fill: `CTAs = Grid/Workgroup = M` = 2048 / 4096 / 8000 / 8192 vs **304 CU** — 6.7x to 27x
  oversubscribed. The GPU is fully fed; grid sizing is not a lever.
- Overhead: **exactly 1 dispatch of `_fp8_mqa_logits_kernel` per call** (4 dispatches for 4
  cases, 76 total dispatches in the profile run are all setup work outside the timed region).
  Latency still scales linearly with work: 78 / 138 / 244 / 250 ns per Melem — no launch floor,
  no dispatch-collapse / HIP-graph lever. (Note the small cases are now *cheaper* per element,
  the opposite of an overhead-bound signature.)
- Spill: `Scratch_Size = 0`.
- Occupancy: 36 + 132 = 168 VGPRs -> `min(8, 512/168)` = **3 waves/SIMD = 37.5%**, up from 25%.
  This is `waves_per_eu=3` doing exactly what it was set to do. `waves_per_eu` 2/3/4 measured
  0.5855 / 0.5567 / 0.5786 ms at M=4096 — 3 is the optimum, both neighbours regress.
- Achieved 429-565 TFLOPS vs the 2.6 PFLOPS fp8 MFMA peak = 16.5-21.7%. A 6x headroom on paper,
  but the ISA says where it went: only 7.2% of issued instructions are MFMA.

**The C1/C2 latency split was re-read after the tile change, as required.** It does not apply:
this configuration is not latency-bound. The prior round's warning ("the kernel will likely flip
to issue-wait-dominant C2 once the loop shrinks") did **not** materialise — the loop shrank 2570
-> 884 but the VALU tax that replaced the conversion chain is dense, independent, and issue-
limited by *instruction count*, not by wave residency. Direct evidence: raising occupancy further
(`waves_per_eu=4`) makes it **slower** (0.5786 vs 0.5567), which is the anti-signature of C2, and
`num_warps` 1/2/4/8 = 10.75 / 2.89 / 1.60 / 3.01 ms at M=8000 with 4 a sharp interior optimum.
More resident waves do not help; fewer instructions do.

## Inner-loop disassembly (the load-bearing measurement)

Extracted from `kernel.metadata` / the `.amdgcn` for the ACTUAL launched config
(BLOCK_KV=128, num_stages=2, nonkdim=16, waves_per_eu=3, shared=16896 — the round-0 dump was of
the stale 64/1/32/2 config). Backward-branch body, 884 instructions:

| count | instruction | what it is |
|---|---|---|
| 128 | `v_max_i16_sdwa` | **the in-loop K 0x80 scrub** (`tl.maximum(kv_bits,-127)`) |
| 106 | `v_mov_b32_e32` | shuffling |
| 96 | `v_or_b32_sdwa` | byte re-pack around the SDWA scrub |
| 96 | `v_mov_b32_dpp` | **cross-lane `tl.sum(axis=0)` over NUM_HEADS=64** |
| 64 | `v_mfma_f32_16x16x32_fp8_fp8` | **the actual math — native fp8, 7.2% of the loop** |
| 64 | `v_mul_f32_e32` | `* kv_scales`, `* w_block` |
| 64 | `v_max_f32_e32` | the ReLU |
| 34 | `v_lshrrev_b32_e32` | byte extraction for the scrub |
| 32 | `v_pk_add_f32` | reduction adds |
| 24 | `v_pk_fma_f32` | epilogue |
| 9 | `global_load_dword` | KV + scales loads |
| 8 | `s_barrier`, 4 `ds_write_b128`, 4+4 `ds_*st64_*` | the pipelined LDS KV tile |

So the scrub costs **~258 instructions (29% of the loop)** — 128 `v_max_i16_sdwa` plus the 96
`v_or_b32_sdwa` + 34 `v_lshrrev` that the SDWA byte-lane handling drags in. The head reduction
costs **~128 (14%)**. Together they are 44% of the loop; the MFMAs are 7%.

## Quantified opportunities (all A/B measured this round)

1. **Host-side K scrub — MEASURED 1.19x, CORRECTNESS-VERIFIED.** The single biggest remaining
   item, and it directly deletes the 258-instruction block above. `k_fp8` is `[N, D]` — only
   `N*D` bytes, whereas the in-loop scrub re-scrubs the same bytes once **per output row**,
   i.e. `M` times over. One `torch.clamp_min(k_fp8.view(torch.int8), -127).view(fp8)` in the
   launcher, then drop `tl.maximum(kv_bits,-127)` from the loop:
   - M=4096: 0.5673 -> **0.4848 ms**; M=8000: 1.9752 -> **1.6510 ms** (1.17-1.20x)
   - Full harness `--correctness`: **ALL PASSED**, `err_ratio=0.0000`, `cos_diff=3.5e-09` on all
     four cases (variant kept at `/tmp/variants/hostscrub.py`; re-create, do not rely on /tmp).
   - The extra host pass **is** inside the timed region and is already paid for in those numbers.
   - Upper bound check: a no-scrub-at-all kernel (INVALID, correctness-failing, probe only) runs
     0.4758 ms at M=4096 vs 0.4848 for the host-scrub version — so the host scrub captures
     essentially **all** of the available scrub headroom (98%). There is nothing left here after
     this lands; do not spend another round on scrub formulations.
   - Note this REVERSES r1_d0's conclusion only in placement, not in substance: r1_d0 was right
     that the scrub is mandatory for correctness and right that `maximum(x,-127)` is the cheapest
     *in-loop* form. It is simply cheaper still to not do it in the loop at all.

2. **Cross-lane head reduction (`tl.sum` over NUM_HEADS=64) — bounded ~1.10x, needs a new idea.**
   96 `v_mov_b32_dpp` + 32 `v_pk_add_f32` = 14% of the loop. I bounded it by replacing `tl.sum`
   with `tl.max` (INVALID, probe only — a same-shape cross-lane op with a cheaper tree): it came
   out *slower* (2.1679 vs 1.9752 at M=8000), so the DPP chain is near the compiler's floor for
   this layout. I also tried the obvious layout fix — transpose the dot to
   `[BLOCK_KV, NUM_HEADS]` so the head axis becomes the contiguous reduction axis: **3.9230 ms
   vs 1.6465, a 2.4x REGRESSION** (numerically fine, maxdiff 6.1e-05 — it is purely a perf loss;
   `tl.trans` on the fp8 operands forces a relayout that costs far more than the reduction).
   Both cheap reformulations are therefore **dead ends, already spent**. What remains untried is
   folding `w_block` into the MFMA itself (scale Q by per-head weight before the dot, so the sum
   over heads becomes part of an accumulation the MFMA already does) — but `relu` sits between
   the dot and the head-sum, so this needs real algebra, not a rewrite. Treat as speculative.

3. **`torch.full(-inf)` prefill — ~1.5%, worth taking only as a freebie.** The `FillFunctor<float>`
   dispatch is inside the timed region: 8.7 / 20.4 / 55.9 / 52.0 us for cases 0-3, against kernel
   times of 160 / 595 / 2198 / 1946 us. That is 5.2% at M=2048 and 2.5% at M=8000 of *kernel*
   time, ~1.5% of end-to-end. r1_d1 already proved the host-side fullness test is a net LOSS
   (a `.item()` sync costs 60-67 us, more than the fill). Direct measurement: `torch.full` 27.2 us
   vs `torch.empty` 9.1 us at M=4096 — so the ceiling here is ~18 us, ~3% at M=4096. The only
   sync-free route is a kernel-side `-inf` epilogue on the out-of-window lanes. Low priority.

4. **`input_precision="ieee"` — RULED OUT, no effect.** The round-0 profile flagged this as a
   cheap A/B. Measured: 0.5651 (dropped) vs 0.5683 ms (kept) at M=4096 — 0.6%, inside noise.
   With native fp8 operands the accumulate is already fp32 and the flag is a no-op. Close it.

5. **Tile / instruction / occupancy axes — ALL AT THEIR OPTIMUM, do not re-tune.** Full sweeps
   at M=4096 (kernel-only, no fill/alloc):
   - `BLOCK_KV`: 64 -> 0.8732, **128 -> 0.5570**, 256 -> 1.1370 ms. Sharp interior optimum;
     r1_d1's corrected LDS gate picks 128 correctly (16896 B measured `shared`, budget 28.8 KiB).
   - `matrix_instr_nonkdim`: **16 -> 0.5570**, 32 -> 0.7062 ms. r1_d1's dot-shape keying is right.
   - `num_stages` (at M=8000): 1 -> 1.6195, **2 -> 1.5834**, 3 -> 2.2918, 4 -> 2.3132 ms.
   - `num_warps` (at M=8000): 1 -> 10.7490, 2 -> 2.8917, **4 -> 1.6026**, 8 -> 3.0146 ms.
   - `waves_per_eu`: 2 -> 0.5855, **3 -> 0.5567**, 4 -> 0.5786 ms.
   Every axis is a strict interior optimum with both neighbours worse. This search space is
   **exhausted** — a round spent re-sweeping it will return zero.

## Realistic remaining headroom

Opportunity #1 is measured and safe: 0.8169 -> ~**0.69 ms** geomean, i.e. 6.77x -> **~8.0x**.
Beyond that, #2 and #3 together are worth at most another ~10-12% and both need new ideas rather
than parameter search. The kernel would then sit around 20-25% of fp8 MFMA peak with an inner
loop that is ~10% MFMA — the structural ceiling for an op whose real work is a rank-128 dot
followed by a 64-way cross-lane reduction per 128 outputs.

## Artifacts

- Profile report: `EVAL_DIR/profile_output_r1/profile_report.txt` (profiler: rocprofv3)
- Kernel stats CSV: `EVAL_DIR/profile_output_r1/rocprofv3/useocpm2m-097-038/11555_kernel_stats.csv`
- Kernel trace CSV: `.../11555_kernel_trace.csv` (per-dispatch VGPR / Scratch / Grid / Workgroup)
- Profiled source: `EVAL_DIR/round_1/profile_ws/triton_fp8_mqa_logits.py`
  (= `round_1/integrate/ws_1786598413_10555/triton_fp8_mqa_logits.py`)
- Metrics JSON: `EVAL_DIR/round_1_metrics.json`
