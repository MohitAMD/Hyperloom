# Baseline Profiling Summary — `fp8_mqa_logits` (ROUND 0)

## Device (detected, not assumed)
`rocminfo`: **gfx942 / CDNA3, 304 CU**, `sramecc+:xnack-`, 196592 MB HBM, L2 4 MB/XCD.
Marketing name blank; PCI model `0x74a1`, 192 GB → **MI300X-class**, HBM peak ≈5.3 TB/s.
8 such GPUs on the box; we lock GPU 0. Triton 3.6.0, torch 2.11.0+gitd0c8b1f.

## Profiler
`rocprof-compute` / `omniperf` are NOT installed → the script degraded (cleanly, **no
`!!! PROFILER FAILED` block**) to **rocprofv3** `--kernel-trace --stats --output-format csv`.
That gives durations, dispatch counts and per-dispatch occupancy fields
(LDS / Scratch / VGPR / Accum_VGPR / SGPR / Workgroup / Grid) but **no Speed-of-Light, cache,
or wavefront-stall sections**. So VALU%/VMEM%/LDS%/L2-hit/dependency-vs-issue-wait are
**unavailable** and are reported as -1 in `baseline_metrics.json` — the classification below is
derived from durations + occupancy fields + the per-case table + **a disassembly of the compiled
kernel** + **a targeted A/B microbenchmark**, which turned out to be far more decisive than SoL
would have been.

Artifacts: `profile_output/profile_report.txt`,
`profile_output/rocprofv3/useocpm2m-097-038/4523_kernel_{stats,trace}.csv`.

## Dispatch shape — no overhead lever here
`_fp8_mqa_logits_kernel`: **4 calls, 94.07 % of all GPU time**, exactly **1 dispatch per call**.
Every other kernel in the trace (`vectorized_elementwise_*`, `distribution_*`, `FillFunctor`,
`float8_copy`, `elementwise_kernel_manual_unroll`) belongs to `setup_inputs`, **outside** the timed
region. Per-case kernel durations from the trace match the harness medians almost exactly:

| case | M=N | kernel dur (rocprofv3) | harness median | Grid_X | CTAs | achieved TFLOP/s | % fp8 peak | % fp16 peak | HBM |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2048 | 1019.3 us | 1.0605 ms | 524288 | 2048 | 64.8 | 2.5 % | 5.0 % | ~48 GB/s (0.9 %) |
| 1 | 4096 | 4109.8 us | 3.9661 ms | 1048576 | 4096 | 69.3 | 2.7 % | 5.3 % | ~42 GB/s (0.8 %) |
| 2 | 8000 | 14846.2 us | 14.7343 ms | 2048000 | 8000 | 71.2 | 2.7 % | 5.5 % | ~39 GB/s (0.7 %) |
| 3 | 8192 | 15059.4 us | 15.1161 ms | 2097152 | 8192 | 72.7 | 2.8 % | 5.6 % | ~40 GB/s (0.8 %) |

Per-case latency scales as ~M·N (us per M·N element is a near-constant 225–253 ns/Melem across a
16× work range) — so **nothing is overhead-bound and nothing is at a launch floor**. There is no
dispatch-count or HIP-graph lever; all four cases are pure kernel time.

## Cheap sanity checks
- **Fill**: CTAs = Grid_X / Workgroup_X = M (2048…8192) vs **304 CU** → 6.7×–27× oversubscribed.
  The GPU is fully filled; more/finer blocks buys nothing.
- **Spill**: `Scratch_Size = 0` on every `_fp8_mqa_logits_kernel` dispatch. No spills.
- **Registers**: `VGPR_Count=108`, `Accum_VGPR_Count=132` (metadata `.vgpr_count: 233`),
  `SGPR=112`, `LDS_Block_Size=0` in the trace but metadata `shared=8192` (8 KiB, the double-buffered
  KV tile). Occupancy ceiling ≈ `512/(108+132)` → **2 waves/SIMD (25 %)** — low, but see below: it
  is not what is costing the 20×.
- **Roofline**: HBM at **<1 % of 5.3 TB/s** and MFMA at **2.5–2.8 % of fp8 peak** — both far below
  any ceiling, so this is neither memory-bandwidth-bound nor genuinely compute-saturated.
  No efficiency >100 %, peaks are sane.
- Correctness-relevant: the tile selector takes the `(BLOCK_KV=64, num_stages=1)` branch for
  H=64/D=128 (default 128/2 rejected by `_gfx942_default_tile_fits_lds`), confirmed by the compiled
  metadata (`num_stages=1`, `matrix_instr_nonkdim=32`, `waves_per_eu=2`, `num_warps=4`).

## THE finding — the `tl.dot` never reaches an fp8 MFMA
Disassembly of the compiled `gfx942` kernel (dumped from `kernel.asm['amdgcn']`) shows the inner KV
loop is **2570 instructions** and contains only **16 `v_mfma_f32_32x32x8_f16`** — i.e. the
**fp16** MFMA, not `v_mfma_f32_32x32x16_fp8_fp8`. The other ~2550 instructions are a software
fp8→fp16 conversion emitted per element:

```
v_cndmask_b32_e64  577      v_cmp_ne_u16_e64   576
s_nop              500      v_and_b32_e32      160
v_mov_b32_dpp       97      v_lshlrev_b16      128
v_or_b32_e32        64      v_add_u16_e32       64
...  v_mfma_f32_32x32x8_f16  16   (0.6 % of the loop body)
```

Cause: the harness feeds **`torch.float8_e4m3fn` (OCP)**, but gfx942/CDNA3 fp8 MFMA is **FNUZ**
(`amd_instinct.md` §3). Triton cannot map an OCP e4m3 operand to a gfx942 MFMA, so it up-converts
to fp16 in software — with a full compare/select NaN-and-denormal fixup chain per value — and runs
the *fp16* matrix instruction. Every wave spends ~99 % of its inner-loop instruction budget on
format conversion, which is exactly why the kernel sits at 2.5 % of fp8 peak with a filled GPU, no
spills and idle HBM.

**A/B microbenchmark confirming it** (same kernel body, M=N=4096, H=64, D=128, GPU 0):

| operand dtype | BLOCK_KV / num_stages | time |
|---|---|---|
| `float8_e4m3fn` (**the baseline path**) | 64 / 1 | **3.771 ms** |
| `float16` | 64 / 1 | 1.241 ms |
| `bfloat16` | 64 / 1 | 1.248 ms |
| `float16` | 128 / 2 | 1.040 ms |
| `bfloat16` | 128 / 2 | 1.040 ms |
| `float8_e4m3fnuz` (**native gfx942 MFMA**) | 64 / 1 | 1.063 ms |
| `float8_e4m3fnuz` | **128 / 2** | **0.618 ms** |
| `float8_e4m3fnuz` | 256 / 2 | 0.798 ms |

Bit-reinterpreting the *same bytes* as FNUZ and taking the native fp8 MFMA is **6.1× faster** than
the baseline path at the best tile. Merely going to fp16 is already 3.0–3.6× faster.

The OCP→FNUZ reinterpretation is also **exactly representable**: e4m3fn and e4m3fnuz share the
same 4-bit exponent / 3-bit mantissa layout and differ only by an exponent bias of 1 (a factor of
2) plus the `0x80` (negative-zero → NaN) codepoint. Probe on seeded data:
`as_fnuz(bits) * 2.0 == e4m3fn_value` **bit-exactly** (max abs diff 0.0, 3320 `0x80` bytes in Q and
47 in K all mapped to +0). So a byte-level `view` + `0x80→0x00` scrub + a **single scalar ×4** folded
into the existing `kv_scales`/`w_block` multiply (2 from Q, 2 from K) preserves exactness — this is a
legitimate format change, not a tolerance relaxation, and it does not read the *values* (only the
`0x80` codepoint), so it does not violate the "don't exploit the fixed seed" rule.

## Classification: **compute-bound** (instruction-issue-bound on VALU, not on MFMA)
Decision-tree walk: HBM <1 % and VMEM clearly low → not memory-bound. GPU is 6.7–27× oversubscribed
with zero spills → not a fill/occupancy problem, so not the classic latency-bound C2. Latency scales
linearly with work across a 16× range → not overhead-bound. What remains is that the SIMDs are
saturated executing VALU work — just the *wrong* VALU work: ~2550 conversion instructions per 16
MFMAs. That is **compute-bound on VALU throughput**, and it is directly removable.

Note on the 25 % occupancy: with a 2570-instruction, dependency-light loop body, 2 waves/SIMD is
mostly adequate to hide issue latency; occupancy is a *second-order* lever here and should only be
revisited after the conversion chain is gone (per `profiling_guide.md`, the C1/C2 split must be
re-read after any tile / `num_stages` / `num_warps` change — and the tile *will* change).

## Ranked opportunities
1. **Eliminate the software fp8→fp16 conversion; feed the gfx942 FNUZ fp8 MFMA.** In the launcher,
   `view(torch.uint8)` → scrub `0x80`→`0x00` → `view(torch.float8_e4m3fnuz)` for both Q and K, and
   fold the resulting ×4 into `kv_scales` (or `w_block`). Measured **6.1×** on the M=4096 probe
   (3.771 → 0.618 ms). This is the whole ballgame; everything else is noise beside it.
   Caveat the implementer must handle: the scrub is two extra elementwise passes over Q (M·H·D) and
   K (N·D) *inside the timed region* — small vs an M·N kernel, but fuse it into the Triton kernel's
   load path (`tl.where(bits == 0x80, 0, bits)` after a `uint8` load + `tl.bitcast`) rather than
   paying host-side torch kernels, which would re-introduce dispatches.
2. **Re-open the tile choice once the fp8 path is native.** `_gfx942_default_tile_fits_lds`'s LDS
   model was written for the fp16-expansion path; with native fp8 operands the KV tile halves.
   `BLOCK_KV=128, num_stages=2` measured **0.618 ms vs 1.063 ms** at BLOCK_KV=64/1 (1.7× on top of
   #1); 256/2 regressed to 0.798 ms and 256/2 in bf16 hit the 64 KiB LDS wall — so 128/2 is the
   sweet spot to target, and the gate function needs updating rather than bypassing.
3. **Drop `input_precision="ieee"` on the `tl.dot`.** With native fp8 operands the accumulate is
   already fp32; `ieee` may be forcing a slower expansion path. Cheap to A/B, must be re-checked
   against the `err_ratio <= 0.02` gate.
4. **Avoid the full `torch.full(-inf)` output prefill.** `FillFunctor<float>` costs 178 us on the
   M=8000 case (4 M·N elements) *inside the timed region*. `cu_starts=0 / cu_ends=N` means the
   kernel writes every element anyway; switch to `torch.empty` when the window is provably full, or
   have the masked tail store `-inf` explicitly. Worth ~1 % now, ~10 % after #1 lands.
5. **Re-read the occupancy/stall split after #1+#2.** 108+132 VGPRs → 2 waves/SIMD; once the
   conversion chain is gone the loop body shrinks ~100×, and the kernel will likely flip to
   issue-wait-dominant (latency C2), at which point cutting `Accum_VGPR` / raising `waves_per_eu`
   becomes a real lever. Do not pre-tune registers now — the current diagnosis will not survive #1.

## Caveats
- No SoL / cache / wavefront-stall counters on this box (rocprofv3 only) — `valu_pct`, `vmem_pct`,
  `lds_pct`, `l2_hit_pct` are genuinely unmeasured, not zero. HBM GB/s is computed by hand from
  bytes/time, not read from a counter.
- The A/B table is a standalone microbenchmark of an equivalent kernel body, not a full-benchmark
  run. Per COMMANDMENT rule 6, the 6.1× is a **direction signal**, not a claimed speedup; only
  `--full-benchmark` decides.
