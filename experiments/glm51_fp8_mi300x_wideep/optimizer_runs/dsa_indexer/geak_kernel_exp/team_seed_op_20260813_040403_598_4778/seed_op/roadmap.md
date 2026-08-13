# Roadmap — `_fp8_mqa_logits_kernel` (DSA sparse-MLA indexer prefill logits), Triton / gfx942

## Kernel summary
One workgroup per query row computes `logits[i,:] = sum_h relu((q[i,h,:].k[j,:]) * kv_scale[j]) * w[i,h]`
over the KV axis. H=64, D=128, fp8_e4m3fn operands, fp32 accumulate/output, M=N up to 8192.
Cost is `2*M*N*H*D` flops and scales quadratically with context. Measured baseline geomean **6.1624 ms**
(cases M=2048 / 8000 / 8192 -> 1.058 / 14.690 / 15.061 ms).

## Bottleneck hypothesis (ISA-verified, not a guess)
**Compute / MFMA-issue bound at ~71 TFLOPS = ~2.7% of MI300X fp8 peak.** Two compounding causes:

1. **The dot never uses fp8 MFMA.** `torch.float8_e4m3fn` is OCP e4m3; CDNA3 has native MFMA only for
   the **fnuz** encodings. Dumped ISA shows `v_mfma_f32_16x16x16_f16` — Triton upconverts to fp16.
   Bitcasting both operands to `tl.float8e4b8` emits `v_mfma_f32_16x16x32_fp8_fp8`: measured
   **7.18 ms -> 1.75 ms at M=8000 (146 -> 600 TFLOPS)**, correct at err_ratio 0.0077.
   The bitcast is exactly a `/2` per operand (bias 7 vs 8), so fold `*4.0` into the existing
   `kv_scales` multiply. The one hazard is byte `0x80` (`-0.0` OCP = NaN fnuz) — must be scrubbed;
   Q in-kernel (free) + K once on the host (0.027 ms) measured 1.760 ms and passes.
2. **The launcher's LDS heuristic picks a pessimal tile.** `(BLOCK_KV=64, num_stages=1,
   matrix_instr_nonkdim=32)` for this shape; `(128, 2, 16)` measured **1.98x** faster in the f16 path
   and is the right tile in the fnuz path too. `num_warps=8` and `waves_per_eu=1` are both hard
   regressions; BLOCK_KV=256 both regresses and breaks the 0.02 err_ratio gate.

Estimated combined ceiling for round 1: **~8x** geomean (14.69 -> ~1.76 ms at M=8000). After that the
op shifts to **L2/KV-reuse bound** (every one of the M workgroups re-reads all of KV: ~8 GB of L2
reads at M=8000, i.e. ~4.6 TB/s at 1.75 ms) plus a 256 MB fp32 output write.

## Multi-round strategy

**Round 1 — land the two orthogonal, already-measured wins (should be near-certain).**
- `compute` / **fnuz fp8 MFMA** (owns `_fp8_mqa_logits_kernel`'s dot + scale line). Bitcast Q and KV
  tiles to `tl.float8e4b8`, fold `*4.0` into the kv_scale multiply, scrub `0x80` on Q in-kernel.
  Verify the ISA shows `v_mfma_f32_*_fp8_fp8` (`TRITON_ALWAYS_COMPILE=1` + `.amdgcn` dump).
  Target: >=3.5x geomean on its own.
- `host_runtime` / **launcher retune + allocation** (owns `fp8_mqa_logits_gfx942` only, must not touch
  the kernel body). Replace/repair `_gfx942_default_tile_fits_lds` so (H=64,D=128) gets
  BLOCK_KV=128 / num_stages=2 / `matrix_instr_nonkdim=16`; keep `num_warps=4`, `waves_per_eu=2`.
  Also do the one-shot host K `0x80` scrub here (0.027 ms) and kill the 0.060 ms
  `torch.full(-inf)` where the `-inf` semantic can be preserved another way. Target: >=1.9x alone.
  These two merge cleanly (disjoint regions) and their product is the ~8x.

**Round 2 — post-integration: attack the new (L2 / KV-reuse) bottleneck.**
- `algorithm` / **Q-row blocking or a 2-D grid**: make one loaded KV tile serve multiple query rows
  (BLOCK_Q>1 with a `[BLOCK_Q*H, BLOCK_KV]` dot + reshape-sum over H, or split the KV axis across a
  2-D grid for better L2 locality/occupancy). **Re-measure**: BLOCK_Q=2 was *not* a win in the f16
  path (7.99 vs 7.18 ms) but the compute/L2 balance is completely different once the MFMA is 4x
  faster — that earlier null result does not carry over.
- `memory` / **output store + KV load path**: the fp32 `[M,N]` store is 256 MB; try non-temporal /
  streaming store cache modifiers, `cache_modifier=".cg"`/`.cv` on the KV loads, and `use_buffer_ops`.
  Also re-check `num_stages` / LDS double-buffering now that the tile is 128 wide.
- Re-sweep the knob grid (BLOCK_KV, nkd, waves_per_eu, num_stages) inside the fnuz path — round 1's
  sweep was done in the f16 path and the optimum can move.

**Round 3+ — `deep_explore`** if the specialists plateau: give it a roofline target (MI300X fp8 dense
peak ~2600 TFLOPS; 600 TFLOPS today = 23%) and let it fuse the tiling, the head reduction (currently a
plain 64-row fp32 sum that could ride the MFMA accumulator layout), the epilogue, and the launcher into
one rewrite. Only reach for this after rounds 1-2 confirm the plateau.

## perf_knowledge pointers (reference only)
`kk_operator=sparse_attention_nsa`, `kk_language=triton`. The cards independently flag the exact issue:
`operators/sparse_attention_nsa/tuning.md` says "fp8 indexer: **fnuz** (`fp8e4b8`/`fp8e5b16`) on
gfx942" and recommends `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`, `waves_per_eu=2-3`,
`use_buffer_ops=ON` — all consistent with the measurements above (except num_stages, where 2 measured
marginally better with BLOCK_KV=128). They also name the indexer / fp8_mqa_logits kernel as *the*
dominant cost center in DeepSeek sparse MLA, and warn about the gfx942 generic-Triton fallback cliff.
Treat these as hypotheses to measure, not verdicts.

## Confirmed dead-ends (do not re-explore)
- `input_precision="tf32"` vs `"ieee"` — identical (operands are fp8).
- `BLOCK_KV=256` — slower AND err_ratio 0.023 > the 0.02 gate, in both the f16 and fnuz paths.
- `num_warps=8` — 15.04 vs 7.18 ms.
- `waves_per_eu=1` — 11.79 vs 7.18 ms.
- Hoisting the `kv_scales` multiply past the ReLU into the post-sum — no measurable effect.
- Scrubbing `0x80` on the KV tile *inside* the loop — correct but costs 0.78 ms (2.52 vs 1.76).
