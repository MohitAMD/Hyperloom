# TechLead Final Report — `moe_fp8_blockscale_g1u1` (GLM-5.1-FP8 MoE stage-1 g1u1 GEMM)

## Summary

| | |
|---|---|
| Kernel | `moe_fp8_blockscale_g1u1.py` — `_moe_g1u1_fp8_kernel` (@triton.jit), launcher `moe_g1u1_fp8` |
| Kernel type | **triton** (triton2triton), gfx942 / CDNA3 / MI300X-class, 304 CU |
| Baseline geomean | **15.0466 ms** |
| Final geomean | **1.6501 ms** |
| **Headline speedup (unweighted geomean — PRIMARY, no WORKLOAD_SPEC)** | **9.11×** (verified 9.1082; independent re-measure at report time 9.1187) |
| Arithmetic mean speedup | 9.14× (re-measure 9.15×) |
| Time-weighted (ratio-of-sums, uniform counts) | 8.92× |
| Rounds run | 2 |
| Budget used | 3 directions (r1_d0, r1_d1, r2_d0) + 1 integrate |
| Correctness | ALL 4 cases PASS, `err_ratio = 0.0000`, `cos_diff 2.1e-10 … 1.3e-9` (bit-exact-class) |
| Final patch | `final_patch.diff` (241 lines, applies cleanly to the pristine baseline commit) |
| Optimized sources | `optimized/moe_fp8_blockscale_g1u1.py` |

Shapes are fixed at K(hidden)=6144, N(moe_intermediate)=2048, E=256, top_k=8, fp8 block [128,128];
the four cases sweep the decode token count T = 16 / 32 / 64 / 128.

---

## Round-by-round

### Round 0 — baseline profile
Bottleneck: **host/overhead**. Dispatch analysis showed `moe_align_block_size` running a
256-iteration Python loop with a `.item()` per expert — 256 device→host syncs per call, a flat
6.9–12.7 ms that scaled with `E=256`, not with tokens. The GEMM itself was secondary at that point.

### Round 1 — two orthogonal specialists

| id | specialty | strategy | verified | verdict |
|---|---|---|---|---|
| `r1_d0` | host_runtime | Vectorized, sync-free `moe_align_block_size` + launcher hygiene + CUDA-graph capture/replay | **2.9152×** (expected 2.8) | **success — confirmed** |
| `r1_d1` | compute | GEMM tile/schedule sweep (`num_stages`, BLOCK_M shrink to cut padded-FLOP waste) | **1.0439×** (expected 1.25) | **partial** — only `num_stages=1` paid; the BLOCK_M-shrink premise was wrong |

- **r1_d0 detail.** `torch.bincount` was replaced by a `scatter_add_` into a `zeros(E)` (bincount
  hides a device→host sync in its internal `max()`); outputs were oversized to a static `EM_max`
  (measured free: 2.960 vs 2.965 ms case0), which is what makes the whole prologue capturable.
  Isolated align-only: case0 7.62 → 0.205 → **0.076 ms**, case3 12.67 → 0.199 → **0.129 ms** (~100×).
  The vectorized rewrite gets to ~0.2 ms; only graph capture removes the last ~0.1 ms of pure
  dispatch count.
- **r1_d1 detail.** Every neighbour of BM16/BN64/BK128/GM1/nw4/ns1 was ≥10% worse. BM=8 is 2.7×
  slower, BM=4 3.3×, BM=32 4.6×, BM=64 2.3× — the 16×16×16 MFMA needs M≥16.
- **Integrate (`r1_integrate`)**: fully orthogonal patches, pure incremental stack, no hand-merge
  needed. Result **3.3826×** vs a multiplicative expectation of 3.043× — *better* than multiplicative,
  because removing the host stall exposes more of the GEMM win. `num_stages` was re-swept in the
  combined context to confirm the optimum had not moved.
- **Round 1 winner:** integrated, **3.3826×**.
- **Bottleneck shift:** overhead → **latency (C2 issue-wait)**. `valu_insts_per_mfma_inst = 160.5`,
  MFMA busy 2.97%, HBM 1020 GB/s (19% of nameplate), `wave_active/(active+wait) = 54.6%`.

### Round 2 — one high-conviction compute direction

| id | specialty | strategy | verified | verdict |
|---|---|---|---|---|
| `r2_d0` | compute | Native fp8 MFMA: bitcast OCP e4m3 → fnuz (`tl.float8e4b8`) to kill the 161:1 VALU dequant chain | **9.1082× cumulative / 2.696× incremental** (expected 2.0) | **success — confirmed, 4.5× over-delivery** |

The round-1 profile's smoking gun was `mfma_variant = v_mfma_f32_16x16x16_f16` — a *fp8* kernel that
was not using the fp8 MFMA at all. Triton-on-AMD only feeds the native 8-bit MFMA from the **fnuz**
types (`fp8e4nv/fp8e5/fp8e5b16/fp8e4b8`); a `torch.float8_e4m3fn` (OCP) tensor was therefore being
software-dequantized to fp16, ~161 VALU ops per MFMA. Since OCP e4m3 (bias 7) and fnuz e4b8 (bias 8)
share the **identical 1/4/3 bit layout**, reinterpreting the bytes is an exact halving of both
operands, corrected by folding a single `* 4.0` (an exact power of two) into `a_scale` — hence
bit-exact. The only encoding clash is `0x80` (−0.0 in OCP, NaN in fnuz), guarded by
`tl.where(b == 0x80, 0, b)` on the loaded bytes; a byte census confirmed `0x7F/0xFF` never occur.

ISA diff (case0): `v_mfma_f32_16x16x16_f16` ×32 → `v_mfma_f32_16x16x32_fp8_fp8` ×8;
`v_cndmask_b32` 1750→0; `v_cmp_ne_u16` 1728→0; `v_lshlrev_b16` 192→0; `v_add_u16` 192→0;
`v_perm_b32` 98→2; VGPR 176 (40 arch + 136 accum) → 104.

- **Integrate:** not required — a single direction ran this round.
- **Round 2 winner:** engineer `r2_d0`, **9.1082×**.
- **Bottleneck shift:** latency → **memory (expert-weight streaming)**. The compute lane is now
  closed: HBM **2.75 TB/s = 64% of the MEASURED 4.32 TB/s achievable ceiling** on this box (the
  5.3 TB/s nameplate is not reachable), L2 hit 2.2%, MFMA busy 5.8%, VALU 21.3%,
  `SQ_WAIT_INST_ANY/SQ_WAIT_ANY` 1.19 → 0.62. The kernel got 2.7× faster while moving the *same*
  bytes, so effective bandwidth rose 1.02 → 2.75 TB/s and memory became the limiter.

---

## Final per-test-case table

| case | params | count | weight | baseline ms | optimized ms | speedup |
|---|---|---|---|---|---|---|
| case=0 | T=16 topk=8 E=256 N=2048 K=6144 | 3 | 0.25 | 10.3734 | 1.0008 | **10.37×** |
| case=1 | T=32 topk=8 E=256 N=2048 K=6144 | 3 | 0.25 | 14.3788 | 1.5558 | **9.24×** |
| case=2 | T=64 topk=8 E=256 N=2048 K=6144 | 3 | 0.25 | 17.6388 | 2.0390 | **8.65×** |
| case=3 | T=128 topk=8 E=256 N=2048 K=6144 | 3 | 0.25 | 19.4824 | 2.3351 | **8.34×** |

- **Geomean (PRIMARY):** 15.0466 ms → 1.6501 ms = **9.12×** (round-2 verified figure 9.1082×; the
  0.1% delta is within the measured ~1.5% run-to-run e2e variance)
- **Arithmetic mean of per-case speedups:** 9.15×
- **Time-weighted (ratio-of-sums, uniform weights — no WORKLOAD_SPEC supplied):** 8.93×

Weights are uniform: no `WORKLOAD_SPEC` was provided, so per the COMMANDMENT the unweighted geomean
is the primary metric.

---

## Key optimizations applied

1. **Native fp8 MFMA via an OCP→fnuz bitcast** (`r2_d0`, largest single win: **2.70× incremental**).
   Load the fp8 weight/activation bytes as `uint8`, sanitize `0x80 → 0x00`, `.to(tl.float8e4b8,
   bitcast=True)`, and fold the exact `* 4.0` into `a_scale`. Swaps the fp16 MFMA + 161-VALU software
   dequant for `v_mfma_f32_16x16x32_fp8_fp8`. Bit-exact.
2. **Vectorized, sync-free `moe_align_block_size`** (`r1_d0`). Removed the 256-iteration `.item()`
   loop; `scatter_add_` instead of `bincount` (which hides a sync); static `EM_max` output sizing
   (free, because the kernel's `pid_m*BLOCK_M >= num_tokens_post_padded` guard reads the device
   tensor and exits immediately). ~7.6 ms → ~0.2 ms on the align term.
3. **CUDA-graph capture/replay of the align prologue** (`r1_d0`). Takes the align term from ~0.2 ms
   to **0.076 ms** — pure dispatch-count elimination that no further op fusion could reach. Cache is
   keyed only on shape/dtype/block_m/num_experts (never tensor identity or content) with real inputs
   copied into static buffers each call.
4. **`num_stages=1`** (`r1_d1`, the only surviving element of the tile sweep) — 1.15–1.20× on the
   isolated GEMM.

Net effect: `moe_align_block_size` fell from **46–65% of wall time to <2.5%**; the GEMM is now
90.7–95.0% of wall and is bandwidth-bound at 64% of the achievable HBM ceiling.

---

## What didn't work (confirmed dead-ends — do not re-explore)

- **Shrinking BLOCK_M to cut padded-FLOP waste.** DISPROVED. BM=8 is 2.7× slower, BM=4 3.3×,
  BM=32 4.6×, BM=64 2.3× (case0, isolated). The 16×16×16 MFMA needs M≥16; below that Triton emits a
  degenerate layout costing far more than the padding saved.
- **The tile/schedule space generally.** CLOSED — swept twice independently and re-swept
  *post*-MFMA-change (12 further variants: BLOCK_N 32/64/128/256, num_warps 2/4/8, num_stages 0–3,
  GROUP_M, kpack, matrix_instr_nonkdim, waves_per_eu, pid orders). All tie or lose; the last three
  are within 1%. Note **BLOCK_K must equal `group_k` (=128) or results are silently WRONG**.
- **Tile compaction / skipping near-empty BLOCK_M=16 blocks.** REJECTED on arithmetic. Each expert's
  weight columns are read exactly once across the 32 disjoint N-blocks, and there are only ~1.0
  (case0) to ~1.2 (case3) M-blocks per touched expert — there is no duplicated weight traffic to
  eliminate. 2.59–6.29 GB is the algorithmic minimum, confirmed two ways 0.8% apart (algorithmic
  2.59 GB vs `TCC_MISS_sum × 128B` = 2.61 GB). With MFMA busy at 5.8%, saved FLOPs are free FLOPs.
- **`torch.bincount` in the prologue** — functionally fine but hides a device→host sync; replaced.
- **Host-side elementwise fixups for the fp8 NaN/-0.0 clash** — rejected in favour of the in-kernel
  `tl.where` guard, which costs nothing on the critical path relative to a whole extra pass.

---

## State at stop, and where the remaining headroom is

Bottleneck is now **memory** (expert-weight streaming). Measured headroom is **~1.58×**
(600 µs roofline vs 949 µs measured, case0) and it is entirely bandwidth *efficiency*, not work
elimination. Per-case active-expert footprint: 2.59 / 4.25 / 5.51 / 6.29 GB (103 / 169 / 219 / 250
of 256 experts touched). Occupancy is at the register ceiling: VGPR 104 → waves/SIMD =
min(8, 512/104) = 4 (50%); LDS 0, scratch 0, 4096 WGs over 304 CU (13× fill) — not a fill problem.

If the run is continued, the two live levers are (a) removing the per-byte `tl.where(b == 0x80, 0, b)`
from the two **weight** operands (still 59 VALU per MFMA in the load→MFMA path) via a once-sanitized
`w1_fp8` buffer cached on `(data_ptr, shape, dtype)` or a packed-32-bit bitwise guard, plus an ISA
check that weight loads are `global_load_dwordx4`; and (b) L2 bypass on the weight stream (2.2% hit
rate → `cache_modifier="cg"` / non-temporal `sc0 sc1`) together with VGPR 104 → <86 to lift
waves/SIMD 4→6, judged by **measured bandwidth, not occupancy**.

## Process notes

- **Orthogonality contract held both rounds.** `r2_d0` touches only the k-loop body, the `a_scale`
  load, and 4 launcher lines; `moe_align_block_size`, the graph-cache keying and the static-buffer
  copy-in are byte-identical to round 1.
- **Workspace drift was found and fixed at report time.** `WORKSPACE/moe_fp8_blockscale_g1u1.py`
  had remained byte-identical to `baseline/` (md5 `17de246d…`) through both rounds. The round-2
  winner has now been applied and committed to the workspace git history (md5 `ee1c402f…`), and
  `final_patch.diff` is the diff from the root baseline commit to HEAD. Correctness and the full
  benchmark were re-run from the promoted workspace to produce the table above.
- **Measurement discipline:** e2e geomean run-to-run variance is ~1.5% (treat sub-2% e2e deltas as
  noise); engineer claims matched verification within 0.1–0.3% in both rounds.
