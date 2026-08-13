# Round 2 re-profile — moe_g1u1_fp8 (bottleneck shift analysis)

Device: **MI300X-class / gfx942 / CDNA3, 304 CU, 8 XCD, 192 GB HBM** (`rocm-smi`: Card Model 0x74a1,
SKU M3000108, GFX Version gfx942). Nameplate HBM ~5.3 TB/s; **measured achievable read BW on this box
= 4324 GB/s** (large bf16 reduction under `gpu_lock`, best of 5). Use 4.32 TB/s as the roofline
ceiling, not 5.3.

Profiler: **rocprofv3** (`--kernel-trace --stats --output-format csv`, no `!!! PROFILER FAILED` block)
plus two hand-driven `rocprofv3 --pmc` counter passes (`SQ_*`/`GRBM_GUI_ACTIVE`, then `TCC_HIT/MISS`).
`rocprof-compute` / `omniperf` are still **not installed** on this box, so there is no Speed-of-Light
or cache-hierarchy section; every SoL-equivalent number below is hand-computed from raw counters.
A third TCC pass with 6 counters (`TCC_EA0_*`) hung the target for >10 min and was killed — the 2-counter
`TCC_HIT_sum`/`TCC_MISS_sum` pass is what is reported.

Candidate profiled = the round-2 winner at
`round_2/engineer_0/workspace/moe_fp8_blockscale_g1u1.py` (md5 `ee1c402f…`; round-1 align + graph
capture + `num_stages=1`, **plus** the OCP-e4m3 → fnuz bitcast that unlocks the native fp8 MFMA).
NOTE: `WORKSPACE/moe_fp8_blockscale_g1u1.py` is **still byte-identical to `baseline/`** — neither the
round-1 nor the round-2 patch has been promoted into the canonical workspace yet. This is now two
rounds of drift; promote before round 3.

## Verified result (FULL_BENCHMARK, my own run, per COMMANDMENT)

Correctness first: all 4 cases PASS, `err_ratio=0.0000`, `cos_diff` 2.1e-10 … 1.3e-9.

| case | T | baseline ms | R1 ms | **R2 ms** | speedup vs baseline |
|---|---|---|---|---|---|
| 0 | 16  | 10.3734 | 2.5848 | **0.9994** | 10.380× |
| 1 | 32  | 14.3788 | 4.1395 | **1.5672** |  9.174× |
| 2 | 64  | 17.6388 | 5.6691 | **2.0239** |  8.716× |
| 3 | 128 | 19.4824 | 6.4427 | **2.3325** |  8.353× |
| geomean | | 15.0466 | 4.4462 | **1.6490** | **9.125×** |

Reproduces the engineer's 9.1363× within 0.1%. Incremental over round 1: **2.696×**.

## BEFORE → AFTER

```
BEFORE (R1): latency-bound, sub-case C2 (issue wait) — VALU 29.8 %, MFMA busy 3.0 %,
             160 VALU insts per MFMA inst (software fp8->fp16 unpack), HBM ~1.02 TB/s
             = 24 % of the 4.32 TB/s achievable ceiling. SQ_WAIT_INST_ANY (1.611e9) >
             SQ_WAIT_ANY (1.354e9): issue side dominated.
AFTER  (R2): MEMORY-bound on expert-weight streaming — HBM 2.75 TB/s = 64 % of the
             4.32 TB/s achievable ceiling (52 % of nameplate), L2 hit 2.2 % (no reuse
             to be had), MFMA busy 5.8 %, VALU 21.3 %. SQ_WAIT_INST_ANY (3.128e8) is now
             only 0.62 × SQ_WAIT_ANY (5.049e8): the issue-side stall that defined C2 is
             gone and what remains is memory wait.
SHIFT:       latency(C2) -> memory, because the bitcast to tl.float8e4b8 replaced
             v_mfma_f32_16x16x16_f16 with the native v_mfma_f32_16x16x32_fp8_fp8 and
             deleted the entire ~161-instruction software dequant chain. The kernel got
             2.7× faster while moving the SAME 2.59-6.29 GB of weights, so effective HBM
             bandwidth rose 1.02 -> 2.75 TB/s and the memory system became the limiter.
NEXT:        Target bandwidth efficiency (2.75 -> ~4.0 TB/s = the last ~1.5×), NOT FLOPs.
```

Dispatch count per call is **unchanged at 28** (27 graph-captured align ops + 1
`_moe_g1u1_fp8_kernel`); the align path is byte-identical to round 1. GEMM share of wall clock,
from the kernel trace vs the benchmark medians: **95.0 / 93.5 / 92.5 / 90.7 %** for cases 0-3
(slightly *lower* than R1's 97-98 % only because the GEMM shrank 2.7× while the ~50-200 us align +
launch tail did not).

## Kernel counters (case 0, steady state, `--pmc` pass on a 20-warmup driver)

Grid 1048576 / WG 256 = **4096 workgroups** vs 304 CU → GPU is filled; not a fill problem.
`LDS_Block_Size = 0`, `Scratch = 0` (no spill), `VGPR = 100 arch + 4 accum = 104` →
waves/SIMD = min(8, 512/104) = **4 (50 % occupancy)**, i.e. sitting exactly at the register ceiling.

| metric | R1 | **R2** | note |
|---|---|---|---|
| VALU utilization | 29.8 % | **21.3 %** | `SQ_ACTIVE_INST_VALU` 2.995e8 |
| MFMA busy | 2.97 % | **5.77 %** | `SQ_VALU_MFMA_BUSY_CYCLES` 8.100e7 — 1.9× |
| VALU insts / MFMA inst | 160.5 | **59.1** | `SQ_INSTS_VALU` 2.990e8 / `SQ_INSTS_MFMA` 5.063e6 |
| SQ_WAIT_ANY | 1.354e9 | **5.049e8** | |
| SQ_WAIT_INST_ANY | 1.611e9 (> WAIT_ANY) | **3.128e8 (0.62 × WAIT_ANY)** | issue-side stall resolved |
| active / (active+wait) | 54.6 % | **37.2 %** | lower because the kernel is now waiting on HBM, not issuing filler VALU |
| VGPR (arch+accum) | 120 | **104** | |
| waves/SIMD | 4 (50 %) | **4 (50 %)** | at the register ceiling |
| SQ_WAVES | 16 384 | **16 384** | |
| L2 hit (`TCC_HIT/(HIT+MISS)`) | 2.2 % | **2.2 %** | weights stream once; L2 is doing nothing |
| HBM read | ~1.02 TB/s | **2.75 TB/s** | see cross-check |
| LDS / scratch | 0 / 0 | 0 / 0 | |

**HBM cross-check (two independent methods agree).**
1. Algorithmic: case 0 touches 103 distinct experts × 2·2048 × 6144 B = **2.59 GB**; kernel duration
   949.4 us (trace) → **2.73 TB/s**. Cases 1-3: 4.25/5.51/6.29 GB in 1464.9/1871.3/2114.9 us →
   2.90 / 2.95 / 2.97 TB/s.
2. Counter: `TCC_MISS_sum` = 2.04e7 × 128 B = **2.61 GB** in 987 us → **2.65 TB/s**. The counter's
   2.61 GB matches the algorithmic 2.59 GB to 0.8 %, which also confirms the 128 B L2 line size is
   the right multiplier here (64 B would give half the known-true traffic).

So: **2.6-3.0 TB/s achieved against a measured 4.32 TB/s ceiling = 61-69 % of roofline.**

## Classification: **memory-bound** (expert-weight streaming)

Decision tree: VALU 21.3 % and MFMA 5.8 % are both far below 40 %, so it is not compute-bound.
The guide warns not to call memory-bound on a small AI alone — but here HBM utilization *is*
actually high (64 % of the measured achievable ceiling, and the traffic is confirmed by two
independent methods), and the residual wait is memory wait, not issue wait
(`SQ_WAIT_INST_ANY / SQ_WAIT_ANY` fell from 1.19 to 0.62). LDS and scratch are zero, the grid fills
the GPU 13× over, and occupancy sits at its register ceiling. Latency-C2 is resolved; the kernel is
memory-bound.

## Correction to the round-2 engineer's steer (important)

The engineer's report recommends round 3 attack the padded/near-empty `BLOCK_M=16` tiles on the
grounds that "padded-FLOP waste is now equivalently PADDED-BANDWIDTH waste". **That is not true in
this shape regime, and the arithmetic says so:**

| case | assignments | distinct experts touched | EM/BLOCK_M = M-blocks | M-blocks per touched expert |
|---|---|---|---|---|
| 0 | 128  | 103 | 2048/16 = 128 (guard exits most) | ~1.0 |
| 3 | 1024 | 250 | 4864/16 = 304 | ~1.2 |

Each expert's weight columns are read by its own M-block(s) exactly once across the 32 N-blocks
(each N-block reads a disjoint 64-column slice). With ~1.0-1.2 M-blocks per expert there is almost
**no duplicated weight traffic to eliminate** — the 2.59-6.29 GB is the algorithmic *minimum* for
touching those experts. Compaction saves padded FLOPs, but MFMA busy is 5.8 %: FLOPs are free here.
Expected payoff from tile compaction: near zero. Do not spend a round on it.

The real ceiling is `2.59 GB / 4.32 TB/s = 600 us` vs 949 us measured for case 0 → **max remaining
headroom ≈ 1.58×, and it is all bandwidth-efficiency, not work-elimination.**

## Ranked opportunities for round 3

1. **Kill the per-byte `tl.where(b == 0x80, 0, b)` guard on the two weight operands** — biggest
   single lever. It still costs 59 VALU insts per MFMA inst (down from 160, but far from zero), and
   those ops sit *in the load → MFMA path*, consuming issue slots that would otherwise be spent
   keeping more global loads in flight. Per k-iteration each lane guards ~64 weight bytes vs ~6 MFMA
   issues. Fixes: (a) sanitize `w1_fp8` **once** into a cleaned buffer cached on
   `(data_ptr, shape, dtype)` — a shape-derived artifact, explicitly allowed by COMMANDMENT rule 5,
   costs 6.4 GB of 192 GB; (b) or express the guard bitwise on packed 32-bit words instead of per
   byte; (c) `a` is 1/8th the bytes — keep its guard, it is nearly free. Expect the residual VALU
   to drop toward the ~24-op/k-iter scale-application floor.
2. **Bypass L2 on the weight loads.** L2 hit rate is **2.2 %** — the 4 MB of L2 is being churned by
   6.3 GB of single-use streaming data for no benefit, and the fill/evict traffic eats real
   bandwidth. Try `cache_modifier` on the two `w_*` `tl.load`s (`"cg"` / non-temporal / `sc0 sc1`
   flags on gfx942) so weights stream past L2 and leave it to the reused `a` tile and scales.
   Cheap to test, directly targets the 2.75 → 4.32 TB/s gap.
3. **Widen the per-wave load granularity on the weight stream.** Each WG pulls 2 × 64 × 128 B = 16 KB
   of weights per k-iteration against only 2 KB of `a`. Check the ISA for `global_load_dwordx4` vs
   narrower loads on the `w_gate`/`w_up` paths; if Triton is emitting dwordx2 or byte-wise loads
   after the uint8 `tl.where`, that alone caps achieved bandwidth. (Opportunity 1 may fix this for
   free — the `where` on uint8 can force a narrower load layout.)
4. **Raise waves/SIMD from 4 to 6-8** to cover HBM latency. VGPR = 104 puts the ceiling at exactly 4;
   getting under 86 → 5-6 waves, under 64 → 8. The accumulators are only 4 accum VGPRs, so the 100
   arch VGPRs are addressing/staging — worth one look at whether the dual gate/up pointer arithmetic
   can be shared. Note this is a genuine C2-style occupancy play and only pays if the extra waves
   translate into more loads in flight; measure BW, not occupancy.
5. **Do NOT re-sweep the tile/schedule space.** Two engineers have now swept it independently
   (BLOCK_N 32/64/128/256, num_warps 2/4/8, num_stages 0/1/2/3, GROUP_M, kpack, nonkdim,
   waves_per_eu, pid orders) — all tie or lose. It is closed at this source.
6. **Promote the patch into `WORKSPACE`.** The canonical workspace is still the pristine baseline;
   round 3 engineers branching from it would silently lose 9.1×.
