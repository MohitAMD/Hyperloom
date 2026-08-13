# Baseline profile — moe_g1u1_fp8 (ROUND 0)

Device: **MI300X-class / gfx942 / CDNA3, 304 CU, 8 XCD, 64KB LDS/CU, ~5.3 TB/s HBM nameplate**
(`rocminfo`: gfx942, Compute Unit 304; `rocm-smi`: Card Model 0x74a1, SKU M3000108, GFX gfx942).
fp8 format on this arch is **FNUZ**-native for MFMA; the harness feeds `torch.float8_e4m3fn` (OCP).

Profiler: **rocprofv3** (`--kernel-trace --stats`), plus two hand-driven `rocprofv3 --pmc` counter
passes. `rocprof-compute`/`omniperf` are **not installed** on this box (`which` returns nothing) —
so no Speed-of-Light / cache-hierarchy section exists. No `!!! PROFILER FAILED` block appeared;
rocprofv3 ran clean and counter collection also worked, so SoL-equivalent numbers below are
computed by hand from raw counters (SQ_*, GRBM_GUI_ACTIVE, TCC_EA0_RDREQ_sum).

## Headline: only ~30% of the wall clock is the GEMM. `moe_align_block_size` is the bottleneck.

Per-call decomposition (20-iter timing, warm, inside `gpu_lock.sh`):

| case | T | full ms | `moe_align_block_size` ms | Triton GEMM ms | align share |
|---|---|---|---|---|---|
| 0 | 16 | 10.013 | **6.936** | 2.995 | **69%** |
| 1 | 32 | 13.934 | **8.841** | 4.836 | 63% |
| 2 | 64 | 17.027 | **10.217** | 6.370 | 60% |
| 3 | 128 | 18.941 | **11.146** | 7.217 | 59% |

`moe_align_block_size` is a pure-python loop over **all 256 experts** with two `int(...item())`
D2H syncs per expert plus a slice-assign. Trace confirms it:

**590 GPU dispatches per `moe_g1u1_fp8()` call** (case 0), of which **469 are
`__amd_rocclr_copyBuffer`** (the `.item()` syncs) + 51 `unrolled_elementwise_kernel<FillFunctor>` +
33/26 `vectorized_elementwise_kernel` + 1 `warpMergeSortKVInPlace` (the argsort). Exactly **1** of
the 590 is the actual `_moe_g1u1_fp8_kernel`. GPU-busy in that call = 4.60 ms, of which 2.99 ms is
the GEMM and 1.61 ms is align scaffolding — and the remaining ~5.4 ms of the 10.0 ms wall is pure
CPU/sync stall between those 590 launches. This is a textbook **overhead-bound** signature: the
latency scales far more slowly than the work (T=16→128 is 8× the tokens but only 1.9× the latency),
because the fixed 256-iteration host loop dominates.

## The GEMM kernel itself (dispatch attrs + counters, case 0)

Grid 843776 / WG 256 = **3296 workgroups** vs 304 CU — the GPU IS filled (not a fill problem).
`LDS_Block_Size=0`, `Scratch_Size=0` (no spill), `VGPR=40 arch + 136 accum = 176` →
waves/SIMD = min(8, 512/176) = **2** (25% occupancy, register-starved by the two
`BLOCK_M×BLOCK_N = 16×64` fp32 accumulators).

Hand-computed SoL (GRBM_GUI_ACTIVE 45.29 M ÷ 8 XCD = 5.66 M cycles @ ~1.89 GHz, 1216 SIMDs):

| metric | value |
|---|---|
| VALU utilization | **23.7 %** (`SQ_ACTIVE_INST_VALU` 1.633e9) |
| MFMA busy | **2.35 %** (`SQ_VALU_MFMA_BUSY_CYCLES` 1.620e8) |
| HBM read BW (`TCC_EA0_RDREQ_sum`×64B/t) | **~434 GB/s** ≈ 8 % of 5.3 TB/s |
| wave active / (active+wait) | **61.5 %** (`SQ_WAIT_ANY` 1.021e9) |
| `SQ_WAIT_INST_ANY` / wave | 37 068 (48 % of total wait → issue-side) |
| SQ_WAVES | 13 184 |
| LDS / scratch | 0 / 0 |

**Killer ratio: `SQ_INSTS_VALU` 1.632e9 vs `SQ_INSTS_MFMA` 1.013e7 — 161 VALU instructions per
MFMA instruction.** The disassembly says why. `_moe_g1u1_fp8_kernel.amdgcn` contains
**`v_mfma_f32_16x16x16_f16` ×32 — an FP16 MFMA, not an FP8 MFMA** — and 1750 `v_cndmask_b32` +
1728 `v_cmp_ne_u16` + 192 `v_lshlrev_b16` + 192 `v_add_u16` + 98 `v_perm_b32`: that is a
**software fp8→fp16 unpack/convert loop** emitted because Triton cannot feed `e4m3fn` (OCP) into a
gfx942 MFMA, whose native 8-bit path is `e4m3**fnuz**`. So the kernel dequantizes every fp8 element
in the VALU and then runs the GEMM at fp16 rate. MFMA busy 2.35% against VALU 23.7% is that cost.

## Padding waste (MoE, effective vs padded FLOPs)

`BLOCK_M=16` with 256 experts and only T·8 routed rows means almost every 16-row block is padding:

| case | valid rows | EM (padded) | occupancy of rows | eff GFLOP | padded GFLOP | waste |
|---|---|---|---|---|---|---|
| 0 | 128 | 1648 | **7.8 %** | 6.4 | 82.9 | **12.9×** |
| 1 | 256 | 2704 | 9.5 % | 12.9 | 136.1 | 10.6× |
| 2 | 512 | 3504 | 14.6 % | 25.8 | 176.4 | 6.8× |
| 3 | 1024 | 4000 | 25.6 % | 51.5 | 201.3 | 3.9× |

Recomputing AI from *effective* FLOPs (per the guide's MoE-padding caveat): case 0 does 6.4 real
GFLOP in 3.0 ms = **2.1 TFLOP/s**, ~0.08 % of the fp8 peak. The kernel is nowhere near any roofline.

## Classification: **overhead-bound** (primary), with a latency-bound C2 GEMM underneath

Not memory-bound: HBM sits at 8 % of peak and VMEM traffic is trivial. Not compute-bound: MFMA busy
is 2.4 %. The wall clock is dominated by 590 dispatches / 469 D2H syncs of host-side alignment, and
per-case latency is nearly flat across an 8× work range. Underneath that, the GEMM is
**latency-bound sub-case C2 (issue wait)**: 176 VGPRs → 2 waves/SIMD, `SQ_WAIT_INST_ANY` is 48 % of
all wait, and there is no LDS use and no spill — too few resident waves to cover the long VALU
dequant chain. (Re-read this split after any tile / `num_stages` / `num_warps` change.)
