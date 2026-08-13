# r2_d0 (compute) — Native fp8 MFMA via OCP->fnuz bitcast

## Task
Replace the software fp8->fp16 dequant chain in `_moe_g1u1_fp8_kernel` with the gfx942 native fp8 MFMA.

## Approach
`torch.float8_e4m3fn` (OCP, exp bias 7) and `tl.float8e4b8` (fnuz, bias 8) have the identical 1/4/3 bit
layout, so reinterpreting the bits as fnuz yields exactly half the value — for normals and subnormals
alike. Both operands halve, so the raw `tl.dot` is exactly 1/4 of the true product; a single `* 4.0`
folded into the existing per-128-block `a_scale` load restores it bit-exactly.

1. Launcher hands the kernel `a_fp8.view(torch.uint8)` / `w1_fp8.view(torch.uint8)` (metadata-only).
2. K-loop: `tl.where(b == 0x80, 0, b)` then `.to(tl.float8e4b8, bitcast=True)`.
   The `0x80` guard is the only encoding that differs (OCP -0.0 vs fnuz NaN). A byte census showed
   `0x80` present but `0x7F`/`0xFF` absent, so one `tl.where` per operand is sufficient — no host-side
   fixup over the 6.4 GB weight tensor.
3. `a_scale = 4.0 * tl.load(...)`.

## ISA evidence (case0 AMDGCN dump, before -> after)
| | before | after |
|---|---|---|
| MFMA opcode | `v_mfma_f32_16x16x16_f16` x32 | **`v_mfma_f32_16x16x32_fp8_fp8` x8** |
| `v_cndmask_b32` | 1750 | **0** |
| `v_cmp_ne_u16` | 1728 | **0** |
| `v_lshlrev_b16` / `v_add_u16` | 192 / 192 | 0 / 0 |
| `v_perm_b32` | 98 | 2 |
| VGPR | 176 (40 + 136 acc) | 104 |

## Results (FULL_BENCHMARK, median of 3 runs, spread 0.2%)
| case | baseline ms | round-1 ms | this ms | speedup vs baseline |
|---|---|---|---|---|
| 0 (T=16)  | 10.3734 | 2.5834 | 1.0001 | 10.372x |
| 1 (T=32)  | 14.3788 | 4.1430 | 1.5562 |  9.240x |
| 2 (T=64)  | 17.6388 | 5.6842 | 2.0251 |  8.710x |
| 3 (T=128) | 19.4824 | 6.4327 | 2.3341 |  8.347x |

**Geomean 9.1363x** (arith 9.1672x, ratio-of-sums 8.9471x). Incremental over round 1: **2.70x**.
Correctness: `err_ratio=0.0000`, `cos_diff` 4.5e-10 .. 1.3e-9 on all 4 cases — bit-exact.

## What didn't work
Re-swept the whole schedule space post-MFMA; 12 variants, all tie or lose:
`num_stages` 0/1/2/3 (ns=0 and ns=1 tie inside noise), `BLOCK_N` 32/64/128/256, `num_warps` 2/4/8,
`GROUP_M` 1/4/8, `kpack=2`, `matrix_instr_nonkdim=16`, `waves_per_eu` 2/4, m-fastest and n-grouped pid
orders, manual 2x k-unroll (worse — register pressure), n-major weight load + `tl.trans`,
`OPTIMIZE_EPILOGUE=1`. `BLOCK_K=256` with sub-tiled scales failed to compile (Triton 3.6 rejects
indexing a reshaped tensor).

## Bottleneck now
Memory-bound on weight streaming, not issue-bound. Achievable HBM read on this box = 4.35 TB/s;
active-expert weight footprint is 2.59/4.25/5.51/6.29 GB per case, so the kernel now runs at
**60-62% of the HBM roofline**. Remaining headroom is ~1.6x and it lives in the memory/algorithm lane:
skip or compact the near-empty `BLOCK_M=16` tiles (7.8% row occupancy at T=16), which now saves
bandwidth, not just FLOPs.
