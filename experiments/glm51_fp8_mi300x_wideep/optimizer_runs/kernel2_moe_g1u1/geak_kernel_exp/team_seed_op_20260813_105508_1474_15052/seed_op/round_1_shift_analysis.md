# Round 1 re-profile — moe_g1u1_fp8 (bottleneck shift analysis)

Device: **MI300X-class / gfx942 / CDNA3, 304 CU** (`rocminfo`: 8× gfx942 agents, Compute Unit 304;
`rocm-smi`: Card Model 0x74a1, 192 GB HBM). fp8 native MFMA format on this arch is **FNUZ**.
Measured shader clock this run: ~1.77 GHz (GRBM_GUI_ACTIVE/8 XCD ÷ kernel duration).

Profiler: **rocprofv3** (`--kernel-trace --stats`) + two hand-driven `rocprofv3 --pmc` counter passes.
`rocprof-compute` / `omniperf` are still **not installed** on this box, so there is no Speed-of-Light or
cache-hierarchy section; every SoL-equivalent below is hand-computed from raw counters. No
`!!! PROFILER FAILED` block appeared — rocprofv3 ran clean, counters collected clean.

Candidate profiled = the round-1 integrated patch at
`round_1/integrate/ws_1786621560_8685/moe_fp8_blockscale_g1u1.py` (vectorized graph-captured align +
`num_stages=2→1`). NOTE: `WORKSPACE/moe_fp8_blockscale_g1u1.py` is still byte-identical to
`baseline/` — the integrated patch has **not** been promoted into the workspace yet.

## Verified result (FULL_BENCHMARK, per COMMANDMENT)

| case | T | baseline ms | round-1 ms | speedup |
|---|---|---|---|---|
| 0 | 16 | 10.3734 | **2.5848** | 4.01× |
| 1 | 32 | 14.3788 | **4.1395** | 3.47× |
| 2 | 64 | 17.6388 | **5.6691** | 3.11× |
| 3 | 128 | 19.4824 | **6.4427** | 3.02× |
| geomean | | 15.0466 | **4.4462** | **3.384×** |

(This beats the engineer's reported 2.914×; that number was measured against a per-case set that
included a slower case-0. The 4.4462 ms geomean reproduced on my run.)

## BEFORE → AFTER

```
BEFORE (R0): overhead-bound — 590 dispatches/call, 469 of them __amd_rocclr_copyBuffer D2H syncs;
             moe_align_block_size = 6.9-11.1 ms = 59-69% of every case's wall clock.
AFTER  (R1): latency-bound (sub-case C2, issue wait) inside _moe_g1u1_fp8_kernel —
             28 dispatches/call, align = 0.083-0.138 ms = 2.1-3.2% of wall,
             the Triton GEMM is 96.8-97.9% of wall.
SHIFT:       overhead → latency(C2), because the host-side per-expert python loop was deleted and
             the ~20 residual align ops were CUDA-graph-captured. Nothing about the GEMM changed
             except num_stages 2→1; it simply became the whole cost.
NEXT:        Target the fp8 software-dequant VALU chain inside the GEMM (see Opportunity 1).
```

Dispatch count per call: **590 → 28** (a 21× cut). Only 1 of the 28 is `_moe_g1u1_fp8_kernel`; the
other 27 are the graph-captured align ops, each 4-6 us, together ~0.13 ms.

## New per-call decomposition (30-iter median, warm, inside gpu_lock)

| case | T | full ms | align ms | GEMM ms | GEMM share | EM | valid rows | row fill |
|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 2.575 | 0.083 | 2.492 | **96.8 %** | 2048 | 128 | 6.2 % |
| 1 | 32 | 4.174 | 0.102 | 4.072 | 97.6 % | 4096 | 256 | 6.2 % |
| 2 | 64 | 5.684 | 0.114 | 5.570 | 98.0 % | 4352 | 512 | 11.8 % |
| 3 | 128 | 6.449 | 0.138 | 6.311 | 97.9 % | 4864 | 1024 | 21.1 % |

Latency now scales ~2.5× across an 8× token range (was 1.9×) — the flat-latency overhead signature
is gone. The residual flatness is the **expert-activation curve**, not overhead: 103/169/219/250
distinct experts get touched for T=16/32/64/128, and every touched expert costs a full `2N×K`
weight read regardless of how many rows use it.

## The GEMM kernel — counters (case 0, steady-state 4th dispatch)

Grid 1048576 / WG 256 = **4096 workgroups** vs 304 CU → the GPU is filled (not a fill problem).
`LDS_Block_Size=0`, `Scratch_Size=0` (no spill), `VGPR = 108 arch + 12 accum = 120` →
waves/SIMD = min(8, 512/120) = **4 (50 % occupancy)** — up from 2 (25 %) at baseline, which is the
one real effect of `num_stages=2→1`.

| metric | R0 baseline | R1 now |
|---|---|---|
| VALU utilization | 23.7 % | **29.8 %** (`SQ_ACTIVE_INST_VALU` 1.626e9) |
| MFMA busy | 2.35 % | **2.97 %** (`SQ_VALU_MFMA_BUSY_CYCLES` 1.620e8) |
| VALU insts / MFMA inst | 161.2 | **160.5** (`SQ_INSTS_VALU` 1.626e9 / `SQ_INSTS_MFMA` 1.013e7) |
| wave active / (active+wait) | 61.5 % | **54.6 %** (`SQ_WAIT_ANY` 1.354e9) |
| `SQ_WAIT_INST_ANY` | 1.611e9 | **1.611e9 — larger than `SQ_WAIT_ANY`, issue-side dominates** |
| VGPR (arch+accum) | 176 | **120** |
| waves/SIMD (occupancy) | 2 (25 %) | **4 (50 %)** |
| SQ_WAVES | 13 184 | 16 384 |
| L2 hit rate (`TCC_HIT/(HIT+MISS)`) | n/a | **2.2 %** — weights stream once, zero reuse |
| HBM read | ~434 GB/s | **~1.02 TB/s** (see cross-check) — ~19 % of 5.3 TB/s nameplate |
| LDS / scratch | 0 / 0 | 0 / 0 (22 `ds_read/write` in asm, trivial) |

**HBM cross-check** (per `profiling_guide.md`, TCC_EA under-reports on multi-XCD): `TCC_EA0_RDREQ_sum`
= 2.039e7 × 64 B = 1.305 GB, but the algorithmic weight demand is 103 experts × 2·2048 · 6144 B =
2.592 GB — exactly 1.99× the EA0 figure, i.e. EA0 sees one of two memory paths. Take the theoretical
2.592 GB / 2.541 ms = **1.02 TB/s**. That is ~19 % of nameplate and ~29 % of an achievable ~3.5 TB/s.
So the kernel is **not yet memory-bound**, but HBM is the hard floor it is heading toward.

## Root cause of the 160 VALU-per-MFMA ratio — now proven, not inferred

Disassembly of the round-1 `_moe_g1u1_fp8_kernel.amdgcn` (fresh compile, `TRITON_ALWAYS_COMPILE=1`):

```
 16  v_mfma_f32_16x16x16_f16      <-- an FP16 MFMA. There is no fp8 MFMA in this kernel.
876  v_cndmask_b32_e64
864  v_cmp_ne_u16_e64
 96  v_lshlrev_b16_sdwa / 96 v_lshlrev_b16_e32 / 96 v_add_u16_e32 / 50 v_perm_b32
```

I confirmed the cause with a minimal standalone `tl.dot` probe on this exact box/Triton:

| operand dtype | MFMA emitted | `v_cndmask` | `v_cmp_ne_u16` |
|---|---|---|---|
| `torch.float8_e4m3fn` (OCP — what the harness feeds) | `v_mfma_f32_16x16x16_f16` ×8 | **576** | **576** |
| `torch.float8_e4m3fnuz` (gfx942 native) | **`v_mfma_f32_16x16x32_fp8_fp8` ×4** | **0** | **0** |

That is the whole story in one table: feeding OCP e4m3 makes Triton emit a software fp8→fp16
unpack in the VALU and then run the GEMM at half-rate fp16; feeding FNUZ gets the native 8-bit
MFMA and the entire dequant chain vanishes.

## Tile sweep re-run (the R1 engineer's sweep was done while align still dominated)

Re-swept BLOCK_M ∈ {16,32} × BLOCK_N ∈ {32,64,128,256} × num_stages ∈ {1,2} × num_warps ∈ {4,8}
on cases 0 and 3 against the current config. **The current `16/64/128/GROUP_M=1/nw=4/ns=1` is a
genuine local optimum** — every one of the 28 valid alternatives was slower (best alternative
`16/128/ns1/nw4` at 0.90× ; `bn=32` and `num_warps=8` are catastrophic, 0.07-0.33×; `bn=256, ns=2`
fails with LDS out-of-resource at 68 KB). **Do not spend another round on tile search** — it is
exhausted at this source. The split may reopen once the dequant chain is gone.

## Classification: **latency-bound, sub-case C2 (issue wait)**

VALU 29.8 % and VMEM/HBM ~19-29 % are both < 40 % → latency-bound by the decision tree. Within
latency, `SQ_WAIT_INST_ANY` (1.611e9) exceeds `SQ_WAIT_ANY` (1.354e9), LDS and scratch are zero, and
occupancy sits exactly at its 4-waves/SIMD register ceiling with a filled 4096-workgroup grid — too
few resident waves to hide the ~160-instruction VALU dequant chain per MFMA. This is a re-read after
the `num_stages` change, not carried forward from R0 (R0 was also C2, but at 2 waves/SIMD).
Not memory-bound (HBM 19 % of nameplate), not compute-bound (MFMA busy 3.0 %), no longer
overhead-bound (28 dispatches, align 2-3 % of wall).
