# Roadmap — GLM-5.1-FP8 MoE stage-1 g1u1 (Triton, gfx942)

## Kernel summary
Expert-sorted, block-padded grouped FP8(e4m3) block-scaled GEMM with a fused SiLU(gate)*up
epilogue. x = [T*8, 6144] fp8 with per-token per-128-K-group fp32 scales; W1 = [256, 4096, 6144]
fp8 with [128x128] block scales; fp32 accumulate; bf16 out [T*8, 2048]. Only
`moe_fp8_blockscale_g1u1.py` is editable (kernel + host align helper + launcher, all three).

Baseline on this box (--full-benchmark): 10.43 / 14.42 / 17.69 / 19.56 ms, **geomean 15.11 ms**.

## Bottleneck hypothesis (measured, not guessed)

**#1 — HOST OVERHEAD, 65-75% of wall clock on every case.** `moe_align_block_size` is a Python
loop over all 256 experts doing `counts[e].item()` / `padded[e].item()` — 512 GPU->host syncs and
~256 slice-assign launches per call. Measured in isolation: **7.85 ms (case 0) / 12.67 ms
(case 3)** vs 2.96 / 7.20 ms for the actual Triton kernel. This is why case 0 (8x less compute
than case 3) is only 1.9x faster than case 3 — the classic overhead-floor signature from
`geomean_levers.md` Levers 1/2. Vectorizing this to sync-free torch ops (or one Triton kernel)
should take it to <0.2 ms and deliver ~**3x geomean on its own**, on ALL cases.

**#2 — the GEMM is HBM-bound on W1, ~6x above its own floor.** Case 0 touches 103 active experts x
2N x K fp8 = 2.6 GB => 0.49 ms at 5.3 TB/s; the kernel takes 2.96 ms. FLOPs are irrelevant here
(6.4 GFLOP = 64 us at fp8 peak) — despite the task brief calling this compute-bound, at decode
shapes it is decisively **weight-streaming bound**. Levers: reduce redundant W1 streaming (one
pass per expert over its full 2N x K, all its tokens resident), improve the fp8 load pattern /
vector width along K, and get more concurrent weight-streaming blocks in flight (grid/occupancy).

**#3 — BLOCK_M=16 padding waste (the brief's headline).** 128 real (token,expert) pairs spread
over 103 active experts = **1.24 tokens per 16-row tile**; ~92% of MFMA M-lanes are padding. But
because the kernel is weight-bound, shrinking BLOCK_M alone does NOT save the dominant cost — it
saves the (already cheap) MFMA work while possibly increasing the number of W1 re-reads. The
correct framing is: pick the smallest M-tile that still lets each expert's weights be streamed
exactly ONCE, i.e. per-expert M-collapse / masked grouped GEMM (DeepGEMM-style) rather than blind
BLOCK_M reduction. Note max tokens/expert is 3 (case 0) and 9 (case 3), so an expert's whole token
set fits in ONE 16-row tile for every case — the pad blocks per expert are already ~1; the waste
is intra-tile, not inter-tile.

Naive tile tuning is nearly exhausted: BN=128/num_stages=1 gives 2.823 vs 2.962 ms (~5%);
BN=256 does not fit LDS (68 KB > 64 KB); num_warps=8 and num_stages=3 both regress (consistent
with the perf_knowledge Triton MoE card: use 4 warps on wave64, 1-2 stages).

## Multi-round strategy

**Round 1 (parallel, orthogonal):**
1. `host_runtime` — **de-Pythonize `moe_align_block_size`** (the whole ballgame). Vectorized
   torch: `padded = ceil(counts,BM)*BM`, `offsets = cumsum`, build `sorted_token_ids` via a single
   scatter using `arange - repeat_interleave(offsets)`, `expert_ids` via
   `repeat_interleave(arange(E), padded//BM)`. ZERO `.item()` / zero device syncs; `EM` must be a
   compile-time-free bound — use a fixed worst-case `EM = E*BM + T*top_k` upper bound plus the
   existing `num_tokens_post_padded` early-exit so no host readback is needed. Also
   `torch.empty` not `torch.zeros` for `out` IF the kernel writes every valid row (it does not
   write pad rows, but pad rows are never read by the oracle — verify against the harness before
   switching). Target: 7.85 -> <0.3 ms on case 0; ~3x geomean.
2. `algorithm` — **one-pass-per-expert / masked grouped GEMM**: restructure the grid so each
   program owns (expert, n-tile) and loops its (few) tokens inside, streaming that expert's W1
   slice exactly once; skip empty experts entirely (only 103/256 active at T=16). Aim at the 2.96
   ms kernel -> ~1.0 ms.
3. `memory` — **fp8 load/LDS efficiency on the W1 stream**: BLOCK_N/BLOCK_K + vector width along
   the contiguous K axis (`stride_wk == 1`), `waves_per_eu`, num_stages=1, `matrix_instr_nonkdim=16`,
   `kpack=2`; avoid the redundant second scale load; keep `a` in registers across both dots.
   Target the 6x gap to the 0.49 ms HBM floor.

**Round 2:** re-profile after integrating round 1. Once host overhead is gone the kernel IS the
wall clock, so the bottleneck shifts to #2/#3 — expect to spend round 2 on split-K / stream-K over
the K=6144 reduction to fill 304 CUs (case 0 has only 103 experts x 16 n-tiles of work), and on a
`compute` direction for occupancy/VGPR after the algorithm rewrite lands. Also a second
`host_runtime` pass: the launcher still allocates `out` and the align tensors every call — consider
a persistent scratch cache keyed by shape.

**Round 3:** if specialists plateau, dispatch `deep_explore` with a roofline target (case 0 at
~0.7 ms total => >14x) letting it fuse align + GEMM + epilogue and choose the layout freely.

## Integration/compounding notes
Direction 1 (host) and directions 2/3 (device) are fully orthogonal — 1 touches only
`moe_align_block_size` + `moe_g1u1_fp8`, 2/3 touch only `_moe_g1u1_fp8_kernel` (+ its grid). They
multiply: 3x (host) x 2-3x (kernel) is the realistic path to ~6-9x geomean. Caution: direction 2
may change the `sorted_token_ids`/`expert_ids` contract; if so it must own the align helper and
direction 1 must be merged into it rather than alongside it — instruct direction 2 to KEEP the
existing align output contract so the merge is clean.

## perf_knowledge levers surfaced (reference hypotheses, to be measured)
- `fused_moe_grouped_gemm/backends/triton.md`: num_warps=4 (NOT 8) on wave64; num_stages 1-2;
  `matrix_instr_nonkdim=16`; `kpack=2`; GROUP_SIZE_M a multiple of XCD=8 (baseline GROUP_M=1 makes
  the swizzle inert — worth testing 8); decode under-fill => masked handling + small BLOCK_M.
- `grouped_gemm_moe/tuning.md`: mfma_16x16 for skinny groups; `ksplit` (K-split) for skinny
  per-expert GEMMs in decode to reach >=1024 tiles across 304 CUs; single fused launch; tile
  round-robin across CUs (stream-K analog).
- `fused_moe_grouped_gemm/tuning.md`: "MoE align&sort redesign cut the sort step 10x" — the
  vendor already collapsed exactly the host cost that dominates HERE; strong prior for direction 1.
- `splitk_streamk_gemm/overview.md` for the round-2 split-K work.
