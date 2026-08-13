# Roadmap — GLM-5.1-FP8 MoE stage-1 g1u1 (Triton, gfx942 / 304 CU)

## Kernel summary
`moe_g1u1_fp8` (launcher) → `moe_align_block_size` (host, pure torch) + `_moe_g1u1_fp8_kernel`
(@triton.jit grouped fp8 e4m3 block-scaled GEMM, per-token per-128K activation scales, [128×128]
weight block scales, fp32 acc, fused `silu(gate)*up` epilogue over the gate/up halves of `W1[e]=[2N,K]`).
E=256, top_k=8, K=6144, N=2048. Decode cases T=16/32/64 dominate.

## Bottleneck hypothesis — CONFIRMED BY MEASUREMENT: it is NOT the GEMM, it is the host align
| case | full | GEMM kernel | host `moe_align_block_size` |
|---|---|---|---|
| 0 (T=16)  | 10.91 ms | 2.97 ms | **7.92 ms — 73%** |
| 1 (T=32)  | 15.05 ms | 4.86 ms | **10.18 ms — 68%** |
| 2 (T=64)  | 18.51 ms | 6.37 ms | **12.11 ms — 65%** |
| 3 (T=128) | 20.45 ms | 7.22 ms | **13.21 ms — 65%** |

`moe_align_block_size` runs a Python `for e in range(256)` loop with two `.item()` device syncs and a
slice-assign launch per expert → ~512 syncs + ~250 micro-kernels per call. It scales with **E**, not
with tokens, which is why it is a near-constant ~8-13 ms floor across all four cases. **The geomean is
host-overhead-dominated.** Everything else is second-order until this is gone.

Secondary (real, but 1/3 of the time): the GEMM runs at ~16% of the HBM roofline. Weight traffic is
`M_blocks·2N·K` fp8 = 2.59 GB (case 0) → a 0.49 ms bandwidth floor vs 2.97 ms measured. BLOCK_M=16
padding waste is 12.9× on case 0 (128 real rows → 1648 padded).

## Prize sizing (already prototyped by the TechLead, bit-exact vs the original)
- A fully vectorized `moe_align_block_size` (argsort + bincount + cumsum + scatter, one `.item()`):
  **0.50 ms** vs 7.82 ms on case 0 → alone ≈ **3.0× end-to-end** on case 0, ≈2.4× on case 3.
- A fused Triton align kernel / sync-free worst-case-sized grid should reach tens of µs → another ~1.4×.
- GEMM tile sweep: `num_stages=1` (BLOCK_N=64, num_warps=4) is a measured free **1.18× / 1.14×** on the
  GEMM for cases 0 / 2. `num_warps=8` and BLOCK_N=128/256 were worse (matches the AMD wave64 guidance).
- Combined realistic target: **5-8× geomean**, with a stretch at ~10× if the GEMM also approaches its
  bandwidth roofline.

## Multi-round strategy

**Round 1 (parallel, orthogonal, all in the one editable file — assign disjoint functions):**
1. `host_runtime` — **rewrite `moe_align_block_size` to be loop-free and sync-free**, plus launcher
   hygiene (`torch.empty` where the kernel fully writes; the pad rows are never written, so an
   `empty` output needs care — pad rows are masked out of the store and never read by the oracle for
   valid rows, so verify). Function scope: `moe_align_block_size` + `moe_g1u1_fp8` body. Target 8-15×
   on that term. THIS IS THE ROUND'S HIGHEST-VALUE DIRECTION.
2. `compute` — GEMM tile/scheduling sweep only: `num_stages`, `num_warps`, `BLOCK_N`, `BLOCK_K`,
   `GROUP_M` (try multiples of XCD=8), `waves_per_eu`, `matrix_instr_nonkdim=16`, `kpack=2`, and a
   shape-dependent config table. Scope: the `@triton.jit` decorator args + launcher kwargs only, no
   algorithm change. Target 1.2-1.4× on the GEMM.
3. `memory`/`algorithm` (pick one; keep it off the two lanes above) — attack the 12.9× BLOCK_M padding:
   masked/variable-M grouped GEMM that skips fully-padded blocks, or a BLOCK_M=8 / GEMV path for
   `m_e==1` experts, or K-split to raise tile occupancy. Scope: `_moe_g1u1_fp8_kernel` body + grid.

**Round 2:** integrate round-1 winners, re-profile. Once the align term is collapsed the GEMM becomes
the dominant term for the first time — re-target with the new per-case table. Expect the useful lever
set to shift toward L2/weight-reuse tile scheduling (order M-blocks so blocks of the same expert are
co-scheduled and share the 4 MB L2) and toward the padding-waste algorithm work.

**Round 3:** wrapper-level HIP-graph capture/replay of the whole op (Lever 6). The harness is a
repeated-call benchmark, so once the per-call CPU work is small it becomes the floor. Also: hoist the
align result when `topk_ids` is unchanged is NOT allowed (it would game the benchmark) — do not cache
across calls keyed on tensor identity; the graph-capture lever must reproduce the real per-call work.

**Round 4 (optional):** `deep_explore` for a ground-up rewrite fusing a device-side align kernel +
persistent grouped GEMM + epilogue into one or two launches, targeting the ~0.5-1.0 ms bandwidth
roofline per case.

## Compounding
Direction 1 and 2 are fully orthogonal (host vs kernel-config) and compose multiplicatively:
~3.0× × ~1.15× ≈ 3.4× after round 1 integration. Direction 3 compounds on top if it lands. All three
edit the same *file* but disjoint *functions* — engineers must be told which function they own.

## perf_knowledge levers surfaced (reference hypotheses, to be measured)
`kk_operator = fused_moe_grouped_gemm`, `kk_language = triton`. The cards say:
- wave64 → `num_warps=4`, NOT 8 (NVIDIA-copied 8 spills VGPRs, 3-5× slower). **Confirmed here.**
- `num_stages` 1-2 only on the AMD stream pipeliner. **Confirmed: ns=1 beats ns=2 by ~15%.**
- `GROUP_SIZE_M` a multiple of XCD=8; `matrix_instr_nonkdim=16`; `kpack=2` on gfx942.
- `block_m` is the padding-waste vs launch-overhead tradeoff — "decode under-fill → padding waste
  dominates; masked handling + small BLOCK_M matter" and "consider a masked grouped GEMM
  (DeepGEMM-style) that skips empty experts instead of padding". `ksplit` for skinny per-expert GEMMs
  to reach ≥1024 tiles across the CUs.
- The align&sort step is explicitly called out as on the critical path even for the Triton backend
  (SGLang's multi-block rewrite gave 7× on MI300X / 10× on the sort). Our measurement is the extreme
  version of exactly that pitfall.
Treat all of the above as dated hints; the verify step measures everything.

## Correctness guardrails (non-negotiable)
- rtol=atol=5e-2, err_ratio ≤ 0.05, all 4 cases. Semantics frozen: `silu(gate)*up`, no router weight.
- Any align rewrite must produce the same *set* semantics; the TechLead's prototype was checked
  bit-exact with `torch.equal` on all three outputs for cases 0 and 1 — engineers should re-assert that.
- `sorted_token_ids` pad value must stay `num_tokens*top_k` (the kernel's `token_mask` depends on it).
- Do not cache/memoize across benchmark iterations keyed on input identity — that is benchmark gaming.
