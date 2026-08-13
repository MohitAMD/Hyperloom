# Kernel #2 — MoE stage-1 g1u1 (fmoe_fp8_blockscale_g1u1) — E2E feasibility verdict

**Date:** 2026-08-13 (UTC-7)  **Node:** useocpm2m-097-038 (inspection only, no GPU workload run)
**Container image:** glm5.1-fp8-disagg:mi300x-fromscratch-aiter017
**Outcome:** NOT INTEGRABLE for an honest in-session E2E A/B → **REJECT** for baking. Stopped before benchmarking, as instructed (a well-evidenced "not integrable" is the result).

## (1) Runtime MoE kernel kind + dispatch path
GLM-5.1-FP8 decode MoE dispatch (traced in the live image):

```
vllm FusedMoE (block-scaled fp8 w8a8, QuantMethod.BLOCK_128x128)
  -> vllm .../fused_moe/experts/rocm_aiter_moe.py: rocm_aiter_fused_experts()
  -> vllm._aiter_ops.rocm_aiter_ops.fused_moe()
  -> aiter.fused_moe.fused_moe()   (per_1x128, Silu, bf16 x fp8 x fp8, isG1U1)
  -> aiter.fmoe_fp8_blockscale_g1u1(moe_buf, a1_q, w1, w2, ...)
```

- `aiter.fmoe_fp8_blockscale_g1u1` is `@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")` — a **compiled AMD ASM/HIP kernel** (`aiter/jit/module_moe_fmoe_asm.so`; symbol `fmoe_g1u1_impl`, config tags `cfg_fmoe_bf16_blockscaleFp8_g1u1_...`). **Not Triton.**
- It is **monolithic / full-fused**: it takes BOTH `w1` (gate|up) AND `w2` (down) and writes `moe_buf = [M, model_dim=6144]` — i.e. it performs stage-1 (gate/up GEMM + SiLU·mul) AND stage-2 (down GEMM + top-k-weighted reduction) in one launch. The "g1u1" in the name only describes the stage-1 activation structure.
- Tuned dispatch confirms decode uses this 1-stage ASM kernel: `configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_1.csv` rows for token=1,2 → `run_1stage=1`, `us2=0`, `kernelName1=_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x128E` (and `_ps_`, `_flat_pf3` a16 variants). These rows also show the real TP=8 per-GPU shape: **model_dim=6144, inter_dim=256, expert=257, topk=9** (moe_intermediate 2048/8 = 256 per GPU).
- The default heuristic for per_1x128 (used when no tuned row matches) is `run_1stage = token > 32`, with the source comment *"for fp8 blockscale, ck has better performance so disable assembly kernel"*. So small decode batches can instead take the **2-stage** path (`fused_moe_2stages`), whose stage-1 is `ck_moe_stage1` / `cktile_moe_stage1` / `_flydsl_stage1_wrapper` / `moe_stage1_g1u1` — all `@compile_ops` **CK/HIP/ASM**, still **not Triton**.

**There is no `@triton.jit` g1u1/stage-1 kernel anywhere in the live GLM MoE path (1-stage or 2-stage).**

## Integration route taken: NONE (feasibility gate failed) — evidence above
- **Route A (direct .py @triton.jit patch)** — IMPOSSIBLE. The GEAK patch targets a standalone Triton `.py` reconstruction that has no counterpart in the runtime; every runtime g1u1/stage-1 kernel is compiled ASM/CK.
- **Route B (route to Triton + inject GEAK)** — NOT feasible as an honest, isolated proof in-session:
  - The GEAK kernel (`moe_g1u1_fp8`) is **stage-1 only**: it outputs the intermediate `[M·topk, N] bf16` (SiLU(gate)·up). The live ASM entry produces the **final** `[M, 6144]` output. Monkeypatching `fmoe_fp8_blockscale_g1u1` → GEAK yields wrong results (missing the entire down projection + reduction).
  - To make it correct you must bolt on stage-2 (down GEMM with w2 fp8 block scales + top-k-weighted scatter-reduce) AND an inter-stage requant of the bf16 intermediate back to fp8 in aiter's exact shuffle/scale layout. That **materializes the [M·topk, 2048] intermediate to HBM** — exactly the traffic the fused ASM kernel avoids — so any measured delta would reflect that reimplementation + extra HBM round-trip, **not** the GEAK stage-1 optimization. Near-certain regression; not an isolation of the win.

## (2) Exactly what full integration would require
1. A stage-2 for the block-scale path: fp8-quantize the GEAK bf16 intermediate to the layout aiter's `ck_moe_stage2`/`cktile_moe_gemm2` consumes (per-1x128 activation scales, expert-sorted), then down-GEMM (w2 fp8 block scale) + top-k-weight reduction to `[M, 6144]`. Equivalent to re-deriving the second half of the vendor fused kernel.
2. Force GLM onto the 2-stage path and replace `ck_moe_stage1` with the GEAK Triton stage-1 emitting stage-2's exact fp8+scale intermediate contract (currently GEAK emits plain bf16, unquantized) — a fragile layout/interleave match against CK.
3. A fair benchmark would ALSO need to reproduce aiter's fnuz weight + shuffle + block-scale conventions to call the vendor kernel head-to-head — itself non-trivial.
This is a multi-day kernel-integration effort with high regression risk; not appropriate to force in-session.

## (3) Node / salloc
- No SLURM allocation taken and no server launched. amd-arad was fully occupied (3 mix + 1 alloc) at check time; the visibly-idle GPUs on 038 sit under another user's job (ravgupta 211684), so not used for a GPU workload. Inspection was done in a device-less docker container (`moe_inspect`, since removed).

## (4) A/B table — NOT RUN
Deliberately not run: there is no correct, isolated way to put the GEAK stage-1 kernel on the GLM decode critical path without reimplementing stage-2, which would contaminate the reading. Per instructions, stopped before benchmarking rather than produce a misleading result.

## (5) Accuracy — N/A (nothing patched to gate).

## (6) KEEP / REJECT: **REJECT** (for baking into the production image)
- The runtime kernel is a hand-tuned monolithic ASM/CK full-fused MoE; the GEAK win is a Triton stage-1-only kernel with no drop-in seam.
- The reported 9.12x is **GEAK-Triton vs a naive-Triton baseline reconstruction**, NOT vs the vendor ASM/CK kernel it would have to beat — so there is *no evidence* GEAK is faster than the production stage-1 it would replace.
- The microbench used N=2048; the real TP=8 per-GPU decode shape is N=256 — off-target.
- Any integration path materializes the intermediate to HBM (undoing the vendor's stage1↔stage2 fusion) → expected E2E regression, not gain.
- Net: no honest, low-risk path to an E2E gain. Do not bake. (Contrast: the DSA indexer kernel WAS a pure-Triton `.py` in the live path → patchable → KEEP-able.)
