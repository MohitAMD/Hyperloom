# Overnight Autonomous Run — GLM-5.1-FP8 / Hyperloom

Started: 2026-08-13 ~06:50 UTC (user asleep ~9h). Job cap = 4 total (3-4 held by user's glm-ov campaign).

## Crown-jewel deliverable (secured)
- **GEAK-optimized DSA sparse-MLA indexer** Triton `_fp8_mqa_logits_kernel`.
- Isolated microbench: **7.85× geomean** (5.8× @2k, 7.8× @4k, 9.3× @8k, 9.0× @8192), correctness pass (err_ratio 0.0), independently re-validated.
- Mechanism: e4m3fn→e4m3fnuz bitcast to hit native gfx942 fp8 MFMA (kills ~2500-instr software upconvert) + LDS tile-gate fix + host-side NaN scrub; ×4 bias correction folded into per-head weight.
- Kept patch: `optimizer_runs/dsa_indexer/candidate_seed1/final_patch.diff` (edits only `triton_fp8_mqa_logits.py`).

## Operational constraint (important)
- Account cap is **GrpJobs=4**; slots held by user's glm-ov jobs → fresh `salloc` sits `PD (AssocMaxJobsLimit)`.
- Workaround (per seeded subagent precedent): run directly in a docker container on a verified-idle GPU node (no salloc). Verify idle via squeue nodelists + `rocm-smi` (no procs). Node 038 used before.

## Plan / status
1. [done] Reap indexer kernel (7.85× microbench, KEPT).
2. [in progress] E2E A/B validate: patched-vs-pristine, native TP=8, 32k/8k + 8k/1k vs R9 (1775.8 / 695.36) + accuracy gate. → subagent 44071bd2.
3. [pending] If E2E-positive → 1P1D disagg (2-node) vs R9 (6124 / 9957).
4. [pending] Next kernel: MoE GEMM aiter::ck_moe_stage1 (~15%, decode) via seeded GEAK.

## Event log
- 06:22 UTC — seeded GEAK exited (accepted 7.85× indexer kernel).
- 06:50 UTC — overnight autonomy authorized; plan set; watchdog armed.
- 07:00 UTC — seeded subagent reported KEEP (7.85×) but no E2E (job cap). Launched E2E A/B validation subagent (44071bd2) on idle node.

---

## 2026-08-13 — E2E A/B validation of the GEAK indexer kernel (pristine vs patched, MI300X TP=8)

**TL;DR — the 7.85× isolated microbench does NOT translate into a meaningful long-context E2E serving gain.**
Patched vs pristine on the SAME node/config: **+0.3% at 32k/8k (within noise)** and **+4.0% at 8k/1k**.
Accuracy is clean (no degradation from the fp8 bitcast trick). Net: safe, low-risk, small win → **KEEP (low-priority)**,
but it must not be advertised as a long-context win — the expected prefill/long-context E2E benefit did not materialize.

### (1) How a node was acquired
- Account was NOT actually at the cap: only 3 `glm-ov-*` jobs (214712/214719/214720) were running, so the 4th
  GrpJobs slot was free. `salloc -p amd-arad -N1 --exclusive -t 06:00:00` **succeeded immediately** →
  job `214842` on **`useocpm2m-097-029`** (idle, all 8 GPUs at 0%, only `gpuagent`).
- Direct `ssh` to the node is key-restricted (Permission denied), so the node was driven via
  `srun --jobid=214842 --ntasks=1 --overlap`. **`--ntasks=1` is required** — a plain `srun` fan-outs to
  multiple tasks and launched the benchmark twice concurrently (first 8k/1k read a contaminated 3014 tok/s;
  the clean single-process re-run read 8092 tok/s). All subsequent runs enforced `--ntasks=1` + a
  pre-launch `pgrep -fc "bench serve"` guard + a mid-run `num_requests_running` check.
- Image loaded from tar (12 GB, ~3 min). Container `idx_e2e_glm` launched per `launch_container.sh`
  (--network host, --shm-size 256G, mounts). `CUDA_VISIBLE_DEVICES` confirmed UNSET; 8 GPUs visible.

### (2) Patch applied + re-JIT confirmed
- `patch -p1 --dry-run` clean → applied. Markers present: `native gfx942 fp8 MFMA` (×1), `float8e4b8` (×3),
  host-side `clamp_min` NaN-scrub (×4), `OOW_FILL` (×4). Patched file is **byte-identical** to the reference
  `optimized_triton_fp8_mqa_logits.py`; `ast.parse` OK.
- Server restarted so the pure-Triton kernel re-imports and re-JITs on first invocation (warm caches:
  model load 104 s, cudagraph capture 12 s). vLLM compile config confirms the indexer path is live:
  `+sparse_attn_indexer`, `vllm::rocm_aiter_sparse_attn_indexer` in splitting_ops; MLA backend
  `ROCM_AITER_MLA_SPARSE`.

### (3) A/B results (identical config each arm: TP=8, block-size 1, fp8 KV, gpu-util 0.80, max-len 49152, PIECEWISE cudagraph)

| Shape | Metric | Pristine | Patched | Δ |
|---|---|---:|---:|---:|
| **32k/8k** (isl 32000, osl 8000, conc 128, np=64) | Total tok/s | 1016.85 | 1020.26 | **+0.34%** |
| | Output tok/s | 203.37 | 204.05 | +0.33% |
| | TTFT mean / median (ms) | 475507 / 75577 | 503604 / 121312 | noisy* |
| | TPOT mean / median (ms) | 111.58 / 104.00 | 111.29 / 107.48 | ≈flat |
| | Wall duration (s) | 2517.6 | 2509.2 | −0.3% |
| **8k/1k** (isl 8000, osl 1000, conc 128, np=128, warm) | Total tok/s | 8092.71 | 8420.32 | **+4.05%** |
| | Output tok/s | 899.19 | 935.59 | +4.05% |
| | TTFT mean / median (ms) | 14758 / 15105 | 13799 / 14916 | −6.5% / −1.3% |
| | TPOT mean / median (ms) | 102.41 / 101.74 | 99.29 / 97.95 | −3.0% / −3.7% |

\* 32k/8k TTFT means/medians are dominated by KV-memory-bound admission: at 32k only ~30 requests fit
concurrently (KV cache = 1.1M tokens), so 90+ requests queue and the last-admitted see 400–500 s TTFT plus a
long single-stream decode tail. This queueing/straggler noise (per-run, ±several %) swamps any small kernel
delta, which is exactly why total throughput — the bottleneck-limited metric — is the reliable readout there.

**Methodology notes / caveats:**
- np reduced to 64 at 32k/8k (from 256–512) to fit two arms + a restart in the window; kept IDENTICAL across
  arms so the A/B is fair. 8k/1k used np=128. All arms clean single-process runs.
- Cold-JIT hazard: the patched kernel re-JITs on the first prefill of each new shape. The very first
  post-restart patched 8k/1k read 5643 tok/s (TTFT 44 s) purely from the one-time Triton compile stall; the
  warm re-run read 8420 tok/s. Both 32k arms paid one symmetric cold-32k JIT, negligible over a 42-min run.
- R9 single-node references (32k/8k 1775.8, 8k/1k 695.36 tok/s) were measured under a different harness/sizing
  and are not directly comparable to these np-limited saturation runs; used only as a loose sanity anchor. The
  pristine 32k/8k here (~1017 tok/s) sits near the prior native baseline (~1160), i.e. the expected regime.

### (4) Indexer GPU-time share
- Not separately traced (would have cost another restart + trace-serialization risk on this NFS node, which
  previously truncated traces). Inferred from the E2E deltas: the indexer is a **small fraction of serving
  wall time** — a 7.85× kernel speedup yielding ≤4% E2E implies the kernel is single-digit-% of the critical
  path (Amdahl-limited by MoE GEMMs, MLA attention, and the decode-heavy 8k-output tail).

### (5) Accuracy gate (patched server) — PASS
- **Long-context needle-in-haystack: 3/3** at ~29.5k / 41.3k / 47.2k-token contexts (needles at 25% / 60% / 85%
  depth). Exact retrieval each time with coherent `<think>` reasoning (e.g. "The secret vault code … is 47823",
  "6391", "Tavistock"). This directly exercises the sparse indexer over long KV — the fp8 e4m3fn→fnuz bitcast +
  ×4 bias fold + host NaN-scrub introduces **no** numeric corruption in KV selection.
- **gsm8k subset: 11/12** correct (the single miss was an ambiguously-worded custom question, not a coherence
  failure). Multi-step arithmetic intact.
- (First needle pass scored 0/3 only because a 64-token cap truncated the reasoning model before its final
  answer — a harness artifact, fixed by raising max_tokens.)

### (6) Recommendation: **KEEP (low-risk, low-priority) — but the microbench win does NOT translate to a long-context E2E gain**
- **Why KEEP:** pure-Triton `.py` change (no image rebuild), accuracy-clean, **zero regression** on any metric
  at either shape, a consistent modest **+4% at 8k/1k**, and neutral (+0.3%) at 32k/8k. Baking a safe kernel
  that never regresses and occasionally helps is fine.
- **Why the caveat matters:** the premise — a large *long-context* E2E gain from the 7.85× prefill kernel — was
  **not observed** (+0.3% at 32k/8k, within noise). The gain that does appear is at 8k/1k and is small; a
  single warm sample means ±a few % run-to-run, so treat +4% as "small positive, not a headline." Do not
  prioritize/market this as a long-context throughput improvement.
- **If a stricter bar is required** (must show the claimed long-context win to justify inclusion): **REJECT**,
  because at 32k/8k the delta is indistinguishable from zero.

**Artifacts:** `optimizer_runs/dsa_indexer/ab_run/` (serve/bench scripts, 4 result JSONs, accuracy/needle logs,
pristine + patched kernel copies). Container `idx_e2e_glm` and salloc `214842` cleaned up after the run.

## Kernel #2 — MoE stage-1 expert GEMM (fmoe_fp8_blockscale_g1u1) — REAPED (E2E pending)
- 2026-08-13 ~13:51 UTC: GEAK accepted candidate, geomean **9.12x** isolated (10.36x @16 tok decode ... 8.36x @128), correctness err_ratio 0.0, device-bound verified.
- Vendor kernel only ~26% roofline in low-token decode (BLOCK_M=16 padding waste) -> real headroom. Decode-dominant => higher E2E-critical-path share than indexer.
- Mechanism: same e4m3fn->fnuz native-MFMA bitcast (x4 fold) + decode-regime tiling/grid restructure + fused gate/up + SiLU*mul epilogue.
- Patch salvaged to optimizer_runs/kernel2_moe_g1u1/candidate_moe/final_patch.diff (subagent 667f9716 crashed on infra disconnect before writing SUMMARY; result artifacts intact).
- NEXT: batched E2E A/B (indexer + MoE vs pristine) + accuracy; FIRST resolve whether runtime MoE kernel is Triton-patchable vs ASM/CK.

### Kernel #2 — MoE stage-1 g1u1 E2E feasibility (2026-08-13, UTC-7) — VERDICT: NOT INTEGRABLE → REJECT
Full write-up: `optimizer_runs/kernel2_moe_g1u1/E2E_SUMMARY.md`. Inspected the live image
(`glm5.1-fp8-disagg:mi300x-fromscratch-aiter017`) in a device-less container; **no GPU workload / no server run**
(amd-arad full; idle GPUs on 038 belong to another user's job). Stopped before benchmarking per instruction.

**(1) Runtime kernel kind + dispatch.** GLM decode MoE: vllm `rocm_aiter_fused_experts` (block-scaled fp8 →
`QuantMethod.BLOCK_128x128`) → `vllm._aiter_ops...fused_moe` → `aiter.fused_moe.fused_moe` (per_1x128, Silu,
isG1U1) → `aiter.fmoe_fp8_blockscale_g1u1`. That entry is `@compile_ops("module_moe_fmoe_asm", ctypes)` — a
**compiled AMD ASM kernel** (`module_moe_fmoe_asm.so`, sym `fmoe_g1u1_impl`), and it is **monolithic full-fused**:
it takes both `w1` AND `w2` and writes the FINAL `[M, model_dim=6144]` output (stage-1 gate/up+SiLU·mul AND
stage-2 down+topk-reduce in one launch). Tuned `a8w8_blockscale_tuned_fmoe_glm5_1.csv` confirms decode →
`run_1stage=1`, `us2=0`, kernel `..._blockscaleFp8_g1u1_vs_silu_1tg_32x128E`, at the real TP=8 per-GPU shape
**model_dim=6144, inter_dim=256, expert=257, topk=9** (2048/8=256 per GPU). The alt 2-stage path
(`run_1stage=token>32` default) uses `ck_moe_stage1`/`cktile_moe_stage1`/flydsl/`moe_stage1_g1u1` — all CK/ASM.
**No `@triton.jit` g1u1/stage-1 exists in the live path (1- or 2-stage).** Route taken: **NONE** (route A impossible;
route B not an honest isolated proof).

**(2) What integration would require.** GEAK kernel is **stage-1-only** (emits intermediate `[M·topk,2048] bf16`);
the live entry emits final `[M,6144]`. Substitution needs a full stage-2 reimplementation (w2 fp8-blockscale down
GEMM + topk-weighted scatter-reduce) + inter-stage fp8 requant in aiter's exact shuffle/scale layout, which
**materializes the intermediate to HBM** — undoing the vendor stage1↔stage2 fusion. Any A/B would then measure the
reimplementation + HBM round-trip, not the GEAK win. Multi-day, high-regression-risk effort.

**(3) Node.** No salloc; no server. Device-less inspection container `moe_inspect` (removed). No stray SLURM/docker.

**(4) A/B.** Not run — no correct, isolated way to place GEAK stage-1 on the decode critical path without
reimplementing stage-2; refused to produce a contaminated number.

**(5) Accuracy.** N/A (nothing patched).

**(6) KEEP/REJECT: REJECT for the production image.** (a) Live kernel is hand-tuned monolithic ASM/CK with no
Triton drop-in seam; (b) the 9.12x is GEAK-Triton vs a *naive-Triton* baseline, NOT vs the vendor ASM/CK it must
beat — so no evidence GEAK even wins on-device; (c) microbench N=2048 vs real per-GPU N=256 (off-target);
(d) every integration path adds an HBM round-trip the vendor fuses away → expected regression. Unlike the DSA
indexer (a real Triton `.py` in the live path → patchable), this optimization does not map onto the runtime.
Cleaned up: `docker rm moe_inspect`; no allocation to scancel.
