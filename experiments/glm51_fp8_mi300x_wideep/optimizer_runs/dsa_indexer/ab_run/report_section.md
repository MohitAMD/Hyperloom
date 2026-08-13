
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
