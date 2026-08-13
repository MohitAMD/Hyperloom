# GLM-5.1-FP8 on AMD MI300X — Hyperloom optimization campaign

Captured run artifacts and reports from a Hyperloom-driven optimization campaign
for **GLM-5.1-FP8** inference on **AMD MI300X** (WideEP, PD-disaggregated vLLM).

- **Hyperloom base version:** `v1.0.0a1` (release wheel
  `hyperloom_inference_optimizer-1.0.0a1`). The framework code changes on this
  branch are an exact, clean diff against tag `v1.0.0a1` (see the
  `framework:` commit touching `src/hyperloom/...`).
- **Topology:** 2-node 1P1D disaggregated (MoRIIO RDMA, EP), driven from a
  single-node Hyperloom via a wrap-harness adapter over SLURM.

## Key deliverables

### DSA (sparse-attention) indexer kernel — **KEEP**
- `optimizer_runs/dsa_indexer/SUMMARY.md` — full technical report
- `optimizer_runs/dsa_indexer/candidate_seed1/final_patch.diff` — the kernel patch
  (`vllm/v1/attention/ops/triton_fp8_mqa_logits.py`)
- Mechanism: `e4m3fn`→`e4m3fnuz` bitcast to hit native gfx942 fp8 MFMA, host-side
  `0x80` NaN scrub, LDS occupancy-gate fix, fused out-of-window `-inf` epilogue.
- Results: **7.85× geomean isolated** speedup, `0.0` err_ratio; E2E **+4.0% @8k/1k**,
  **+0.3% @32k/8k**; accuracy pass (needle 3/3, gsm8k 11/12).
- Productized as **Recipe 11** (Dockerfile overlay on Recipe 10) — see
  `MohitAMD/MAD` branch `mdeopuja/GLM51_Recipe11_GEAK_dsa_indexer`.

### MoE (g1u1) kernel — **REJECT**
- `optimizer_runs/kernel2_moe_g1u1/E2E_SUMMARY.md`,
  `candidate_moe/{final_patch.diff,tech_lead_report.md}`
- GEAK reaped a **9.12×** isolated speedup, but the runtime MoE path is monolithic
  vendor ASM (`module_moe_fmoe_asm.so`) with no Triton seam to patch — not
  integrable without reimplementing vendor fusion. Kept for reference only.

## Reports
- `optimizer_runs/OVERNIGHT_REPORT.md` — consolidated autonomous overnight log
- `optimizer_runs/geak_cost_shrink_SUMMARY.md` — GEAK benchmark cost-shrink work
- `optimizer_runs/dsa_indexer/SUMMARY.md` — DSA indexer deep-dive

## Framework changes (this branch, vs `v1.0.0a1`)
| File | Purpose |
|---|---|
| `agents/kernel/tools/apply_and_bench.py` | microbench-gate + batched cheap E2E confirm (reps 5→3) |
| `agents/kernel/tools/kernel_optimization.py` | informational `e2e_confirm` annotations |
| `agents/kernel/tools/data/op_to_source.json` | map `vllm::rocm_aiter_sparse_attn_indexer` → Triton `_fp8_mqa_logits_kernel` |
| `agents/kernel/tools/tracelens_analysis.py` | streaming byte-scanner kernel counter (fixes OOM on multi-GB traces) |
| `orchestrator/actions/executors/_workload_envs.py`, `_grid_server_args.py` | preserve operator-baked custom `benchmark_script` (disagg adapter) |
| `agents/kernel/scripts/install.sh` | `/root/.claude` → `${HOME}/.claude` (non-root installs) |

## Excluded from this capture
Large GPU traces (`*.json.gz`) and rocprof counter dumps (`*.dat`) were excluded
(hundreds of MB, not reviewable). Everything else from the run tree is preserved.
