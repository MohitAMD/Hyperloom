#!/bin/bash
# Single-op GEAK optimize run for the GLM-5.1-FP8 MoE stage-1 GEMM (g1u1), DIRECT
# on the validated harness -- bypasses the whole optimize/trace/serve pipeline.
#
# Drives GEAK's per-kernel Workflow (kernel_workflow.js, mode=optimize) via the
# claude CLI's Workflow tool (mirrors the dsa_indexer run_seed_geak.sh). The
# kernel_workflow.js optimize path:
#   director(setup: freeze baseline copy) -> tech_lead(analyze) ->
#   benchmark_engineer(build COMMANDMENT from seed_op/config.yaml+test_harness.py)
#   -> profile_engineer(baseline) -> LOOP[ engineer(generate/edit) || verify
#      (compile+correctness+microbench) -> integrate -> commit ] ->
#   tech_lead(report) -> director(validate).
# No server, no weight load, no trace -> per-candidate work is a seconds-scale
# CUDA-event microbench (test_harness.py emits GEAK_RESULT_LATENCY_MS), compared
# to the frozen original moe_fp8_blockscale_g1u1.py (~15.05 ms geomean seed).
#
# Runs INSIDE the Recipe 9 container. Usage:
#   run_seed_geak.sh <tag> <budget> <timeout_hours> [gpu_ids]
set +e
TAG="${1:?tag}"; BUDGET="${2:-3}"; TIMEOUT_H="${3:-5}"; GPU_IDS="${4:-0}"
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true

# Compute nodes cannot reach statsig; null-route so the claude CLI startup
# feature-gate call fails fast instead of blocking ~180s. Idempotent.
grep -qi "statsig.anthropic.com" /etc/hosts 2>/dev/null || printf '127.0.0.1 statsig.anthropic.com\n127.0.0.1 api.statsig.com\n127.0.0.1 events.statsigapi.net\n127.0.0.1 featuregates.org\n127.0.0.1 featureassets.org\n' >> /etc/hosts 2>/dev/null || true
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1
# The Workflow tool runs the per-kernel pipeline as a background task. The claude
# CLI otherwise terminates background tasks after 600s ("Background tasks still
# running after 600s; terminating"), killing the run ~10 min in. 0 = wait
# indefinitely (until our outer `timeout` fires).
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

# Secrets (gateway key) + kernel-agent runtime env (GEAK_ROOT, ANTHROPIC_BASE_URL).
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

# Scrub CUDA_VISIBLE_DEVICES on ROCm (a set value tripped a vllm rocm.py
# logger.warning(scope=) crash in a model-inspection subprocess in prior runs).
unset CUDA_VISIBLE_DEVICES

# GEAK per-kernel Workflow location (from kernel-agent.env.sh GEAK_ROOT).
WF_DIR="$GEAK_ROOT/kernel_workflow"
WF_SCRIPT="$WF_DIR/kernel_workflow.js"
KERNEL_PATH=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/seed_op
EXP_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1/geak_kernel_exp
mkdir -p "$EXP_ROOT"

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/kernel2_moe_g1u1
LOG="$LOGDIR/seed_geak_${TAG}.log"
TIMEOUT_S=$(( TIMEOUT_H * 3600 ))

WORKFLOW_SETTINGS='{"enableWorkflows": true, "ultracode": true}'

read -r -d '' ARGS_JSON <<EOF
{"kernel_path": "$KERNEL_PATH", "workflow_dir": "$WF_DIR", "mode": "optimize", "budget": $BUDGET, "gpu_ids": "$GPU_IDS", "exp_root": "$EXP_ROOT", "apply_to_original": "false", "task": "Optimize the Triton _moe_g1u1_fp8_kernel and its moe_g1u1_fp8 launcher in moe_fp8_blockscale_g1u1.py -- the GLM-5.1-FP8 MoE stage-1 expert GEMM (aiter::fmoe_fp8_blockscale_g1u1, g1u1 = fused gate+up projections with a SiLU(gate)*up epilogue). It is an expert-sorted, block-padded grouped FP8 (e4m3) block-scaled GEMM: activations x are [num_tokens*top_k, K] fp8 with per-token per-128-K-group scales; per-expert weights W1 are [E, 2N, K] fp8 with [128x128] block scales; FP32 accumulation. GLM-5.1-FP8 shapes: K(hidden)=6144, N(moe_intermediate)=2048, E=256 experts, top_k=8, fp8 block=[128,128]. The DOMINANT/priority target is the DECODE regime: num_tokens=16/32/64 (cases 0-2), where each active expert sees only ~1-2 tokens, so BLOCK_M=16 block-padding wastes most MFMA lanes -- this is exactly why the vendor kernel only reaches ~26% of roofline. Tuning levers to prioritize: (1) shrink/split BLOCK_M or restructure the grid so ~1-token experts do not pay full-tile padding; (2) BLOCK_N / BLOCK_K / GROUP_M tiling, num_warps, num_stages, waves_per_eu; (3) fuse the gate/up loads and the SiLU*mul epilogue efficiently (single K-loop, avoid redundant re-reads of x); (4) fp8 tl.dot accumulation width and LDS pressure. config.yaml + test_harness.py define correctness (pure-torch fp32 dequant oracle, rtol=atol=5e-2, err_ratio<=0.05) and the CUDA-event benchmark (emits GEAK_RESULT_LATENCY_MS, geomean over cases). The speedup denominator is the frozen original kernel (~15.05 ms geomean; decode case-0 ~10.35 ms). Keep correctness within tolerance for ALL cases. Do NOT change the numerics/semantics (still silu(gate)*up, no router weight in stage-1)."}
EOF

PROMPT="You are a headless GPU-kernel optimization driver. Invoke the Workflow tool EXACTLY ONCE with:
  scriptPath: \"$WF_SCRIPT\"
  args: $ARGS_JSON
CRITICAL: pass \`args\` as a real JSON OBJECT (a mapping), NOT a JSON-encoded string; do not wrap it in quotes or json.dumps it, or the workflow cannot read args.workflow_dir and aborts. This is the single-language per-kernel optimize pipeline (Setup -> Analyze -> Benchmark -> Profile -> optimize LOOP -> Report -> Director validate); it runs on ONE GPU with no server. Wait for it to finish, then print EXACTLY ONE final line of compact JSON that is the Workflow tool's full return value (it includes eval_dir, final_geomean, final_patch, validation_status). Do not do any GPU work yourself; recovering a wedged port or failed Workflow call is NOT your job -- just report."

echo "=== $(date -u +%FT%TZ) launching seed GEAK kernel_workflow tag=$TAG budget=$BUDGET timeout=${TIMEOUT_H}h gpu=$GPU_IDS ===" | tee "$LOG"
echo "WF_SCRIPT=$WF_SCRIPT" | tee -a "$LOG"
echo "KERNEL_PATH=$KERNEL_PATH" | tee -a "$LOG"
echo "EXP_ROOT=$EXP_ROOT" | tee -a "$LOG"
echo "GEAK_ROOT=$GEAK_ROOT  MODEL=${GEAK_CLAUDE_MODEL:-Claude-Opus-5}" | tee -a "$LOG"
echo "ARGS_JSON=$ARGS_JSON" | tee -a "$LOG"

cd "$WF_DIR" || cd "$HOME"
IS_SANDBOX=1 timeout "$TIMEOUT_S" claude -p "$PROMPT" \
  --output-format json \
  --settings "$WORKFLOW_SETTINGS" \
  --model "${GEAK_CLAUDE_MODEL:-Claude-Opus-5}" \
  --allowed-tools Workflow,Bash,Read,Write \
  --permission-mode auto \
  >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) seed GEAK tag=$TAG EXIT=$? ===" | tee -a "$LOG"
