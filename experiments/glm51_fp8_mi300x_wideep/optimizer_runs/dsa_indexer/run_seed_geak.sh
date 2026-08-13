#!/bin/bash
# Single-op GEAK optimize run for the DSA sparse-MLA indexer, DIRECT on the
# validated harness -- bypasses the whole optimize/trace/serve pipeline.
#
# Drives GEAK's per-kernel Workflow (kernel_workflow.js, mode=optimize) via the
# claude CLI's Workflow tool (mirrors GEAK interface/run_e2e.py _invoke_via_cli,
# but points at the PER-KERNEL workflow, not the e2e whole-pipeline one). The
# kernel_workflow.js optimize path:
#   director(setup: freeze baseline copy) -> tech_lead(analyze) ->
#   benchmark_engineer(build COMMANDMENT from seed_op/config.yaml+test_harness.py)
#   -> profile_engineer(baseline) -> LOOP[ engineer(generate/edit) || verify
#      (compile+correctness+microbench) -> integrate -> commit ] ->
#   tech_lead(report) -> director(validate).
# No server, no weight load, no trace -> per-candidate work is a seconds-scale
# CUDA-event microbench (test_harness.py emits GEAK_RESULT_LATENCY_MS), compared
# to the frozen original triton_fp8_mqa_logits.py (~14.9 ms @ seqlen 8000).
#
# Runs INSIDE the Recipe 9 container. Usage:
#   run_seed_geak.sh <tag> <budget> <timeout_hours> [gpu_ids]
set +e
TAG="${1:?tag}"; BUDGET="${2:-3}"; TIMEOUT_H="${3:-4}"; GPU_IDS="${4:-0}"
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true

# Compute nodes cannot reach statsig; null-route so the claude CLI startup
# feature-gate call fails fast instead of blocking ~180s. Idempotent.
grep -qi "statsig.anthropic.com" /etc/hosts 2>/dev/null || printf '127.0.0.1 statsig.anthropic.com\n127.0.0.1 api.statsig.com\n127.0.0.1 events.statsigapi.net\n127.0.0.1 featuregates.org\n127.0.0.1 featureassets.org\n' >> /etc/hosts 2>/dev/null || true
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1

# Secrets (gateway key) + kernel-agent runtime env (GEAK_ROOT, ANTHROPIC_BASE_URL).
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

# DSA sparse-indexer critical serving env (the kernel reads these). Without the
# 64MB logits cap the default builds a multi-hundred-MB buffer + grid=(N,)
# kernel that GPU-faults gfx942 on a long prefill.
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
# Scrub CUDA_VISIBLE_DEVICES on ROCm (a set value tripped a vllm rocm.py
# logger.warning(scope=) crash in a model-inspection subprocess in prior runs).
unset CUDA_VISIBLE_DEVICES

# GEAK per-kernel Workflow location (from kernel-agent.env.sh GEAK_ROOT).
WF_DIR="$GEAK_ROOT/kernel_workflow"
WF_SCRIPT="$WF_DIR/kernel_workflow.js"
KERNEL_PATH=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/seed_op
EXP_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/geak_kernel_exp
mkdir -p "$EXP_ROOT"

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer
LOG="$LOGDIR/seed_geak_${TAG}.log"
TIMEOUT_S=$(( TIMEOUT_H * 3600 ))

# Workflow primitives are gated behind these settings on public claude builds
# (>=2.1.x reject --effort ultracode; the Workflow/phase/parallel tools live
# behind enableWorkflows+ultracode settings). Mirrors run_e2e.py WORKFLOW_SETTINGS.
WORKFLOW_SETTINGS='{"enableWorkflows": true, "ultracode": true}'

# args passed to the Workflow tool as a REAL JSON OBJECT (not a string).
read -r -d '' ARGS_JSON <<EOF
{"kernel_path": "$KERNEL_PATH", "workflow_dir": "$WF_DIR", "mode": "optimize", "budget": $BUDGET, "gpu_ids": "$GPU_IDS", "exp_root": "$EXP_ROOT", "apply_to_original": "false", "task": "Optimize the Triton _fp8_mqa_logits_kernel and its fp8_mqa_logits_gfx942 launcher in triton_fp8_mqa_logits.py -- the DSA sparse-MLA indexer prefill logits kernel (num_heads=64, head_dim=128, fp8 e4m3fn, long-context prefill; cost scales with KV length). config.yaml + test_harness.py define correctness (pure-torch fp32 oracle, rtol=atol=5e-2) and the CUDA-event benchmark (emits GEAK_RESULT_LATENCY_MS). The speedup denominator is the frozen original kernel (~14.9 ms at seqlen 8000). Prioritize the 8000/8192 long-context cases (BLOCK_KV / num_stages / num_warps / waves_per_eu tiling, LDS pressure, fp8 dot accumulation, grid/masking). Keep correctness within tolerance."}
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
