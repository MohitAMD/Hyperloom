#!/bin/bash
# Phase D RESUME launcher: continue an existing GLM-5.1-FP8 32k/8k session with
# the corrected roofline profile-window env (MAX_ITERS=16 / DELAY=3000). The
# landed baseline in state.json is preserved; PRELUDE re-enqueues roofline and
# proceeds to GEAK. Usage: run_resume.sh <session_dir> <max_hours>
set +e
SESSION_DIR="${1:?session_dir}"; MAXH="${2:?max_hours}"; shift 2 || true
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true
grep -qi "statsig.anthropic.com" /etc/hosts 2>/dev/null || printf '127.0.0.1 statsig.anthropic.com\n127.0.0.1 api.statsig.com\n127.0.0.1 events.statsigapi.net\n127.0.0.1 featuregates.org\n127.0.0.1 featureassets.org\n' >> /etc/hosts 2>/dev/null || true
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"
export INFERENCE_OPTIMIZER_ASSET_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD/asset_root
_DIST=/usr/local/lib/python3.12/dist-packages
_ROOTS=""
for d in "$_DIST/vllm" "$_DIST/aiter" "$_DIST/aiter_meta" /app/vllm; do
  [ -e "$d" ] && _ROOTS="${_ROOTS:+$_ROOTS:}$d"
done
export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS="$_ROOTS"
export VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_ROCM_USE_AITER_MLA=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 GPU_MEMORY_UTILIZATION=0.80
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC:-14400}"
export INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC:-16200}"
# ROOT CAUSE #2 fix: tiny profiled window so the trace serializes fast (no NFS
# watchdog truncation). 16 captured iters after 3000 warmup iters.
export HYPERLOOM_PROFILE_MAX_ITERS="${HYPERLOOM_PROFILE_MAX_ITERS:-16}"
export HYPERLOOM_PROFILE_DELAY_ITERS="${HYPERLOOM_PROFILE_DELAY_ITERS:-3000}"
export GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --compilation-config {\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
cd "$HOME" || cd /

LOG=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD/optimize_resume.log
echo "=== $(date -u +%FT%TZ) RESUME session=$SESSION_DIR max_hours=$MAXH MAX_ITERS=$HYPERLOOM_PROFILE_MAX_ITERS DELAY=$HYPERLOOM_PROFILE_DELAY_ITERS ===" | tee "$LOG"

python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --resume-from "$SESSION_DIR" \
  --kernel-claude \
  --no-explore --no-framework-agent \
  --profile-osl 512 \
  --claude-model Claude-Opus-5 \
  --server-args "$GLM_SERVER_ARGS" \
  --max-hours "$MAXH" \
  "$@" >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) resume EXIT=$? ===" | tee -a "$LOG"
