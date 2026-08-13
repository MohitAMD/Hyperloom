#!/bin/bash
# Phase D native single-node optimize launcher for GLM-5.1-FP8 @ 32k/8k (TP=8).
# Runs INSIDE the Recipe 9 container. GEAK-first budgeting: roofline ON,
# EXPLORE + FRAMEWORK cut so the Kernel-agent gets the whole post-baseline wall.
# Usage: run_optimize.sh <tag> <max_hours> [extra args...]
set +e
TAG="${1:?tag}"; MAXH="${2:?max_hours}"; shift 2 || true
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true
# This node (151) cannot reach statsig.anthropic.com; the bundled Claude Code
# CLI blocks ~180s on a startup statsig feature-gate call then returns
# "Execution error" with no intents -> the Coordinator never seeds and no
# baseline/roofline/GEAK ever dispatches. Null-route statsig so the call fails
# fast and the CLI proceeds. Idempotent.
grep -qi "statsig.anthropic.com" /etc/hosts 2>/dev/null || printf '127.0.0.1 statsig.anthropic.com\n127.0.0.1 api.statsig.com\n127.0.0.1 events.statsigapi.net\n127.0.0.1 featuregates.org\n127.0.0.1 featureassets.org\n' >> /etc/hosts 2>/dev/null || true
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

# Phase D PRIVATE asset root (native single-node baseline_vllm.yaml -> local
# vllm serve via vllm_mi300x.sh). Isolated from phaseC / phaseC2 so parallel
# runs never collide.
export INFERENCE_OPTIMIZER_ASSET_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD/asset_root

# GEAK / kernel-agent overlays against the RUNTIME framework install (NOT
# /opt/rocm): the vllm + aiter python packages the live serve actually imports.
_DIST=/usr/local/lib/python3.12/dist-packages
_ROOTS=""
for d in "$_DIST/vllm" "$_DIST/aiter" "$_DIST/aiter_meta" /app/vllm; do
  [ -e "$d" ] && _ROOTS="${_ROOTS:+$_ROOTS:}$d"
done
export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS="$_ROOTS"

# --- GLM-5.1-FP8 recipe env (Recipe 9) for native colocated TP=8 ---
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MLA=1
# DSA sparse-indexer fp32 logits buffer cap (64MB): the default builds a
# multi-hundred-MB buffer + grid=(N,) kernel that GPU-faults gfx942 on a long
# prefill -> worker death. At 32k prefill this matters even more than 8k.
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export GPU_MEMORY_UTILIZATION=0.80
# Long-ctx NCCL watchdog: don't tear down DP group on slow sparse-MLA collectives.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Long-context benchmark wall: 32k/8k boot + decode legitimately exceeds
# the default 130/150 min baseline caps; raise so the warmup round can't time
# out and skip the measured baseline + roofline. GEAK microbench validations
# reuse the hot server so they stay well inside this. ---
export INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC:-14400}"
export INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC:-16200}"

# --- Roofline profile: keep the serialized trace SMALL (critical at 32k) ---
# ROOT CAUSE #2 (confirmed by parallel run): the stock 128-iter WITH-python-stack
# capture is ~16 GB/rank at 8k/1k (LARGER at 32k/8k); serializing 8x that over
# this node's NFS took >15 min, tripping the EngineCore watchdog
# (EngineDeadError / shm_broadcast dequeue TimeoutError) which killed the
# workers mid-write and TRUNCATED all 8 traces. Shrink the captured window hard:
# 16 iters (splitter steady-state floor is 8) after a 3000-iter warmup delay.
export HYPERLOOM_PROFILE_MAX_ITERS="${HYPERLOOM_PROFILE_MAX_ITERS:-16}"
export HYPERLOOM_PROFILE_DELAY_ITERS="${HYPERLOOM_PROFILE_DELAY_ITERS:-3000}"

# Serve flags (full mode): 0.80 headroom, block-size 1 (DSA indexer requires),
# fp8 KV cache, PIECEWISE decode cudagraphs (prefill full-capture deadlocks).
if [ "$TAG" = "full" ]; then
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --compilation-config {\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
else
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --enforce-eager}"
fi
cd "$HOME" || cd /

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseD
LOG="$LOGDIR/optimize_${TAG}.log"
echo "=== $(date -u +%FT%TZ) launching optimize tag=$TAG max_hours=$MAXH ===" | tee "$LOG"
echo "ASSET_ROOT=$INFERENCE_OPTIMIZER_ASSET_ROOT" | tee -a "$LOG"
echo "FRAMEWORK_ROOTS=$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS" | tee -a "$LOG"

python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework vllm --nodes 1 --gpu-type mi300x \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --tp 8 --isl 32000 --osl 8000 --conc 128 --max-model-len 49152 \
  --profile-osl 512 \
  --precision fp8 \
  --claude-model Claude-Opus-5 \
  --kernel-claude \
  --no-explore --no-framework-agent \
  --server-args "$GLM_SERVER_ARGS" \
  --max-hours "$MAXH" \
  "$@" >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) optimize tag=$TAG EXIT=$? ===" | tee -a "$LOG"
