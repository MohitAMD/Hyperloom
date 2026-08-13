#!/bin/bash
# Phase C-2 native single-node optimize launcher (runs INSIDE the Recipe 9 container).
# Full stack: Kernel Agent + GEAK + framework patching + roofline (roofline now FIXED).
# Usage: run_optimize.sh <tag> <max_hours> [extra args...]
set +e
TAG="${1:?tag}"; MAXH="${2:?max_hours}"; shift 2 || true
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

# Phase C-2 PRIVATE asset root (native single-node baseline_vllm.yaml -> vllm_mi300x.sh).
# Isolated copy so the 213855 session collector is never touched.
export INFERENCE_OPTIMIZER_ASSET_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC2/asset_root

# Pin framework patch/source roots to the runtime packages (+ /app/vllm git
# tree), NOT /opt/rocm. The CLI preflight auto-re-probes and would otherwise
# reset kernel-agent.env.sh to include /opt/rocm; export AFTER sourcing so this
# wins for the framework agent.
export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=/usr/local/lib/python3.12/dist-packages/vllm/:/usr/local/lib/python3.12/dist-packages/aiter/:/usr/local/lib/python3.12/dist-packages/aiter_meta/:/app/vllm

# --- GLM-5.1-FP8 recipe env (Recipe 9 models.yaml) for native colocated TP=8 ---
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MLA=1
# DSA sparse-indexer OOM mitigation (cap fp32 logits buffer at 64MB).
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export GPU_MEMORY_UTILIZATION=0.80
# Long-ctx NCCL watchdog: don't tear down DP group on slow sparse-MLA collectives.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Roofline capture-stability fix (Phase C-2) ---
# The stock roofline captures 128 profiled iterations WITH python stacks
# (torch_profiler_with_stack) -> ~15.9GB/rank / 55M events. On this node's NFS
# that took >15min to serialize; the 8 workers block on the gzip export, go
# unresponsive, and the EngineCore watchdog trips (shm_broadcast dequeue
# TimeoutError -> EngineDeadError) and kills them MID-WRITE, leaving truncated
# traces (all ranks ~1.07GB, "unexpected end of file"). tracelens then sees a
# corrupt trace. Capture a much smaller steady-state window so the trace
# serializes in ~2min and the engine survives; keep the deep delay so the
# window is still steady-state decode (splitter floor=8; 16>=8 keeps mixed/
# decode/prefilldecode windows). Also makes each roofline cycle cheaper ->
# more GEAK budget.
export HYPERLOOM_PROFILE_MAX_ITERS=16
export HYPERLOOM_PROFILE_DELAY_ITERS=3000

# Native serve flags. full = PIECEWISE decode cudagraphs (recipe win); smoke = eager.
if [ "$TAG" = "full" ]; then
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --compilation-config {\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
else
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --enforce-eager}"
fi
cd "$HOME" || cd /

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC2
LOG="$LOGDIR/optimize_${TAG}.log"
echo "=== $(date -u +%FT%TZ) launching optimize tag=$TAG max_hours=$MAXH ===" | tee "$LOG"
echo "ASSET_ROOT=$INFERENCE_OPTIMIZER_ASSET_ROOT" | tee -a "$LOG"
echo "USER_DATA_PATH=$USER_DATA_PATH  FRAMEWORK_ROOTS=$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS" | tee -a "$LOG"
echo "GLM_SERVER_ARGS=$GLM_SERVER_ARGS" | tee -a "$LOG"

python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework vllm --nodes 1 --gpu-type mi300x \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --tp 8 --isl 8000 --osl 1000 --conc 128 --max-model-len 40960 \
  --precision fp8 \
  --claude-model Claude-Opus-5 \
  --server-args "$GLM_SERVER_ARGS" \
  --max-hours "$MAXH" \
  "$@" >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) optimize tag=$TAG EXIT=$? ===" | tee -a "$LOG"
