#!/bin/bash
# Phase C native single-node optimize launcher (runs INSIDE the Recipe 9 container).
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
# Phase C native single-node: use a PRIVATE asset root whose baseline_vllm.yaml
# launches a LOCAL vllm serve (vllm_mi300x.sh) instead of the Phase B 2-node
# MoRIIO disagg wrap. Isolated from the shared assets so Phase B (213701) is
# untouched.
export INFERENCE_OPTIMIZER_ASSET_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC/asset_root

# --- GLM-5.1-FP8 recipe env (Recipe 9 models.yaml) for native colocated TP=8 ---
# DSA sparse-MLA + MoE needs these to boot without OOM / GPU-fault:
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MLA=1
# Cap the DSA indexer fp32 logits buffer (64MB) — the default lets an 8k prefill
# build a 268MB buffer + grid=(8192,) kernel that GPU-faults gfx942 -> worker death.
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
# 0.80 headroom (the minimal native serve OOMed at the hardcoded 0.95).
export GPU_MEMORY_UTILIZATION=0.80
# Long-ctx NCCL watchdog: don't tear down DP group on slow sparse-MLA collectives.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Serve flags: block-size 1 (DSA indexer requires it) + fp8 KV cache. cudagraph:
# smoke = enforce-eager (guaranteed boot, no capture deadlock); full = PIECEWISE
# decode graphs (recipe: prefill full-capture deadlocks, decode PIECEWISE is a win).
# NOTE: vllm_mi300x.sh hardcodes --gpu-memory-utilization 0.95; we append 0.80
# here so argparse (last-wins) uses the recipe's 0.80 headroom.
if [ "$TAG" = "full" ]; then
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --compilation-config {\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
else
  GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --enforce-eager}"
fi
cd "$HOME" || cd /

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC
LOG="$LOGDIR/optimize_${TAG}.log"
echo "=== $(date -u +%FT%TZ) launching optimize tag=$TAG max_hours=$MAXH ===" | tee "$LOG"
echo "USER_DATA_PATH=$USER_DATA_PATH  FRAMEWORK_ROOTS=$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS" | tee -a "$LOG"

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
