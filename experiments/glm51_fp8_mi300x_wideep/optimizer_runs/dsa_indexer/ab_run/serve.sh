#!/bin/bash
# Launch GLM-5.1-FP8 vLLM server (Recipe-9 flags) inside the container.
# Usage: serve.sh <logfile>
set +e
LOG="${1:?logfile}"
export HOME=/home/mdeopuja
unset CUDA_VISIBLE_DEVICES
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export GPU_MEMORY_UTILIZATION=0.80
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

vllm serve /shared_inference/models_blog/GLM-5.1-FP8 \
  --tensor-parallel-size 8 \
  --block-size 1 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 49152 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}' \
  --host 0.0.0.0 --port 8000 \
  >> "$LOG" 2>&1
