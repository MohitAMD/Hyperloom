#!/bin/bash
# Run a vllm bench serve shape against the local server.
# Usage: bench.sh <isl> <osl> <conc> <num_prompts> <resultjson>
set +e
export HOME=/home/mdeopuja
ISL="${1:?}"; OSL="${2:?}"; CONC="${3:?}"; NP="${4:?}"; OUT="${5:?}"
vllm bench serve \
  --backend vllm \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --max-concurrency "$CONC" \
  --num-prompts "$NP" \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-filename "$OUT" 2>&1
