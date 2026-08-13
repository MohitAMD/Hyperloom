#!/bin/bash
# DSA sparse-MLA indexer kernel-only GEAK launcher for GLM-5.1-FP8.
# Targets the vllm::rocm_aiter_sparse_attn_indexer op (Triton _fp8_mqa_logits_kernel,
# vllm/v1/attention/ops/triton_fp8_mqa_logits.py) newly mapped in op_to_source.json.
# The indexer is a LONG-CONTEXT / prefill kernel whose cost scales with KV length
# (not OSL/concurrency) -> profile it at a CHEAP serving shape.
# Runs INSIDE the Recipe 9 container.  Modeled on phaseD/run_optimize.sh.
# Usage: run_optimize_dsa.sh <tag> <max_hours> [extra args...]
set +e
TAG="${1:?tag}"; MAXH="${2:?max_hours}"; shift 2 || true
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true

# Compute nodes cannot reach statsig.anthropic.com; the bundled Claude Code CLI
# blocks ~180s on a startup statsig feature-gate call then returns "Execution
# error" with no intents -> the Coordinator never seeds and no baseline/roofline/
# GEAK ever dispatches. Null-route statsig so the call fails fast. Idempotent.
grep -qi "statsig.anthropic.com" /etc/hosts 2>/dev/null || printf '127.0.0.1 statsig.anthropic.com\n127.0.0.1 api.statsig.com\n127.0.0.1 events.statsigapi.net\n127.0.0.1 featuregates.org\n127.0.0.1 featureassets.org\n' >> /etc/hosts 2>/dev/null || true
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

# PRIVATE asset root for the dsa_indexer run (isolated from phaseC/phaseD so the
# sibling subagent's parallel run never collides).
export INFERENCE_OPTIMIZER_ASSET_ROOT=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer/asset_root
mkdir -p "$INFERENCE_OPTIMIZER_ASSET_ROOT"

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
# CRITICAL: DSA sparse-indexer fp32 logits buffer cap (64MB). Without it the
# default builds a multi-hundred-MB buffer + grid=(N,) kernel that GPU-faults
# gfx942 on a long prefill -> worker death. MUST stay exported.
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export GPU_MEMORY_UTILIZATION=0.80
# Long-ctx NCCL watchdog: don't tear down DP group on slow sparse-MLA collectives.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Warmup / cold-start caps: the CHEAP shape boots + decodes far faster than 32k,
# but keep generous caps so a slow NFS load can't time out the measured baseline.
export INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_WARM_TIMEOUT_SEC:-7200}"
export INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC="${INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC:-9000}"

# Roofline profile: keep the serialized trace SMALL. 16 iters after a 3000-iter
# warmup delay (splitter steady-state floor is 8).
export HYPERLOOM_PROFILE_MAX_ITERS="${HYPERLOOM_PROFILE_MAX_ITERS:-16}"
export HYPERLOOM_PROFILE_DELAY_ITERS="${HYPERLOOM_PROFILE_DELAY_ITERS:-3000}"

# KERNEL-FOCUSED run: disable the roofline full-gsm8k accuracy gate. On GLM-5.1
# (~53 tok/s aggregate decode) the 1319-problem gsm8k eval takes ~2-3h and by
# itself exhausts the --max-hours budget BEFORE the kernel phase can dispatch.
# _workload_envs.py defaults RUN_EVAL to "true" when unset; force it false so the
# roofline round captures the trace and the coordinator proceeds straight to the
# kernel GEAK phase. Correctness is still validated by the isolated GEAK harness.
export RUN_EVAL=false

# Serve flags: 0.80 headroom, block-size 1 (DSA indexer requires), fp8 KV cache,
# enforce-eager (cheap shape, no cudagraph capture needed).
GLM_SERVER_ARGS="${GLM_SERVER_ARGS:---gpu-memory-utilization 0.80 --block-size 1 --kv-cache-dtype fp8 --enforce-eager}"
cd "$HOME" || cd /

LOGDIR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/dsa_indexer
LOG="$LOGDIR/optimize_${TAG}.log"
echo "=== $(date -u +%FT%TZ) launching DSA-indexer optimize tag=$TAG max_hours=$MAXH ===" | tee "$LOG"
echo "ASSET_ROOT=$INFERENCE_OPTIMIZER_ASSET_ROOT" | tee -a "$LOG"
echo "FRAMEWORK_ROOTS=$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS" | tee -a "$LOG"
echo "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" | tee -a "$LOG"

# CHEAP long-context shape: 8k prefill / 256 decode / conc 16, max-model-len 12288.
# GEAK per-candidate work is microbench (seconds); at most one cheap E2E confirm.
python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework vllm --nodes 1 --gpu-type mi300x \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --tp 8 --isl 8000 --osl 256 --conc 16 --max-model-len 12288 \
  --profile-osl 256 \
  --precision fp8 \
  --claude-model Claude-Opus-5 \
  --kernel-claude \
  --no-explore --no-framework-agent \
  --server-args "$GLM_SERVER_ARGS" \
  --max-hours "$MAXH" \
  "$@" >> "$LOG" 2>&1
echo "=== $(date -u +%FT%TZ) DSA-indexer optimize tag=$TAG EXIT=$? ===" | tee -a "$LOG"
