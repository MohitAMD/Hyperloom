#!/bin/bash
# Phase C-2 roofline-fix verification (runs INSIDE the Recipe 9 container).
# Proves the streaming-preflight fix: tracelens_analysis on a REAL prior trace
# (521,520 GPU kernel events) now (a) passes the GPU-kernel pre-flight and
# (b) produces a deterministic roofline analysis instead of trace_analyze_failed.
set +e
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/home/mdeopuja/.nvm/versions/node/v22.22.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
git config --global --add safe.directory '*' 2>/dev/null || true
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"

KA_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-$REPO_ROOT/hyperloom/agents/kernel}"
TL_ROOT="${TRACELENS_ROOT:?TRACELENS_ROOT unset}"

# Default probe: the rank0 trace from the prior GPU-blind-misdiagnosed run.
TRACE_IN="${1:-/shared_inference/mdeopuja/Hyperloom/GLM-5.1-FP8/20260811T090449Z/runs/roofline/3df49df2268748c8a242766ecbacc29b/benchmark_vllm_20260811_094117/torch_trace/dp0_pp0_tp0_dcp0_ep0_rank0.1786443329043716575.pt.trace.json.gz}"
WS=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC2/verify_ws
mkdir -p "$WS"
LOG=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC2/verify_roofline_fix.log

echo "=== $(date -u +%FT%TZ) verify roofline fix ===" | tee "$LOG"
echo "KA_ROOT=$KA_ROOT" | tee -a "$LOG"
echo "TL_ROOT=$TL_ROOT" | tee -a "$LOG"
echo "TRACE_IN=$TRACE_IN" | tee -a "$LOG"

python3 "$KA_ROOT/tools/tracelens_analysis.py" \
  --trace-input "$TRACE_IN" \
  --tracelens-root "$TL_ROOT" \
  --session-id verifyfix \
  --workspace-path "$WS" \
  --model-name GLM-5.1-FP8 \
  --framework vllm \
  --target-platform mi300x \
  --split-conc 128 --split-osl 1000 --split-r 1 \
  --steady-state-mode mixed \
  --analysis-route deterministic \
  --top-k 10 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== tracelens_analysis EXIT=$RC ===" | tee -a "$LOG"
echo "--- kernel-agent status/artifacts ---" | tee -a "$LOG"
find "$WS/kernel-agent/runs" -name "*.json" -path "*tracelens*" 2>/dev/null | tail -5 | tee -a "$LOG"
exit $RC
