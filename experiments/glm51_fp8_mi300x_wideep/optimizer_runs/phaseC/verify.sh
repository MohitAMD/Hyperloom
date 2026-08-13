#!/bin/bash
set +e
export HOME=/home/mdeopuja
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export PATH="/home/mdeopuja/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
echo "=== imports ==="
python3 - <<'PY'
mods=["vllm","magpie","torch","ray","tracelens","geak","claude_agent_sdk"]
import importlib
for m in mods:
    try:
        mod=importlib.import_module(m)
        print(f"OK  {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"ERR {m}: {type(e).__name__}: {e}")
import hyperloom, os
print("hyperloom", hyperloom.__file__)
PY
echo "=== magpie scripts dir ==="
ls "$MAGPIE_PATH/Magpie/scripts/benchmark/vllm_mi300x.sh" 2>&1 | head -1
echo "=== optimize --help (first line) ==="
python3 -m hyperloom.inference_optimizer.cli optimize --help 2>&1 | head -3
echo "=== claude CLI ==="
"$GEAK_CLAUDE_BIN" --version 2>&1 | head -1
echo "=== TraceLens CLI ==="
which TraceLens_generate_perf_report_pytorch_inference 2>&1 | head -1
echo "=== framework roots ==="
echo "$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS"
echo "=== GEAK e2e runner ==="
ls "$GEAK_E2E_RUNNER" 2>&1 | head -1
echo "DONE"
