#!/bin/bash
set +e
REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
TEST='from vllm.platforms import current_platform; print("OK", current_platform.get_device_name())'

run() { echo "--- $1 ---"; shift; env -i "$@" /usr/bin/python3.12 -c "$TEST" 2>&1 | tail -4; echo; }

run "clean" PATH=/usr/local/bin:/usr/bin HOME=/root
run "PYTHONPATH=REPO" PATH=/usr/local/bin:/usr/bin HOME=/root PYTHONPATH=$REPO_ROOT

echo "=== full run env (source session+kernel env) ==="
export HOME=/home/mdeopuja
export PATH="/home/mdeopuja/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null
export PYTHONPATH="$REPO_ROOT"
/usr/bin/python3.12 -c "$TEST" 2>&1 | tail -6
echo
echo "=== dump VLLM_/PYTHON/LOG related env from full ==="
env | grep -iE "VLLM|PYTHONPATH|PYTHONSTARTUP|LOGGING|LOG_" | sort
