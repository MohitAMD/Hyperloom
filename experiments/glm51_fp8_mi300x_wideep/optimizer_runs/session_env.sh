#!/bin/bash
# Source this in any shell that drives Hyperloom for the GLM-5.1-FP8 1p1d wrap.
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH="$REPO_ROOT"
export PYTHON="$(command -v python3)"
# node/claude for the forge kernel backend live in the user tree, not system PATH
export PATH="$HOME/.local/bin:$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
# gateway creds + model (Claude-Opus-5)
set -a; . "$REPO_ROOT/.env"; set +a
# generated runtime env (TraceLens/GEAK/InferenceX paths, Ray, gateway aliases)
. "$REPO_ROOT/runtime/kernel-agent.env.sh" 2>/dev/null || true
