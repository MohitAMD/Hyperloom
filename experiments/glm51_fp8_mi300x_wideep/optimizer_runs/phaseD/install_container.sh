#!/bin/bash
# Phase C: install Hyperloom/Magpie/GEAK/TraceLens toolchain into the
# Recipe 9 container's Python 3.12 (system /usr/bin/python3, which owns vllm).
set +e
export REPO_ROOT=/shared_inference/mdeopuja/Hyperloom
export USER_DATA_PATH=/shared_inference/mdeopuja/Hyperloom
export HOME=/home/mdeopuja
export PATH="/home/mdeopuja/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Container runs as root but the shared-FS repos are owned by the user; allow git.
git config --global --add safe.directory '*' 2>/dev/null || true
# Do NOT let the packaged extras pull a non-ROCm torch/vllm/aiter from PyPI.
# Pin them to the already-installed in-image versions via a constraints file.
CONSTR=/shared_inference/mdeopuja/Hyperloom/optimizer_runs/phaseC/pip_constraints.txt
python3 - <<'PY' > "$CONSTR"
import importlib.metadata as m
for pkg in ("torch","vllm","aiter","torchvision","torchaudio","triton","pytorch-triton-rocm","flash_attn"):
    try:
        v=m.version(pkg); print(f"{pkg}=={v}")
    except Exception:
        pass
PY
echo "=== constraints ==="; cat "$CONSTR"
export PIP_CONSTRAINT="$CONSTR"

echo "=== python that will be used ==="; command -v python3; python3 --version
echo "=== running master install.sh ==="
bash /shared_inference/mdeopuja/Hyperloom/hyperloom/inference_optimizer/assets/install.sh 2>&1
echo "=== INSTALL_EXIT=$? ==="
