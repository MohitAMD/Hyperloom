#!/bin/bash
# Container-side probe for Phase C
set +e
export PYTHONPATH=/shared_inference/mdeopuja/Hyperloom
echo "=== python ==="; python3 --version
echo "=== hyperloom core import ==="
python3 -c 'import hyperloom; print("hyperloom", hyperloom.__file__)'
echo "=== CLI help top ==="
python3 -m hyperloom.inference_optimizer.cli --help 2>&1 | head -25
