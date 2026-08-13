#!/bin/bash
# Idempotently patch the container-local Magpie vllm_mi300x.sh so it does NOT
# derive HIP_VISIBLE_DEVICES from ROCR (Hyperloom convention = ROCR-only). The
# HIP->CUDA propagation in vllm serve otherwise makes the arch-inspection child
# hit vLLM rocm.py's early `scope=` logger bug (crash before model load).
set +e
F=/usr/local/lib/python3.12/dist-packages/Magpie/scripts/benchmark/vllm_mi300x.sh
python3 - "$F" <<'PY'
import io, sys
f = sys.argv[1]
src = open(f).read()
marker = "PHASE_C_ROCR_ONLY"
if marker in src:
    print("already patched"); sys.exit(0)
old = (
'if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then\n'
'    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, \'{print NF}\')\n'
'    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))\n'
'fi\n'
)
new = (
'# ' + marker + ': Hyperloom convention is ROCR-only. Do NOT set HIP here;\n'
'# vllm serve would copy HIP->CUDA_VISIBLE_DEVICES and the arch-inspection\n'
'# child would trip vLLM rocm.py _sync_hip_cuda_env_vars scope= logger bug.\n'
'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES\n'
)
if old not in src:
    print("WARN: expected block not found; inserting unset after shebang as fallback")
    # Fallback: force-unset near the top (after set -x is later; put before serve)
    src2 = src.replace('export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}',
                       'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES  # ' + marker + '\nexport VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}', 1)
    if src2 == src:
        print("ERROR: fallback anchor not found either"); sys.exit(2)
    src = src2
else:
    src = src.replace(old, new, 1)
open(f, "w").write(src)
print("patched OK")
PY
# Also make the hardcoded --gpu-memory-utilization honor $GPU_MEMORY_UTILIZATION
python3 - "$F" <<'PY'
import sys
f=sys.argv[1]; s=open(f).read()
old="--gpu-memory-utilization 0.95 \\"
new="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.9} \\"
if old in s and "GPU_MEMORY_UTILIZATION:-" not in s:
    s=s.replace(old,new); open(f,"w").write(s); print("gpu-mem-util now env-driven")
elif "GPU_MEMORY_UTILIZATION:-" in s:
    print("gpu-mem-util already env-driven")
else:
    print("WARN: gpu-mem-util anchor not found")
PY
echo "--- verify ---"
grep -nE "PHASE_C_ROCR_ONLY|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|gpu-memory-utilization" "$F" | head
