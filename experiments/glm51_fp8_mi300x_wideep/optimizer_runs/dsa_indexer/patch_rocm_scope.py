#!/usr/bin/env python3
"""Container-local workaround for an upstream vLLM logging bug on this image.

vllm/platforms/rocm.py::_sync_hip_cuda_env_vars() calls logger.warning(..., scope="process")
but this build's logger.warning() does not accept a `scope=` kwarg (only warning_once does),
so it raises `TypeError: Logger._log() got an unexpected keyword argument 'scope'` whenever
CUDA_VISIBLE_DEVICES is set. This crashes the fresh model-inspection subprocess vLLM spawns
during `vllm serve` startup (which inherits a non-empty CUDA_VISIBLE_DEVICES), aborting the
baseline server boot. We drop the unsupported kwarg. Idempotent.
"""
import io
import sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py"
src = io.open(F, encoding="utf-8").read()
needle = '\n            scope="process",'
n = src.count(needle)
out = src.replace(needle, "")
io.open(F, "w", encoding="utf-8").write(out)
print("removed_occurrences=%d changed=%s" % (n, src != out))
