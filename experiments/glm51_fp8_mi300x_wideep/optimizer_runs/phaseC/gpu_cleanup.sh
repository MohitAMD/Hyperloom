#!/bin/bash
set +e
echo "=== VRAM before ==="
rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head -8
echo "=== GPU-holding pids (rocm-smi) ==="
timeout 20 rocm-smi --showpids 2>/dev/null | head -40
echo "=== killing vllm/worker/engine/magpie procs ==="
pkill -9 -f "inference_optimizer.cli"
pkill -9 -f "vllm serve"
pkill -9 -f "EngineCore"
pkill -9 -f "VllmWorker"
pkill -9 -f "multiproc_executor"
pkill -9 -f "from multiprocessing.spawn"
pkill -9 -f "vllm.v1.engine"
pkill -9 -f "\-m Magpie"
pkill -9 -f "pt_main_thread"
# kill any python holding /dev/kfd that isn't ray infra
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  if grep -qE "kfd" /proc/$pid/maps 2>/dev/null; then
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$cmd" in
      *ray/*|*raylet*|*gcs_server*|*dashboard*|*plasma*|*log_monitor*|*runtime_env*) : ;;  # keep ray infra
      *) echo "killing kfd pid $pid: ${cmd:0:80}"; kill -9 "$pid" 2>/dev/null ;;
    esac
  fi
done
sleep 8
echo "=== VRAM after ==="
rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head -8
