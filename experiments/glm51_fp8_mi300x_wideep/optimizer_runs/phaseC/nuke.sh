#!/bin/bash
# Full teardown of all Phase C optimize/specialist/vllm processes so the GPUs
# return to a clean state. Keeps the Ray head infra alive.
set +e
echo "=== killing launchers/orchestrators ==="
pkill -9 -f "run_optimize.sh"
pkill -9 -f "inference_optimizer.cli"
pkill -9 -f "hyperloom.orchestrator"
echo "=== killing framework/kernel specialists (claude worktree agents) ==="
pkill -9 -f "runs/specialist/"
pkill -9 -f "shell-snapshots/snapshot-bash"
pkill -9 -f "/home/mdeopuja/.local/share/claude/versions"
pkill -9 -f "claude --model"
pkill -9 -f "\.claude/.*claude"
echo "=== killing all vllm servers/workers ==="
pkill -9 -f "vllm serve"
pkill -9 -f "VLLM::"
pkill -9 -f "EngineCore"
pkill -9 -f "VllmWorker"
pkill -9 -f "multiproc_executor"
pkill -9 -f "vllm.v1.engine"
pkill -9 -f "\-m Magpie"
sleep 3
# Kill anything still mapping /dev/kfd except ray infra
for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  if grep -qE "kfd" /proc/$pid/maps 2>/dev/null; then
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$cmd" in
      *ray/*|*raylet*|*gcs_server*|*dashboard*|*plasma*|*log_monitor*|*runtime_env*|*ray::IDLE*) : ;;
      *) echo "kfd-kill $pid: ${cmd:0:70}"; kill -9 "$pid" 2>/dev/null ;;
    esac
  fi
done
sleep 6
echo "=== clearing stale aiter JIT baton/ninja locks (dead-pid deadlock guard) ==="
# A crashed worker can leave lock_module_* holding a dead PID + an incomplete
# ninja build; the persistent /opt/vllm_cache preserves it and the next boot's
# workers wait forever. Remove baton locks + ninja locks; drop any module build
# dir that has no final .so (incomplete) so it recompiles cleanly.
find /opt/vllm_cache/aiter_jit/build -maxdepth 1 -name 'lock_module_*' -delete 2>/dev/null
find /opt/vllm_cache/aiter_jit/build -name '.ninja_lock' -delete 2>/dev/null
for d in /opt/vllm_cache/aiter_jit/build/module_*; do
  [ -d "$d" ] || continue
  if ! ls "$d"/*.so >/dev/null 2>&1; then
    echo "  dropping incomplete build: $(basename "$d")"
    rm -rf "$d"
  fi
done
sleep 2
echo "=== VRAM after nuke ==="
rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head -8
echo "=== remaining optimize/vllm/specialist procs (should be none) ==="
pgrep -af "inference_optimizer.cli|vllm serve|VLLM::|runs/specialist" | grep -v pgrep | head
