#!/bin/bash
set -uo pipefail
cd /shared_inference/mdeopuja/Hyperloom
source optimizer_runs/session_env.sh

# --- Attach to the FRESH persistent 2-node allocation 214096 ----------------
# The wrap-harness adapter runs `bash run_xPyD_models.slurm`, which resolves the
# nodes via `scontrol show hostnames "$SLURM_JOB_NODELIST"` and launches srun
# steps inside this allocation. Exporting only SLURM_JOB_ID is NOT enough:
# SLURM_JOB_NODELIST / NNODES / NTASKS must be present or the harness fails fast
# (and the adapter would otherwise fall back to a stale log). Populate the full
# set from the live allocation. (Phase B's 213701 expired mid-run; this is a
# fresh 12h allocation.)
export SLURM_JOB_ID=214096
export SLURM_JOBID=214096
export SLURM_JOB_NODELIST="useocpm2m-097-[077,125]"
export SLURM_NODELIST="useocpm2m-097-[077,125]"
export SLURM_NNODES=2
export SLURM_JOB_NUM_NODES=2
export SLURM_NTASKS=2
export SLURM_NPROCS=2
export SLURM_TASKS_PER_NODE="1(x2)"
export SLURM_JOB_PARTITION="amd-rccl"
export SLURM_JOB_ACCOUNT="amd-rccl"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE:-}"
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST:-$(hostname)}"

unset GPU_TYPE
# Ensure default_baseline_config() resolves the vLLM asset (which bakes the
# hl_benchmark_disagg.sh wrap adapter) rather than the sglang fallback.
export FRAMEWORK=vllm

echo "=== launch env sanity ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODELIST=$SLURM_JOB_NODELIST NNODES=$SLURM_NNODES GPU_TYPE=${GPU_TYPE:-<unset>} CLAUDE_MODEL=${CLAUDE_MODEL:-<unset>}"
squeue -j 214096 || true
echo "=== starting optimize (32k/8k con=128) ==="
python3 -m hyperloom.inference_optimizer.cli optimize \
  --framework vllm --nodes 1 \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --isl 32000 --osl 8000 --conc 128 --tp 1 --ep 8 \
  --max-model-len 49152 \
  --no-kernel \
  --no-enable-roofline \
  --target-tput 382.8 \
  --max-hours 8 \
  --claude-model Claude-Opus-5
echo "=== optimize exited rc=$? ==="
