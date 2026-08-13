#!/bin/bash
set -uo pipefail
cd /shared_inference/mdeopuja/Hyperloom
source optimizer_runs/session_env.sh

# --- Attach to the persistent 2-node allocation 213701 ----------------------
# The wrap-harness adapter runs `bash run_xPyD_models.slurm`, which resolves the
# nodes via `scontrol show hostnames "$SLURM_JOB_NODELIST"` and launches srun
# steps inside this allocation. Exporting only SLURM_JOB_ID is NOT enough:
# SLURM_JOB_NODELIST / NNODES / NTASKS must be present or the harness fails fast
# (and the adapter would otherwise fall back to a stale log). Populate the full
# set from the live allocation.
export SLURM_JOB_ID=213701
export SLURM_JOBID=213701
export SLURM_JOB_NODELIST="useocpm2m-097-[019,025]"
export SLURM_NODELIST="useocpm2m-097-[019,025]"
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

echo "=== launch env sanity ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODELIST=$SLURM_JOB_NODELIST NNODES=$SLURM_NNODES GPU_TYPE=${GPU_TYPE:-<unset>} CLAUDE_MODEL=${CLAUDE_MODEL:-<unset>}"
squeue -j 213701 || true
echo "=== starting optimize ==="
python3 -m hyperloom.inference_optimizer.cli optimize \
  --framework vllm --nodes 1 \
  --model /shared_inference/models_blog/GLM-5.1-FP8 \
  --isl 8000 --osl 1000 --conc 128 --tp 1 --ep 8 \
  --max-model-len 40960 \
  --no-kernel \
  --no-enable-roofline \
  --target-tput 9957.68 \
  --max-hours 6 \
  --claude-model Claude-Opus-5
echo "=== optimize exited rc=$? ==="
