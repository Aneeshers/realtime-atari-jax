#!/bin/bash
#SBATCH --job-name=arppo                   # Default job name (sweep overrides)
#SBATCH -c 2                               # CPU cores per task
#SBATCH -t 0-03:10                         # Runtime (D-HH:MM)
#SBATCH -p seas_gpu                    # Partition
#SBATCH --account=gershman_lab
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --mem=32G                          # RAM for the job
#SBATCH -o slurm-%x-%j.out                 # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e slurm-%x-%j.err                 # STDERR

set -euo pipefail

echo "Job:  ${SLURM_JOB_NAME:-arppo} (${SLURM_JOB_ID:-noid})"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Cores per task: ${SLURM_CPUS_PER_TASK:-2}"

# (Optional) keep CPU threads aligned with -c
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
# Avoid grabbing all GPU memory at start (often helpful with JAX)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Load modules / env
module load python/3.10.9-fasrc01

# If you rely on conda commands:
# source ~/.bashrc || true
# conda activate torch || true

# Use explicit Python path from your torch env (as in your example)
~/.conda/envs/torch/bin/python \
  /n/home04/amuppidi/realtime-atari-jax/examples/minatar-ppo/N_actions_train.py \
  "$@"
