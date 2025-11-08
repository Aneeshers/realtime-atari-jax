#!/usr/bin/env bash
set -euo pipefail

# Where to put slurm logs (must exist before sbatch opens files)
LOGDIR="/n/home04/amuppidi/realtime-atari-jax/slurm_logs"
mkdir -p "${LOGDIR}"

RUN_SCRIPT="/n/home04/amuppidi/realtime-atari-jax/examples/minatar-ppo/run.sh"

ENVS=("minatar-freeway" "minatar-breakout")
for ENV in "${ENVS[@]}"; do
  for N in 1 2 3 4 5 6; do
    JOB_NAME="${ENV}-N${N}"
    echo "Submitting: ${JOB_NAME}"
    sbatch \
      --job-name="${JOB_NAME}" \
      --output="${LOGDIR}/%x-%j.out" \
      --error="${LOGDIR}/%x-%j.err" \
      "${RUN_SCRIPT}" \
        --env_name "${ENV}" \
        --plan_horizon "${N}"
    sleep 0.5
  done
done
