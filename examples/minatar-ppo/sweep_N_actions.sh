#!/usr/bin/env bash
set -euo pipefail

# Where to put slurm logs (must exist before sbatch opens files)
LOGDIR="/n/home04/amuppidi/realtime-atari-jax/slurm_logs"
mkdir -p "${LOGDIR}"

RUN_SCRIPT="/n/home04/amuppidi/realtime-atari-jax/examples/minatar-ppo/run_N_actions.sh"

ENVS=("minatar-freeway" "minatar-breakout")
for ENV in "${ENVS[@]}"; do
  for FS in 1 2 3 4 5 6; do
    JOB_NAME="${ENV}-fs${FS}"
    echo "Submitting: ${JOB_NAME}"
    sbatch \
      --job-name="${JOB_NAME}" \
      --output="${LOGDIR}/%x-%j.out" \
      --error="${LOGDIR}/%x-%j.err" \
      "${RUN_SCRIPT}" \
        --env_name "${ENV}" \
        --frame_skip "${FS}"
    sleep 0.5
  done
done
