#!/bin/bash
# Script to run MinAtar PPO training with local pgx library

# Set the path to the local pgx library
export PYTHONPATH=/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax:$PYTHONPATH

# Default to offline mode for wandb (remove this line to sync with wandb)
export WANDB_MODE=offline

# Run the training script with all arguments passed through
# Example usage:
#   ./run_train.sh env_name=minatar-breakout total_timesteps=20000000
#   ./run_train.sh env_name=minatar-freeway num_envs=1024
#   ./run_train.sh env_name=minatar-space_invaders seed=42

python3 train.py "$@"