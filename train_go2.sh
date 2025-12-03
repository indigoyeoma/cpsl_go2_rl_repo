#!/bin/bash

# Go2 Training Script for Terrain Steps
# This script trains the Go2 robot only on terrain_steps (parkour_step)

export LD_LIBRARY_PATH=/home/nvidiasims/miniconda3/envs/go2gym/lib
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)

cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym/legged_gym/scripts

# Train Go2 on terrain steps
python train.py \
    --task go2 \
    --exptid go2-01-terrain_steps \
    --device cuda:0 \
    --num_envs 4096 \
    --max_iterations 15000 \
    --proj_name go2_parkour

echo "Training completed! Logs saved in legged_gym/logs/go2_parkour/go2-01-terrain_steps"
