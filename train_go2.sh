#!/bin/bash

# Go2 Training Script for Terrain Steps
# This script trains the Go2 robot only on terrain_steps (parkour_step)

export LD_LIBRARY_PATH=/home/nvidiasims/miniconda3/envs/go2gym/lib
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)

cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym/legged_gym/scripts

# Train Go2 on terrain steps
python train.py --task go2 --exptid go2_teacher --headless --no_wandb --max_iterations 20002

  python train.py --task go2_student --exptid go2_student_001 --load_run ../../logs/parkour_new/go2_teacher_001 --use_camera --headless --no_wandb --max_iterations 15002