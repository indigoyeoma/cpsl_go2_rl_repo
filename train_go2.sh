#!/bin/bash

# Go2 Training Script for Terrain Steps
# This script trains the Go2 robot only on terrain_steps (parkour_step)

export LD_LIBRARY_PATH=/home/nvidiasims/miniconda3/envs/go2gym/lib
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)


FOLDER="parkour_new_v3"

cd /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym/legged_gym/scripts

# Resume Go2 teacher training from checkpoint (9k -> 20k, ~11k more iterations)
python train.py --task go2 --exptid go2_teacher --proj_name parkour_new_v3 --headless --no_wandb --max_iterations 20005 

# Train student after teacher completes
python train.py --task go2_student --exptid go2_student --proj_name parkour_new_v3 --load_run ../../logs/parkour_new_v3/go2_teacher --use_camera --headless --no_wandb --max_iterations 15002

# python train.py --task go2_student_ghn --exptid go2_student_ghn --no_wandb --headless --use_camera --num_envs 512 --load_run /home/nvidiasims/ws_go2/cpsl_go2_rl_repo/legged_gym/logs/parkour_new/go2_teacher
