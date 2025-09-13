#!/bin/bash

# Set library path
export LD_LIBRARY_PATH=/home/nvidiasims/miniconda3/envs/go2rl_edge/lib

# Use all CPU cores
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)

# Run training
python legged_gym/scripts/train.py --task=hyper_go2 --sim_device=cuda:0 --rl_device=cuda:0 # --headless