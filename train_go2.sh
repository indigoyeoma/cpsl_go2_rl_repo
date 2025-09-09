#!/bin/bash

# Set library path
export LD_LIBRARY_PATH=/home/jiwoo/miniforge3/envs/go2_rl/lib

# Use all CPU cores
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)

# Run training
python legged_gym/scripts/train.py --task=go2 --sim_device=cuda:0 --rl_device=cuda:0