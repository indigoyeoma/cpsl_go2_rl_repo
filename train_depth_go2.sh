#!/bin/bash

# Set library path
export LD_LIBRARY_PATH=/home/nvidiasims/miniconda3/envs/go2rl_edge/lib

# Use all CPU cores
export OMP_NUM_THREADS=$(nproc)
export MAX_JOBS=$(nproc)

# Run depth-based visual RL training
# Uses:
# - Task: depth_go2 (GO2DepthRobot with GO2DepthCfg)  
# - Architecture: State(48→256) + Depth(84x84→256) = 512 → [512,512,256]
# - Single-frame synchronous depth processing
# - VisualActorCritic with SimpleCNN depth encoder + MLP state encoder
python legged_gym/scripts/train.py --task=depth_go2 --sim_device=cuda:0 --rl_device=cuda:0 