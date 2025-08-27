import os
import numpy as np
from datetime import datetime
import sys

# Fix camera buffer creation issues before importing IsaacGym
#MESA_VK_DEVICE_SELECT=list vulkaninfo
#use this code to get the gpu
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:2231'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import cv2

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
