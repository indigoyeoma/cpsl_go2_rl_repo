from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config import GO2RoughCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

class GO2Robot(LeggedRobot):
    """GO2 quadruped robot environment for state-space RL"""
    
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)