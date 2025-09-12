from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config import GO2RoughCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

class GO2Robot(LeggedRobot):
    """GO2 quadruped robot environment for state-space RL"""
    
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def create_sim(self):
        """Creates simulation with trimesh terrain"""
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = self.cfg.terrain.terrain_kwargs.pop('type', 'plane')
        self.terrain = self.cfg.terrain.terrain_kwargs.pop('type', 'plane')
        self._create_ground_plane()
        self._create_envs()
        
    def _get_observations(self):
        """Compute observations without depth images"""
        base_obs = super()._get_observations()
        return base_obs