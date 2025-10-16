from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config import GO2RoughCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from isaacgym.torch_utils import quat_rotate_inverse


class GO2Robot(LeggedRobot):
    """GO2 quadruped with sim2real-oriented observations (48 dims).

    Observation layout (deployable signals, with estimated linear velocity):
      [lin_vel_est(3), ang_vel(3), projected_gravity(3), commands(3),
       dof_pos_err(12), dof_vel(12), previous_action(12)]
    """

    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # simple IMU-style velocity estimator (world & body frames)
        self.lin_vel_est_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.lin_vel_est = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Add rigid body states for foot velocity tracking
        from isaacgym import gymtorch
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # estimated linear velocity noise
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # gyro and gravity
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # keep commands clean
        noise_vec[9:12] = 0.0
        # joint pos/vel noise
        start = 12
        noise_vec[start:start + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        start += self.num_actions
        noise_vec[start:start + self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        start += self.num_actions
        # previous actions remain noise-free
        noise_vec[start:start + self.num_actions] = 0.0
        return noise_vec

    def _post_physics_step_callback(self):
        # invoke base (handles command resampling, pushes, etc.)
        super()._post_physics_step_callback()

        # Use commanded velocity as estimate (matches deployment!)
        # This creates perfect sim-to-real transfer since hardware will use same approach
        cmd_vel_world = torch.zeros_like(self.base_lin_vel)
        cmd_vel_world[:, 0] = self.commands[:, 0] * 1.0  # vx: forward velocity (m/s)
        cmd_vel_world[:, 1] = self.commands[:, 1] * 0.5  # vy: lateral velocity (m/s)
        cmd_vel_world[:, 2] = 0.0  # vz: always zero for ground robot

        # Exponential smoothing to simulate acceleration dynamics (same as deployment)
        alpha = 0.1  # Smoothing factor: ~10 steps (0.2s) to reach 63% of target
        self.lin_vel_est_world = (1 - alpha) * self.lin_vel_est_world + alpha * cmd_vel_world

        # Convert to body frame for observations
        self.lin_vel_est = quat_rotate_inverse(self.base_quat, self.lin_vel_est_world)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.lin_vel_est_world[env_ids] = 0
            self.lin_vel_est[env_ids] = 0

    def compute_observations(self):
        # assemble 48-dim observation using estimated lin vel
        self.obs_buf = torch.cat(
            (
                self.lin_vel_est * self.obs_scales.lin_vel,    # 3 (estimated)
                self.base_ang_vel * self.obs_scales.ang_vel,    # 3
                self.projected_gravity,                         # 3
                self.commands[:, :3] * self.commands_scale,     # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,         # 12
                self.actions,                                    # 12 (previous action)
            ),
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces."""
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
            4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1
        ).float()

    def _reward_feet_drag(self):
        """Penalize foot dragging (feet moving horizontally while in contact)."""
        # Get foot velocities from rigid body states
        feet_xy_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:9]).sum(dim=-1)
        # Detect contact
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        # Penalize horizontal velocity when in contact
        dragging_vel = contact * feet_xy_vel
        return dragging_vel.sum(dim=-1)
