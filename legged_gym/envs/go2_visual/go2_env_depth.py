import math
import threading

import numpy as np
import torch
import torchvision.transforms
from isaacgym import gymapi, gymtorch

from legged_gym.envs.go2.go2_env import GO2Robot
from .go2_config_depth import GO2DepthCfg


class GO2DepthRobot(GO2Robot):
    """GO2 quadruped with depth camera for visual RL.

    Extends base GO2Robot with depth camera processing for visual observations.
    Uses flat plane terrain from base GO2 configuration.

    Observation layout (48 state dims + 7056 depth dims = 7104 total):
      State: [lin_vel_est(3), ang_vel(3), projected_gravity(3), commands(3),
              dof_pos_err(12), dof_vel(12), previous_action(12)]
      Depth: [84x84 depth image flattened = 7056]
    """

    def __init__(self, cfg: GO2DepthCfg, sim_params, physics_engine, sim_device, headless):
        # Camera/image holders and flags (set before parent ctor so we can init buffers afterwards)
        self.check_camera = False
        self.depth_image = None
        self.rgb_image = None
        self.raw_depth_image = None
        self.all_rgb_images = {}
        self.all_depth_images = {}
        self.cam_handles = []
        self.show_camera = False
        self.camera_window_name = "GO2 Depth Camera Feed"
        self.display_thread = None
        self.stop_display = threading.Event()
        self.camera_env_id = 0
        self.show_all_cameras = False

        if cfg.depth.use_camera:
            self.resize_transform = torchvision.transforms.Resize(
                (cfg.depth.resized[1], cfg.depth.resized[0]),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            print(f"Depth processing: {cfg.depth.original} -> {cfg.depth.resized}")

        # Parent constructor creates sim/envs and base buffers
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Depth buffer per env (single-frame buffer_len=1 by default)
        if cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(
                self.num_envs, cfg.depth.buffer_len, cfg.depth.resized[1], cfg.depth.resized[0]
            ).to(self.device)
            self.global_counter = 0

    def _get_noise_scale_vec(self, cfg):
        """Override to create noise only for 48 state dims, not full 7104 obs."""
        noise_vec = torch.zeros(48, device=self.device)

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

    def create_sim(self):
        """Override to add camera setup after environment creation."""
        # Call parent to create sim and environments
        super().create_sim()

        # Add cameras if configured
        if self.cfg.depth.use_camera:
            self._setup_cameras()

    def _setup_cameras(self):
        """Set up one depth camera per environment and attach to robot."""
        print(f"Setting up {self.num_envs} depth cameras...")
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.depth.original[0]
        camera_props.height = self.cfg.depth.original[1]
        camera_props.horizontal_fov = self.cfg.depth.horizontal_fov
        camera_props.near_plane = self.cfg.depth.near_clip
        camera_props.far_plane = self.cfg.depth.far_clip
        camera_props.enable_tensors = True
        # Speed: no supersampling; render collision meshes (sufficient for depth)
        try:
            camera_props.supersampling_horizontal = 1
            camera_props.supersampling_vertical = 1
            camera_props.use_collision_geometry = True
        except Exception:
            pass

        for i in range(self.num_envs):
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            cam_pos = gymapi.Vec3(*self.cfg.depth.position)

            # Pitch rotation around Y axis (per-env randomized if range provided)
            pitch = self.cfg.depth.angle[0]
            if hasattr(self.cfg.depth, 'angle_pitch_range_deg') and self.cfg.depth.angle_pitch_range_deg is not None:
                pr = self.cfg.depth.angle_pitch_range_deg
                pitch = math.radians(np.random.uniform(pr[0], pr[1]))
            quat = gymapi.Quat.from_euler_zyx(0, pitch, 0)

            # Attach to Head_upper if available, otherwise base
            body_handle = self.gym.find_actor_rigid_body_handle(
                self.envs[i], self.actor_handles[i], "Head_upper"
            )
            attachment_point = "Head_upper"

            self.gym.attach_camera_to_body(
                cam_handle,
                self.envs[i],
                body_handle,
                gymapi.Transform(cam_pos, quat),
                gymapi.FOLLOW_TRANSFORM,
            )

            self.cam_handles.append(cam_handle)
            if i == 0:
                self.camera_attachment = attachment_point

        print(f"Camera setup complete: {len(self.cam_handles)} cameras initialized")
        print(
            f"Camera config - FOV: {self.cfg.depth.horizontal_fov}, Near: {self.cfg.depth.near_clip}, Far: {self.cfg.depth.far_clip}"
        )
        print(f"Camera position: {self.cfg.depth.position}, angle: {self.cfg.depth.angle}")
        print(f"Camera tilt: {self.cfg.depth.angle[0]} rad ({self.cfg.depth.angle[0]*57.3:.1f} degrees)")
        print(f"Camera attached to: {getattr(self, 'camera_attachment', 'unknown')}")

    def _update_depth_buffer(self):
        """Render and store latest depth for each env into `self.depth_buffer` (optimized GPU processing)."""
        if not hasattr(self, "cam_handles") or len(self.cam_handles) == 0:
            return

        # Render all cameras once
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # Collect per-env depth into a list to allow cropping then batch-resize
        depth_list = []
        cl = getattr(self.cfg.depth, 'crop_left', 0)
        cr = getattr(self.cfg.depth, 'crop_right', 0)
        ct = getattr(self.cfg.depth, 'crop_top', 0)
        cb = getattr(self.cfg.depth, 'crop_bottom', 0)
        for i in range(self.num_envs):
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
            )
            if depth_tensor is None:
                continue
            depth_img = gymtorch.wrap_tensor(depth_tensor)
            # Replace invalid with far clip and keep negative sign
            depth_img = torch.where(torch.isfinite(depth_img), depth_img, -self.cfg.depth.far_clip)
            # Crop
            if ct or cb or cl or cr:
                h, w = depth_img.shape[-2], depth_img.shape[-1]
                top = ct
                bottom = h - cb if cb > 0 else h
                left = cl
                right = w - cr if cr > 0 else w
                depth_img = depth_img[top:bottom, left:right]
            depth_list.append(depth_img)

        self.gym.end_access_image_tensors(self.sim)

        if len(depth_list) == 0:
            return

        # Stack and resize to 84x84 (batch on GPU)
        depth_stack = torch.stack(depth_list, dim=0)  # [N, Hc, Wc]
        depth_stack = depth_stack.unsqueeze(1)        # [N, 1, Hc, Wc]
        depth_resized = torch.nn.functional.interpolate(
            depth_stack, size=(84, 84), mode='bilinear', align_corners=False
        ).squeeze(1)  # [N, 84, 84]

        # Add Gaussian noise if configured (meters, still negative)
        if self.cfg.depth.dis_noise > 0:
            depth_resized = depth_resized + torch.randn_like(depth_resized) * self.cfg.depth.dis_noise
        # Clamp to valid negative range
        depth_resized = depth_resized.clamp(-self.cfg.depth.far_clip, -self.cfg.depth.near_clip)

        # Normalize to [-0.5, 0.5]
        depth_m = -depth_resized
        depth_norm = (depth_m - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip) - 0.5

        # Store in buffer
        self.depth_buffer[:, 0] = depth_norm

    def compute_observations(self):
        """Override to append depth observations to base state observations."""
        # Build 48-dim state obs using estimated lin vel (same as parent but without noise yet)
        state_obs = torch.cat(
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

        # Add noise only to state observations
        if self.add_noise:
            state_obs += (2 * torch.rand_like(state_obs) - 1) * self.noise_scale_vec

        # Add depth observations if camera is enabled
        if self.cfg.depth.use_camera:
            # Decimated depth updates: render only on intervals, reuse last frame otherwise
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self._update_depth_buffer()
            current_depth = self.depth_buffer[:, 0, :, :].flatten(start_dim=1)
            self.obs_buf = torch.cat([state_obs, current_depth], dim=1)
            self.global_counter += 1
        else:
            self.obs_buf = state_obs
