import math
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, torch_rand_float, to_torch, quat_apply

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config_obsavoid_depth import GO2ObsAvoidDepthCfg
from .go2_terrain_obsavoid import build_obsavoid_heightfield, heightfield_to_trimesh


class GO2ObsAvoidDepthRobot(LeggedRobot):
    """GO2 quadruped with depth camera for obstacle avoidance.

    Student observation layout (48 base state + 7056 depth = 7104 dims):
      State: [lin_vel_est(3), ang_vel(3), projected_gravity(3), commands(3),
              dof_pos_err(12), dof_vel(12), previous_action(12)]
      Depth: [84x84 depth image flattened = 7056]

    Privileged teacher observation additionally appends 49 terrain height samples.

    Obstacles are static geometry (not actors) - they never move or need resetting.
    """

    def __init__(self, cfg: GO2ObsAvoidDepthCfg, sim_params, physics_engine, sim_device, headless):
        # Height sampling configuration (needed before parent constructor)
        self.height_sample_points = None
        x_cnt = len(getattr(cfg.terrain, "measured_points_x", []))
        y_cnt = len(getattr(cfg.terrain, "measured_points_y", []))
        self.num_height_samples = x_cnt * y_cnt if getattr(cfg.terrain, "measure_heights", False) else 0
        self.extra_goal_obs_dim = 5
        self.goal_metrics = None
        self.head_body_index = None
        self.goal_indicator_geom = None

        # Camera/image holders and flags (set before parent ctor so we can init buffers afterwards)
        self.check_camera = False
        self.depth_image = None
        self.rgb_image = None
        self.raw_depth_image = None
        self.all_rgb_images = {}
        self.all_depth_images = {}
        self.cam_handles = []
        self.show_camera = False
        self.camera_window_name = "GO2 All RGB Cameras Feed"
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
        hip_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        self.hip_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_names if name in self.dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.last_torques_buffer = torch.zeros_like(self.torques)

        if getattr(self.cfg.terrain, "measure_heights", False) and self.num_height_samples > 0:
            self._init_height_measurement_points()

        # Depth buffer per env (single-frame buffer_len=1 by default)
        if cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(
                self.num_envs, cfg.depth.buffer_len, cfg.depth.resized[1], cfg.depth.resized[0]
            ).to(self.device)
            self.global_counter = 0

        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.speed_targets = torch.zeros(self.num_envs, device=self.device)
        self.goal_forward_min = float(self.cfg.goal.forward_range[0])
        self.goal_forward_max = float(self.cfg.goal.forward_range[1])
        self.goal_lateral_min = float(self.cfg.goal.lateral_range[0])
        self.goal_lateral_max = float(self.cfg.goal.lateral_range[1])
        self.goal_reach_eps = float(getattr(self.cfg.goal, 'reach_epsilon', 0.5))
        self.forward_ref = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.head_forward_ref = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        if not headless:
            if hasattr(gymutil, "WireframeSphereGeometry"):
                self.goal_indicator_geom = gymutil.WireframeSphereGeometry(0.25, 12, 12, None, color=(1.0, 0.0, 0.0))
            elif hasattr(gymutil, "AxesGeometry"):
                self.goal_indicator_geom = gymutil.AxesGeometry(0.25)
        self.add_noise = self.cfg.noise.add_noise

    def _init_buffers(self):
        """Initialize buffers for single-actor environments."""
        super()._init_buffers()

        # Velocity estimator buffers (specific to this class)
        self.lin_vel_est_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.lin_vel_est = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Rigid body states for foot velocity tracking
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)


    def _get_noise_scale_vec(self, cfg):
        """Define noise scaling for state observations (depth remains clean)."""
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        device = self.device

        components = [
            torch.full((3,), noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel, device=device),
            torch.full((3,), noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel, device=device),
            torch.full((3,), noise_scales.gravity * noise_level, device=device),
            torch.zeros(3, device=device),  # commands remain noise-free
            torch.full((self.num_actions,), noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos, device=device),
            torch.full((self.num_actions,), noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel, device=device),
            torch.zeros(self.num_actions, device=device),  # previous actions noise-free
        ]

        base_noise = torch.cat(components)

        if getattr(self.cfg.terrain, "measure_heights", False) and self.num_height_samples > 0:
            base_noise = torch.cat([base_noise, torch.zeros(self.num_height_samples, device=device)])

        if self.cfg.depth.use_camera:
            depth_h, depth_w = self.cfg.depth.resized
            depth_len = depth_h * depth_w * self.cfg.depth.buffer_len
            base_noise = torch.cat([base_noise, torch.zeros(depth_len, device=device)])

        return base_noise

    def _update_goal_commands(self, metrics, env_ids=None):
        """Encode goal tracking into command space (vx, vy, yaw rate)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if metrics is None:
            self.commands[env_ids, :3] = 0.0
            if self.cfg.commands.heading_command and self.commands.shape[1] > 3:
                self.commands[env_ids, 3] = 0.0
            return

        dir_body = metrics["dir_body"][env_ids]
        speed_targets = self.speed_targets[env_ids].unsqueeze(1)
        dir_body_xy = dir_body[:, :2]

        # Guard against division-by-zero for coincident goals
        valid = torch.norm(dir_body_xy, dim=1, keepdim=True) > 1e-6
        cmd_xy = torch.zeros_like(dir_body_xy)
        cmd_xy[valid.squeeze(1)] = speed_targets[valid.squeeze(1)] * dir_body_xy[valid.squeeze(1)]

        self.commands[env_ids, 0] = cmd_xy[:, 0]
        self.commands[env_ids, 1] = cmd_xy[:, 1]

        heading_error = torch.zeros(env_ids.numel(), device=self.device)
        heading_error[valid.squeeze(1)] = torch.atan2(dir_body_xy[valid.squeeze(1), 1], dir_body_xy[valid.squeeze(1), 0])
        self.commands[env_ids, 2] = torch.clamp(heading_error, -1.0, 1.0)

        if self.cfg.commands.heading_command and self.commands.shape[1] > 3:
            desired_heading = torch.zeros(env_ids.numel(), device=self.device)
            dir_world = metrics["dir_world"][env_ids]
            desired_heading[valid.squeeze(1)] = torch.atan2(dir_world[valid.squeeze(1), 1], dir_world[valid.squeeze(1), 0])
            self.commands[env_ids, 3] = desired_heading

    def _update_velocity_estimate(self):
        """Update linear velocity estimate using commanded motion (sim-to-real parity)."""
        cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_body[:, 0] = self.commands[:, 0]
        cmd_body[:, 1] = self.commands[:, 1]
        cmd_vel_world = quat_apply(self.base_quat, cmd_body)
        cmd_vel_world[:, 1] *= 0.5  # match deployment lateral dynamics scaling

        alpha = 0.1
        self.lin_vel_est_world = (1 - alpha) * self.lin_vel_est_world + alpha * cmd_vel_world
        self.lin_vel_est = quat_rotate_inverse(self.base_quat, self.lin_vel_est_world)

    def _post_physics_step_callback(self):
        """Update after physics step with command resampling and velocity estimation.

        - Call base _post_physics_step_callback (it resamples commands and updates estimator).
        - Update goal tracking and compute target positions
        """
        super()._post_physics_step_callback()

        if self.goal_positions is None:
            return

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.goal_metrics = self._compute_goal_metrics()
        dist = self.goal_metrics["dist_xy"].squeeze(1)
        env_ids = torch.nonzero(dist < self.goal_reach_eps, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._resample_commands(env_ids)
            self.goal_metrics = self._compute_goal_metrics()
        self._update_goal_commands(self.goal_metrics)
        self._update_velocity_estimate()
        self._draw_goal_indicator()

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        super().reset_idx(env_ids)

        if len(env_ids) > 0:
            env_ids = env_ids.to(dtype=torch.long)
            self.lin_vel_est_world[env_ids] = 0
            self.lin_vel_est[env_ids] = 0

        # Call curriculum update after reset
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)

    def update_command_curriculum(self, env_ids):
        """Implements REVERSE curriculum - narrows command ranges over time.

        Starts with broad ranges (easy - robot can walk anywhere) and gradually
        narrows to precise control (hard - specific velocity targets).

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.cfg.commands.curriculum:
            return

        reward_name = "tracking_lin_vel"
        if reward_name not in self.reward_scales or self.reward_scales[reward_name] == 0.0:
            return
        if reward_name not in self.episode_sums:
            return

        # If tracking reward is above 80% threshold, narrow the ranges
        rewards = self.episode_sums[reward_name]
        if rewards.numel() == 0:
            return
        mean_reward = torch.mean(rewards[env_ids]) / self.max_episode_length
        if mean_reward > 0.8 * self.reward_scales[reward_name]:

            # Get increment values from config (or use defaults)
            increment_x = getattr(self.cfg.commands.crclm_incremnt, "lin_vel_x", 0.1)
            increment_y = getattr(self.cfg.commands.crclm_incremnt, "lin_vel_y", 0.1)
            increment_heading = getattr(self.cfg.commands.crclm_incremnt, "heading", 0.5)

            # Get max (target) ranges from config
            max_x_min = self.cfg.commands.max_ranges.lin_vel_x[0]
            max_x_max = self.cfg.commands.max_ranges.lin_vel_x[1]
            max_y_min = self.cfg.commands.max_ranges.lin_vel_y[0]
            max_y_max = self.cfg.commands.max_ranges.lin_vel_y[1]
            max_heading_min = self.cfg.commands.max_ranges.heading[0]
            max_heading_max = self.cfg.commands.max_ranges.heading[1]

            # REVERSE CURRICULUM: Narrow ranges by moving bounds INWARD
            # lin_vel_x: move min upward, max downward
            new_x_min = self.command_ranges["lin_vel_x"][0] + increment_x
            new_x_max = self.command_ranges["lin_vel_x"][1] - increment_x

            # lin_vel_y: move min upward, max downward
            new_y_min = self.command_ranges["lin_vel_y"][0] + increment_y
            new_y_max = self.command_ranges["lin_vel_y"][1] - increment_y

            # heading: move min upward, max downward
            new_heading_min = self.command_ranges["heading"][0] + increment_heading
            new_heading_max = self.command_ranges["heading"][1] - increment_heading

            # Clamp to max_ranges (don't go narrower than target)
            # For min: don't go higher than max_ranges min
            # For max: don't go lower than max_ranges max
            self.command_ranges["lin_vel_x"][0] = np.clip(new_x_min, max_x_min, self.command_ranges["lin_vel_x"][1])
            self.command_ranges["lin_vel_x"][1] = np.clip(new_x_max, self.command_ranges["lin_vel_x"][0], max_x_max)

            self.command_ranges["lin_vel_y"][0] = np.clip(new_y_min, max_y_min, self.command_ranges["lin_vel_y"][1])
            self.command_ranges["lin_vel_y"][1] = np.clip(new_y_max, self.command_ranges["lin_vel_y"][0], max_y_max)

            self.command_ranges["heading"][0] = np.clip(new_heading_min, max_heading_min, self.command_ranges["heading"][1])
            self.command_ranges["heading"][1] = np.clip(new_heading_max, self.command_ranges["heading"][0], max_heading_max)

    def _sample_goal_positions(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(dtype=torch.long)
        forward = torch_rand_float(self.goal_forward_min, self.goal_forward_max, (len(env_ids), 1), device=self.device)
        lateral = torch_rand_float(self.goal_lateral_min, self.goal_lateral_max, (len(env_ids), 1), device=self.device)
        offsets_local = torch.zeros(len(env_ids), 3, device=self.device)
        offsets_local[:, 0] = forward.squeeze(1)
        offsets_local[:, 1] = lateral.squeeze(1)
        base_quat = self.root_states[env_ids, 3:7]
        offset_world = quat_apply(base_quat, offsets_local)
        base_pos = self.root_states[env_ids, 0:3]
        goal_world = base_pos + offset_world
        goal_world[:, 2] = base_pos[:, 2]
        self.goal_positions[env_ids] = goal_world

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(dtype=torch.long)
        self._sample_goal_positions(env_ids)
        speed_min, speed_max = getattr(self.cfg.goal, "speed_range", (0.5, 1.0))
        speeds = torch_rand_float(float(speed_min), float(speed_max), (len(env_ids), 1), device=self.device).squeeze(1)
        self.speed_targets[env_ids] = speeds
        self.goal_metrics = None

    def _compute_goal_metrics(self):
        if self.goal_positions is None:
            return None
        deltas = self.goal_positions - self.root_states[:, 0:3]
        deltas_xy = deltas[:, :2]
        dist_xy = torch.norm(deltas_xy, dim=1, keepdim=True)
        safe_dist = torch.clamp(dist_xy, min=1e-6)
        goal_dir_xy = deltas_xy / safe_dist
        goal_dir_xy = torch.where(dist_xy > 1e-6, goal_dir_xy, torch.zeros_like(goal_dir_xy))
        goal_dir_world = torch.cat(
            (goal_dir_xy, torch.zeros(self.num_envs, 1, device=self.device, dtype=deltas.dtype)),
            dim=1,
        )
        goal_dir_body = quat_rotate_inverse(self.root_states[:, 3:7], goal_dir_world)
        vel_xy = self.root_states[:, 7:9]
        speed_mag = torch.norm(vel_xy, dim=1, keepdim=True)
        speed_along = torch.sum(vel_xy * goal_dir_xy, dim=1, keepdim=True)
        metrics = {
            "delta_world": deltas,
            "dist_xy": dist_xy,
            "dir_xy_world": goal_dir_xy,
            "dir_world": goal_dir_world,
            "dir_body": goal_dir_body,
            "speed_mag": speed_mag,
            "speed_along": speed_along,
        }
        if (
            self.head_body_index is not None
            and self.head_body_index >= 0
            and self.head_body_index < self.rigid_body_states.shape[1]
        ):
            head_quat = self.rigid_body_states[:, self.head_body_index, 3:7]
            head_forward = quat_apply(head_quat, self.head_forward_ref.expand(self.num_envs, -1))
            head_forward_xy = F.normalize(head_forward[:, :2], dim=1)
            metrics["head_forward_xy"] = head_forward_xy
            metrics["head_alignment"] = torch.sum(head_forward_xy * goal_dir_xy, dim=1, keepdim=True)
        return metrics


    def _reset_root_states(self, env_ids):
        # Base reset sets positions (near env origins) and velocities
        super()._reset_root_states(env_ids)

        # Random yaw per environment
        yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        half_yaw = 0.5 * yaw
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 2] = torch.sin(half_yaw)  # z component
        quat[:, 3] = torch.cos(half_yaw)  # w component
        self.root_states[env_ids, 3:7] = quat

        # Randomize XY within [-1, 1] square around env origin
        xy_offset = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2] + xy_offset

        # Commit updated root states for these envs
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _build_and_add_terrain(self):
        """Build obstacle terrain as a single static mesh/heightfield."""
        terrain_cfg = self.cfg.terrain
        mesh_type = getattr(terrain_cfg, "mesh_type", "trimesh")
        mesh_type_lower = mesh_type.lower() if isinstance(mesh_type, str) else "trimesh"

        # Allow disabling custom terrain for pretraining on a flat plane.
        if mesh_type_lower in ("none", "plane"):
            self.custom_origins = False
            if hasattr(self, "terrain_env_origins"):
                delattr(self, "terrain_env_origins")
            if mesh_type_lower == "plane":
                plane_params = gymapi.PlaneParams()
                plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
                plane_params.static_friction = getattr(terrain_cfg, "static_friction", 1.0)
                plane_params.dynamic_friction = getattr(terrain_cfg, "dynamic_friction", 1.0)
                plane_params.restitution = getattr(terrain_cfg, "restitution", 0.0)
                self.gym.add_ground(self.sim, plane_params)
            return

        num_cubes_setting = getattr(self.cfg.obstacles, "num_cubes", 0)
        if isinstance(num_cubes_setting, (list, tuple)):
            per_group = list(num_cubes_setting)
            while len(per_group) < self.num_envs:
                per_group.extend(per_group)
            per_env_cubes = per_group[: self.num_envs]
        else:
            per_env_cubes = [int(num_cubes_setting)] * self.num_envs
        self.num_cubes_per_env = torch.tensor(per_env_cubes, dtype=torch.long, device="cpu")

        hf, env_origins, _ = build_obsavoid_heightfield(
            terrain_cfg,
            self.num_envs,
            self.cfg.env.env_spacing,
            cube_size=getattr(self.cfg.obstacles, "cube_size", (0.5, 0.5, 0.5)),
            num_cubes=per_env_cubes,
            spawn_area=getattr(self.cfg.obstacles, "spawn_area", (-4.0, 4.0, -4.0, 4.0)),
            min_distance_robot=getattr(self.cfg.obstacles, "min_distance_robot", 0.0),
            min_distance_between=getattr(self.cfg.obstacles, "min_distance_between", 0.0),
            seed=getattr(self.cfg.obstacles, "seed", None),
            num_goals=0,  # Don't generate goals
        )

        env_origins_flat = env_origins.reshape(-1, 3)

        if env_origins_flat.shape[0] < self.num_envs:
            raise RuntimeError("Terrain grid does not provide enough tiles for all environments")

        # Store terrain env origins
        self.terrain_env_origins = to_torch(env_origins_flat[: self.num_envs], device=self.device)
        self.custom_origins = True

        # Cache heightfield for fast sampling
        if getattr(self.cfg.terrain, "measure_heights", False) and self.num_height_samples > 0:
            self.heightfield = torch.from_numpy(hf).to(self.device, dtype=torch.float32)
            self.heightfield_shape = hf.shape
            self.heightfield_flat = self.heightfield.view(-1)
            self.heightfield_horizontal_scale = float(terrain_cfg.horizontal_scale)
            self.heightfield_vertical_scale = float(terrain_cfg.vertical_scale)

        if mesh_type_lower in ("trimesh", "heightfield"):
            horizontal_scale = float(terrain_cfg.horizontal_scale)
            vertical_scale = float(terrain_cfg.vertical_scale)

            if mesh_type_lower == "trimesh":
                vertices, triangles = heightfield_to_trimesh(
                    hf,
                    horizontal_scale,
                    vertical_scale,
                    getattr(terrain_cfg, "slope_treshold", 1.0),
                )
                vertices = vertices.astype(np.float32)
                triangles = triangles.astype(np.uint32)

                params = gymapi.TriangleMeshParams()
                params.nb_vertices = vertices.shape[0]
                params.nb_triangles = triangles.shape[0]
                params.transform = gymapi.Transform()

                self.gym.add_triangle_mesh(
                    self.sim,
                    vertices.flatten(),
                    triangles.flatten(),
                    params,
                )
            else:  # heightfield
                hf_props = gymapi.HeightFieldProperties()
                hf_props.rows = hf.shape[0]
                hf_props.columns = hf.shape[1]
                hf_props.row_scale = horizontal_scale
                hf_props.column_scale = horizontal_scale
                hf_props.height_scale = vertical_scale
                hf_props.min_height = float(np.min(hf) * vertical_scale)
                hf_props.max_height = float(np.max(hf) * vertical_scale)
                hf_props.transform = gymapi.Transform()

                self.gym.add_heightfield(self.sim, hf.flatten(), hf_props)


    def _create_envs(self):
        """Create environments with robot actor; terrain already baked in."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        if self.head_body_index is None:
            head_candidates = getattr(self.cfg.asset, "head_body_names", ["Head_lower", "Head_upper", "head"])
            found_idx = None
            for candidate in head_candidates:
                for idx, name in enumerate(body_names):
                    if candidate in name:
                        found_idx = idx
                        break
                if found_idx is not None:
                    break
            self.head_body_index = found_idx
            self.head_body_name = body_names[found_idx] if found_idx is not None else None
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        if hasattr(self, "terrain_env_origins"):
            self.env_origins = self.terrain_env_origins.clone()
            self.custom_origins = True
        else:
            self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Robot spawn position: env_origin + small random offset
            pos = self.env_origins[i].clone()
            if not self.custom_origins:
                pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            pos_cpu = pos.detach().cpu().numpy()
            start_pose.p = gymapi.Vec3(float(pos_cpu[0]), float(pos_cpu[1]), float(pos_cpu[2]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        print(f"Created {self.num_envs} environments with 1 actor (robot) each")

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            global_idx = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )
            # Convert global body index to local (robot-only) index
            # In multi-actor envs, body indices from find_actor_rigid_body_handle are per-env offsets
            self.termination_contact_indices[i] = global_idx


    def create_sim(self):
        """Create simulation, inject terrain, create envs, and attach cameras."""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        # Build baked terrain obstacles
        self._build_and_add_terrain()

        # Create environments (robots only)
        self._create_envs()

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
                # Use numpy for random pitch (obstacle_rng is commented out)
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

        # Stack and resize to target resolution (batch on GPU)
        target_size = (self.cfg.depth.resized[1], self.cfg.depth.resized[0])  # (H, W)
        depth_stack = torch.stack(depth_list, dim=0)  # [N, Hc, Wc]
        depth_stack = depth_stack.unsqueeze(1)        # [N, 1, Hc, Wc]
        depth_resized = torch.nn.functional.interpolate(
            depth_stack, size=target_size, mode='bilinear', align_corners=False
        ).squeeze(1)  # [N, H, W]

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
        """Build observations synchronously; append a single depth frame each step."""
        metrics = self.goal_metrics
        if metrics is None and self.goal_positions is not None:
            metrics = self._compute_goal_metrics()
            self.goal_metrics = metrics
        if metrics is not None:
            self._update_goal_commands(metrics)

        lin_vel_feature = self.lin_vel_est * self.obs_scales.lin_vel
        core_state = torch.cat(
            (
                lin_vel_feature,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        state_obs = core_state

        # Add noise only to state observations (keep depth clean / privileged clean)
        if self.add_noise:
            state_noise = (2 * torch.rand_like(state_obs) - 1) * self.noise_scale_vec[: state_obs.shape[1]]
            state_obs = state_obs + state_noise

        # Terrain height samples for teacher (privileged) observations only
        teacher_state = state_obs
        height_measurements = None
        if getattr(self.cfg.terrain, "measure_heights", False) and self.num_height_samples > 0:
            height_measurements = self._sample_height_measurements()
            if height_measurements is not None:
                scaled_heights = height_measurements * self.obs_scales.height_measurements
                teacher_state = torch.cat([teacher_state, scaled_heights], dim=-1)

        if self.privileged_obs_buf is not None:
            if teacher_state.shape[1] != self.privileged_obs_buf.shape[1]:
                raise RuntimeError(
                    f"Teacher observation dim mismatch: expected {self.privileged_obs_buf.shape[1]}, got {teacher_state.shape[1]}"
                )
            self.privileged_obs_buf[:] = teacher_state

        # Hyper-style decimated depth updates: render only on intervals, reuse last frame otherwise
        if self.cfg.depth.use_camera:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self._update_depth_buffer()
            current_depth = self.depth_buffer[:, 0, :, :].flatten(start_dim=1)
            self.obs_buf = torch.cat([state_obs, current_depth], dim=1)
            self.global_counter += 1
        else:
            self.obs_buf = state_obs

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        if not getattr(self.cfg.viewer, 'show_goal', True):
            return
        if self.viewer is None or self.goal_positions is None:
            return
        self._draw_goal_indicator()

    def _draw_goal_indicator(self):
        if self.goal_indicator_geom is None:
            return
        if len(self.envs) == 0:
            return
        self.gym.clear_lines(self.viewer)
        env_id = 0
        goal = self.goal_positions[env_id]
        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(float(goal[0]), float(goal[1]), float(goal[2] + 0.1))
        gymutil.draw_lines(self.goal_indicator_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)


    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_drag(self):
        # Penalize horizontal motion of feet while in contact
        feet_xy_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:9]).sum(dim=-1)
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        dragging_vel = contact * feet_xy_vel
        return dragging_vel.sum(dim=-1)

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    # ------------------------------------------------------------------
    # Height sampling helpers
    # ------------------------------------------------------------------
    def _init_height_measurement_points(self):
        x_points = torch.tensor(self.cfg.terrain.measured_points_x, dtype=torch.float, device=self.device)
        y_points = torch.tensor(self.cfg.terrain.measured_points_y, dtype=torch.float, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
        self.height_sample_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        if not getattr(self.cfg.viewer, 'show_goal', True):
            return
        if self.viewer is None or self.goal_positions is None:
            return
        if self.goal_indicator_geom is None:
            return
        if len(self.envs) == 0:
            return
        self.gym.clear_lines(self.viewer)
        env_id = 0
        goal = self.goal_positions[env_id]
        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(float(goal[0]), float(goal[1]), float(goal[2] + 0.1))
        gymutil.draw_lines(self.goal_indicator_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

    def _reward_goal_distance(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        dist = self.goal_metrics["dist_xy"].squeeze(1)
        return 1.0 / (1.0 + dist)

    def _reward_tracking_goal_vel(self):
        if self.goal_metrics is None or not hasattr(self, "speed_targets"):
            return torch.zeros(self.num_envs, device=self.device)
        desired = self.speed_targets
        speed_along = self.goal_metrics["speed_along"].squeeze(1)
        speed_error = desired - speed_along
        return torch.exp(-torch.square(speed_error) / 0.09)

    def _reward_tracking_yaw(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        goal_dir = self.goal_metrics["dir_xy_world"]
        heading_vec = quat_apply(self.root_states[:, 3:7], self.forward_ref.expand(self.num_envs, -1))
        heading_xy = F.normalize(heading_vec[:, :2], dim=1)
        alignment = torch.sum(goal_dir * heading_xy, dim=1)
        return (alignment + 1.0) * 0.5

    def _reward_goal_alignment(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        goal_dir = self.goal_metrics["dir_xy_world"]
        heading_vec = quat_apply(self.root_states[:, 3:7], self.forward_ref.expand(self.num_envs, -1))
        heading_xy = heading_vec[:, :2]
        heading_xy = F.normalize(heading_xy, dim=1)
        alignment = torch.sum(goal_dir * heading_xy, dim=1)
        return (alignment + 1.0) * 0.5

    def _reward_goal_speed(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        speed_mag = self.goal_metrics["speed_mag"].squeeze(1)
        speed_along = self.goal_metrics["speed_along"].squeeze(1)
        desired = self.speed_targets
        speed_error = torch.square(speed_mag - desired)
        backward_penalty = torch.clamp(-speed_along, min=0.0)
        return -(speed_error + backward_penalty)

    def _reward_head_alignment(self):
        if self.goal_metrics is None or "head_alignment" not in self.goal_metrics:
            return torch.zeros(self.num_envs, device=self.device)
        alignment = self.goal_metrics["head_alignment"].squeeze(1)
        return (alignment + 1.0) * 0.5

    def _reward_delta_torques(self):
        delta = self.torques - self.last_torques_buffer
        self.last_torques_buffer = self.torques.clone()
        return torch.sum(torch.square(delta), dim=1)

    def _reward_hip_pos(self):
        if self.hip_dof_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        hip_error = (self.dof_pos[:, self.hip_dof_indices] - self.default_dof_pos[:, self.hip_dof_indices])
        return torch.norm(hip_error, dim=1)

    def _reward_dof_error(self):
        dof_err = self.dof_pos - self.default_dof_pos
        return torch.norm(dof_err, dim=1)

    def _reward_feet_edge(self):
        if not hasattr(self, "x_edge_mask") or self.feet_indices is None:
            return torch.zeros(self.num_envs, device=self.device)
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.cfg.terrain.border_size) /
                       self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clamp(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clamp(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        feet_edge_contact = torch.logical_and(contact, feet_at_edge)
        return torch.sum(feet_edge_contact.float(), dim=1)

    def _sample_height_measurements(self):
        if self.height_sample_points is None or not hasattr(self, "heightfield"):
            return None

        base_xy = self.root_states[:, :2]
        sample_xy = base_xy.unsqueeze(1) + self.height_sample_points.unsqueeze(0)

        scale = self.heightfield_horizontal_scale
        rows, cols = self.heightfield_shape
        px = torch.clamp((sample_xy[..., 0] / scale).long(), 0, rows - 1)
        py = torch.clamp((sample_xy[..., 1] / scale).long(), 0, cols - 1)

        flat_idx = px * cols + py
        heights = torch.take(self.heightfield_flat, flat_idx)
        heights = heights.view(self.num_envs, -1) * self.heightfield_vertical_scale

        base_height = self.root_states[:, 2].unsqueeze(1)
        return base_height - heights
