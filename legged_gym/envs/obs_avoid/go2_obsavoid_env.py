import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import torch_rand_float, to_torch, quat_apply, quat_rotate_inverse

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_env import GO2Robot
from .go2_obsavoid_config import GO2ObsAvoidCfg
from .go2_terrain_obsavoid import (
    build_obsavoid_heightfield,
    heightfield_to_trimesh,
)


class GO2ObsAvoidRobot(GO2Robot):
    """State-only GO2 obstacle-avoidance environment (48-dim deployable state + terrain samples)."""

    def __init__(self, cfg: GO2ObsAvoidCfg, sim_params, physics_engine, sim_device, headless):
        self.height_sample_points = None
        self.num_height_samples = 0
        if getattr(cfg.terrain, "measure_heights", False):
            x_count = len(getattr(cfg.terrain, "measured_points_x", []))
            y_count = len(getattr(cfg.terrain, "measured_points_y", []))
            self.num_height_samples = x_count * y_count
        self.goal_metrics = None
        self.head_body_index = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.add_noise = self.cfg.noise.add_noise
        hip_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        self.hip_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_names if name in self.dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.last_torques_buffer = torch.zeros_like(self.torques)

        if getattr(self.cfg.terrain, "measure_heights", False):
            self._init_height_measurement_points()

        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.speed_targets = torch.zeros(self.num_envs, device=self.device)
        self.goal_forward_min = float(self.cfg.goal.forward_range[0])
        self.goal_forward_max = float(self.cfg.goal.forward_range[1])
        self.goal_lateral_min = float(self.cfg.goal.lateral_range[0])
        self.goal_lateral_max = float(self.cfg.goal.lateral_range[1])
        self.goal_reach_eps = float(getattr(self.cfg.goal, 'reach_epsilon', 0.5))
        self.goal_reached_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.forward_ref = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.head_forward_ref = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.goal_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        self.goal_indicator_geom = None
        if not headless:
            if hasattr(gymutil, "WireframeSphereGeometry"):
                self.goal_indicator_geom = gymutil.WireframeSphereGeometry(
                    0.25,
                    12,
                    12,
                    None,
                    color=(1.0, 0.0, 0.0),
                )
            elif hasattr(gymutil, "AxesGeometry"):
                self.goal_indicator_geom = gymutil.AxesGeometry(0.25)

    def create_sim(self):
        """Create simulation with baked obstacle terrain and robot actors."""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        self._build_and_add_terrain()
        self._create_envs()

    def _build_and_add_terrain(self):
        """Generate obstacle field using the shared obs-avoid terrain helper."""
        terrain_cfg = self.cfg.terrain
        mesh_type = getattr(terrain_cfg, "mesh_type", "trimesh")
        mesh_type_lower = mesh_type.lower() if isinstance(mesh_type, str) else "trimesh"

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
            num_goals=0,
        )

        env_origins_flat = env_origins.reshape(-1, 3)
        if env_origins_flat.shape[0] < self.num_envs:
            raise RuntimeError("Terrain grid does not provide enough tiles for all environments")

        self.terrain_env_origins = to_torch(env_origins_flat[: self.num_envs], device=self.device)
        self.custom_origins = True

        self.heightfield = torch.from_numpy(hf).to(self.device, dtype=torch.float32)
        self.heightfield_shape = hf.shape
        self.heightfield_flat = self.heightfield.reshape(-1)
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
            else:
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

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.goal_positions is None:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.goal_metrics = self._compute_goal_metrics()
        dist = self.goal_metrics["dist_xy"].squeeze(1)
        env_ids = torch.nonzero(dist < self.goal_reach_eps, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            # Keep current goal when reached; just refresh metrics for reward computation.
            self.goal_metrics = self._compute_goal_metrics()
        self._update_goal_commands(self.goal_metrics)
        self._update_velocity_estimate()
        in_goal = dist < self.goal_reach_eps
        self.goal_reached_mask = in_goal
        self._draw_goal_indicator()

    def check_termination(self):
        """Extend default termination with goal-hold based reset."""
        super().check_termination()
        if self.goal_reached_mask is None:
            return
        if torch.any(self.goal_reached_mask):
            reached_ids = torch.nonzero(self.goal_reached_mask, as_tuple=False).squeeze(-1)
            self.reset_buf[reached_ids] = True
            self.goal_reached_mask[reached_ids] = False

    def _create_envs(self):
        """Create environments with robots positioned on the obstacle terrain."""
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

        grid_size = int(np.ceil(np.sqrt(self.num_envs)))
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, grid_size)

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
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _reset_root_states(self, env_ids):
        """Reset robot bases with random yaw/xy offsets inside each tile."""
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return

        yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        half_yaw = 0.5 * yaw
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 2] = torch.sin(half_yaw)
        quat[:, 3] = torch.cos(half_yaw)
        self.root_states[env_ids, 3:7] = quat

        xy_offset = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2] + xy_offset

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.goal_initialized[env_ids] = False
        self.goal_reached_mask[env_ids] = False
        super().reset_idx(env_ids)

    def _sample_goal_positions(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(dtype=torch.long)
        # Sample goals in WORLD coordinates (absolute) to avoid circular motion
        # Use env_origins as reference for each environment's tile center
        env_center = self.env_origins[env_ids, :2]
        x_offset = torch_rand_float(self.goal_forward_min, self.goal_forward_max, (len(env_ids), 1), device=self.device)
        y_offset = torch_rand_float(self.goal_lateral_min, self.goal_lateral_max, (len(env_ids), 1), device=self.device)
        goal_xy = env_center + torch.cat([x_offset, y_offset], dim=1)
        base_pos = self.root_states[env_ids, 0:3]
        goal_world = torch.zeros(len(env_ids), 3, device=self.device)
        goal_world[:, :2] = goal_xy
        goal_world[:, 2] = base_pos[:, 2]
        self.goal_positions[env_ids] = goal_world

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(dtype=torch.long)
        fresh_mask = ~self.goal_initialized[env_ids]
        fresh_ids = env_ids[fresh_mask]
        if fresh_ids.numel() == 0:
            return
        self._sample_goal_positions(fresh_ids)
        speed_min, speed_max = getattr(self.cfg.goal, "speed_range", (0.5, 1.0))
        speeds = torch_rand_float(float(speed_min), float(speed_max), (len(fresh_ids), 1), device=self.device).squeeze(1)
        self.speed_targets[fresh_ids] = speeds
        self.goal_initialized[fresh_ids] = True
        self.goal_metrics = None
        metrics = self._compute_goal_metrics()
        self.goal_metrics = metrics
        self._update_goal_commands(metrics, fresh_ids)

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
        if self.head_body_index is not None and self.head_body_index < self.rigid_body_states.shape[1]:
            head_quat = self.rigid_body_states[:, self.head_body_index, 3:7]
            head_forward = quat_apply(head_quat, self.head_forward_ref.expand(self.num_envs, -1))
            head_forward_xy = F.normalize(head_forward[:, :2], dim=1)
            metrics["head_forward_xy"] = head_forward_xy
            metrics["head_alignment"] = torch.sum(head_forward_xy * goal_dir_xy, dim=1, keepdim=True)
        return metrics

    def _init_height_measurement_points(self):
        x_points = torch.tensor(self.cfg.terrain.measured_points_x, dtype=torch.float, device=self.device)
        y_points = torch.tensor(self.cfg.terrain.measured_points_y, dtype=torch.float, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
        self.height_sample_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        self.num_height_samples = self.height_sample_points.shape[0]

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        self._draw_goal_indicator()

    def _draw_goal_indicator(self):
        if not getattr(self.cfg.viewer, 'show_goal', True):
            return
        if self.viewer is None or self.goal_positions is None:
            return
        if self.goal_positions.shape[0] == 0:
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
        speed_error = torch.abs(desired - speed_along)
        # Smooth bell-shaped reward centered at desired speed
        reward = torch.exp(-torch.square(speed_error) / 0.09)
        # Zero out reward if moving backwards or largely stationary
        reverse_mask = speed_along < 0.0
        low_speed_mask = speed_mag < 0.1
        reward = torch.where(reverse_mask | low_speed_mask, torch.zeros_like(reward), reward)
        return reward

    def _reward_goal_reached(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        dist = self.goal_metrics["dist_xy"].squeeze(1)
        return (dist <= self.goal_reach_eps).float()

    def _reward_goal_lateral_drift(self):
        if self.goal_metrics is None:
            return torch.zeros(self.num_envs, device=self.device)
        speed_mag = self.goal_metrics["speed_mag"].squeeze(1)
        speed_along = self.goal_metrics["speed_along"].squeeze(1)
        lateral_sq = torch.clamp(speed_mag ** 2 - speed_along ** 2, min=0.0)
        lateral_speed = torch.sqrt(lateral_sq)
        # Normalize by total speed to express lateral drift ratio (safe if stationary)
        norm = torch.clamp(speed_mag, min=0.1)
        drift_ratio = lateral_speed / norm
        return torch.exp(-torch.square(drift_ratio) * 4.0)

    def _reward_head_alignment(self):
        if self.goal_metrics is None or "head_alignment" not in self.goal_metrics:
            return torch.zeros(self.num_envs, device=self.device)
        alignment = self.goal_metrics["head_alignment"].squeeze(1)
        return (alignment + 1.0) * 0.5

    def _reward_foot_contact_balance(self):
        """Reward keeping all feet grounded with balanced load."""
        if self.contact_forces is None or self.feet_indices is None:
            return torch.zeros(self.num_envs, device=self.device)
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        contact_mask = contact_forces > 20.0
        contact_ratio = contact_mask.float().mean(dim=1)
        centered = contact_forces - contact_forces.mean(dim=1, keepdim=True)
        variance = torch.mean(centered ** 2, dim=1)
        stability = torch.exp(-variance / 1500.0)
        return contact_ratio * stability

    def _reward_foot_height_asymmetry(self):
        """Penalize large height disparities between feet to discourage high stepping of a single leg."""
        if self.rigid_body_states is None or self.feet_indices is None:
            return torch.zeros(self.num_envs, device=self.device)
        foot_heights = self.rigid_body_states[:, self.feet_indices, 2]
        mean_height = torch.mean(foot_heights, dim=1, keepdim=True)
        height_dev = torch.abs(foot_heights - mean_height)
        return torch.mean(height_dev, dim=1)

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
        if not hasattr(self, "contact_filt"):
            return torch.zeros(self.num_envs, device=self.device)
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.cfg.terrain.border_size) /
                       self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clamp(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clamp(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        if not hasattr(self, "contact_forces"):
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        feet_edge_contact = torch.logical_and(contact, feet_at_edge)
        return torch.sum(feet_edge_contact.float(), dim=1)

    def _reward_feet_air_time(self):
        """Override base class: Reward proper swing phase (0.1-0.3s air time) to prevent rapid bouncing."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt

        # Only reward air time in the proper range (0.1s to 0.3s) to discourage rapid bouncing
        # Use Gaussian-like reward: peak at 0.2s, zero outside [0.1, 0.3]
        target_air_time = 0.2
        air_time_clamped = torch.clamp(self.feet_air_time, 0.1, 0.3)
        air_time_quality = 1.0 - torch.abs(air_time_clamped - target_air_time) / 0.1
        rew_airTime = torch.sum(air_time_quality * first_contact, dim=1)

        # Only give reward when robot is moving toward goal
        if self.goal_metrics is not None:
            speed_mag = self.goal_metrics["speed_mag"].squeeze(1)
            rew_airTime *= (speed_mag > 0.2)  # Only reward gait when actually moving
        self.feet_air_time *= ~contact_filt
        return rew_airTime

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
        heights = heights.view(self.num_envs, -1).float() * self.heightfield_vertical_scale

        base_height = self.root_states[:, 2].unsqueeze(1)
        return (base_height - heights)

    def _get_noise_scale_vec(self, cfg):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        device = self.device

        components = []
        components.append(torch.full((3,), noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel, device=device))
        components.append(torch.full((3,), noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel, device=device))
        components.append(torch.full((3,), noise_scales.gravity * noise_level, device=device))
        components.append(torch.zeros(3, device=device))
        components.append(torch.full((self.num_actions,), noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos, device=device))
        components.append(torch.full((self.num_actions,), noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel, device=device))
        components.append(torch.zeros(self.num_actions, device=device))

        base_noise = torch.cat(components)

        if self.num_height_samples == 0:
            return base_noise
        height_noise = torch.zeros(self.num_height_samples, device=device)
        return torch.cat([base_noise, height_noise])

    def _update_goal_commands(self, metrics, env_ids=None):
        """Map goal tracking objectives into command space."""
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

        valid = torch.norm(dir_body_xy, dim=1, keepdim=True) > 1e-6
        cmd_xy = torch.zeros_like(dir_body_xy)
        cmd_xy[valid.squeeze(1)] = speed_targets[valid.squeeze(1)] * dir_body_xy[valid.squeeze(1)]

        self.commands[env_ids, 0] = cmd_xy[:, 0]
        self.commands[env_ids, 1] = cmd_xy[:, 1]

        heading_error = torch.zeros(env_ids.numel(), device=self.device)
        heading_error[valid.squeeze(1)] = torch.atan2(dir_body_xy[valid.squeeze(1), 1], dir_body_xy[valid.squeeze(1), 0])
        self.commands[env_ids, 2] = torch.clamp(heading_error, -1.0, 1.0)

        if self.cfg.commands.heading_command and self.commands.shape[1] > 3:
            dir_world = metrics["dir_world"][env_ids]
            desired_heading = torch.zeros(env_ids.numel(), device=self.device)
            desired_heading[valid.squeeze(1)] = torch.atan2(dir_world[valid.squeeze(1), 1], dir_world[valid.squeeze(1), 0])
            self.commands[env_ids, 3] = desired_heading

    def _update_velocity_estimate(self):
        """Update linear velocity estimate using commands (mirrors deployment estimator)."""
        cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_body[:, 0] = self.commands[:, 0]
        cmd_body[:, 1] = self.commands[:, 1]
        cmd_world = quat_apply(self.base_quat, cmd_body)
        cmd_world[:, 1] *= 0.5

        alpha = 0.1
        self.lin_vel_est_world = (1 - alpha) * self.lin_vel_est_world + alpha * cmd_world
        self.lin_vel_est = quat_rotate_inverse(self.base_quat, self.lin_vel_est_world)

    def compute_observations(self):
        metrics = self.goal_metrics
        if metrics is None and self.goal_positions is not None:
            metrics = self._compute_goal_metrics()
            self.goal_metrics = metrics
        if metrics is not None:
            self._update_goal_commands(metrics)

        state_obs = torch.cat(
            (
                self.lin_vel_est * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            noise = (2 * torch.rand_like(state_obs) - 1) * self.noise_scale_vec[: state_obs.shape[1]]
            state_obs = state_obs + noise

        height_measurements = self._sample_height_measurements()
        if height_measurements is not None:
            scaled_heights = height_measurements * self.obs_scales.height_measurements
            self.obs_buf = torch.cat([state_obs, scaled_heights], dim=-1)
        else:
            self.obs_buf = state_obs
