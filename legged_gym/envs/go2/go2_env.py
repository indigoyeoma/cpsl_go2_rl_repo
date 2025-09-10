from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config import GO2RoughCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
import torchvision.transforms
import cv2

class GO2Robot(LeggedRobot):
    """GO2 quadruped robot environment with depth camera for visual RL"""
    
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):
        self.check_camera = False  # CRITICAL: Disable to avoid GPU→CPU transfers during training
        self.depth_image = None  # Single frame depth image
        
        # IMPROVEMENT from Helpful-Doggybot: Use depth buffer for temporal history
        # self.depth_buffer = torch.zeros(num_envs, buffer_len, height, width)
        # This would store last N frames for better motion understanding
        
        # Initialize camera-related attributes
        self.cam_handles = []
        if cfg.depth.use_camera:
            # torchvision.Resize expects (height, width) but cfg has (width, height)  
            # Current: (cfg.depth.resized[1], cfg.depth.resized[0]) swaps to get (height, width)
            # This is correct if cfg.depth.resized = (84, 84) for square images
            self.resize_transform = torchvision.transforms.Resize(
                (cfg.depth.resized[1], cfg.depth.resized[0]), 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize depth buffer for raw image storage
        if cfg.depth.use_camera:
            print(f"🔍 Debug: Initializing depth buffer")
            print(f"   - num_envs: {self.num_envs}")
            print(f"   - buffer_len: {cfg.depth.buffer_len}")
            print(f"   - resized dimensions: {cfg.depth.resized} → height={cfg.depth.resized[1]}, width={cfg.depth.resized[0]}")
            
            self.depth_buffer = torch.zeros(
                self.num_envs,
                cfg.depth.buffer_len,
                cfg.depth.resized[1],  # height
                cfg.depth.resized[0]   # width
            ).to(self.device)
            
            print(f"   - depth_buffer shape: {self.depth_buffer.shape}")
            print(f"   - Expected flattened size per env: {cfg.depth.resized[0] * cfg.depth.resized[1]}")
            # Initialize global counter for depth buffer updates
            self.global_counter = 0
            # Update noise scale vector after initialization to account for depth observations
            self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
    
    def create_sim(self):
        """Creates simulation with trimesh terrain and camera support"""
        self.up_axis_idx = 2
        
        # Required for camera creation in headless mode
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id
            
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # Create terrain based on mesh_type
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'trimesh':
            from legged_gym.utils.terrain import Terrain
            self.terrain = Terrain(self.cfg.terrain, self.cfg.env.num_envs)
            self._create_trimesh()
        elif mesh_type == 'heightfield':
            from legged_gym.utils.terrain import Terrain
            self.terrain = Terrain(self.cfg.terrain, self.cfg.env.num_envs)
            self._create_heightfield()
        elif mesh_type == 'plane':
            self._create_ground_plane()
            
        self._create_envs()
    
    def _get_env_origins(self):
        """Simple grid placement for flat terrain with configured spacing"""
        # Let parent class handle the grid placement with env_spacing
        super()._get_env_origins()
    
  
    def _reset_root_states(self, env_ids):
        """Override to add random yaw if configured"""
        # Call parent reset
        super()._reset_root_states(env_ids)
        
        # Add random yaw if configured
        if hasattr(self.cfg.init_state, 'random_yaw') and self.cfg.init_state.random_yaw:
            for env_id in env_ids:
                # Random yaw orientation for variety
                random_yaw = torch.rand(1, device=self.device) * 2 * np.pi  # 0 to 2π radians
                
                quat = torch.tensor([0.0, 0.0, torch.sin(random_yaw / 2), torch.cos(random_yaw / 2)], device=self.device)
                self.root_states[env_id, 3:7] = quat
            
            # Apply the reset with new orientations
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation with obstacles"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )
    
    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation"""
        hf_params = gymapi.HeightFieldParams()
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size  
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        # Isaac Gym add_heightfield takes (sim, height_array, HeightFieldParams)
        # Scaling is handled by the height values and terrain generation
        self.gym.add_heightfield(
            self.sim, 
            self.terrain.heightsamples,
            hf_params
        )
        
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)  
            .to(self.device)
        )
    
    def show_depth_image(self, robot_id=0):
        """Display depth image from robot's camera for debugging"""
        if not self.cfg.depth.use_camera or not hasattr(self, 'cam_handles') or robot_id >= len(self.cam_handles):
            print(f"Camera not available for robot {robot_id}")
            return None
            
        # Get current depth image directly from camera
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        depth_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.envs[robot_id], 
            self.cam_handles[robot_id],
            gymapi.IMAGE_DEPTH
        )
        depth_image = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        
        self.gym.end_access_image_tensors(self.sim)
        
        # Debug info
        print(f"Depth image stats: min={depth_image.min():.3f}, max={depth_image.max():.3f}, mean={depth_image.mean():.3f}")
        
        # Simple normalization for display
        depth_display = np.clip(-depth_image * 255 / self.cfg.depth.far_clip, 0, 255).astype(np.uint8)
        depth_display = cv2.resize(depth_display, (320, 240))
        
        cv2.imshow(f'GO2 Robot {robot_id} Depth Camera - Live Training Feed', depth_display)
        cv2.waitKey(1)
        return depth_image
    
    def step(self, actions):
        """Override step to update depth buffer and show camera feed"""
        # Increment global counter for depth buffer timing
        if hasattr(self, 'global_counter'):
            self.global_counter += 1
        
        # Update depth buffer before stepping (ensures sync with compute_observations)
        self.update_depth_buffer()
        
        result = super().step(actions)
        
        # Show depth camera at same rate as GUI rendering
        if not self.headless and hasattr(self, 'viewer') and self.viewer and self.check_camera:
            self.show_depth_image(robot_id=0)
            
        return result
    
    def update_depth_buffer(self):
        """Update depth buffer with raw images - following Helpful-Doggybot pattern"""
        if not self.cfg.depth.use_camera:
            return
        
        # Debug: Check camera setup
        if not hasattr(self, '_debug_printed'):
            print(f"🔍 Debug: update_depth_buffer called")
            print(f"   - num_envs: {self.num_envs}")
            print(f"   - cam_handles length: {len(self.cam_handles) if hasattr(self, 'cam_handles') else 'No cam_handles'}")
            print(f"   - depth_buffer shape: {self.depth_buffer.shape if hasattr(self, 'depth_buffer') else 'No depth_buffer'}")
            self._debug_printed = True
        
        # Only update depth buffer at specified intervals for efficiency
        if hasattr(self, 'global_counter') and self.global_counter % self.cfg.depth.update_interval != 0:
            return
            
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        # Batch collect all depth tensors first
        raw_depth_tensors = []
        valid_env_indices = []
        
        for i in range(self.num_envs):
            if i < len(self.cam_handles):
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i], 
                    self.cam_handles[i],
                    gymapi.IMAGE_DEPTH
                )
                raw_depth_tensors.append(gymtorch.wrap_tensor(depth_tensor))
                valid_env_indices.append(i)
        
        self.gym.end_access_image_tensors(self.sim)
        
        # Batch process all images at once
        if raw_depth_tensors:
            batch_raw_depth = torch.stack(raw_depth_tensors, dim=0)  # [N, H, W]
            batch_processed_depth = self.process_depth_batch(batch_raw_depth)  # [N, 84, 84]
            
            # Vectorized buffer update - much faster than loop
            valid_env_tensor = torch.tensor(valid_env_indices, device=self.device, dtype=torch.long)
            init_flags = self.episode_length_buf[valid_env_tensor] <= 1
            
            # For initialized environments, just copy the new frame
            if self.cfg.depth.buffer_len == 1:
                # Single frame buffer - direct assignment
                self.depth_buffer[valid_env_tensor] = batch_processed_depth.unsqueeze(1)
            else:
                # Multi-frame buffer - needs shifting
                for idx, env_i in enumerate(valid_env_indices):
                    if init_flags[idx]:
                        self.depth_buffer[env_i] = batch_processed_depth[idx].unsqueeze(0).repeat(self.cfg.depth.buffer_len, 1, 1)
                    else:
                        self.depth_buffer[env_i, :-1] = self.depth_buffer[env_i, 1:].clone()
                        self.depth_buffer[env_i, -1] = batch_processed_depth[idx]

    def process_depth_batch(self, batch_raw_depth):
        """Process batch of raw depth images - much faster than individual processing"""
        # Clean up depth images - replace inf/nan with camera range values
        near_clip = self.cfg.depth.near_clip  # 0.3m
        far_clip = self.cfg.depth.far_clip    # 3.0m
        
        # Clamp to camera depth range for entire batch
        depth_batch = torch.clamp(batch_raw_depth, min=-far_clip, max=-near_clip)
        # Replace nan/inf with valid depth values within camera range
        depth_batch = torch.nan_to_num(depth_batch, nan=-far_clip, posinf=-far_clip, neginf=-near_clip)
        
        # Add D435i realistic noise if configured
        if self.cfg.depth.dis_noise > 0 and self.add_noise:
            # Distance-dependent noise (more noise at farther distances)
            distance_normalized = (depth_batch - (-far_clip)) / (far_clip - near_clip)  # 0 to 1
            noise_scale = self.cfg.depth.dis_noise * (1.0 + distance_normalized * 2.0)  # Scale noise with distance
            noise = torch.randn_like(depth_batch) * noise_scale
            depth_batch = depth_batch + noise
            # Re-clamp after adding noise
            depth_batch = torch.clamp(depth_batch, min=-far_clip, max=-near_clip)
        
        # Normalize depth values to [-1, 1] for better learning
        # Depth values are negative (farther = more negative)
        # Convert from [-far_clip, -near_clip] to [0, 1] then to [-1, 1]
        depth_normalized = (depth_batch - (-far_clip)) / (far_clip - near_clip)  # [0, 1]
        depth_batch = depth_normalized * 2.0 - 1.0  # [-1, 1]
        
        # Batch resize from 480×270 to 84×84 - much faster than individual resizes
        if hasattr(self, 'resize_transform'):
            depth_batch = self.resize_transform(depth_batch)  # Process all images at once
        
        return depth_batch  # [N, 84, 84]

    def process_depth_image(self, raw_depth):
        """Process raw depth image into normalized format - kept for compatibility"""
        # Clean up depth image - replace inf/nan with camera range values
        near_clip = self.cfg.depth.near_clip  # 0.3m
        far_clip = self.cfg.depth.far_clip    # 3.0m
        
        # Clamp to camera depth range
        depth_image = torch.clamp(raw_depth, min=-far_clip, max=-near_clip)
        # Replace nan/inf with valid depth values within camera range
        depth_image = torch.nan_to_num(depth_image, nan=-far_clip, posinf=-far_clip, neginf=-near_clip)
        
        # Normalize depth values to [-1, 1] for better learning
        # Depth values are negative (farther = more negative)
        # Convert from [-far_clip, -near_clip] to [0, 1] then to [-1, 1]
        depth_normalized = (depth_image - (-far_clip)) / (far_clip - near_clip)  # [0, 1]
        depth_image = depth_normalized * 2.0 - 1.0  # [-1, 1]
        
        # Resize from 480×270 to 84×84 if needed
        if hasattr(self, 'resize_transform'):
            depth_image = self.resize_transform(depth_image.unsqueeze(0)).squeeze(0)
        
        return depth_image  # [84, 84]

    def get_current_depth_obs(self):
        """Get current depth observations as [N, 1, 84, 84] for CNN processing"""
        if not self.cfg.depth.use_camera or not hasattr(self, 'depth_buffer'):
            return torch.zeros(self.num_envs, 1, 84, 84, device=self.device)
        
        # Get most recent depth image from buffer and add channel dimension
        current_depth = self.depth_buffer[:, -1]  # [N, 84, 84]
        return current_depth.unsqueeze(1)  # [N, 1, 84, 84]

    def compute_observations(self):
        """Compute proprioceptive observations - depth handled separately"""
        # Compute base observations without noise (copy from base class)
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # For backward compatibility with non-HyperPPO systems, add flattened depth
        if self.cfg.depth.use_camera and hasattr(self, 'depth_buffer'):
            current_depth = self.depth_buffer[:, -1]  # [N, 64, 64]
            depth_flat = current_depth.view(self.num_envs, -1)  # [N, 4096]
            self.obs_buf = torch.cat([self.obs_buf, depth_flat], dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _create_envs(self):
        """Override to attach cameras when creating environments"""
        super()._create_envs()
        
        # Attach cameras to all environments if using depth camera
        if self.cfg.depth.use_camera:
            for i in range(self.num_envs):
                if i < len(self.actor_handles):
                    self.attach_camera_to_robot(i, self.envs[i], self.actor_handles[i])
    
    def attach_camera_to_robot(self, env_id, env_handle, actor_handle):
        """Attach depth camera to robot with random positioning"""
        config = self.cfg.depth
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.depth.original[0]
        camera_props.height = self.cfg.depth.original[1]
        camera_props.enable_tensors = True
        camera_horizontal_fov = self.cfg.depth.horizontal_fov 
        camera_props.horizontal_fov = camera_horizontal_fov
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        self.cam_handles.append(camera_handle)
        
        local_transform = gymapi.Transform()
        
        camera_position_center = np.copy(config.position)
        camera_position = np.random.uniform(camera_position_center-config.position_rand, camera_position_center+config.position_rand)
        camera_angle = np.random.uniform(config.angle[0], config.angle[1])
        
        local_transform.p = gymapi.Vec3(*camera_position)
        local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
        root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
        
        self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    
    def _get_noise_scale_vec(self, cfg):
        """Override to handle depth image noise scaling"""
        # Set the add_noise attribute (normally set by base class)
        self.add_noise = cfg.noise.add_noise
        
        # Create noise vector manually instead of calling super() to avoid size issues
        noise_vec = torch.zeros(48, device=self.device)  # Base observations only
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        
        # Fill in noise for base observations (same as base class logic)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        
        # If using depth camera, extend noise vector for depth dimensions
        if cfg.depth.use_camera:
            depth_size = cfg.depth.resized[0] * cfg.depth.resized[1]  # 84*84 = 7056
            # Add zero noise for depth images (depth images shouldn't have noise added)
            depth_noise = torch.zeros(depth_size, device=self.device)
            # Concatenate: [48 base obs noise + 7056 depth noise (zeros)]
            result = torch.cat([noise_vec, depth_noise])
            return result
        else:
            return noise_vec
