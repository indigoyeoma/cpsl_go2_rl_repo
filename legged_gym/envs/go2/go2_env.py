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
        self.check_camera = False  # Disable camera display by default
        self.depth_image = None  # Single frame depth image
        
        # Initialize camera-related attributes
        self.cam_handles = []
        if cfg.depth.use_camera:
            self.resize_transform = torchvision.transforms.Resize(
                (cfg.depth.resized[1], cfg.depth.resized[0]), 
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            )
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Update noise scale vector after initialization to account for depth observations
        if cfg.depth.use_camera:
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
        """Place robots in individual corridors or randomly for wall-based terrain"""
        if self.cfg.terrain.mesh_type in ['trimesh', 'heightfield']:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            # Check if we're using individual corridors terrain
            if len(self.cfg.terrain.terrain_proportions) > 5 and self.cfg.terrain.terrain_proportions[5] > 0.0:
                # Individual corridors layout
                corridors_per_row = int(np.sqrt(self.num_envs))  # 11x11 grid for 128 robots
                corridor_width = 3.0  # Match terrain generation
                
                # Calculate terrain bounds
                terrain_width = self.cfg.terrain.terrain_width
                terrain_length = self.cfg.terrain.terrain_length
                
                # Spacing between corridor centers
                corridor_spacing_x = terrain_width / corridors_per_row
                corridor_spacing_y = terrain_length / corridors_per_row
                
                for robot_id in range(self.num_envs):
                    row = robot_id // corridors_per_row
                    col = robot_id % corridors_per_row
                    
                    # Place robot at the start of its corridor (centered)
                    start_x = col * corridor_spacing_x - terrain_width/2 + corridor_spacing_x/2
                    start_y = row * corridor_spacing_y - terrain_length/2 + 2.0  # 2m from corridor start
                    
                    self.env_origins[robot_id, 0] = start_x
                    self.env_origins[robot_id, 1] = start_y
                    self.env_origins[robot_id, 2] = 0.0
                
                print(f"Placed {self.num_envs} robots in individual corridors ({corridors_per_row}x{corridors_per_row})")
                return
            
            # Original random placement for wall-based terrain
            # Get terrain grid boundaries - use actual terrain origins
            grid_bounds = self.terrain.env_origins
            min_x = np.min(grid_bounds[:, :, 0]) - self.terrain.cfg.terrain_length/2
            max_x = np.max(grid_bounds[:, :, 0]) + self.terrain.cfg.terrain_length/2  
            min_y = np.min(grid_bounds[:, :, 1]) - self.terrain.cfg.terrain_width/2
            max_y = np.max(grid_bounds[:, :, 1]) + self.terrain.cfg.terrain_width/2
            
            grid_center_x = (min_x + max_x) / 2
            grid_center_y = (min_y + max_y) / 2
            
            # Place robots at the edge of terrain, ready to enter obstacle course
            terrain_size = max(max_x - min_x, max_y - min_y)
            radius = terrain_size * 0.6  # Just at the edge of obstacle area
            num_cols = 8
            spacing = 2.0  # Increased spacing to prevent inter-robot visibility
            
            # Safe spawning parameters - balanced for good distribution and safety
            min_robot_distance = 3.0  # Increased spacing between robots
            safe_wall_distance = 0.8  # Safe margin from walls
            
            # Generate safe spawn positions avoiding walls
            spawn_positions = torch.zeros(self.num_envs, 2, device=self.device)
            
            # Keep trying until all robots are placed safely
            for robot_idx in range(self.num_envs):
                max_attempts = 1000  # Much higher attempt count
                placed = False
                
                for attempt in range(max_attempts):
                    # Random position within terrain bounds
                    pos_x = torch.rand(1, device=self.device) * (max_x - min_x - 2*safe_wall_distance) + (min_x + safe_wall_distance)
                    pos_y = torch.rand(1, device=self.device) * (max_y - min_y - 2*safe_wall_distance) + (min_y + safe_wall_distance)
                    candidate_pos = torch.tensor([pos_x, pos_y], device=self.device).squeeze()
                    
                    # Check if position is safe from walls
                    safe_from_walls = True
                    if hasattr(self.terrain, 'wall_positions') and len(self.terrain.wall_positions) > 0:
                        for wall_x, wall_y, wall_radius in self.terrain.wall_positions:
                            wall_distance = torch.sqrt((candidate_pos[0] - wall_x)**2 + (candidate_pos[1] - wall_y)**2)
                            if wall_distance < (wall_radius + safe_wall_distance):
                                safe_from_walls = False
                                break
                    
                    # Check distance from other robots
                    safe_from_robots = True
                    for other_idx in range(robot_idx):
                        robot_distance = torch.norm(candidate_pos - spawn_positions[other_idx])
                        if robot_distance < min_robot_distance:
                            safe_from_robots = False
                            break
                    
                    if safe_from_walls and safe_from_robots:
                        spawn_positions[robot_idx] = candidate_pos
                        placed = True
                        break
                
                if not placed:
                    print(f"Warning: Could not place robot {robot_idx} after {max_attempts} attempts!")
            
            print(f"Robot placement completed")
            
            # Set final positions on flat terrain
            self.env_origins[:, :2] = spawn_positions 
            self.env_origins[:, 2] = 0.0  # Flat terrain at height 0
                
        else:
            super()._get_env_origins()
    
  
    def _reset_root_states(self, env_ids):
        """Override to set safe initial positions without random offset"""
        # Set base position WITHOUT random offset - keep safe spawn positions
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # Set zero initial velocities for stable start
        self.root_states[env_ids, 7:13] = torch.zeros((len(env_ids), 6), device=self.device)
        
        # Set random orientations for robustness (fully random yaw)
        for env_id in env_ids:
            # Fully random yaw orientation for maximum robustness
            random_yaw = torch.rand(1, device=self.device) * 2 * np.pi  # 0 to 2π radians
            
            quat = torch.tensor([0.0, 0.0, torch.sin(random_yaw / 2), torch.cos(random_yaw / 2)], device=self.device)
            self.root_states[env_id, 3:7] = quat
        
        # Apply the reset
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
        """Override step to show live camera feed during training"""
        result = super().step(actions)
        
        # Show depth camera at same rate as GUI rendering
        if not self.headless and hasattr(self, 'viewer') and self.viewer and self.check_camera:
            self.show_depth_image(robot_id=0)
            
        return result
    
    def compute_observations(self):
        """Override to include depth features in observations"""
        # Compute base observations without noise (copy from base class)
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # Add depth images if using camera
        if self.cfg.depth.use_camera:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            depth_images = []
            for i in range(self.num_envs):
                if i < len(self.cam_handles):
                    # Get depth image directly from camera
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        self.envs[i], 
                        self.cam_handles[i],
                        gymapi.IMAGE_DEPTH
                    )
                    depth_image = gymtorch.wrap_tensor(depth_tensor)
                    
                    # Clean up depth image - replace inf/nan with camera range values
                    near_clip = self.cfg.depth.near_clip  # 0.3m
                    far_clip = self.cfg.depth.far_clip    # 3.0m
                    
                    # Clamp to camera depth range
                    depth_image = torch.clamp(depth_image, min=-far_clip, max=far_clip)
                    # Replace nan/inf with valid depth values within camera range
                    depth_image = torch.nan_to_num(depth_image, nan=far_clip, posinf=far_clip, neginf=near_clip)
                    
                    # Resize to 64x64 if needed, then flatten immediately
                    if hasattr(self, 'resize_transform'):
                        depth_image = self.resize_transform(depth_image.unsqueeze(0)).squeeze(0)
                    
                    # Flatten: [64, 64] → [4096]
                    depth_flat = depth_image.view(-1)
                    depth_images.append(depth_flat)
            
            self.gym.end_access_image_tensors(self.sim)
            
            if depth_images:
                # Stack and add to observations: [N, 4096] 
                depth_batch = torch.stack(depth_images, dim=0)  # [N, 4096]
                self.obs_buf = torch.cat([self.obs_buf, depth_batch], dim=-1)
        
        # Add noise if needed (applied to full observation including depth)
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
            depth_size = cfg.depth.resized[0] * cfg.depth.resized[1]  # 64*64 = 4096
            # Add zero noise for depth images (depth images shouldn't have noise added)
            depth_noise = torch.zeros(depth_size, device=self.device)
            # Concatenate: [48 base obs noise + 4096 depth noise (zeros)]
            result = torch.cat([noise_vec, depth_noise])
            return result
        else:
            return noise_vec
