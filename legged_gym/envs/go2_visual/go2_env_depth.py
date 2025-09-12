from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config_depth import GO2DepthCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
import torchvision.transforms
import cv2
import threading

class GO2DepthRobot(LeggedRobot):
    """
    GO2 quadruped robot environment with depth camera support for visual RL.
    Extends the base LeggedRobot class to add depth camera functionality.
    """
    
    def __init__(self, cfg: GO2DepthCfg, sim_params, physics_engine, sim_device, headless):
        # Disable camera checks to avoid GPU-CPU transfers during training
        self.check_camera = False
        self.depth_image = None
        self.rgb_image = None
        self.raw_depth_image = None
        self.all_rgb_images = {}  # Store RGB images from all environments
        self.all_depth_images = {}  # Store depth images from all environments
        
        # Initialize camera-related attributes
        self.cam_handles = []
        
        # Camera display attributes (only for play mode)
        self.show_camera = False
        self.camera_window_name = "GO2 All RGB Cameras Feed"
        self.display_thread = None
        self.stop_display = threading.Event()
        self.camera_env_id = 0  # Which environment's camera to display
        self.show_all_cameras = False  # Show all cameras simultaneously
        
        if cfg.depth.use_camera:
            # Setup image resize transform for depth processing
            self.resize_transform = torchvision.transforms.Resize(
                (cfg.depth.resized[1], cfg.depth.resized[0]),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            print(f"Depth processing: {cfg.depth.original} -> {cfg.depth.resized}")
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize depth buffer for synchronous processing
        if cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(
                self.num_envs,
                cfg.depth.buffer_len,
                cfg.depth.resized[1],
                cfg.depth.resized[0]
            ).to(self.device)
            self.global_counter = 0
    
    def create_sim(self):
        """
        Creates simulation and sets up cameras.
        Calls parent create_sim and then initializes depth cameras if enabled.
        """
        super().create_sim()
        
        # Setup cameras after environment creation
        if self.cfg.depth.use_camera:
            self._setup_cameras()
    
    def _setup_cameras(self):
        """
        Setup depth cameras for all environments.
        Creates one camera per environment and attaches it to the robot base.
        """
        print(f"Setting up {self.num_envs} depth cameras...")
        
        # Configure camera properties based on D435i specifications
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.depth.original[0]
        camera_props.height = self.cfg.depth.original[1]
        camera_props.horizontal_fov = self.cfg.depth.horizontal_fov
        camera_props.near_plane = self.cfg.depth.near_clip
        camera_props.far_plane = self.cfg.depth.far_clip
        camera_props.enable_tensors = True  # Enable GPU tensor access
        
        for i in range(self.num_envs):
            # Create camera sensor for this environment
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            
            # Set camera position relative to robot base
            cam_pos = gymapi.Vec3(*self.cfg.depth.position)
            
            # Create camera rotation from angles (pitch, yaw)
            import math
            pitch = self.cfg.depth.angle[0]  # Rotation around y-axis (up/down)
            yaw = self.cfg.depth.angle[1]    # Rotation around z-axis (left/right)
            
            # Convert to quaternion (pitch rotation around Y axis)
            quat = gymapi.Quat()
            quat = gymapi.Quat.from_euler_zyx(0, pitch, 0)  # (roll, pitch, yaw)
            
            # Attach camera back to robot head
            try:
                body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], "Head_upper")
                attachment_point = "Head_upper"
            except:
                body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], 0)  # base body
                attachment_point = "base"
            self.gym.attach_camera_to_body(
                cam_handle, self.envs[i], body_handle, 
                gymapi.Transform(cam_pos, quat), 
                gymapi.FOLLOW_TRANSFORM
            )
            
            self.cam_handles.append(cam_handle)
            
            # Store attachment info for debug (only for first camera)
            if i == 0:
                self.camera_attachment = attachment_point
        
        print(f"Camera setup complete: {len(self.cam_handles)} cameras initialized")
        print(f"Camera config - FOV: {self.cfg.depth.horizontal_fov}, Near: {self.cfg.depth.near_clip}, Far: {self.cfg.depth.far_clip}")
        print(f"Camera position: {self.cfg.depth.position}, angle: {self.cfg.depth.angle}")
        print(f"Camera tilt: {self.cfg.depth.angle[0]} rad ({self.cfg.depth.angle[0]*57.3:.1f} degrees)")
        print(f"Camera attached to: {getattr(self, 'camera_attachment', 'unknown')}")
    
    
    def _update_depth_buffer(self):
        """
        Update depth buffer with new depth images from all cameras.
        Renders all camera sensors and processes depth data for each environment.
        """
        if not hasattr(self, 'cam_handles') or len(self.cam_handles) == 0:
            return
            
        # Render all cameras in single call for performance
        self.gym.render_all_camera_sensors(self.sim)
        
        # Process depth image for each environment
        for i in range(self.num_envs):
            if i < len(self.cam_handles):
                # Get both RGB and depth images
                rgb_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_COLOR
                )
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
                )
                
                if depth_tensor is None:
                    print(f"Warning: No depth data from camera {i}")
                    continue
                
                # Store RGB for display
                if rgb_tensor is not None:
                    if self.show_all_cameras:
                        self.all_rgb_images[i] = gymtorch.wrap_tensor(rgb_tensor)
                    elif i == self.camera_env_id:
                        self.rgb_image = gymtorch.wrap_tensor(rgb_tensor)
                
                # Convert depth to PyTorch tensor
                depth_image = gymtorch.wrap_tensor(depth_tensor)
                
                # Isaac Gym depth values are often negative and need special handling
                # Convert from Isaac Gym depth format to proper depth values
                depth_image = -depth_image  # Isaac Gym often gives negative depth
                
                # Clamp invalid values and replace with far clip distance
                depth_image = torch.where(torch.isfinite(depth_image) & (depth_image > 0), 
                                        depth_image, 
                                        torch.tensor(self.cfg.depth.far_clip, device=depth_image.device))
                
                # Store raw depth for display (before resize)
                if self.show_all_cameras:
                    self.all_depth_images[i] = depth_image.clone()
                elif i == self.camera_env_id:
                    self.raw_depth_image = depth_image.clone()
                
                if i == 0 and self.show_camera:  # Debug first environment always
                    depth_stats = depth_image.detach()
                    print(f"Env {i} processed depth range: {torch.min(depth_stats):.3f} - {torch.max(depth_stats):.3f}")
                    if rgb_tensor is not None:
                        rgb_stats = gymtorch.wrap_tensor(rgb_tensor).detach()
                        print(f"Env {i} RGB stats: min={torch.min(rgb_stats):.3f}, max={torch.max(rgb_stats):.3f}, mean={torch.mean(rgb_stats.float()):.3f}")
                
                # Resize from original resolution to target size
                depth_image = depth_image.unsqueeze(0).unsqueeze(0)
                depth_resized = torch.nn.functional.interpolate(
                    depth_image, size=(84, 84), mode='bilinear', align_corners=False
                ).squeeze()
                
                # Add realistic depth noise if configured
                if self.cfg.depth.dis_noise > 0:
                    noise = torch.randn_like(depth_resized) * self.cfg.depth.dis_noise
                    depth_resized = torch.clamp(depth_resized + noise, 
                                              self.cfg.depth.near_clip, self.cfg.depth.far_clip)
                
                # Store processed depth in buffer
                self.depth_buffer[i, 0] = depth_resized
        
        # Update camera display if enabled (only show first environment's camera)
        if self.show_camera and not self.stop_display.is_set():
            if self.display_thread is None or not self.display_thread.is_alive():
                self.stop_display.clear()
                self.display_thread = threading.Thread(target=self._display_camera_feed)
                self.display_thread.daemon = True
                self.display_thread.start()
                print(f"Camera display thread started - depth buffer shape: {self.depth_buffer.shape}")
    
    def compute_observations(self):
        """
        Computes observations including both state and depth data.
        Builds base state observations, applies noise, then appends depth data.
        """
        # Build base state observations (exactly as parent class)
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                self.base_ang_vel * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.commands[:, :3] * self.commands_scale,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                                ), dim=-1)
        
        # Add noise only to state observations (first 48 dimensions)
        if self.add_noise:
            # Create noise tensor with same size as current obs_buf (state only)
            state_noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:self.obs_buf.shape[1]]
            self.obs_buf += state_noise
        
        # Append depth observations if camera is enabled
        if self.cfg.depth.use_camera:
            # Update depth buffer at specified interval
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self._update_depth_buffer()
            
            # Get current depth frame and flatten for observation
            current_depth = self.depth_buffer[:, 0, :, :].flatten(start_dim=1)
            
            # Concatenate state observations with depth data
            self.obs_buf = torch.cat([self.obs_buf, current_depth], dim=1)
            
            self.global_counter += 1
    
    def enable_camera_display(self, env_id=0, show_all=False):
        """Enable live camera feed display for play mode only."""
        if not self.cfg.depth.use_camera:
            print("Camera not enabled in config, cannot show display")
            return
        
        if self.num_envs == 0:
            print("No environments available for camera display")
            return
        
        self.show_all_cameras = show_all
        
        if show_all:
            self.show_camera = True
            print(f"Live camera feed enabled - displaying ALL {self.num_envs} cameras simultaneously")
            print("Press 'q' in camera window to close, or ESC to exit")
        else:
            # Clamp env_id to valid range
            env_id = max(0, min(env_id, self.num_envs - 1))
            self.camera_env_id = env_id
                
            self.show_camera = True
            print(f"Live camera feed enabled - displaying camera from environment {env_id} (total envs: {self.num_envs})")
            print("Press 'q' in camera window to close, or ESC to exit")
    
    def disable_camera_display(self):
        """Disable camera feed display and clean up."""
        if self.show_camera:
            self.show_camera = False
            self.stop_display.set()
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join()
            cv2.destroyAllWindows()
            print("Camera display disabled")
    
    def _display_camera_feed(self):
        """Display live RGB camera feeds from environments."""
        while not self.stop_display.is_set() and self.show_camera:
            try:
                if self.show_all_cameras:
                    self._display_all_cameras()
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                        
            except Exception as e:
                print(f"Error in camera display: {e}")
                break
                
            # Small delay to prevent excessive CPU usage
            cv2.waitKey(1)
        
        # Cleanup
        cv2.destroyAllWindows()
    
    def _display_all_cameras(self):
        """Display all camera feeds in a grid layout."""
        if not self.all_rgb_images:
            return
            
        # Create grid layout (4x3 for 10 cameras, with 2 empty slots)
        grid_rows, grid_cols = 4, 3
        cell_size = 200
        
        # Create large canvas
        canvas = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)
        
        # Place each camera feed in grid
        for env_id in range(min(self.num_envs, grid_rows * grid_cols)):
            if env_id in self.all_rgb_images:
                row = env_id // grid_cols
                col = env_id % grid_cols
                
                # Get and process RGB image
                rgb_img = self.all_rgb_images[env_id].cpu().numpy()
                if rgb_img.size > 0 and not np.all(rgb_img == 0):
                    # Convert from RGBA to BGR for OpenCV display
                    if rgb_img.shape[-1] == 4:  # RGBA
                        rgb_img = rgb_img[:, :, :3]  # Remove alpha channel
                    
                    # Convert RGB to BGR for OpenCV
                    rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    
                    # Resize to cell size
                    rgb_resized = cv2.resize(rgb_img_bgr, (cell_size, cell_size))
                    
                    # Add environment label
                    cv2.putText(rgb_resized, f"Env {env_id}", (5, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Place in canvas
                    y_start = row * cell_size
                    y_end = y_start + cell_size
                    x_start = col * cell_size
                    x_end = x_start + cell_size
                    canvas[y_start:y_end, x_start:x_end] = rgb_resized
        
        # Add title
        cv2.putText(canvas, f"All {self.num_envs} Robot Cameras - Press 'q' to close", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow(self.camera_window_name, canvas)