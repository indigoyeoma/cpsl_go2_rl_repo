from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config_hyper import GO2HyperCfg
from isaacgym import gymapi, gymtorch
import torch
import torchvision.transforms
import threading

class GO2HyperRobot(LeggedRobot):
    """
    GO2 quadruped robot environment with HyperPPO and depth camera support.
    Extends the base LeggedRobot class to add depth camera functionality for HyperPPO visual RL.
    """
    
    def __init__(self, cfg: GO2HyperCfg, sim_params, physics_engine, sim_device, headless):
        # Disable camera checks to avoid GPU-CPU transfers during training
        self.check_camera = False
        self.depth_image = None
        self.rgb_image = None
        self.raw_depth_image = None
        self.all_rgb_images = {}
        self.all_depth_images = {}
        
        # Initialize camera-related attributes
        self.cam_handles = []
        
        # Camera display attributes (only for play mode)
        self.show_camera = False
        self.camera_window_name = "GO2 HyperPPO Camera Feed"
        self.display_thread = None
        self.stop_display = threading.Event()
        self.camera_env_id = 0
        self.show_all_cameras = False
        
        if cfg.depth.use_camera:
            # Setup image resize transform for depth processing
            self.resize_transform = torchvision.transforms.Resize(
                (cfg.depth.resized[1], cfg.depth.resized[0]),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            print(f"HyperPPO depth processing: {cfg.depth.original} -> {cfg.depth.resized}")
        
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
        print(f"Setting up {self.num_envs} depth cameras for HyperPPO...")
        
        # Configure camera properties based on D435i specifications
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.depth.original[0]
        camera_props.height = self.cfg.depth.original[1]
        camera_props.horizontal_fov = self.cfg.depth.horizontal_fov
        camera_props.near_plane = self.cfg.depth.near_clip
        camera_props.far_plane = self.cfg.depth.far_clip
        camera_props.enable_tensors = True
        
        for i in range(self.num_envs):
            # Create camera sensor for this environment
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            
            # Set camera position relative to robot base
            cam_pos = gymapi.Vec3(*self.cfg.depth.position)
            
            # Create camera rotation from angles (pitch, yaw)
            pitch = self.cfg.depth.angle[0]
            
            # Convert to quaternion (pitch rotation around Y axis)
            quat = gymapi.Quat()
            quat = gymapi.Quat.from_euler_zyx(0, pitch, 0)
            
            # Attach camera to robot head
            try:
                body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], "Head_upper")
                attachment_point = "Head_upper"
            except:
                body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], 0)
                attachment_point = "base"
            self.gym.attach_camera_to_body(
                cam_handle, self.envs[i], body_handle, 
                gymapi.Transform(cam_pos, quat), 
                gymapi.FOLLOW_TRANSFORM
            )
            
            self.cam_handles.append(cam_handle)
            
            if i == 0:
                self.camera_attachment = attachment_point
        
        print(f"HyperPPO camera setup complete: {len(self.cam_handles)} cameras initialized")
        print(f"Camera config - FOV: {self.cfg.depth.horizontal_fov}, Near: {self.cfg.depth.near_clip}, Far: {self.cfg.depth.far_clip}")
        print(f"Camera position: {self.cfg.depth.position}, angle: {self.cfg.depth.angle}")
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
                # Get depth image
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
                )
                
                if depth_tensor is None:
                    continue
                
                # Convert depth to PyTorch tensor
                depth_image = gymtorch.wrap_tensor(depth_tensor)
                
                # Isaac Gym depth values handling
                depth_image = -depth_image
                
                # Clamp invalid values
                depth_image = torch.where(torch.isfinite(depth_image) & (depth_image > 0), 
                                        depth_image, 
                                        torch.tensor(self.cfg.depth.far_clip, device=depth_image.device))
                
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
    
    def compute_observations(self):
        """
        Computes observations including both state and depth data for HyperPPO.
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