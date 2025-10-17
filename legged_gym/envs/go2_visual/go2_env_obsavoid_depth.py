from legged_gym.envs.base.legged_robot import LeggedRobot
from .go2_config_obsavoid_depth import GO2ObsAvoidDepthCfg
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
import torchvision.transforms
import cv2
import threading
from isaacgym.torch_utils import quat_rotate_inverse


class GO2ObsAvoidDepthRobot(LeggedRobot):
    """GO2 quadruped with depth camera for obstacle avoidance.

    Observation layout (48 state dims + 7056 depth dims = 7104 total):
      State: [lin_vel_est(3), ang_vel(3), projected_gravity(3), commands(3),
              dof_pos_err(12), dof_vel(12), previous_action(12)]
      Depth: [84x84 depth image flattened = 7056]
    """

    def __init__(self, cfg: GO2ObsAvoidDepthCfg, sim_params, physics_engine, sim_device, headless):
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
            # Reset velocity estimator
            self.lin_vel_est_world[env_ids] = 0
            self.lin_vel_est[env_ids] = 0

            # Randomize yaw orientation (keep position at 0,0)
            # Generate random yaw angles for each environment being reset
            random_yaw = torch.rand(len(env_ids), device=self.device) * 2 * 3.14159  # 0 to 2π

            # Convert yaw to quaternion (rotation around z-axis)
            # quat = [x, y, z, w] where rotation is around z-axis
            quat_w = torch.cos(random_yaw / 2)
            quat_z = torch.sin(random_yaw / 2)

            # Set the base orientation (indices 3:7 are quaternion x,y,z,w)
            self.root_states[env_ids, 3] = 0  # quat_x
            self.root_states[env_ids, 4] = 0  # quat_y
            self.root_states[env_ids, 5] = quat_z
            self.root_states[env_ids, 6] = quat_w

            # Keep position at (0, 0) relative to env origin - override any randomization
            self.root_states[env_ids, 0] = self.env_origins[env_ids, 0]  # x = 0
            self.root_states[env_ids, 1] = self.env_origins[env_ids, 1]  # y = 0

            # Apply the updated root states
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )

    def _get_env_origins(self):
        """
        Override base class to center the grid around origin.
        Grid will range from [-5*num_rows, -5*num_cols] to [5*num_rows, 5*num_cols]
        instead of starting at [0, 0].
        """
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        # Create a grid of robots
        num_cols = int(np.floor(np.sqrt(self.num_envs)))
        num_rows = int(np.ceil(self.num_envs / num_cols))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
        spacing = self.cfg.env.env_spacing

        # Center the grid by subtracting half the grid dimensions
        x_offset = (num_rows - 1) / 2.0 * spacing
        y_offset = (num_cols - 1) / 2.0 * spacing

        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs] - x_offset
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs] - y_offset
        self.env_origins[:, 2] = 0.

        print(f"Environment grid centered: X range [{self.env_origins[:, 0].min():.1f}, {self.env_origins[:, 0].max():.1f}], "
              f"Y range [{self.env_origins[:, 1].min():.1f}, {self.env_origins[:, 1].max():.1f}]")

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

            # Create camera rotation from angles (pitch only, no yaw needed)
            pitch = self.cfg.depth.angle[0]  # Rotation around y-axis (up/down tilt)

            # Convert to quaternion (pitch rotation around Y axis)
            quat = gymapi.Quat.from_euler_zyx(0, pitch, 0)  # (roll, pitch, yaw)

            # Attach camera to robot head
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
        Each environment gets its own independent depth image.
        """
        if not hasattr(self, 'cam_handles') or len(self.cam_handles) == 0:
            return

        # Render all cameras in single call for performance
        self.gym.render_all_camera_sensors(self.sim)

        # Process depth image for EACH environment independently
        for i in range(self.num_envs):
            if i < len(self.cam_handles):
                # Get depth image for THIS specific environment's camera
                # IMPORTANT: Each env gets its own camera view based on robot position/orientation
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
                )

                if depth_tensor is None:
                    print(f"Warning: No depth data from camera {i}")
                    continue

                # Get RGB for display (optional)
                rgb_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_COLOR
                )

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

                # Store processed depth in buffer for THIS environment
                # Each environment index 'i' gets its own unique depth observation
                self.depth_buffer[i, 0] = depth_resized

        # Verification: Check that different environments have different depth data
        if self.global_counter % 100 == 0 and self.num_envs > 1:
            # Compare env 0 vs env 1 to verify they're independent
            diff = torch.abs(self.depth_buffer[0, 0] - self.depth_buffer[1, 0]).mean()
            print(f"[Step {self.global_counter}] Depth difference between env0 and env1: {diff:.4f}m (should be >0 if different)")

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
        Uses command-based velocity estimation instead of ground truth.
        """
        # Build base state observations using ESTIMATED velocity (not ground truth)
        self.obs_buf = torch.cat((
            self.lin_vel_est * self.obs_scales.lin_vel,            # 3 (estimated velocity)
            self.base_ang_vel * self.obs_scales.ang_vel,           # 3
            self.projected_gravity,                                 # 3
            self.commands[:, :3] * self.commands_scale,            # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,                # 12
            self.actions                                            # 12 (previous action)
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

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces."""
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
            4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1
        ).float()

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
