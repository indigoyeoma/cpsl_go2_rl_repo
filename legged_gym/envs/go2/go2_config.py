from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 100  # 100 robots to avoid placement issues
        # Base observations: 48 (proprioception, commands, actions)
        # Depth observations: 64*64 = 4096 (flattened depth image) 
        num_observations = 48 + 64*64  # 48 + 4096 = 4144 total
        num_privileged_obs = 48
        episode_length_s = 30  # Increase from default 20s to give more learning time
        
        # GPU acceleration
        sim_device = 'cuda:0'
        rl_device = 'cuda:0'

        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2_locomotion"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]  # Only terminate on hard collisions
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # 100% walls only - simple dodging
        border_size = 10
        num_rows = 8  # 8x16 grid for reduced training
        num_cols = 16  # 8x16 grid for reduced training
        terrain_length = 5.  # Keep 5m patches
        terrain_width = 5.
        curriculum = False
    
    class depth:
        use_camera = True  # Enabled for visual RL
        camera_num_envs = 100  # Cameras for all 100 robots
        camera_terrain_num_rows = 8  # 8x16 grid for 128 robots
        camera_terrain_num_cols = 16  # 8x16 grid for 128 robots
        
        # Camera mounting position on Go2
        position = [0.27, 0, 0.08]  # 27cm forward, 8cm up (lower for better ground view)
        position_rand = 0.01  # Small position randomization for robustness
        angle = [0, 0]  # camera angle [min, max] (positive pitch down)
        
        # Camera resolution settings
        original = (106, 60)   # Original camera image size
        resized = (64, 64)     # Resized image size for processing
        horizontal_fov = 87    # Wide FOV for obstacle detection
        
        # Depth range for obstacle detection
        near_clip = 0.3   # 30cm minimum range
        far_clip = 3.0    # 3m maximum range for wall detection
        dis_noise = 0.0  # depth noise magnitude
        
        # Collision detection thresholds
        # collision_threshold = 1.0  # Trigger penalty when walls closer than 1m
        # front_collision_weight = 1.0  # Weight for front-facing collision detection
        
        # No buffer needed - single frame processing
        update_interval = 1    # Process every frame
        buffer_len = 2         # temporal buffer length
  
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [0.8, 1.2]      # Still mostly forward, slight variation
            lin_vel_y = [-1.0, 1.0]     # Side-stepping for dodging walls
            ang_vel_yaw = [-1.0, 1.0]   # Moderate yaw for turning around obstacles
            heading = [-1.0, 1.0]       # Allow heading changes for navigation
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):
            
            # Original essentials
            torques = -0.0002
            dof_pos_limits = -10.0

            # Core locomotion rewards
            # tracking_lin_vel = 5.0      # Start VERY low - just learn to stand first!
            # standing = 3.0               # Removed - not helping
            
            # Movement encouragement
            # feet_air_time = 0.1        # Reward stepping motion
            # action_rate = -0.01         # Smooth actions
            
            # Collision handling  
            # collision = -1.0            # Start gentle - learn to stand first!
            # depth_wall_avoidance = 1.0  # Visual wall proximity penalty
            # termination = -20.0         # Big penalty for falling completely
            
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    # class algorithm( LeggedRobotCfgPPO.algorithm ):
        # entropy_coef = 0.005  # Fixed: was 0.001 - need exploration!
        # learning_rate = 1.e-4  # Reduced from default 1.e-3 for stability
        
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [1024, 512, 256]  # Good balance for locomotion control
        critic_hidden_dims = [1024, 512, 256]  # Matching actor network size
        activation = 'relu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        depth_image_shape = (64, 64)  # Standard square input
        depth_latent_dim = 256  # Sufficient for wall detection features
        scan_encoder_dims = [128, 64, 32]  # Standard encoder size
        priv_encoder_dims = [64, 20]  # Standard privileged encoder
        tanh_encoder_output = False
        share_cnn = True  # Share CNN encoder between actor and critic
        
        # CNN architecture optimized for wall detection (if customizable)
        cnn_channels = [32, 64, 128]  # 3-layer CNN: edge detection → shapes → objects
        
    class estimator:
        hidden_dims = [256, 128, 64]
        dagger_update_freq = 10

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        policy_class_name = 'ActorCritic'  # Use original ActorCritic with SimpleCNN depth processing
        save_interval = 500  
        max_iterations = 1000000

  
