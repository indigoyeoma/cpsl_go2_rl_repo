from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 100  # 100 robots for training
        num_privileged_obs = 48
        num_actions = 12
        episode_length_s = 30  # Longer episodes for navigation learning
        
        # GPU acceleration
        sim_device = 'cuda:0'
        rl_device = 'cuda:0'
        
        # Visual RL parameters
        n_proprio = 48  # proprioceptive observations
        n_scan = 132    # scan/lidar observations (if used)
        n_priv = 6      # privileged observations 
        n_priv_latent = 4 # privileged latent dimensions
        history_len = 10  # history length for temporal encoding
        
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
        num_rows = 10  # 2x increase from 5
        num_cols = 10  # 2x increase from 5
        terrain_length = 5.  # Keep 5m patches
        terrain_width = 5.
        curriculum = False
    
    class depth:
        use_camera = True  # Enabled for visual RL
        camera_num_envs = 100  # Cameras for all 100 robots
        camera_terrain_num_rows = 10  # 10x10 grid for 100 robots
        camera_terrain_num_cols = 10  # 10x10 grid for 100 robots
        
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
        collision_threshold = 1.0  # Trigger penalty when walls closer than 1m
        front_collision_weight = 1.0  # Weight for front-facing collision detection
        
        # No buffer needed - single frame processing
        update_interval = 1    # Process every frame
        buffer_len = 2         # temporal buffer length
  
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [1.0, 1.0]   # constant forward velocity [m/s] - no variation
            lin_vel_y = [-0.3, 0.3]  # minimal lateral for dodging [m/s] 
            ang_vel_yaw = [-0.2, 0.2]  # small turns only for dodging [rad/s]
            heading = [0.0, 0.0]    # always straight ahead
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):
            # Original essentials
            torques = -0.0002
            dof_pos_limits = -10.0

            # Your task: move forward, avoid walls
            tracking_lin_vel = 5.0   # Big reward for moving forward
            collision = -2.0         # Big penalty for hitting walls
            
            # Maintain good posture while walking
            base_height = -1.0       # Light penalty for wrong height (target is 0.25m)
            orientation = -1.0       # Light penalty for tilting

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-4  # Reduced from default 1.e-3 for stability
        
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]  # Original architecture
        critic_hidden_dims = [512, 256, 128]  # Original architecture
        activation = 'relu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        depth_image_shape = (64, 64)  # Standard square input
        depth_latent_dim = 128  # Reasonable latent dimension for 64x64 input
        scan_encoder_dims = [128, 64, 32]
        priv_encoder_dims = [64, 20]
        tanh_encoder_output = False
        share_cnn = True  # Share CNN encoder between actor and critic
        
    class estimator:
        hidden_dims = [256, 128, 64]
        dagger_update_freq = 10

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        policy_class_name = 'VisualActorCritic'  # Visual policy for depth camera input
        save_interval = 500  # Save checkpoint every 500 iterations
        max_iterations = 1000000

  
