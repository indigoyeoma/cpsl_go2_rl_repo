from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 1024  # Set to 1024 environments
        num_observations = 48 + 84*84  # 48 + 7056 = 7104 total
        num_privileged_obs = 48
        env_spacing = 8.0  # 5m spacing between robots for safety

        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        random_yaw = True  # Enable random yaw for varied initial orientations
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
        mesh_type = 'plane'  # Simple infinite plane for fastest collection
        curriculum = False
    
    class depth:
        use_camera = True  # Enabled for visual RL
        camera_num_envs = 1024  # Cameras for 1024 robots
        camera_terrain_num_rows = 32   # Grid for 1024 robots (32x32 = 1024)
        camera_terrain_num_cols = 32   # Grid for 1024 robots
        
        # Camera mounting position on Go2
        position = [0.27, 0, 0.08]  # 27cm forward, 8cm up (lower for better ground view)
        position_rand = 0.01  # Small position randomization for robustness
        angle = [0, 0]  # camera angle [min, max] (positive pitch down)
        
        # Camera resolution settings (D435i specs) - KEEP 84x84 FOR SIM2REAL
        original = (424, 240)  # D435i depth resolution
        resized = (84, 84)     # Keep 84x84 for sim2real transfer
        horizontal_fov = 87    # D435i horizontal FOV (87 degrees)
        
        # Depth range for obstacle detection (D435i specs)
        near_clip = 0.3   # 30cm minimum range
        far_clip = 3.0    # 3m maximum range 
        dis_noise = 0.01   # D435i realistic noise: ~1cm std dev (average across range)
        
        # No buffer needed - single frame processing
        update_interval = 1    # Process every frame for best sim2real
        buffer_len = 1         # single frame only
  
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [0.6, 1.0]      # Wider speed range for variety
            lin_vel_y = [-0.1, 0.1]     # Moderate side-stepping
            ang_vel_yaw = [0.0, 0.0]   # Allow turning for orientation correction
            heading = [0.0, 0.0]     # Full rotation range for diverse training
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25  # Increased from 0.3 to prevent body dragging
        only_positive_rewards = True  # Allow negative rewards for better gradient signal

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        # HyperPPO network dimensions 
        actor_hidden_dims = [512, 256, 128]  
        critic_hidden_dims = [512, 256, 128]  
        activation = 'elu'
        
        # HyperPPO specific parameters
        architecture_config_path = "rsl_rl/rsl_rl/hyperppo/configs_go2/architecture_go2_baseline.json"
        meta_batch_size = 2  # Number of architectures per iteration
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'hyper_go2'
        # HyperPPO visual RL with CNN+MLP
        policy_class_name = 'HyperPPOActorCritic'  # HyperPPO ActorCritic
        algorithm_class_name = 'HyperPPO'          # HyperPPO algorithm
        save_interval = 500  
        max_iterations = 100_000
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # Optimized for visual RL with CNN processing
        num_learning_epochs = 3  # Increased for better visual feature learning
        num_mini_batches = 16  # 1024 envs ÷ 16 = 64 samples per batch (optimal for visual learning)
        
        # HyperPPO specific settings
        value_loss_coef = 1.0    # Standard value loss coefficient
        entropy_coef = 0.01      # Standard entropy coefficient
        learning_rate = 3e-4     # Standard learning rate

  
