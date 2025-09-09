from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 64  # Reduced for faster training
        # Base observations: 48 (proprioception, commands, actions)
        # Depth observations: 84*84 = 7056 (flattened depth image from 480x270 raw → 84x84 processed)
        num_observations = 48 + 84*84  # 48 + 7056 = 7104 total
        num_privileged_obs = 48
        # IMPORTANT: This triggers automatic visual mode in ActorCritic (observations > 48)
        # episode_length_s = 30  # Increase from default 20s to give more learning time
        
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
        border_size = 0  # No border - walls extend to entire area
        num_rows = 12  # 12x12 terrain grid (robots spawn in middle 8x8)
        num_cols = 12  # 12x12 terrain grid (robots spawn in middle 8x8)
        terrain_length = 5.  # Keep 5m patches
        terrain_width = 5.
        curriculum = False
    
    class depth:
        use_camera = True  # Enabled for visual RL
        camera_num_envs = 64  # Cameras for all 64 robots
        camera_terrain_num_rows = 8  # Robots spawn in middle 8x8 of 12x12 terrain
        camera_terrain_num_cols = 8  # Robots spawn in middle 8x8 of 12x12 terrain
        
        # Camera mounting position on Go2
        position = [0.27, 0, 0.08]  # 27cm forward, 8cm up (lower for better ground view)
        # IMPROVEMENT from Helpful-Doggybot: position = [0.3, 0, 0.147]  # 30cm forward, 14.7cm up
        position_rand = 0.01  # Small position randomization for robustness
        angle = [0, 0]  # camera angle [min, max] (positive pitch down)
        # IMPROVEMENT from Helpful-Doggybot: angle = [27, 32]  # Better downward angle for terrain
        
        # Camera resolution settings (following ManiSkill HyperPPO pattern)
        original = (480, 270)  # Raw camera image size (16:9 aspect ratio)
        resized = (84, 84)     # Final processed image size for CNN (matching ManiSkill)
        # Processing pipeline: 480x270 → 84x84 (bilinear interpolation downsampling)
        horizontal_fov = 87    # Wide FOV for obstacle detection
        # IMPROVEMENT: Could use 86 degrees to match references
        
        # Depth range for obstacle detection
        near_clip = 0.3   # 30cm minimum range
        # IMPROVEMENT: near_clip = 0.01 for closer obstacle detection
        far_clip = 3.0    # 3m maximum range for wall detection
        # IMPROVEMENT: far_clip = 3.5 for slightly longer range
        dis_noise = 0.0  # depth noise magnitude
        
        # No buffer needed - single frame processing
        update_interval = 1    # Process every frame
        buffer_len = 2         # temporal buffer length
  
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [0.9, 1.1]      # Still mostly forward, slight variation
            lin_vel_y = [-0.5, 0.5]     # Side-stepping for dodging walls
            ang_vel_yaw = [-0., 0.]     # No rotation - pure strafing
            heading = [0.0, 0.0]        # Always face forward
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            
            tracking_lin_vel = 5.0  # Add reward for tracking commanded velocities
            # orientation = -0.5
            
            # Height and stability
            base_height = -8.0          # Strong penalty for height deviation
            orientation = -2.0          # Keep body level
            lin_vel_z = -2.0           # Minimize vertical velocity

            # Collision avoidance
            collision = -15.0           # Very strong penalty for collisions
            
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        # Standard network dimensions 
        actor_hidden_dims = [512, 256, 128]  
        critic_hidden_dims = [512, 256, 128]  
        activation = 'elu'
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'hyper_go2'
        # Use standard ActorCritic - HyperAgent wrapper will handle HyperPPO
        policy_class_name = 'ActorCritic'  
        algorithm_class_name = 'PPO'  
        save_interval = 100  # Save more frequently for monitoring
        max_iterations = 10000  # Reduced for initial testing
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # Standard PPO hyperparameters
        learning_rate = 1e-4  
        entropy_coef = 0.01  
        num_learning_epochs = 3  
        num_mini_batches = 4  
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        max_grad_norm = 0.5  
        desired_kl = 0.01
        schedule = 'adaptive'

  
