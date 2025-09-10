from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 128  # Balanced for performance and training
        # Base observations: 48 (proprioception, commands, actions)
        # Depth observations: 84*84 = 7056 (flattened depth image from 424x240 → 84x84)
        num_observations = 48 + 84*84  # 48 + 7056 = 7104 total
        num_privileged_obs = 48
        

        
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
        num_rows = 13  # 13x13 terrain grid (robots spawn in middle 11x11)
        num_cols = 13  # 13x13 terrain grid (robots spawn in middle 11x11)
        terrain_length = 5.  # Keep 5m patches
        terrain_width = 5.
        curriculum = False
    
    class depth:
        use_camera = True  # Enabled for visual RL
        camera_num_envs = 128  # Cameras for all 128 robots
        camera_terrain_num_rows = 11  # Robots spawn in middle 11x11 of 13x13 terrain
        camera_terrain_num_cols = 11  # Robots spawn in middle 11x11 of 13x13 terrain
        
        # Camera mounting position on Go2
        position = [0.27, 0, 0.08]  # 27cm forward, 8cm up (lower for better ground view)
        position_rand = 0.01  # Small position randomization for robustness
        angle = [0, 0]  # camera angle [min, max] (positive pitch down)
        
        # Camera resolution settings (D435i specs)
        original = (424, 240)  # D435i depth resolution
        resized = (84, 84)     # Final processed image size for CNN
        horizontal_fov = 87    # D435i horizontal FOV (87 degrees)
        
        # Depth range for obstacle detection (D435i specs)
        near_clip = 0.3   # 30cm minimum range
        far_clip = 3.0    # 3m maximum range 
        dis_noise = 0.01   # D435i realistic noise: ~1cm std dev (average across range)
        
        # No buffer needed - single frame processing
        update_interval = 1    # Process every frame
        buffer_len = 1         # single frame only
  
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [0.9, 1.1]      # Still mostly forward, slight variation
            lin_vel_y = [-0.5, 0.5]     # Side-stepping for dodging walls
            ang_vel_yaw = [-0., 0.]     # No rotation - pure strafing
            heading = [0.0, 0.0]        # Always face forward
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False  # Allow negative rewards for better gradient signal

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
            collision = -15.0     
            
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]  # Reduced network size for visual input
        critic_hidden_dims = [512, 256, 128]  # Reduced network size for visual input
        activation = 'elu'  # ELU often works better than ReLU for RL
        
        # HyperPPO specific parameters (commented out - using original)
        # meta_batch_size = 2  # Number of architectures per iteration
        # architecture_config_path = None  # Use default GO2 architectures
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'visual_go2'
        # Standard visual RL with CNN+MLP
        policy_class_name = 'ActorCritic'  # Standard ActorCritic
        algorithm_class_name = 'PPO'       # Standard PPO
        save_interval = 500  
        max_iterations = 100_000
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # Optimized for visual RL with CNN processing
        num_learning_epochs = 3  
        num_mini_batches = 4  # 256 envs ÷ 16 = 16 samples per batch (better for CNN training)

  
