from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2HyperCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 48 + 84*84  # 48 state + 7056 depth = 7104 total
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2_hyper"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf","body"]
        terminate_after_contacts_on = []  # Removed early termination to ensure full episodes
        self_collisions = 1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005

    class depth:
        use_camera = True
        camera_num_envs = 1024
        position = [0.05, 0, 0.02]
        position_rand = 0.01
        angle = [0, 0]
        original = (424, 240)
        resized = (84, 84)
        horizontal_fov = 87
        near_clip = 0.3
        far_clip = 3.0
        dis_noise = 0.01
        update_interval = 10
        buffer_len = 1
  
    
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            
class GO2HyperCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        
        critic_hidden_dims = [512, 256, 128]  # Only critic dims are fixed (architecture-conditioned)
        activation = 'elu'
        
        # HyperPPO specific parameters
        architecture_config_path = "rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth84.json"
        meta_batch_size = 8
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        schedule = 'fixed'
        learning_rate = 7e-4
        num_learning_epochs = 3
        num_mini_batches = 16
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'hyper_go2'
        policy_class_name = 'HyperPPOActorCritic'
        algorithm_class_name = 'HyperPPO'
        max_iterations = 10000

  
