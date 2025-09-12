from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2DepthCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512
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
        name = "go2_depth"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # Ensure we have a floor for the depth camera
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]

    class depth:
        use_camera = True
        camera_num_envs = 512
        position = [0.05, 0, 0.02]  # D435i on head front: 5cm forward, 2cm up from head center
        position_rand = 0.01
        angle = [0, 0]  # Level with head - D435i typically mounted horizontally
        original = (424, 240)
        resized = (84, 84)
        horizontal_fov = 87
        near_clip = 0.3
        far_clip = 3.0
        dis_noise = 0.01
        update_interval = 4
        buffer_len = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

class GO2DepthCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        # Fixed learning rate for stability  
        schedule = 'fixed'
        learning_rate = 7e-4
        
        # num_mini_batches = 3
        # num_learning_epochs = 8
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'depth_go2'
        policy_class_name = 'VisualActorCritic'
        algorithm_class_name = 'PPO'
        max_iterations = 10000