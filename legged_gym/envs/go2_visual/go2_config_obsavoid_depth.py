from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2ObsAvoidDepthCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 512
        num_observations = 48 + 84*84  # 48 state + 7056 depth = 7104 total
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 20.0  # 10m spacing to prevent cameras seeing neighboring robots
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m] - fixed at origin
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat] - will be randomized in reset_idx
        default_joint_angles = {  # target angles when action = 0.0
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 30.0}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2_obsavoid_depth"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

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

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.5, 4.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.5       
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27
        only_positive_rewards = False  # Allow negative rewards

        class scales(LeggedRobotCfg.rewards.scales):
            # Tracking rewards (main positive rewards)
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            # Regularization (keep robot stable)
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height = -0.3

            # Smoothness (prevent jittery motion)
            dof_acc = -2.5e-7
            action_rate = -0.01
            torques = -1e-5

            # Joint limits and collisions
            dof_pos_limits = -1.0
            collision = -0.5

            # Foot contact (encourage natural gait)
            feet_air_time = 0.5
            feet_stumble = -0.2
            feet_drag = -0.05

class GO2ObsAvoidDepthCfgPPO(LeggedRobotCfgPPO):
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

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'obsavoid_depth_go2'
        policy_class_name = 'VisualActorCritic'
        algorithm_class_name = 'PPO'
        save_interval = 1000
        max_iterations = 20000
