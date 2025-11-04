from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

NUM_ENVS = 1536


class GO2ObsAvoidCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = NUM_ENVS
        num_observations = 48 + 49  # 48 deployable base state + 49 terrain heights
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 20.0
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        horizontal_scale = 0.25
        vertical_scale = 0.02
        slope_treshold = 1.0
        terrain_length = 20.0
        terrain_width = 20.0
        num_rows = 0
        num_cols = 0
        border_size = 10.0
        selected = False
        curriculum = False
        measure_horizontal_noise = 0.0
        measured_points_x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        measured_points_y = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    class obstacles:
        cube_size = (1.0, 0.5, 0.5)  # (length, width, height) for rectangular obstacles
        num_cubes = [50, 50, 30, 20]
        spawn_area = (-10.0, 10.0, -10.0, 10.0)
        min_distance_robot = 2.0
        min_distance_between = 2.0
        seed = None

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        randomize_yaw = True
        yaw_range = [-3.1416, 3.1416]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.6}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_upper"]
        terminate_after_contacts_on = ["base", "thigh", "calf","Head_upper"]  # Terminate on leg/body collisions
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1., 4.0]
        push_robots = False
        push_interval_s = 5.0
        max_push_vel_xy = 0.5

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

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 1e9  # Keep commands/goals fixed throughout an episode
        heading_command = True

        lin_vel_clip = 0.2
        ang_vel_clip = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]


    class goal:
        forward_range = [-5.0, 10.0]
        lateral_range = [-8.0, 8.0]
        reach_epsilon = 0.5
        speed_range = [0.55, 0.65]
        hold_time_s = 0.0  # Immediate reset once the goal radius is entered

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        # tracking_sigma = 0.2
        soft_dof_pos_limit = 0.9
        # soft_dof_vel_limit = 1.0
        # soft_torque_limit = 0.4
        base_height_target = 0.3
        

        class scales(LeggedRobotCfg.rewards.scales):
            # Goal tracking
            tracking_goal_vel = 1.4
            tracking_yaw = 0.5
            goal_alignment = 0.25  # Encourage body heading to align with goal direction

            # Regularization / safety
            lin_vel_z = -1.2
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -10.0
            action_rate = -0.15
            delta_torques = -2.0e-7
            torques = -1.0e-5
            hip_pos = -0.6
            dof_error = -0.04
            feet_stumble = -1.2
            feet_edge = -1.2

    class viewer(LeggedRobotCfg.viewer):
        show_goal = False
        ref_env = 0
        pos = [43.0, 20.0, 8.0]
        lookat = [20.0, 20.0, 1.0]


class GO2ObsAvoidCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        schedule = "fixed"
        learning_rate = 3e-4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "state_obsavoid_height49"
        experiment_name = "obsavoid_state_height49"
        save_interval = 1000
        max_iterations = 3000
