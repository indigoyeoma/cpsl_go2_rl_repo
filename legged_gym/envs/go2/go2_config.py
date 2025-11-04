from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 48
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
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
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

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

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # class commands:
    #     curriculum = False
    #     max_curriculum = 1.
    #     num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    #     resampling_time = 10. # time before command are changed[s]
    #     heading_command = True # if true: compute ang vel command from heading error
    #     class ranges:
    #         lin_vel_x = [0.0, 1.0] # min max [m/s]
    #         lin_vel_y = [-0.2, 0.2]   # min max [m/s]
    #         ang_vel_yaw = [-1, 1]    # min max [rad/s]
    #         heading = [-3.14, 3.14]
            
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35

        class scales(LeggedRobotCfg.rewards.scales):
            # Tracking rewards (main positive rewards)
            tracking_lin_vel = 1.0


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
            collision = -10.0

            # Foot contact (encourage natural gait)
            feet_air_time = 0.5
            feet_stumble = -0.2
            feet_drag = -0.05
            

class GO2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'teacher'
        save_interval = 1000
        max_iterations = 2000
