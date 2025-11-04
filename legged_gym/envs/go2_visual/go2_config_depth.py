from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

NUM_ENVS = 1024


class GO2DepthCfg(LeggedRobotCfg):
    """GO2 locomotion configuration extended with a depth camera input.

    Inherits base GO2 settings (flat plane terrain, default rewards) and adds depth camera.
    """

    class env(LeggedRobotCfg.env):
        num_envs = NUM_ENVS
        num_observations = 48 + 84*84
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 20.0
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

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.5, 4.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27


        class scales(LeggedRobotCfg.rewards.scales):
            # === PRIMARY OBJECTIVES (Positive Rewards) ===
            tracking_lin_vel = 2.5      # Main task: follow X,Y velocity commands
            tracking_ang_vel = 0.5      # Angular velocity tracking
            feet_air_time = 1.5         # Encourage natural gait

            # === STABILITY (Medium Penalties) ===
            orientation = -1.0          # Stay upright (critical!)
            base_height = -0.5          # Maintain proper height
            lin_vel_z = -2.0            # Prevent bouncing
            ang_vel_xy = -0.1           # Prevent roll/pitch oscillation

            # === SMOOTHNESS (Small Penalties) ===
            action_rate = -0.01         # Discourage jerky movements
            dof_acc = -2.5e-7           # Penalize joint acceleration
            torques = -5e-6             # Slightly increased for efficiency

            # === SAFETY (Strong Penalties) ===
            collision = -1.0            # Penalty for collisions
            dof_pos_limits = -10.0      # Prevent joint limit violations
            feet_stumble = -2.0         # Penalize foot hitting vertical surfaces
            feet_drag = -0.5            # Penalize dragging feet
            termination = -1.0          # Discourage early termination

    class depth:
        use_camera = True
        camera_num_envs = NUM_ENVS
        # Intel RealSense D435i depth config
        original = (424, 240)         # Render resolution
        resized = (84, 84)            # Downsampled input for CNN
        horizontal_fov = 86           # degrees
        near_clip = 0.3
        far_clip = 3.0
        # Mount pose (front, slight downward pitch)
        position = [0.3, 0.0, 0.147]
        position_rand = 0.01
        angle = [0.506, 0.0]          # default pitch (rad), yaw (rad)
        angle_pitch_range_deg = [24.0, 34.0]  # per-env randomized pitch in degrees
        # Cropping margins before resizing (pixels)
        crop_left = 0
        crop_right = 0
        crop_top = 0
        crop_bottom = 0
        # Noise and update
        dis_noise = 0.01
        update_interval = 8           # Render depth every 8 steps
        buffer_len = 1


class GO2DepthCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for GO2 with depth camera."""

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 7e-4
        entropy_coef = 0.01
        schedule = "fixed"
        num_mini_batches = 8

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "test"
        policy_class_name = "VisualActorCritic"
        algorithm_class_name = "PPO"
        max_iterations = 800
        save_interval = 800
