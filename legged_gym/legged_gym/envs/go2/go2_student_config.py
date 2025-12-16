# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2StudentParkourCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # Proprio history for temporal info (gait phase, acceleration, contacts)
        # Flattened as MLP input (no RNN) - matches teacher config
        history_len = 10
        history_encoding = True

        # Observation dimensions
        # proprio(53) + scan(132) + priv_latent(29) + priv(9) + history(53*10=530) = 753
        n_scan = 132
        n_priv = 3 + 3 + 3  # 9
        n_priv_latent = 4 + 1 + 12 + 12  # 29
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 53
        num_observations = n_proprio + n_scan + n_priv_latent + n_priv + history_len * n_proprio  # 753

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class noise( LeggedRobotCfg.noise ):
        # Enable sensor noise for sim-to-real transfer
        # Values calibrated from real-world measurements (parkour repo)
        add_noise = True
        noise_level = 1.0  # Global scaling factor for all noise
        quantize_height = True  # Quantize height measurements to simulate LiDAR resolution

        class noise_scales:
            # IMU noise - based on real-world measurements
            rotation = 0.05   # Roll/pitch noise [rad] - IMU orientation uncertainty
            ang_vel = 0.2     # Angular velocity noise [rad/s] - measured ~0.02, use 0.2 for robustness
            gravity = 0.06    # Gravity vector estimation noise - measured ~0.05

            # Joint encoder noise - measured from real Go1/Go2
            dof_pos = 0.0006  # Joint position noise [rad] - measured ~0.0002
            dof_vel = 0.02    # Joint velocity noise [rad/s] - measured ~0.015

            # Base velocity estimation noise - from state estimator
            lin_vel = 0.0     # Student doesn't use privileged lin_vel

            # Height/terrain perception noise
            height_measurements = 0.0  # Student uses depth camera, not height scans
            forward_depth = 0.0  # Depth has its own noise model below

        # Stereo depth camera noise model (D435i characteristics)
        class forward_depth:
            # Stereo matching limitations
            stereo_min_distance = 0.175  # [m] Below this, stereo matching fails
            stereo_far_distance = 2.0    # [m] Beyond this, noise increases significantly

            # Depth noise (distance-dependent)
            stereo_far_noise_std = 0.08   # Noise std for far pixels (>stereo_far_distance)
            stereo_near_noise_std = 0.02  # Noise std for near pixels

            # Block artifacts (stereo matching failures)
            stereo_full_block_artifacts_prob = 0.004  # Prob of full block artifacts
            stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]  # Possible artifact values
            stereo_full_block_height_mean_std = [62, 1.5]  # Block height distribution
            stereo_full_block_width_mean_std = [3, 0.01]   # Block width distribution

            # Spark artifacts (random bright pixels)
            stereo_half_block_spark_prob = 0.02
            stereo_half_block_value = 3000  # Max depth value

            # Sky artifacts (incorrect far readings)
            sky_artifacts_prob = 0.0001
            sky_artifacts_far_distance = 2.0  # Pixels beyond this may have sky artifacts
            sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            sky_artifacts_height_mean_std = [2, 3.2]
            sky_artifacts_width_mean_std = [2, 3.2]

    class depth( LeggedRobotCfg.depth ):
        # STUDENT CONFIG: Uses depth camera (D435i)
        use_camera = True

        # D435i mounted on robot's head

        # Position relative to base_link: [forward, left/right, up]
        # From official Unitree URDF: front_camera_joint xyz="0.32715 -0.00003 0.04297"
        position = [0.327, 0.0, 0.043]  # Official Go2 front camera position
        angle = [-15, 15]  # Pitch angle range (degrees) - positive is down

        # D435i specifications (matching parkour deployment settings)
        # Native resolution: 640x480 in deployment
        # Simulation uses smaller resolution for efficiency
        original = (106, 60)  # Simulation camera resolution (before crop)
        resized = (87, 58)    # Final output resolution (after crop & resize)
        horizontal_fov = 87   # D435i horizontal FOV: 87°

        # Cropping settings (matching parkour go2_visual.py ratios)
        # Parkour uses 640x480 with crop_top=48, crop_left=28, crop_right=36, crop_bottom=0
        # Proportions: top=10%, left=4.4%, right=5.6%, bottom=0%
        crop_top = 6      # 60 * 0.10 = 6 (matches parkour 10% top crop)
        crop_bottom = 0   # No bottom crop (matches parkour)
        crop_left = 5     # 106 * 0.044 ≈ 5 (matches parkour 4.4% left crop)
        crop_right = 6    # 106 * 0.056 ≈ 6 (matches parkour 5.6% right crop)

        # D435i depth range: 0.3m (min) to 3m (optimal max)
        near_clip = 0.3  # D435i minimum depth distance
        far_clip = 3.0   # D435i maximum reliable depth distance

        # Use only single frame (no temporal buffering)
        buffer_len = 1  # Single frame, feedforward encoder

        update_interval = 4  # Camera update frequency
        dis_noise = 0.01  # Depth sensor noise for sim-to-real (applied in env)

        scale = 1
        invert = True

        # Camera latency simulation (USB2.0 D435i on Go2)
        # Real-world latency: 250-300ms for USB2.0, 80-140ms for USB3.0
        latency_range = [0.08, 0.142]  # [s] USB3.0 latency range
        latency_resampling_time = 5.0  # [s] Resample latency periodically
        refresh_duration = 0.1  # [s] Camera refresh rate (10 Hz)

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # Domain randomization for sim-to-real transfer (STUDENT)
        # Same as teacher but NO push_robots (distillation should be stable)
        randomize_friction = True
        friction_range = [0.2, 2.0]  # Wide range for robustness

        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]  # Add up to 3kg payload

        randomize_base_com = True
        added_com_range = [-0.2, 0.2]  # COM offset in x,y,z

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]  # Motor strength variation

        # NO push during distillation - student learns from stable teacher demos
        push_robots = False

        # Gravity bias randomization (simulates IMU drift)
        randomize_gravity_bias = True
        gravity_bias_range = [-0.1, 0.1]  # Gravity vector bias

        # Action delay (simulates communication latency)
        action_delay = True
        action_delay_range = [0, 2]  # Delay in policy steps (0-2 steps)

        # Proprioception latency (5-45ms typical for Go2)
        proprioception_latency_range = [0.005, 0.045]  # [s]
        proprioception_latency_resampling_time = 5.0  # [s]

    class terrain( LeggedRobotCfg.terrain ):
        # Train on ALL 5 parkour terrains equally (matches extreme-parkour)
        terrain_dict = {
            "smooth slope": 0.,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.,
            "rough stairs down": 0.,
            "discrete": 0.,
            "stepping stones": 0.0,
            "gaps": 0.,
            "smooth flat": 0,
            "pit": 0.0,
            "wall": 0.0,
            "platform": 0.,
            "large stairs up": 0.,
            "large stairs down": 0.,
            "parkour": 0.2,           # 20% - mixed parkour
            "parkour_hurdle": 0.2,    # 20% - hurdles to jump over
            "parkour_flat": 0.2,      # 20% - flat walking sections
            "parkour_step": 0.2,      # 20% - steps/stairs
            "parkour_gap": 0.2,       # 20% - gaps to leap across
            "demo": 0.0,
        }
        terrain_proportions = list(terrain_dict.values())

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25


class Go2StudentParkourCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class depth_encoder( LeggedRobotCfgPPO.depth_encoder ):
        if_depth = True
        learning_rate = 1e-3
        num_steps_per_env = 24

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_student'
        # For training: pass --load_run /path/to/teacher to load teacher checkpoint
        # For playing: will load from logs/go2_student/go2_student automatically
        resume = True
        load_run = -1  # Override via --load_run for training
        checkpoint = -1  # Use latest checkpoint
