# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Go2 joint limits from SDK
# go2_const_dof_range = dict(
#     Hip_max=1.0472, Hip_min=-1.0472,
#     Front_Thigh_max=3.4907, Front_Thigh_min=-1.5708,
#     Rear_Thigh_max=4.5379, Rear_Thigh_min=-0.5236,
#     Calf_max=-0.83776, Calf_min=-2.7227,
# )


class Go2ParkourCfg(LeggedRobotCfg):
    """Go2 Teacher config - uses privileged terrain scans (no depth camera)."""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        history_len = 10
        history_encoding = True

        # Observation: proprio(53) + scan(132) + priv_latent(29) + priv(9) + history(530) = 753
        n_scan = 132
        n_priv = 9            # friction(3) + body_vel(3) + body_height(3)
        n_priv_latent = 29    # feet_contact(4) + terrain_height(1) + feet_pos(12) + thigh_contact(12)
        n_proprio = 53        # projected_gravity(3) + commands(2) + joint_angles(3) + joint_vels(4) + actions(36) + contact(5)
        num_observations = n_proprio + n_scan + n_priv_latent + n_priv + history_len * n_proprio

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,   'FR_hip_joint': -0.1,
            'RL_hip_joint': 0.1,   'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}
        damping = {'joint': 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        # sdk_dof_range = go2_const_dof_range
        # dof_velocity_override = 35.

    class noise(LeggedRobotCfg.noise):
        add_noise = False  # Teacher uses privileged info
        noise_level = 1.0
        quantize_height = True

        class noise_scales:
            rotation = 0.05
            ang_vel = 0.2
            gravity = 0.06
            dof_pos = 0.0006
            dof_vel = 0.02
            lin_vel = 0.0
            height_measurements = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 2.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        randomize_gravity_bias = True
        gravity_bias_range = [-0.1, 0.1]
        action_delay = True
        action_delay_range = [0, 2]

    class depth(LeggedRobotCfg.depth):
        use_camera = False  # Teacher uses privileged scans

    class terrain(LeggedRobotCfg.terrain):
        terrain_dict = {
            "smooth slope": 0.,
            "rough slope up": 0.,
            "rough slope down": 0.,
            "rough stairs up": 0.,
            "rough stairs down": 0.,
            "discrete": 0.,
            "stepping stones": 0.,
            "gaps": 0.,
            "smooth flat": 0.,
            "pit": 0.,
            "wall": 0.,
            "platform": 0.,
            "large stairs up": 0.,
            "large stairs down": 0.,
            "parkour": 0.2,
            "parkour_hurdle": 0.2,
            "parkour_flat": 0.2,
            "parkour_step": 0.2,
            "parkour_gap": 0.2,
            "demo": 0.,
        }
        terrain_proportions = list(terrain_dict.values())

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25


class Go2ParkourCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_teacher'
