# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Go2 GHN (Graph HyperNetwork) Training Config
#
# This config trains a GHN to predict depth encoder weights for arbitrary architectures.
# The GHN learns to generalize across different backbone architectures.
#
# Training flow:
# 1. Sample random backbone architectures for each environment
# 2. GHN predicts weights for each architecture
# 3. Student executes actions using GHN-predicted backbones (DAgger)
# 4. Teacher provides action labels (privileged)
# 5. Loss = ||student_action - teacher_action||
# 6. Backprop through GHN to update its weights
#
# For regular fixed-backbone student training, use go2_student_config.py instead.

from legged_gym.envs.go2.go2_student_config import Go2StudentParkourCfg, Go2StudentParkourCfgPPO


class Go2StudentGHNCfg(Go2StudentParkourCfg):
    """
    Environment config for GHN training.
    Inherits ALL settings from Go2StudentParkourCfg (same depth, noise, etc.)
    Sets num_envs = camera_num_envs so all envs have cameras (no wasted simulation).
    """

    class env(Go2StudentParkourCfg.env):
        num_envs = 256  # Match camera_num_envs for GHN training

    class depth(Go2StudentParkourCfg.depth):
        camera_num_envs = 256  # All envs have cameras for GHN


class Go2StudentGHNCfgPPO(Go2StudentParkourCfgPPO):
    """
    PPO config with GHN training.
    Inherits ALL settings from Go2StudentParkourCfgPPO, adds GHN-specific settings.
    """

    class depth_encoder(Go2StudentParkourCfgPPO.depth_encoder):
        if_depth = True
        learning_rate = 1e-3
        num_steps_per_env = 24

        # === GHN Training Settings ===
        num_parallel_architectures = 8  # Number of random architectures sampled per batch
        # GHN hyperparameters (hid=64, num_classes=32) are set in on_policy_runner.py
        # Increase num_parallel_architectures for more diversity (but slower training)

    class runner(Go2StudentParkourCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_student_ghn'
        resume = True   # Must be True to load teacher checkpoint!
        load_run = -1   # Pass --load_run /path/to/teacher to load teacher checkpoint
        checkpoint = -1
