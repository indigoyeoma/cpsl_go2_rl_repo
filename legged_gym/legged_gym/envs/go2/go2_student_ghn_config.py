# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Go2 Student Config with GHN-sampled Depth Encoder
#
# This config uses a depth encoder architecture sampled from the GHN search space
# instead of the hand-designed SimpleDepthEncoder.

from legged_gym.envs.go2.go2_student_config import Go2StudentParkourCfg, Go2StudentParkourCfgPPO


class Go2StudentGHNCfg(Go2StudentParkourCfg):
    """
    Same as Go2StudentParkourCfg but uses GHN-sampled depth encoder.
    """

    class env(Go2StudentParkourCfg.env):
        pass  # Same as parent

    class depth(Go2StudentParkourCfg.depth):
        pass  # Same as parent


class Go2StudentGHNCfgPPO(Go2StudentParkourCfgPPO):
    """
    PPO config with GHN depth encoder architecture.
    """

    class algorithm(Go2StudentParkourCfgPPO.algorithm):
        entropy_coef = 0.01

    class depth_encoder(Go2StudentParkourCfgPPO.depth_encoder):
        if_depth = True
        learning_rate = 1e-3
        num_steps_per_env = 24

        # === GHN Encoder Config ===
        use_ghn_encoder = True  # Use GHN-sampled architecture instead of SimpleDepthEncoder

        # Architecture configuration (from rsl_rl.ghn2.DepthEncoderConfig)
        # Set to None to sample randomly, or specify a config dict
        ghn_config = {
            'channels': [32, 64],       # Channel sizes per conv layer
            'kernel_sizes': [5, 3],     # Kernel sizes per conv layer
            'strides': [1, 1],          # Strides per conv layer
            'pool_type': 'max',         # 'max', 'avg', or 'none'
            'pool_positions': [0],      # Which layers to add pooling after
            'activation': 'elu',        # 'elu', 'relu', 'lrelu', 'gelu'
            'norm': 'bn',               # 'bn', 'ln', 'none'
            'fc_hidden': 128,           # FC hidden dim (0 = direct projection)
            'dropout': 0.0,             # Dropout rate
        }

        # Alternative: use preset configs
        # Options: 'baseline', 'deep', 'wide', 'light', 'random'
        ghn_preset = None  # Set to override ghn_config

        # Input/output dimensions (should match environment)
        input_shape = (58, 87)  # Depth image resolution
        latent_dim = 32         # Output latent dimension (matches scan_encoder output)

    class runner(Go2StudentParkourCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_student_ghn'
        resume = True
        load_run = -1
        checkpoint = -1


# ============================================================================
# Preset Architecture Configs
# ============================================================================

class Go2StudentGHNDeepCfgPPO(Go2StudentGHNCfgPPO):
    """Deeper architecture variant."""

    class depth_encoder(Go2StudentGHNCfgPPO.depth_encoder):
        if_depth = True
        use_ghn_encoder = True
        ghn_config = {
            'channels': [32, 64, 64, 128],
            'kernel_sizes': [5, 3, 3, 3],
            'strides': [1, 1, 1, 1],
            'pool_type': 'max',
            'pool_positions': [0, 2],
            'activation': 'elu',
            'norm': 'bn',
            'fc_hidden': 128,
            'dropout': 0.0,
        }

    class runner(Go2StudentGHNCfgPPO.runner):
        experiment_name = 'go2_student_ghn_deep'


class Go2StudentGHNWideCfgPPO(Go2StudentGHNCfgPPO):
    """Wider architecture variant."""

    class depth_encoder(Go2StudentGHNCfgPPO.depth_encoder):
        if_depth = True
        use_ghn_encoder = True
        ghn_config = {
            'channels': [64, 128],
            'kernel_sizes': [5, 3],
            'strides': [1, 1],
            'pool_type': 'max',
            'pool_positions': [0],
            'activation': 'elu',
            'norm': 'bn',
            'fc_hidden': 256,
            'dropout': 0.0,
        }

    class runner(Go2StudentGHNCfgPPO.runner):
        experiment_name = 'go2_student_ghn_wide'


class Go2StudentGHNLightCfgPPO(Go2StudentGHNCfgPPO):
    """Lightweight architecture for fast inference."""

    class depth_encoder(Go2StudentGHNCfgPPO.depth_encoder):
        if_depth = True
        use_ghn_encoder = True
        ghn_config = {
            'channels': [16, 32],
            'kernel_sizes': [3, 3],
            'strides': [2, 2],
            'pool_type': 'none',
            'pool_positions': [],
            'activation': 'relu',
            'norm': 'bn',
            'fc_hidden': 64,
            'dropout': 0.0,
        }

    class runner(Go2StudentGHNCfgPPO.runner):
        experiment_name = 'go2_student_ghn_light'
