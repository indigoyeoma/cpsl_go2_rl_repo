# Depth Encoder Architecture Search
#
# Sample diverse sequential CNN architectures for depth encoding.
#
# Usage:
#     from rsl_rl.ghn2 import sample_config, build_depth_encoder, DepthEncoder
#
#     # Sample a random architecture
#     config = sample_config()
#     encoder = build_depth_encoder(config, input_shape=(58, 87), latent_dim=32)
#
#     # Or use a baseline
#     from rsl_rl.ghn2 import BASELINE_CONFIG
#     encoder = build_depth_encoder(BASELINE_CONFIG)
#
#     # Forward: depth [B, 58, 87] -> latent [B, 32]
#     latent = encoder(depth_image)

from .config import (
    DepthEncoderConfig,
    sample_config,
    sample_configs,
    SEARCH_SPACE,
    BASELINE_CONFIG,
    DEEP_CONFIG,
    WIDE_CONFIG,
    LIGHT_CONFIG,
)

from .depth_encoder_net import (
    DepthEncoder,
    build_depth_encoder,
    build_depth_encoders_from_configs,
)

from .ops import ConvBlock, get_activation, get_norm, get_pool

__all__ = [
    # Config
    'DepthEncoderConfig',
    'sample_config',
    'sample_configs',
    'SEARCH_SPACE',
    'BASELINE_CONFIG',
    'DEEP_CONFIG',
    'WIDE_CONFIG',
    'LIGHT_CONFIG',
    # Network (backbone only, wrap with SimpleDepthEncoder for full encoder)
    'DepthEncoder',
    'build_depth_encoder',
    'build_depth_encoders_from_configs',
    # Ops
    'ConvBlock',
    'get_activation',
    'get_norm',
    'get_pool',
]
