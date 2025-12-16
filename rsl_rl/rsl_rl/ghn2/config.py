# Depth Encoder Architecture Search - Configuration
#
# Configuration space and sampling for sequential depth encoder architectures.

import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DepthEncoderConfig:
    """
    Configuration for a sequential depth encoder architecture.

    Architecture: depth [1, 58, 87] -> CNN -> latent [32]

    Attributes:
        channels: list of output channels for each conv layer
        kernel_sizes: list of kernel sizes for each conv layer
        strides: list of strides for each conv layer (for downsampling)
        pool_type: pooling type after conv blocks ('max', 'avg', 'none')
        pool_positions: which layers to add pooling after (indices)
        activation: activation function ('elu', 'relu', 'lrelu', 'gelu')
        norm: normalization type ('bn', 'ln', 'none')
        fc_hidden: hidden dim for FC layer (0 = direct projection)
        dropout: dropout rate (0 = no dropout)
    """
    channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1])
    pool_type: str = 'max'
    pool_positions: List[int] = field(default_factory=lambda: [0])
    activation: str = 'elu'
    norm: str = 'bn'
    fc_hidden: int = 128
    dropout: float = 0.0

    @property
    def num_layers(self):
        return len(self.channels)

    def __str__(self):
        ch_str = '-'.join(map(str, self.channels))
        ks_str = '-'.join(map(str, self.kernel_sizes))
        return f"ch{ch_str}_k{ks_str}_{self.activation}_{self.pool_type}"


# ============================================================================
# Search Space Definition
# ============================================================================

SEARCH_SPACE = {
    'num_layers': [2, 3, 4, 5],
    'channels': [16, 32, 64, 128],
    'kernel_sizes': [3, 5, 7],
    'strides': [1, 2],
    'pool_type': ['max', 'avg', 'none'],
    'activation': ['elu', 'relu', 'lrelu'],
    'norm': ['bn', 'none'],
    'fc_hidden': [0, 64, 128, 256],
    'dropout': [0.0, 0.1, 0.2],
}


def sample_config(
    num_layers: int = None,
    channels_range: Tuple[int, int] = (16, 128),
    kernel_options: List[int] = [3, 5],
    activation_options: List[str] = ['elu', 'relu'],
    pool_options: List[str] = ['max', 'avg'],
    norm_options: List[str] = ['bn'],
    fc_hidden_options: List[int] = [128],
    dropout_options: List[float] = [0.0],
    seed: int = None,
) -> DepthEncoderConfig:
    """
    Sample a random depth encoder configuration.

    Args:
        num_layers: number of conv layers (None = random 2-5)
        channels_range: (min, max) channels per layer
        kernel_options: list of kernel sizes to sample from
        activation_options: list of activations to sample from
        pool_options: list of pooling types to sample from
        norm_options: list of norm types to sample from
        fc_hidden_options: list of FC hidden dims to sample from
        dropout_options: list of dropout rates to sample from
        seed: random seed for reproducibility

    Returns:
        DepthEncoderConfig: sampled configuration
    """
    if seed is not None:
        random.seed(seed)

    # Number of layers
    if num_layers is None:
        num_layers = random.choice([2, 3, 4, 5])

    # Sample channels (generally increasing)
    min_ch, max_ch = channels_range
    channels = []
    ch = random.choice([min_ch, min_ch * 2])
    for i in range(num_layers):
        channels.append(min(ch, max_ch))
        if random.random() > 0.3:  # 70% chance to increase
            ch = min(ch * 2, max_ch)

    # Sample kernel sizes
    kernel_sizes = [random.choice(kernel_options) for _ in range(num_layers)]

    # Sample strides (at most 2 stride-2 layers to not downsample too much)
    strides = [1] * num_layers
    stride_positions = random.sample(range(num_layers), min(2, num_layers))
    for pos in stride_positions:
        if random.random() > 0.5:
            strides[pos] = 2

    # Pool positions (after early layers, before final)
    pool_type = random.choice(pool_options)
    pool_positions = []
    if pool_type != 'none' and num_layers > 1:
        # Add pooling after 1-2 early layers
        n_pools = random.randint(0, min(2, num_layers - 1))
        pool_positions = sorted(random.sample(range(num_layers - 1), n_pools))

    return DepthEncoderConfig(
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pool_type=pool_type,
        pool_positions=pool_positions,
        activation=random.choice(activation_options),
        norm=random.choice(norm_options),
        fc_hidden=random.choice(fc_hidden_options),
        dropout=random.choice(dropout_options),
    )


def sample_configs(n: int, **kwargs) -> List[DepthEncoderConfig]:
    """Sample n unique configurations."""
    configs = []
    seen = set()
    attempts = 0
    max_attempts = n * 10

    while len(configs) < n and attempts < max_attempts:
        cfg = sample_config(**kwargs)
        cfg_str = str(cfg)
        if cfg_str not in seen:
            seen.add(cfg_str)
            configs.append(cfg)
        attempts += 1

    return configs


# ============================================================================
# Baseline Configurations (known good architectures)
# ============================================================================

# Current baseline (SimpleDepthEncoder style)
BASELINE_CONFIG = DepthEncoderConfig(
    channels=[32, 64],
    kernel_sizes=[5, 3],
    strides=[1, 1],
    pool_type='max',
    pool_positions=[0],
    activation='elu',
    norm='bn',
    fc_hidden=128,
    dropout=0.0,
)

# Deeper variant
DEEP_CONFIG = DepthEncoderConfig(
    channels=[32, 64, 64, 128],
    kernel_sizes=[5, 3, 3, 3],
    strides=[1, 1, 1, 1],
    pool_type='max',
    pool_positions=[0, 2],
    activation='elu',
    norm='bn',
    fc_hidden=128,
    dropout=0.0,
)

# Wider variant
WIDE_CONFIG = DepthEncoderConfig(
    channels=[64, 128],
    kernel_sizes=[5, 3],
    strides=[1, 1],
    pool_type='max',
    pool_positions=[0],
    activation='elu',
    norm='bn',
    fc_hidden=256,
    dropout=0.0,
)

# Lightweight variant
LIGHT_CONFIG = DepthEncoderConfig(
    channels=[16, 32],
    kernel_sizes=[3, 3],
    strides=[2, 2],
    pool_type='none',
    pool_positions=[],
    activation='relu',
    norm='bn',
    fc_hidden=64,
    dropout=0.0,
)
