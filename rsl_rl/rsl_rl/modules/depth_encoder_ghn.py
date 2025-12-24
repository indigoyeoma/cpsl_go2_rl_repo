# Depth Encoder Architecture with Trainable GHN
#
# This module provides tools to:
# 1. Sample diverse simple CNN architectures for depth encoding
# 2. Build depth encoder networks from configs
# 3. Train a GHN to predict good initial weights for depth encoders
#
# Two modes of operation:
# A) GHN Training Mode: Train a GHN that learns to predict weights
#    - GHN is updated via backprop through weight prediction
#    - After training, GHN can initialize any new architecture instantly
#
# B) Direct Training Mode: Train encoders directly without GHN
#    - Multiple architectures trained in parallel
#    - Each architecture's weights are directly updated
#
# Usage:
#     from rsl_rl.modules.depth_encoder_ghn import (
#         sample_depth_encoder_config,
#         build_depth_backbone,
#         TrainableGHN,
#     )
#
#     # Create trainable GHN
#     ghn = TrainableGHN(device='cuda')
#
#     # Sample architectures and predict weights
#     configs = [sample_depth_encoder_config() for _ in range(8)]
#     backbones = [build_depth_backbone(cfg) for cfg in configs]
#     backbones = ghn.predict_weights(backbones)  # GHN predicts weights
#
#     # Forward pass through backbones, compute loss, backprop
#     # Gradients flow through backbone weights to GHN

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random


@dataclass
class DepthEncoderConfig:
    """Configuration for a simple CNN depth encoder backbone.

    The backbone processes depth images and outputs a latent vector.
    Architecture: Conv → BN → Act → Pool → Conv → ... → Flatten → FC → output_dim
    """
    # Input (48x64 matches go2_student_config.py resized=(64, 48))
    input_height: int = 48
    input_width: int = 64
    input_channels: int = 1
    output_dim: int = 32

    # Architecture
    num_layers: int = 2
    channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1])

    # Pooling (applied after each conv)
    pool_type: str = 'max'  # 'max', 'avg', or 'none'
    pool_size: int = 2
    pool_positions: List[int] = field(default_factory=lambda: [0])  # Which layers to pool after

    # Activation
    activation: str = 'elu'  # 'elu', 'relu', 'gelu'

    # FC layers
    fc_hidden: int = 128  # Hidden dim before output

    def __post_init__(self):
        """Validate config."""
        assert len(self.channels) == self.num_layers
        assert len(self.kernel_sizes) == self.num_layers
        assert len(self.strides) == self.num_layers

    def to_dict(self):
        return {
            'num_layers': self.num_layers,
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
            'pool_positions': self.pool_positions,
            'fc_hidden': self.fc_hidden,
        }

    def __repr__(self):
        return (f"DepthEncoderConfig(layers={self.num_layers}, "
                f"ch={self.channels}, k={self.kernel_sizes}, s={self.strides}, "
                f"pool@{self.pool_positions})")


# Search space for architecture sampling
SEARCH_SPACE = {
    'num_layers': [2, 3, 4],
    'channels': [16, 32, 64, 128],
    'kernel_sizes': [3, 5, 7],
    'strides': [1, 2],
    'pool_type': ['max', 'avg'],
    'fc_hidden': [64, 128, 256],
}


# Preset configurations
BASELINE_CONFIG = DepthEncoderConfig(
    num_layers=2,
    channels=[32, 64],
    kernel_sizes=[5, 3],
    strides=[1, 1],
    pool_positions=[0],
    fc_hidden=128,
)

DEEP_CONFIG = DepthEncoderConfig(
    num_layers=4,
    channels=[32, 64, 128, 128],
    kernel_sizes=[5, 3, 3, 3],
    strides=[1, 1, 1, 1],
    pool_positions=[0, 2],
    fc_hidden=128,
)

WIDE_CONFIG = DepthEncoderConfig(
    num_layers=2,
    channels=[64, 128],
    kernel_sizes=[5, 3],
    strides=[1, 1],
    pool_positions=[0],
    fc_hidden=256,
)

LIGHT_CONFIG = DepthEncoderConfig(
    num_layers=2,
    channels=[16, 32],
    kernel_sizes=[5, 3],
    strides=[2, 1],
    pool_positions=[0],
    fc_hidden=64,
)


def sample_depth_encoder_config(
    num_layers_range: Tuple[int, int] = (2, 4),
    channel_options: List[int] = None,
    kernel_options: List[int] = None,
    stride_options: List[int] = None,
    seed: int = None,
) -> DepthEncoderConfig:
    """Sample a random depth encoder configuration.

    Args:
        num_layers_range: (min, max) number of conv layers
        channel_options: Choices for channel counts (default: [16, 32, 64, 128])
        kernel_options: Choices for kernel sizes (default: [3, 5, 7])
        stride_options: Choices for strides (default: [1, 2])
        seed: Random seed for reproducibility

    Returns:
        DepthEncoderConfig with randomly sampled architecture
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    channel_options = channel_options or SEARCH_SPACE['channels']
    kernel_options = kernel_options or SEARCH_SPACE['kernel_sizes']
    stride_options = stride_options or SEARCH_SPACE['strides']
    fc_options = SEARCH_SPACE['fc_hidden']

    # Sample number of layers
    num_layers = random.randint(num_layers_range[0], num_layers_range[1])

    # Sample architecture
    # Channels should generally increase or stay same
    channels = []
    prev_ch = random.choice([16, 32])
    for i in range(num_layers):
        if i == 0:
            ch = prev_ch
        else:
            # 70% chance to increase, 30% to stay same
            if random.random() < 0.7:
                ch = min(prev_ch * 2, max(channel_options))
            else:
                ch = prev_ch
        channels.append(ch)
        prev_ch = ch

    # First kernel often larger
    kernel_sizes = [random.choice([5, 7])] + [random.choice(kernel_options) for _ in range(num_layers - 1)]

    # Strides: usually 1, occasionally 2
    strides = [random.choice(stride_options) if random.random() < 0.3 else 1 for _ in range(num_layers)]

    # Pool positions: usually after first layer, maybe also middle
    pool_positions = [0]
    if num_layers >= 3 and random.random() < 0.5:
        pool_positions.append(num_layers // 2)

    return DepthEncoderConfig(
        num_layers=num_layers,
        channels=channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pool_positions=pool_positions,
        fc_hidden=random.choice(fc_options),
    )


def sample_depth_encoder_configs(n: int, unique: bool = True, **kwargs) -> List[DepthEncoderConfig]:
    """Sample multiple depth encoder configurations.

    Args:
        n: Number of configs to sample
        unique: If True, ensure all configs are different
        **kwargs: Passed to sample_depth_encoder_config

    Returns:
        List of DepthEncoderConfig
    """
    configs = []
    seen = set()
    max_attempts = n * 10
    attempts = 0

    while len(configs) < n and attempts < max_attempts:
        cfg = sample_depth_encoder_config(**kwargs)
        key = str(cfg.to_dict()) if unique else None

        if not unique or key not in seen:
            configs.append(cfg)
            if unique:
                seen.add(key)
        attempts += 1

    return configs


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        'elu': nn.ELU(),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.1),
    }
    return activations.get(name.lower(), nn.ELU())


def compute_output_size(h: int, w: int, config: DepthEncoderConfig) -> Tuple[int, int]:
    """Compute the spatial size after all conv/pool layers."""
    for i in range(config.num_layers):
        k = config.kernel_sizes[i]
        s = config.strides[i]
        p = k // 2  # Same padding

        # Conv
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1

        # Pool
        if i in config.pool_positions:
            h = h // config.pool_size
            w = w // config.pool_size

    return h, w


class DepthBackboneWrapper(nn.Module):
    """Wrapper that adds unsqueeze to match DepthOnlyFCBackbone58x87 interface.

    Takes [B, H, W] input and adds channel dim internally.
    """
    def __init__(self, sequential: nn.Sequential, config: DepthEncoderConfig):
        super().__init__()
        self.sequential = sequential
        self.config = config
        # (H, W) only - wrapper adds channel dim in forward()
        # GHN Graph will create [1, H, W] input, then unsqueeze makes [1, 1, H, W]
        self.expected_input_sz = (config.input_height, config.input_width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: [B, H, W] -> [B, 1, H, W]
        # Matches DepthOnlyFCBackbone58x87.forward() behavior
        images = images.unsqueeze(1)
        return self.sequential(images)


def build_depth_backbone(config: DepthEncoderConfig) -> nn.Module:
    """Build a depth encoder backbone from config.

    Args:
        config: DepthEncoderConfig specifying the architecture

    Returns:
        nn.Module that takes [B, H, W] depth image and outputs [B, output_dim]
        (matches DepthOnlyFCBackbone58x87 interface)
    """
    layers = []
    in_ch = config.input_channels
    h, w = config.input_height, config.input_width

    activation = get_activation(config.activation)

    for i in range(config.num_layers):
        out_ch = config.channels[i]
        k = config.kernel_sizes[i]
        s = config.strides[i]
        p = k // 2

        # Conv + BN + Activation
        layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation)

        # Update spatial size
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1

        # Pool
        if i in config.pool_positions:
            if config.pool_type == 'max':
                layers.append(nn.MaxPool2d(config.pool_size))
            elif config.pool_type == 'avg':
                layers.append(nn.AvgPool2d(config.pool_size))
            h = h // config.pool_size
            w = w // config.pool_size

        in_ch = out_ch

    # Flatten
    layers.append(nn.Flatten())
    flat_size = config.channels[-1] * h * w

    # FC layers
    layers.append(nn.Linear(flat_size, config.fc_hidden))
    layers.append(activation)
    layers.append(nn.Linear(config.fc_hidden, config.output_dim))

    sequential = nn.Sequential(*layers)

    # Wrap with unsqueeze handler (matches DepthOnlyFCBackbone58x87 interface)
    backbone = DepthBackboneWrapper(sequential, config)

    return backbone


def build_depth_backbones(configs: List[DepthEncoderConfig]) -> List[nn.Module]:
    """Build multiple depth encoder backbones from configs."""
    return [build_depth_backbone(cfg) for cfg in configs]


class TrainableGHN(nn.Module):
    """Trainable GHN for learning to predict depth encoder weights.

    Unlike using a pre-trained GHN for initialization, this class is designed
    for training the GHN from scratch. The GHN learns to predict good weights
    for depth encoders by backpropagating through the weight prediction.

    Training loop:
        1. Sample diverse architectures
        2. GHN predicts weights for each architecture
        3. Forward pass through depth encoders with predicted weights
        4. Compute loss (DAgger from teacher)
        5. Backprop through encoders AND GHN
        6. Update GHN parameters

    After training, the GHN can instantly predict good weights for any new
    depth encoder architecture.

    Usage:
        ghn = TrainableGHN(device='cuda')
        optimizer = Adam(ghn.parameters(), lr=1e-3)

        # Training loop
        backbones = [build_depth_backbone(cfg) for cfg in configs]
        backbones = ghn.predict_weights(backbones)  # Differentiable!
        outputs = [backbone(depth_images) for backbone in backbones]
        loss = compute_loss(outputs, targets)
        loss.backward()  # Gradients flow to GHN
        optimizer.step()
    """

    def __init__(
        self,
        max_shape: Tuple[int, int, int, int] = (128, 128, 7, 7),
        num_classes: int = 32,  # Output dim of depth encoder
        hid: int = 64,  # Larger hidden dim for better capacity
        hypernet: str = 'gatedgnn',
        decoder: str = 'conv',
        weight_norm: bool = True,
        ve: bool = True,  # Virtual edges
        device: str = 'cuda',
    ):
        """Initialize the trainable GHN.

        Args:
            max_shape: Maximum shape of conv kernels (out, in, h, w)
            num_classes: Output dimension (should match encoder output_dim)
            hid: Hidden dimension for GNN (larger = more capacity)
            hypernet: Type of hypernet ('gatedgnn' or 'mlp')
            decoder: Type of decoder ('conv' or 'mlp')
            weight_norm: Whether to normalize predicted weights
            ve: Whether to use virtual edges
            device: Device to place the GHN on
        """
        super().__init__()

        from ..ppuda_ghn.ghn.nn import GHN

        self.ghn = GHN(
            max_shape=max_shape,
            num_classes=num_classes,
            hypernet=hypernet,
            decoder=decoder,
            weight_norm=weight_norm,
            ve=ve,
            hid=hid,
        )
        self.device = device
        self.ve = ve
        self.to(device)

    def predict_weights(
        self,
        models: List[nn.Module],
        return_graphs: bool = False,
    ) -> List[nn.Module]:
        """Predict weights for multiple depth encoder backbones.

        This is differentiable - gradients will flow back to GHN parameters.

        Args:
            models: List of nn.Module backbones (created by build_depth_backbone)
            return_graphs: If True, also return the GraphBatch for reuse

        Returns:
            List of models with predicted weights (same objects, modified in-place)
            If return_graphs=True, also returns (models, graphs)
        """
        from ..ppuda_ghn.deepnets1m.graph import Graph, GraphBatch
        from ..ppuda_ghn.deepnets1m.net import named_layered_modules

        # Build computation graphs for all models
        graphs = GraphBatch([Graph(m, ve_cutoff=50 if self.ve else 1) for m in models])
        graphs.to_device(self.device)

        # Move models to device
        models = [m.to(self.device) for m in models]

        # Pre-compute _layered_modules for training mode
        # This is required by PPUDA GHN when self.training=True
        for m in models:
            m._layered_modules = named_layered_modules(m)

        # GHN in training mode for gradient flow
        self.ghn.train()

        # Predict weights - this modifies models in-place
        result = self.ghn(models, graphs=graphs)

        if return_graphs:
            return result, graphs
        return result

    def predict_weights_with_graphs(
        self,
        models: List[nn.Module],
        graphs,
    ) -> List[nn.Module]:
        """Predict weights using pre-computed graphs (faster for repeated calls).

        Args:
            models: List of nn.Module backbones
            graphs: Pre-computed GraphBatch from predict_weights(return_graphs=True)

        Returns:
            List of models with predicted weights
        """
        from ..ppuda_ghn.deepnets1m.net import named_layered_modules

        # Pre-compute _layered_modules for training mode
        for m in models:
            m._layered_modules = named_layered_modules(m)

        self.ghn.train()
        return self.ghn(models, graphs=graphs)

    def forward(self, models: List[nn.Module]) -> List[nn.Module]:
        """Alias for predict_weights for nn.Module compatibility."""
        return self.predict_weights(models)

    def save(self, path: str):
        """Save GHN checkpoint."""
        torch.save({
            'ghn_state_dict': self.ghn.state_dict(),
            'config': {
                'max_shape': self.ghn.max_shape,
                'num_classes': self.ghn.num_classes,
                've': self.ve,
            }
        }, path)
        print(f"Saved TrainableGHN to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'TrainableGHN':
        """Load GHN from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        ghn = cls(
            max_shape=config['max_shape'],
            num_classes=config['num_classes'],
            ve=config['ve'],
            device=device,
        )
        ghn.ghn.load_state_dict(checkpoint['ghn_state_dict'])
        print(f"Loaded TrainableGHN from {path}")
        return ghn


# Keep old name for backwards compatibility
DepthEncoderGHN = TrainableGHN


# Preset for quick testing
def get_preset_configs() -> List[DepthEncoderConfig]:
    """Get a list of preset configurations for testing."""
    return [BASELINE_CONFIG, DEEP_CONFIG, WIDE_CONFIG, LIGHT_CONFIG]


if __name__ == '__main__':
    # Test the module
    print("Testing depth encoder GHN module...")

    # Sample configs
    configs = sample_depth_encoder_configs(4)
    for i, cfg in enumerate(configs):
        print(f"Config {i}: {cfg}")

    # Build backbones
    backbones = build_depth_backbones(configs)

    # Test forward pass
    x = torch.randn(2, 1, 58, 87)
    for i, backbone in enumerate(backbones):
        y = backbone(x)
        print(f"Backbone {i}: input {x.shape} -> output {y.shape}")
        print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")
