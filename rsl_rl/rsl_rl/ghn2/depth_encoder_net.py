# Depth Encoder Architecture Search - Network Builder
#
# Build depth encoder networks from configuration.

import torch
import torch.nn as nn
from .ops import ConvBlock, get_activation, get_pool
from .config import DepthEncoderConfig, BASELINE_CONFIG


class DepthEncoder(nn.Module):
    """
    Sequential depth encoder built from configuration.

    Architecture:
        Input: [B, 1, H, W] depth image (default 58x87)
        -> Conv blocks with optional pooling
        -> Flatten
        -> FC layers
        -> Output: [B, latent_dim] depth latent (default 32)

    This is a drop-in replacement for SimpleDepthEncoder in depth_backbone.py
    """

    def __init__(
        self,
        config: DepthEncoderConfig = None,
        input_shape: tuple = (58, 87),
        latent_dim: int = 32,
    ):
        """
        Args:
            config: DepthEncoderConfig specifying architecture
            input_shape: (H, W) of input depth image
            latent_dim: output latent dimension (32 for depth encoder)
        """
        super().__init__()

        if config is None:
            config = BASELINE_CONFIG

        self.config = config
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build conv layers
        layers = []
        in_channels = 1  # depth image

        for i, (out_ch, ks, stride) in enumerate(zip(
            config.channels, config.kernel_sizes, config.strides
        )):
            # Conv block
            layers.append(ConvBlock(
                in_channels, out_ch,
                kernel_size=ks,
                stride=stride,
                activation=config.activation,
                norm=config.norm,
            ))

            # Optional pooling
            if i in config.pool_positions and config.pool_type != 'none':
                layers.append(get_pool(config.pool_type, kernel_size=2, stride=2))

            in_channels = out_ch

        self.features = nn.Sequential(*layers)

        # Compute feature size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            feat = self.features(dummy)
            self.feature_size = feat.numel()
            self.feature_shape = feat.shape[1:]  # [C, H, W]

        # FC layers
        if config.fc_hidden > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size, config.fc_hidden),
                get_activation(config.activation),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(config.fc_hidden, latent_dim),
            )
        else:
            self.fc = nn.Linear(self.feature_size, latent_dim)

    def forward(self, depth_image):
        """
        Forward pass.

        Args:
            depth_image: [B, H, W] or [B, 1, H, W] depth image

        Returns:
            [B, latent_dim] depth latent vector
        """
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)

        x = self.features(depth_image)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"DepthEncoder(\n"
            f"  config={self.config},\n"
            f"  input_shape={self.input_shape},\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  feature_shape={self.feature_shape},\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


def build_depth_encoder(
    config: DepthEncoderConfig = None,
    input_shape: tuple = (58, 87),
    latent_dim: int = 32,
) -> DepthEncoder:
    """
    Factory function to build a depth encoder.

    Args:
        config: architecture configuration (None = baseline)
        input_shape: (H, W) of input depth image
        latent_dim: output latent dimension

    Returns:
        DepthEncoder: the network
    """
    return DepthEncoder(config, input_shape, latent_dim)


def build_depth_encoders_from_configs(
    configs: list,
    input_shape: tuple = (58, 87),
    latent_dim: int = 32,
) -> list:
    """
    Build multiple depth encoders from a list of configs.

    Args:
        configs: list of DepthEncoderConfig
        input_shape: (H, W) of input depth image
        latent_dim: output latent dimension

    Returns:
        list of DepthEncoder networks
    """
    return [build_depth_encoder(cfg, input_shape, latent_dim) for cfg in configs]
