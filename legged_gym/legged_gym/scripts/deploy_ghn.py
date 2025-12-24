# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Deploy a trained GHN to predict weights for any depth encoder architecture
#
# Architecture (matches training):
#   GHN backbone → 32-dim visual → combination_mlp (fuse with proprio) → output_mlp → (32 latent + 2 yaw)
#
# Usage:
#   python deploy_ghn.py --ghn_checkpoint /path/to/ghn_model_1000.pt --arch random
#   python deploy_ghn.py --ghn_checkpoint /path/to/ghn_model_1000.pt --arch baseline
#   python deploy_ghn.py --ghn_checkpoint /path/to/ghn_model_1000.pt --arch custom --num_layers 3 --channels 32,64,128

import torch
import torch.nn as nn
import argparse
import os


class GHNDepthEncoder(nn.Module):
    """
    Depth encoder using GHN-predicted backbone weights.

    Architecture (matches training):
        1. backbone: depth → 32-dim visual features
        2. combination_mlp: (visual + proprio) → 32-dim fused
        3. output_mlp: fused → 34-dim (32 latent + 2 yaw)
    """

    def __init__(self, backbone, n_proprio, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.device = device

        # Shared encoder layers (same as training)
        activation = nn.ELU()
        last_activation = nn.Tanh()

        # Proprio fusion: [32 visual + n_proprio] → [32 fused]
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        # Output: [32 fused] → [34] (32 latent + 2 yaw)
        self.output_mlp = nn.Sequential(
            nn.Linear(32, 32 + 2),
            last_activation
        )

        self.to(device)

    def forward(self, depth_image, proprioception):
        """
        Args:
            depth_image: [batch, H, W] or [batch, 1, H, W] depth image
            proprioception: [batch, n_proprio] robot state (with yaw zeroed out)

        Returns:
            depth_latent_and_yaw: [batch, 34] = [32 latent, 2 yaw]
        """
        # Add channel dim if needed
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)

        # Backbone: depth → 32-dim visual
        visual = self.backbone(depth_image)

        # Fuse with proprio
        fused = self.combination_mlp(torch.cat([visual, proprioception], dim=-1))

        # Output
        output = self.output_mlp(fused)

        return output

    def get_latent_and_yaw(self, depth_image, proprioception):
        """Convenience method to get separate latent and yaw."""
        output = self.forward(depth_image, proprioception)
        depth_latent = output[:, :-2]
        yaw = 1.5 * output[:, -2:]  # Scale yaw same as training
        return depth_latent, yaw


def load_ghn_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained GHN checkpoint.

    Returns:
        checkpoint: Full checkpoint dict
        trainable_ghn: TrainableGHN with loaded weights
    """
    from rsl_rl.modules.depth_encoder_ghn import TrainableGHN

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create GHN with same params as training
    trainable_ghn = TrainableGHN(
        max_shape=(128, 128, 7, 7),
        num_classes=32,
        hid=64,
        device=device,
    )
    trainable_ghn.ghn.load_state_dict(checkpoint['ghn_state_dict'])
    trainable_ghn.eval()

    print(f"Loaded trained GHN from {checkpoint_path}")
    print(f"  GHN params: {sum(p.numel() for p in trainable_ghn.parameters()):,}")

    return checkpoint, trainable_ghn


def get_architecture_config(arch_type: str, **kwargs):
    """Get architecture config based on type."""
    from rsl_rl.modules.depth_encoder_ghn import (
        sample_depth_encoder_config,
        DepthEncoderConfig,
        BASELINE_CONFIG, DEEP_CONFIG, WIDE_CONFIG, LIGHT_CONFIG
    )

    if arch_type == 'random':
        return sample_depth_encoder_config()
    elif arch_type == 'baseline':
        return BASELINE_CONFIG
    elif arch_type == 'deep':
        return DEEP_CONFIG
    elif arch_type == 'wide':
        return WIDE_CONFIG
    elif arch_type == 'light':
        return LIGHT_CONFIG
    elif arch_type == 'custom':
        return DepthEncoderConfig(
            num_layers=kwargs.get('num_layers', 2),
            channels=kwargs.get('channels', [32, 64]),
            kernel_sizes=kwargs.get('kernel_sizes', [5, 3]),
            strides=kwargs.get('strides', [1, 1]),
            pool_positions=kwargs.get('pool_positions', [0]),
            fc_hidden=kwargs.get('fc_hidden', 128),
        )
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")


def create_ghn_depth_encoder(
    checkpoint,
    trainable_ghn,
    arch_config,
    n_proprio=53,
    device='cuda'
):
    """
    Create a GHNDepthEncoder with GHN-predicted backbone and loaded MLP weights.

    Args:
        checkpoint: Loaded checkpoint dict
        trainable_ghn: Trained TrainableGHN
        arch_config: DepthEncoderConfig for the backbone architecture
        n_proprio: Number of proprioception dimensions
        device: Device

    Returns:
        GHNDepthEncoder ready for inference
    """
    from rsl_rl.modules.depth_encoder_ghn import build_depth_backbone

    # Build backbone (on CPU for Graph construction)
    backbone = build_depth_backbone(arch_config)

    # GHN predicts weights (moves to device internally)
    backbone = trainable_ghn.predict_weights([backbone])[0]

    # Create encoder
    encoder = GHNDepthEncoder(backbone, n_proprio, device)

    # Load trained MLP weights from checkpoint
    if 'combination_mlp_state_dict' in checkpoint:
        encoder.combination_mlp.load_state_dict(checkpoint['combination_mlp_state_dict'])
        encoder.output_mlp.load_state_dict(checkpoint['output_mlp_state_dict'])
        print("  Loaded combination_mlp and output_mlp from checkpoint")
    else:
        print("  WARNING: No MLP weights in checkpoint, using random initialization")

    encoder.eval()

    print(f"Created GHN depth encoder with config: {arch_config}")
    print(f"  Backbone params: {sum(p.numel() for p in backbone.parameters()):,}")

    return encoder


class GHNDepthPolicy:
    """
    Complete policy wrapper for deploying GHN-based depth policy.

    Includes:
        - GHN for instant weight prediction
        - Depth encoder (backbone + combination_mlp + output_mlp)
        - Depth actor for action generation

    Can switch architectures on-the-fly using switch_architecture().
    """

    def __init__(
        self,
        ghn_checkpoint_path: str,
        arch_type: str = 'random',
        n_proprio: int = 53,
        device: str = 'cuda',
        **arch_kwargs
    ):
        self.device = device
        self.n_proprio = n_proprio

        # Load checkpoint and GHN
        self.checkpoint, self.trainable_ghn = load_ghn_checkpoint(
            ghn_checkpoint_path, device
        )

        # Create encoder with specified architecture
        self.arch_config = get_architecture_config(arch_type, **arch_kwargs)
        self.encoder = create_ghn_depth_encoder(
            self.checkpoint, self.trainable_ghn, self.arch_config,
            n_proprio, device
        )

        # Depth actor needs to be loaded separately (requires actor architecture)
        self.depth_actor = None
        if 'depth_actor_state_dict' in self.checkpoint:
            print("  depth_actor_state_dict available in checkpoint")
            print("  NOTE: To use depth_actor, load it with the correct architecture from training")

    def switch_architecture(self, arch_type: str = 'random', **arch_kwargs):
        """
        Switch to a different backbone architecture.

        This is the key benefit of GHN - instant weight prediction for new architectures!
        The combination_mlp and output_mlp stay the same (shared across architectures).
        """
        from rsl_rl.modules.depth_encoder_ghn import build_depth_backbone

        self.arch_config = get_architecture_config(arch_type, **arch_kwargs)

        # Build new backbone
        backbone = build_depth_backbone(self.arch_config)

        # GHN predicts weights
        backbone = self.trainable_ghn.predict_weights([backbone])[0]

        # Create new encoder with shared MLP weights
        self.encoder = GHNDepthEncoder(backbone, self.n_proprio, self.device)

        # Load MLP weights
        if 'combination_mlp_state_dict' in self.checkpoint:
            self.encoder.combination_mlp.load_state_dict(
                self.checkpoint['combination_mlp_state_dict']
            )
            self.encoder.output_mlp.load_state_dict(
                self.checkpoint['output_mlp_state_dict']
            )

        self.encoder.eval()
        print(f"Switched to architecture: {self.arch_config}")
        print(f"  Backbone params: {sum(p.numel() for p in self.encoder.backbone.parameters()):,}")

    def get_depth_latent(self, depth_image: torch.Tensor, proprioception: torch.Tensor):
        """
        Get depth latent and yaw from depth image.

        Args:
            depth_image: [batch, H, W] depth image
            proprioception: [batch, n_proprio] robot state (yaw should be zeroed at indices 6:8)

        Returns:
            depth_latent: [batch, 32] depth encoding
            yaw: [batch, 2] yaw estimate
        """
        with torch.no_grad():
            return self.encoder.get_latent_and_yaw(depth_image, proprioception)


def main():
    parser = argparse.ArgumentParser(description='Deploy trained GHN for depth encoding')
    parser.add_argument('--ghn_checkpoint', type=str, required=True,
                        help='Path to trained GHN checkpoint (ghn_model_*.pt)')
    parser.add_argument('--arch', type=str, default='random',
                        choices=['random', 'baseline', 'deep', 'wide', 'light', 'custom'],
                        help='Architecture type')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of conv layers (for custom arch)')
    parser.add_argument('--channels', type=str, default='32,64',
                        help='Channels per layer, comma-separated (for custom arch)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--n_proprio', type=int, default=53,
                        help='Number of proprioception dimensions')

    args = parser.parse_args()

    # Parse custom architecture args
    arch_kwargs = {}
    if args.arch == 'custom':
        arch_kwargs['num_layers'] = args.num_layers
        arch_kwargs['channels'] = [int(c) for c in args.channels.split(',')]
        arch_kwargs['kernel_sizes'] = [5] + [3] * (args.num_layers - 1)
        arch_kwargs['strides'] = [1] * args.num_layers
        arch_kwargs['pool_positions'] = [0]

    # Create policy
    policy = GHNDepthPolicy(
        args.ghn_checkpoint,
        arch_type=args.arch,
        n_proprio=args.n_proprio,
        device=args.device,
        **arch_kwargs
    )

    # Test forward pass
    print("\n=== Testing forward pass ===")
    dummy_depth = torch.randn(1, 58, 87, device=args.device)
    dummy_proprio = torch.randn(1, args.n_proprio, device=args.device)
    dummy_proprio[:, 6:8] = 0  # Zero out yaw

    depth_latent, yaw = policy.get_depth_latent(dummy_depth, dummy_proprio)

    print(f"  Input depth: {dummy_depth.shape}")
    print(f"  Input proprio: {dummy_proprio.shape}")
    print(f"  Output depth_latent: {depth_latent.shape}")
    print(f"  Output yaw: {yaw.shape}")

    # Demonstrate switching architectures
    print("\n=== Demonstrating instant architecture switching ===")
    for arch in ['baseline', 'deep', 'wide', 'light', 'random']:
        policy.switch_architecture(arch)

        # Test forward pass with new architecture
        depth_latent, yaw = policy.get_depth_latent(dummy_depth, dummy_proprio)
        print(f"    Forward pass OK: latent={depth_latent.shape}, yaw={yaw.shape}")


if __name__ == '__main__':
    main()
