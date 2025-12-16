# Depth Encoder Architecture Search - Sequential Operations
#
# Simple CNN operations for building diverse depth encoder architectures.

import torch.nn as nn


def get_activation(name):
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'lrelu': nn.LeakyReLU(0.1, inplace=True),
        'selu': nn.SELU(inplace=True),
        'gelu': nn.GELU(),
        'none': nn.Identity(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


def get_norm(name, channels):
    """Get normalization layer by name."""
    if name == 'bn':
        return nn.BatchNorm2d(channels)
    elif name == 'ln':
        return nn.LayerNorm(channels)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {name}. Choose from ['bn', 'ln', 'none']")


def get_pool(name, kernel_size=2, stride=2):
    """Get pooling layer by name."""
    if name == 'max':
        return nn.MaxPool2d(kernel_size, stride=stride)
    elif name == 'avg':
        return nn.AvgPool2d(kernel_size, stride=stride)
    elif name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown pool: {name}. Choose from ['max', 'avg', 'none']")


class ConvBlock(nn.Module):
    """Conv -> Norm -> Activation block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, activation='elu', norm='bn'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=(norm == 'none'))
        self.norm = get_norm(norm, out_ch)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
