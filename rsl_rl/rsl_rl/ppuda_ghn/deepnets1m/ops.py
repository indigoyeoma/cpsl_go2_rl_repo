# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified operations for simple CNN depth encoders.
"""

import torch
import torch.nn as nn
from .light_ops import BatchNorm2dLight, LayerNormLight


# Normalization layer types that GHN recognizes
NormLayers = [nn.BatchNorm2d, nn.LayerNorm, BatchNorm2dLight, LayerNormLight]

# Try to add ConvNeXt's LayerNorm2d if available
try:
    import torchvision
    NormLayers.append(torchvision.models.convnext.LayerNorm2d)
except Exception:
    pass


class PosEnc(nn.Module):
    """Positional encoding for transformers (not used for simple CNNs)."""
    def __init__(self, C, ks, light=False):
        super().__init__()
        fn = torch.empty if light else torch.randn
        self.weight = nn.Parameter(fn(1, C, ks, ks))

    def forward(self, x):
        return x + self.weight
