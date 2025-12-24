# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified utils for GHN depth encoder training.
"""

import random
import numpy as np
import torch


def set_seed(seed, only_torch=False):
    """Set random seeds for reproducibility."""
    if not only_torch:
        random.seed(seed)
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capacity(model, is_grad=True):
    """Count number of parameters in a model."""
    c, n = 0, 0
    for name, p in model.named_parameters():
        if not is_grad or (is_grad and p.requires_grad):
            c += 1
            sz = p if isinstance(p, (tuple, list)) else p.shape
            n += np.prod(sz)
    return c, int(n)


def default_device():
    """Get default device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
