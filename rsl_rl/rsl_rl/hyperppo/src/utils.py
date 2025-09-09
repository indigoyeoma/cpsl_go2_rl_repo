import numpy as np
import torch

def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

PRIMITIVES_DEEPNETS1M = [
    'max_pool',
    'avg_pool',
    'sep_conv',
    'dil_conv',
    'conv',
    'msa',
    'cse',
    'sum',
    'concat',
    'input',
    'bias',
    'bn',
    'ln',
    'pos_enc',
    'glob_avg',
]


def capacity(model):
    c, n = 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c += 1
            n += np.prod(p.shape)
    return c, int(n)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Simplified placeholder for CNN+MLP networks."""
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def adjust_net(model, large_input=True):
    """Simplified adjust_net for CNN+MLP networks."""
    return model


def rand_choice(choices, n=1):
    """Random choice function."""
    if isinstance(choices, torch.Tensor):
        if n == 1:
            return choices[torch.randint(len(choices), (1,))].item()
        else:
            indices = torch.randint(len(choices), (n,))
            return choices[indices[torch.randint(len(indices), (1,))]]
    else:
        return np.random.choice(choices, size=n if n > 1 else None)