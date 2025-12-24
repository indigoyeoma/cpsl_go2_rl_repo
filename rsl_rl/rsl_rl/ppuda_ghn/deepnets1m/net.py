# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper functions for GHN to work with neural networks.
Simplified version for depth encoder training.
"""


def get_cell_ind(param_name, layers=1):
    """
    Get the cell index from a parameter name.
    For simple sequential networks, this usually returns 0.
    """
    if param_name.find('cells.') >= 0:
        pos1 = len('cells.')
        pos2 = pos1 + param_name[pos1:].find('.')
        cell_ind = int(param_name[pos1: pos2])
    elif param_name.startswith('classifier') or param_name.startswith('auxiliary'):
        cell_ind = layers - 1
    elif layers == 1 or param_name.startswith('stem') or param_name.startswith('pos_enc'):
        cell_ind = 0
    else:
        cell_ind = None

    return cell_ind


def named_layered_modules(model):
    """
    Create a mapping of module names to their metadata for GHN weight prediction.
    """
    if hasattr(model, 'module'):  # in case of multigpu model
        model = model.module
    layers = model._n_cells if hasattr(model, '_n_cells') else 1
    layered_modules = [{} for _ in range(layers)]
    cell_ind = 0
    for module_name, m in model.named_modules():

        cell_ind = m._cell_ind if hasattr(m, '_cell_ind') else cell_ind

        is_layer_scale = hasattr(m, 'layer_scale') and m.layer_scale is not None
        is_proj_w = hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None
        is_pos_enc = hasattr(m, 'pos_embedding') and m.pos_embedding is not None
        is_w = (hasattr(m, 'weight') and m.weight is not None) or is_proj_w or is_pos_enc or is_layer_scale
        is_proj_b = hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None
        is_b = (hasattr(m, 'bias') and m.bias is not None) or is_proj_b

        if is_w or is_b:
            if module_name.startswith('module.'):
                module_name = module_name[module_name.find('.') + 1:]

            if is_w:
                key = module_name + ('.layer_scale' if is_layer_scale else ('.in_proj_weight'
                                                                            if is_proj_w else
                                                                            ('.pos_embedding.weight'
                                                                             if is_pos_enc else '.weight')))
                w = m.layer_scale if is_layer_scale else (m.in_proj_weight if is_proj_w else
                                                          (m.pos_embedding if is_pos_enc else m.weight))
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': True,
                                                  'sz': tuple(w) if isinstance(w, (list, tuple)) else w.shape}
            if is_b:
                key = module_name + ('.in_proj_bias' if is_proj_b else '.bias')
                b = m.in_proj_bias if is_proj_b else m.bias
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': False,
                                                  'sz': tuple(b) if isinstance(b, (list, tuple)) else b.shape}

    return layered_modules
