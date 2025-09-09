import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEnc(nn.Module):
    def __init__(self, C, ks):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, C, ks, ks))

    def forward(self, x):
        return  x + self.weight
        
NormLayers = [nn.BatchNorm2d, nn.LayerNorm]


def get_cell_ind(param_name, layers=1):
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


class CnnMlpNetwork(nn.Module):
    """CNN+MLP network compatible with generate_architectures.py output"""
    
    def __init__(self, cnn_config, cnn_mlp_config, mlp_config, input_channels=3, output_dim=8, input_size=64):
        super(CnnMlpNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.input_size = input_size
        
        # Build CNN layers
        cnn_layers = []
        in_channels = input_channels
        
        for layer_config in cnn_config:
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config['channels'],
                kernel_size=layer_config['kernel'],
                stride=layer_config['stride'],
                padding=layer_config['padding']
            ))
            cnn_layers.append(nn.ReLU(inplace=True))
            in_channels = layer_config['channels']
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size from architecture config
        self.cnn_output_size = self._calculate_cnn_output_size(cnn_config, input_size)
        
        # CNN post-processing MLP (flattened CNN → fixed dimension)
        self.cnn_output_dim = cnn_mlp_config[0] if cnn_mlp_config else mlp_config[0]
        self.cnn_mlp = nn.Linear(self.cnn_output_size, self.cnn_output_dim)
        
        # Main MLP layers
        mlp_layers = []
        for i in range(len(mlp_config) - 1):
            mlp_layers.append(nn.Linear(mlp_config[i], mlp_config[i+1]))
            mlp_layers.append(nn.ReLU(inplace=True))
        
        # Final output layer
        mlp_layers.append(nn.Linear(mlp_config[-1], output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize layered modules for GHN compatibility
        from rsl_rl.modules.hyper.ghn_modules import named_layered_modules
        self._layered_modules = named_layered_modules(self)
        
    def _calculate_cnn_output_size(self, cnn_config, input_size):
        """Calculate the flattened CNN output size from architecture config"""
        h, w = input_size, input_size
        out_channels = cnn_config[-1]['channels']  # Final CNN layer channels
        
        for layer_config in cnn_config:
            kernel_size = layer_config['kernel']
            stride = layer_config['stride']
            padding = layer_config['padding']
            
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
        
        return out_channels * h * w
        
    def forward(self, x):
        # CNN forward pass
        cnn_out = self.cnn(x)
        
        # Flatten CNN output - use reshape instead of view for non-contiguous tensors
        flattened = cnn_out.reshape(cnn_out.size(0), -1)
        
        # Verify size matches calculation
        actual_size = flattened.size(1)
        if actual_size != self.cnn_output_size:
            print(f"⚠️ CNN output size mismatch: expected {self.cnn_output_size}, got {actual_size}")
        
        # CNN post-processing
        cnn_processed = F.relu(self.cnn_mlp(flattened))
        
        # Main MLP forward pass
        output = self.mlp(cnn_processed)
        
        return output