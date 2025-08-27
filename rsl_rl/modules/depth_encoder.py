import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDepthEncoder(nn.Module):
    """Simple CNN encoder to process depth images and output scan-like latent features"""
    def __init__(self, output_dim=32, input_height=58, input_width=87):
        super(SimpleDepthEncoder, self).__init__()
        self.output_dim = output_dim
        
        # CNN layers to process depth image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate the size after convolutions
        # For 87x58 input: conv1 -> 21x14, conv2 -> 9x6, conv3 -> 7x4
        conv_output_size = 7 * 4 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, depth_image, proprioception=None):
        """
        Args:
            depth_image: [batch_size, height, width] depth image
            proprioception: [batch_size, n_proprio] proprioceptive info (optional, for compatibility)
        Returns:
            latent: [batch_size, output_dim] latent features
        """
        # Add channel dimension if needed
        if len(depth_image.shape) == 3:
            depth_image = depth_image.unsqueeze(1)  # [batch_size, 1, height, width]
            
        x = self.activation(self.conv1(depth_image))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x