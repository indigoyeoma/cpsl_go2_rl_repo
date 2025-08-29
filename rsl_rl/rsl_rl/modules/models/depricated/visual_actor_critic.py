import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class SimpleDepthEncoder(nn.Module):
    """CNN encoder for depth images - handles 64x64 input properly"""
    def __init__(self, output_dim=256, input_height=64, input_width=64):
        super(SimpleDepthEncoder, self).__init__()
        self.output_dim = output_dim
        
        # CNN layers optimized for 64x64 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # 64x64 -> 32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 8x8 -> 4x4
        self.bn4 = nn.BatchNorm2d(256)
        
        # Define activation BEFORE using it
        self.activation = nn.ReLU()
        
        # Calculate conv output size dynamically  
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            x = self._forward_conv(dummy_input)
            conv_output_size = x.numel()
        
        # Larger FC layers for better feature extraction
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, output_dim)
    
    def _forward_conv(self, x):
        """Forward pass through conv layers only"""
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        return x
        
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
            
        # Forward through conv layers
        x = self._forward_conv(depth_image)
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

class VisualActorCritic(nn.Module):
    """Actor-Critic for visual RL using depth images + proprioception with shared encoder"""
    is_recurrent = False
    
    def __init__(self, 
                 num_actor_obs,  # Standard parameter name like ActorCritic
                 num_privileged_obs,
                 num_actions,
                 depth_image_shape=(96, 128),
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 depth_latent_dim=32,
                 activation='elu',
                 init_noise_std=1.0,
                 use_depth_for_critic=False,  # Option to use depth for critic
                 **kwargs):
        super(VisualActorCritic, self).__init__()
        
        activation_fn = self.get_activation(activation)
        self.use_depth_for_critic = use_depth_for_critic
        
        # Shared depth encoder to process depth images for both actor and critic
        self.depth_encoder = SimpleDepthEncoder(output_dim=depth_latent_dim, 
                                               input_height=depth_image_shape[0], 
                                               input_width=depth_image_shape[1])
        
        # Actor network: observations + depth latent features -> actions  
        actor_input_dim = num_actor_obs + depth_latent_dim
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                actor_layers.append(activation_fn)
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network: privileged observations (+ optionally depth) -> value
        critic_input_dim = num_privileged_obs
        if use_depth_for_critic:
            critic_input_dim += depth_latent_dim
            
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Store dimensions
        self.num_actor_obs = num_actor_obs
        self.depth_latent_dim = depth_latent_dim
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Disable validation for speedup
        Normal.set_default_validate_args = False
    
    def _initialize_weights(self):
        """Proper weight initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, depth_images=None):
        # Use full observations (proprioception + other sensor data)
        actor_obs = observations[:, :self.num_actor_obs]
        
        # Process depth image if available
        if depth_images is not None:
            # Debug: Print depth image stats before CNN
            # print(f"[Before CNN] Depth shape: {depth_images.shape}, "
            #       f"min={depth_images.min().item():.3f}, "
            #       f"max={depth_images.max().item():.3f}, "
            #       f"mean={depth_images.mean().item():.3f}")
            
            depth_latent = self.depth_encoder(depth_images)
        else:
            # Use zeros if no depth image
            depth_latent = torch.zeros(actor_obs.shape[0], self.depth_latent_dim, 
                                     device=actor_obs.device)
        
        # Combine observations and depth features
        actor_input = torch.cat([actor_obs, depth_latent], dim=1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, depth_images=None, **kwargs):
        self.update_distribution(observations, depth_images)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, depth_images=None, **kwargs):
        # Use full observations
        actor_obs = observations[:, :self.num_actor_obs]
        
        # Process depth image if available
        if depth_images is not None:
            depth_latent = self.depth_encoder(depth_images)
        else:
            # Use zeros if no depth image
            depth_latent = torch.zeros(actor_obs.shape[0], self.depth_latent_dim, 
                                     device=actor_obs.device)
        
        # Combine observations and depth features
        actor_input = torch.cat([actor_obs, depth_latent], dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def evaluate(self, critic_observations, depth_images=None, **kwargs):
        if self.use_depth_for_critic and depth_images is not None:
            # Process depth image and concatenate with critic observations
            depth_latent = self.depth_encoder(depth_images)
            critic_input = torch.cat([critic_observations, depth_latent], dim=1)
            value = self.critic(critic_input)
        else:
            value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data