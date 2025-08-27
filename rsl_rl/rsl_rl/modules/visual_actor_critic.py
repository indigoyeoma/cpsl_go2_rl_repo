import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class SimpleDepthEncoder(nn.Module):
    """Simple CNN encoder to process depth images and output scan-like latent features"""
    def __init__(self, output_dim=32, input_height=24, input_width=32):
        super(SimpleDepthEncoder, self).__init__()
        self.output_dim = output_dim
        
        # CNN layers to process depth image (adjusted for 16x12 input)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 16x12 -> 8x6
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 8x6 -> 4x3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0) # 4x3 -> 3x2
        
        # Calculate the size after convolutions dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            conv_output_size = x.numel()
        
        # Fully connected layers - increased capacity for larger latent dim
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

class VisualActorCritic(nn.Module):
    """Actor-Critic for visual RL using depth images + proprioception"""
    is_recurrent = False
    
    def __init__(self, 
                 num_proprio,
                 num_privileged_obs,
                 num_actions,
                 depth_image_shape=(96, 128),
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 depth_latent_dim=32,
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super(VisualActorCritic, self).__init__()
        
        activation_fn = self.get_activation(activation)
        
        # Depth encoder to process depth images
        self.depth_encoder = SimpleDepthEncoder(output_dim=depth_latent_dim, 
                                               input_height=depth_image_shape[0], 
                                               input_width=depth_image_shape[1])
        
        # Actor network: proprioception + depth latent features -> actions
        actor_input_dim = num_proprio + depth_latent_dim
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
        
        # Critic network: privileged observations -> value
        critic_layers = []
        critic_layers.append(nn.Linear(num_privileged_obs, critic_hidden_dims[0]))
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
        self.num_proprio = num_proprio
        self.depth_latent_dim = depth_latent_dim
        
        # Disable validation for speedup
        Normal.set_default_validate_args = False
    
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
        # Extract proprioception
        proprioception = observations[:, :self.num_proprio]
        
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
            depth_latent = torch.zeros(proprioception.shape[0], self.depth_latent_dim, 
                                     device=proprioception.device)
        
        # Combine proprioception and depth features
        actor_input = torch.cat([proprioception, depth_latent], dim=1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, depth_images=None, **kwargs):
        self.update_distribution(observations, depth_images)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, depth_images=None, **kwargs):
        # Extract proprioception
        proprioception = observations[:, :self.num_proprio]
        
        # Process depth image if available
        if depth_images is not None:
            depth_latent = self.depth_encoder(depth_images)
        else:
            # Use zeros if no depth image
            depth_latent = torch.zeros(proprioception.shape[0], self.depth_latent_dim, 
                                     device=proprioception.device)
        
        # Combine proprioception and depth features
        actor_input = torch.cat([proprioception, depth_latent], dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data