# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.models.simple_cnn import SimpleCNN

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class VisualActorCritic(nn.Module):
    """Actor-Critic for visual RL using single-frame depth images with synchronous processing
    - Depth: 424x240 → 84x84 → 128 latent via SimpleCNN
    - State: 48 → 128 latent via MLP encoder  
    - Combined: 256 latent → [512, 256, 128] MLP"""
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 depth_image_shape=(84, 84),  # [H, W] - 424x240 → 84x84
                 depth_latent_dim=128,        # Depth → 128 latent
                 state_latent_dim=128,        # State → 128 latent  
                 actor_hidden_dims=[512, 256, 128],  # Combined processing (256 input → larger MLP)
                 critic_hidden_dims=[512, 256, 128], # Combined processing (256 input → larger MLP)
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        
        if kwargs:
            print("VisualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        
        super(VisualActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.depth_latent_dim = depth_latent_dim
        self.state_latent_dim = state_latent_dim
        
        # Calculate observation splits - depth is always appended at the end: [base_obs, depth_flat]
        self.depth_image_size = depth_image_shape[0] * depth_image_shape[1]  # 84*84 = 7056
        self.base_actor_obs_size = num_actor_obs - self.depth_image_size  # 48 
        self.base_critic_obs_size = num_critic_obs - self.depth_image_size  # 48
        
        print(f"VisualActorCritic initialized:")
        print(f"  - Actor obs: {num_actor_obs} (state: {self.base_actor_obs_size}, depth: {self.depth_image_size})")
        print(f"  - Critic obs: {num_critic_obs} (state: {self.base_critic_obs_size}, depth: {self.depth_image_size})")
        print(f"  - Depth: 424x240 → {depth_image_shape} → {depth_latent_dim} latent")
        print(f"  - State: {self.base_actor_obs_size} → {state_latent_dim} latent")
        print(f"  - Combined: {depth_latent_dim + state_latent_dim} → {actor_hidden_dims}")
        
        activation_fn = get_activation(activation)
        
        # Create observation space wrapper for SimpleCNN
        class ObservationSpace:
            def __init__(self, depth_shape):
                # SimpleCNN expects [H, W, C] format
                self.spaces = {"depth": type('obj', (object,), {'shape': (depth_shape[0], depth_shape[1], 1)})}
        
        # Depth encoder: 84x84 → 256 latent using SimpleCNN
        obs_space = ObservationSpace(depth_image_shape)
        self.depth_encoder = SimpleCNN(obs_space, depth_latent_dim)
        print(f"Depth Encoder (SimpleCNN): {depth_image_shape} → {depth_latent_dim} latent")
        
        # State encoder: 48 → 256 latent using MLP
        self.state_encoder = nn.Sequential(
            nn.Linear(self.base_actor_obs_size, 128),
            activation_fn,
            nn.Linear(128, state_latent_dim),
            activation_fn
        )
        print(f"State Encoder (MLP): {self.base_actor_obs_size} → {state_latent_dim} latent")
        
        # Combined input dimension: 128 + 128 = 256
        combined_latent_dim = depth_latent_dim + state_latent_dim  # 256
        
        # Actor network: 256 → [256, 256, 128] → actions
        actor_layers = []
        actor_layers.append(nn.Linear(combined_latent_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network: 256 → [256, 256, 128] → 1 value
        critic_layers = []
        critic_layers.append(nn.Linear(combined_latent_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Using depth for both actor and critic")
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
    
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
    
    def _process_depth_image(self, depth_images):
        """Process depth images for SimpleCNN
        Args:
            depth_images: [batch_size, 84, 84] 
        Returns:
            depth_dict: Dictionary with 'depth' key for SimpleCNN
        """
        # Convert [batch, H, W] -> [batch, H, W, 1] for SimpleCNN
        depth_images = depth_images.unsqueeze(-1)  # [B, 84, 84, 1]
        return {"depth": depth_images}
    
    def _extract_depth_from_obs(self, observations):
        """Extract depth features from observations buffer
        The depth is flattened in obs_buf, we reshape it back to image format for CNN.
        
        Args:
            observations: [batch_size, 7104] = [base_obs(48) + depth_flat(7056)]
        Returns:
            base_obs: [batch_size, 48] base observations
            depth_images: [batch_size, 84, 84] resized depth images
        """
        # Split observations: [base_obs, depth_flat]
        base_obs = observations[:, :self.base_actor_obs_size]  # [B, 48] 
        depth_flat = observations[:, self.base_actor_obs_size:]  # [B, 7056]
        
        # Reshape flattened depth back to image format for CNN processing
        depth_images = depth_flat.view(-1, 84, 84)  # [B, 84, 84]
        
        return base_obs, depth_images
    
    def _encode_depth(self, depth_images):
        """Encode depth images using shared SimpleCNN encoder"""
        depth_dict = self._process_depth_image(depth_images)
        return self.depth_encoder(depth_dict)
    
    def update_distribution(self, observations, **kwargs):
        """Update action distribution - encode state and depth separately then combine"""
        # Extract base observations and depth from observation buffer
        base_obs, depth_images = self._extract_depth_from_obs(observations)
        
        # Encode state: 48 → 128 latent
        state_features = self.state_encoder(base_obs)
        
        # Encode depth: 84x84 → 128 latent  
        depth_features = self._encode_depth(depth_images)
        
        # Combine: 256 + 256 = 512 latent
        combined_features = torch.cat([state_features, depth_features], dim=1)
        
        mean = self.actor(combined_features)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, **kwargs):
        """Sample action from distribution - extract depth from observations"""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, **kwargs):
        """Get deterministic action for inference - encode state and depth separately"""
        # Extract base observations and depth from observation buffer
        base_obs, depth_images = self._extract_depth_from_obs(observations)
        
        # Encode state: 48 → 128 latent
        state_features = self.state_encoder(base_obs)
        
        # Encode depth: 84x84 → 128 latent
        depth_features = self._encode_depth(depth_images)
        
        # Combine: 256 + 256 = 512 latent
        combined_features = torch.cat([state_features, depth_features], dim=1)
        
        return self.actor(combined_features)
    
    def evaluate(self, critic_observations, **kwargs):
        """Compute value function - encode state and depth separately"""
        # Extract depth from critic observations (same format as actor)
        base_critic_obs, depth_images = self._extract_depth_from_obs(critic_observations)
        
        # Encode state: 48 → 128 latent
        state_features = self.state_encoder(base_critic_obs)
        
        # Encode depth: 84x84 → 128 latent
        depth_features = self._encode_depth(depth_images)
        
        # Combine: 256 + 256 = 512 latent
        combined_features = torch.cat([state_features, depth_features], dim=1)
        
        return self.critic(combined_features)
    
    def reset_std(self, std, num_actions, device):
        """Reset action noise standard deviation"""
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data