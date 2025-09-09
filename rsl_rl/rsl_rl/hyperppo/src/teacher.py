# Teacher model architecture extracted from ppo_rgb.py
# Used for kickstarting guidance in HyperPPO training

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, sample_obs, image_size=128, feature_size=256):
        super().__init__()
        
        extractors = {}
        self.out_features = 0
        self.feature_size = feature_size
        
        # RGB processing
        if "rgb" in sample_obs:
            in_channels = sample_obs["rgb"].shape[-1]
            original_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
            self.image_size = image_size
            self.needs_resize = (original_size[0] != image_size or original_size[1] != image_size)
            
            if self.needs_resize:
                print(f"NatureCNN: Will resize from {original_size} to {image_size}x{image_size}")
            else:
                print(f"NatureCNN: Using native {image_size}x{image_size} input size")


            # NatureCNN architecture optimized for visual RL
            # Adaptable architecture - automatically calculates dimensions for any input size
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Automatically calculate output dimensions for any image_size
            with torch.no_grad():
                dummy_rgb = torch.zeros(1, in_channels, self.image_size, self.image_size)
                n_flatten = cnn(dummy_rgb).shape[1]
                print(f"NatureCNN: CNN output features: {n_flatten} (for {image_size}x{image_size} input)")
                fc = nn.Sequential(nn.Linear(n_flatten, self.feature_size), nn.ReLU())
            
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += self.feature_size

        # State processing (if available)
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            print(f"NatureCNN: Processing {state_size}D state → {self.feature_size}D features")
            extractors["state"] = nn.Linear(state_size, self.feature_size)
            self.out_features += self.feature_size
        else:
            print("NatureCNN: No state information available")

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            # RGB preprocessing is now handled externally for consistency
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, image_size=128, feature_size=256):
        super().__init__()
        # Initialize NatureCNN with configurable image_size and feature_size
        self.feature_net = NatureCNN(sample_obs=sample_obs, image_size=image_size, feature_size=feature_size)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=np.sqrt(2)),  # Same scale as GHN
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
        
    def get_features(self, x):
        return self.feature_net(x)
        
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
        
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
        
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)