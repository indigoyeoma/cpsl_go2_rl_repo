import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional

from rsl_rl.hyperppo.src.core import hyperActor


class MinimalHyperWrapper(nn.Module):
    """
    Minimal wrapper around working hyperActor from @hyperhelp/src/
    Just handles observation preprocessing and delegates to hyperActor
    """
    def __init__(self, 
                 act_dim, 
                 obs_dim, 
                 architecture_config_path: Optional[str] = None,
                 meta_batch_size: int = 2,
                 device: str = "cuda",
                 **kwargs):
        """
        Minimal wrapper for hyperActor integration with Isaac Gym
        """
        super(MinimalHyperWrapper, self).__init__()
        
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.base_obs_dim = 48  # State observations
        self.depth_obs_dim = obs_dim - self.base_obs_dim  # 7056 for 84x84 depth
        
        # Create the working hyperActor directly
        self.hyper_actor = hyperActor(
            act_dim=act_dim,
            obs_dim=obs_dim, 
            architecture_config_path=architecture_config_path,
            meta_batch_size=meta_batch_size,
            device=device,
            multi_gpu=False,
            architecture_sampling_mode="uniform"
        )
        
        # Store references for compatibility with optimizer setup
        self.ghn = self.hyper_actor.ghn
        self.log_std = self.hyper_actor.log_std
        
    def change_graph(self, repeat_sample=False):
        """Delegate to hyperActor"""
        return self.hyper_actor.change_graph(repeat_sample=repeat_sample)
        
    def sample_arc_indices(self):
        """Delegate to hyperActor"""
        return self.hyper_actor.sample_arc_indices()
    
    def parameters(self):
        """Return only GHN and log_std parameters (not generated weights)"""
        ghn_params = list(self.hyper_actor.ghn.parameters())
        std_params = list(self.hyper_actor.log_std.parameters())
        return iter(ghn_params + std_params)
    
    def forward(self, observations, track=True):
        """
        Preprocess observations for Isaac Gym and delegate to hyperActor
        
        Args:
            observations: [batch_size, 48 + 7056] = [state + flattened depth]
        Returns:
            mu, log_std from hyperActor
        """
        # Extract depth and state from Isaac Gym observations
        batch_size = observations.shape[0]
        state_obs = observations[:, :self.base_obs_dim]  # [batch_size, 48]
        depth_flat = observations[:, self.base_obs_dim:]  # [batch_size, 7056] 
        
        # Reshape depth to image format for hyperActor
        depth_images = depth_flat.reshape(batch_size, 1, 84, 84)  # [batch_size, 1, 84, 84]
        
        # Call hyperActor with properly formatted inputs
        mu, log_std = self.hyper_actor.forward(obs=depth_images, state_obs=state_obs, track=track)
        
        return mu, log_std
    
    @property 
    def arch_descriptors_per_state(self):
        """Get architecture descriptors per state from hyperActor"""
        return getattr(self.hyper_actor, 'arch_descriptors_per_state', None)
    
    @property
    def sampled_indices(self):
        """Get sampled indices from hyperActor"""  
        return getattr(self.hyper_actor, 'sampled_indices', None)