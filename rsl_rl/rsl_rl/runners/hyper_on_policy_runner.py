# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# HyperPPO extension for on-policy training with Graph HyperNetworks
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

# Import HyperPPO components from existing infrastructure
from rsl_rl.hyperppo.src.core import hyperActor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layers with orthogonal weights"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ArchConditionedCritic(nn.Module):
    """Architecture-conditioned critic network with Depth + State + Architecture inputs (asymmetric design)"""
    
    def __init__(self, state_dim, arch_descriptor_dim, device):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        
        # Process depth observations with CNN (privileged info for critic) - matching ManiSkill pattern
        depth_features_dim = 256
        self.depth_encoder = nn.Sequential(
            # Input: [batch, 1, 84, 84] - following ManiSkill 84x84 standard
            layer_init(nn.Conv2d(1, 32, 8, stride=4, padding=0)),  # → [batch, 32, 20, 20]
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), # → [batch, 64, 10, 10] 
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, padding=1)), # → [batch, 128, 5, 5]
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 3, stride=1, padding=1)), # → [batch, 256, 5, 5]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                           # → [batch, 256, 2, 2]
            nn.Flatten(),                                           # → [batch, 1024]
            layer_init(nn.Linear(1024, depth_features_dim))         # → [batch, 256]
        )
        
        # Process state information (privileged info for critic)
        state_features_dim = 128
        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, state_features_dim))
        )
        
        # Process architecture descriptor
        arch_embedding_dim = 128
        self.arch_embedding = nn.Sequential(
            layer_init(nn.Linear(arch_descriptor_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, arch_embedding_dim))
        )
        
        # Combine all features: Depth + State + Architecture
        combined_dim = depth_features_dim + state_features_dim + arch_embedding_dim  # 256 + 128 + 128 = 512
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )
        
    def forward(self, obs, arch_descriptors):
        # Split GO2 observations: [state(48) + flattened_depth(?)]
        state_obs = obs[:, :48]  # First 48 are proprioceptive
        
        if obs.shape[1] > 48:
            depth_flat = obs[:, 48:]  # Remaining are flattened depth
            batch_size = obs.shape[0]
            depth_elements = depth_flat.shape[1]
            
            # Check if we have expected 84×84 size
            expected_elements = 84 * 84  # 7056
            if depth_elements == expected_elements:
                # Perfect! Environment is using new config
                depth_obs = depth_flat.view(batch_size, 1, 84, 84)
            else:
                # Fallback: Environment still using old config, detect actual size and upsample
                import math
                img_size = int(math.sqrt(depth_elements))
                depth_obs = depth_flat.view(batch_size, 1, img_size, img_size)
                import torch.nn.functional as F
                depth_obs = F.interpolate(depth_obs, size=(84, 84), mode='bilinear', align_corners=False)
        else:
            batch_size = obs.shape[0]
            depth_obs = torch.zeros(batch_size, 1, 84, 84, device=obs.device)
        
        # Process all inputs separately
        depth_features = self.depth_encoder(depth_obs)  # [batch, 256]
        state_features = self.state_encoder(state_obs)  # [batch, 128] 
        arch_embedding = self.arch_embedding(arch_descriptors)  # [batch, 128]
        
        # Combine all privileged information
        combined = torch.cat([depth_features, state_features, arch_embedding], dim=1)  # [batch, 512]
        return self.critic(combined).squeeze(-1)


class GO2HyperAgent(nn.Module):
    """HyperAgent wrapper with architecture-conditioned critic for GO2"""
    
    def __init__(self, hyper_actor, device, state_dim, arch_descriptor_dim):
        super().__init__()
        self.hyper_actor = hyper_actor
        self.device = device
        
        # Initialize asymmetric architecture-conditioned critic (Depth + State + Architecture)
        self.critic = ArchConditionedCritic(state_dim, arch_descriptor_dim, device).to(device)
        
    def forward(self, obs):
        """Actor forward pass that splits GO2 observations"""
        # Split GO2 observations: [state(48) + flattened_depth(?)]
        state_obs = obs[:, :48]  # First 48 are proprioceptive
        
        if obs.shape[1] > 48:
            depth_flat = obs[:, 48:]  # Remaining are flattened depth
            batch_size = obs.shape[0]
            depth_elements = depth_flat.shape[1]
            
            # Expected: 84×84 = 7056 elements (after downsampling from 480×270)
            expected_elements = 84 * 84
            if depth_elements == expected_elements:
                # Perfect! Environment is properly downsampling 480×270 → 84×84
                depth_obs = depth_flat.view(batch_size, 1, 84, 84)
            else:
                # Environment not yet updated - detect current size and downsample
                import math
                
                # Check for 480×270 raw input first
                if depth_elements == 480 * 270:  # 129600
                    print(f"📷 Raw 480×270 input detected, downsampling to 84×84")
                    depth_obs = depth_flat.view(batch_size, 1, 270, 480)
                else:
                    # Detect other common sizes and upsample/downsample to 84×84
                    common_sizes = [
                        (44, 68), (68, 44),  # Current environment
                        (55, 55), (64, 64),  # Legacy configs
                    ]
                    
                    found_size = None
                    for h, w in common_sizes:
                        if h * w == depth_elements:
                            found_size = (h, w)
                            break
                    
                    if found_size:
                        h, w = found_size
                        print(f"📷 Detected {h}×{w}, resampling to 84×84")
                        depth_obs = depth_flat.view(batch_size, 1, h, w)
                    else:
                        # Last resort
                        img_size = int(math.sqrt(depth_elements))
                        print(f"📷 Assuming {img_size}×{img_size}, resampling to 84×84")  
                        depth_obs = depth_flat.view(batch_size, 1, img_size, img_size)
                
                # Downsample/upsample to target 84×84 
                import torch.nn.functional as F
                depth_obs = F.interpolate(depth_obs, size=(84, 84), mode='bilinear', align_corners=False)
        else:
            batch_size = obs.shape[0]
            depth_obs = torch.zeros(batch_size, 1, 84, 84, device=obs.device)
        
        # Use hyperActor with proper depth and state inputs
        return self.hyper_actor.forward(depth_obs, state_obs)
    
    def get_value(self, obs):
        """Get value estimation using architecture-conditioned critic"""
        # Need to do a forward pass to get architecture descriptors
        with torch.no_grad():
            # Extract depth and state from observations
            state_obs = obs[:, :48]  
            if obs.shape[1] > 48:
                depth_flat = obs[:, 48:]  # Variable size flattened depth
                batch_size = obs.shape[0]
                depth_elements = depth_flat.shape[1]
                
                # Expected: 84×84 = 7056 elements (after downsampling from 480×270)
                expected_elements = 84 * 84
                if depth_elements == expected_elements:
                    # Perfect! Environment is properly downsampling 480×270 → 84×84
                    depth_obs = depth_flat.view(batch_size, 1, 84, 84)
                else:
                    # Environment not yet updated - detect current size and downsample
                    import math
                    
                    # Check for 480×270 raw input first
                    if depth_elements == 480 * 270:  # 129600
                        depth_obs = depth_flat.view(batch_size, 1, 270, 480)
                    else:
                        # Detect other common sizes
                        common_sizes = [
                            (44, 68), (68, 44),  # Current environment
                            (55, 55), (64, 64),  # Legacy configs
                        ]
                        
                        found_size = None
                        for h, w in common_sizes:
                            if h * w == depth_elements:
                                found_size = (h, w)
                                break
                        
                        if found_size:
                            h, w = found_size
                            depth_obs = depth_flat.view(batch_size, 1, h, w)
                        else:
                            # Last resort
                            img_size = int(math.sqrt(depth_elements))
                            depth_obs = depth_flat.view(batch_size, 1, img_size, img_size)
                    
                    # Downsample/upsample to target 84×84 
                    import torch.nn.functional as F
                    depth_obs = F.interpolate(depth_obs, size=(84, 84), mode='bilinear', align_corners=False)
            else:
                batch_size = obs.shape[0]
                depth_obs = torch.zeros(batch_size, 1, 84, 84, device=obs.device)
            
            # Forward pass to get architecture descriptors
            self.hyper_actor(depth_obs, state_obs, track=True)
            arch_descriptors = self.hyper_actor.arch_descriptors_per_state
            
            # Critic uses the full concatenated observations
            return self.critic(obs, arch_descriptors)
        
    def change_graph(self, repeat_sample=True):
        """Change architecture and regenerate network weights"""
        self.hyper_actor.change_graph(repeat_sample=repeat_sample)
    


class HyperOnPolicyRunner:
    """
    HyperPPO On-Policy Runner with Graph HyperNetwork support
    Following the ManiSkill HyperPPO integration pattern
    """

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # Handle privileged observations
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
            
        # Create standard ActorCritic
        policy_class_name = self.cfg["policy_class_name"]
        
        if policy_class_name == "ActorCritic":
            actor_critic = ActorCritic(
                self.env.num_obs,
                num_critic_obs,
                self.env.num_actions,
                **self.policy_cfg
            ).to(self.device)
        elif policy_class_name == "ActorCriticRecurrent":
            actor_critic = ActorCriticRecurrent(
                self.env.num_obs,
                num_critic_obs,
                self.env.num_actions,
                **self.policy_cfg
            ).to(self.device)
        else:
            raise ValueError(f"Unknown policy class: {policy_class_name}")
        
        # Create standard PPO algorithm
        algorithm_class_name = self.cfg["algorithm_class_name"]
        if algorithm_class_name == "PPO":
            self.alg = PPO(
                actor_critic,
                device=self.device,
                **self.alg_cfg
            )
        else:
            raise ValueError(f"Unknown algorithm class: {algorithm_class_name}")
        
        # Initialize HyperPPO components
        config_path = "/home/jiwoo/ws/go2_rl_jw/rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth84_kernel3.json"
        
        # Initialize HyperActor for weight generation
        hyper_actor_module = hyperActor(
            act_dim=self.env.num_actions,
            obs_dim=48,  # GO2 state dimension
            architecture_config_path=config_path,
            meta_batch_size=4,
            device=self.device,
            multi_gpu=False,
            architecture_sampling_mode="uniform"
        )
        
        # Get architecture descriptor dimension
        arch_descriptor_dim = hyper_actor_module.arch_max_len
        
        # Create GO2HyperAgent with architecture-conditioned critic
        self.go2_agent = GO2HyperAgent(
            hyper_actor=hyper_actor_module,
            device=self.device,
            state_dim=48,  # GO2 state dimension
            arch_descriptor_dim=arch_descriptor_dim
        )
        
        # Replace the actor in ActorCritic with our custom actor
        self.alg.actor_critic.actor = self.go2_agent
        
        # Replace the critic in ActorCritic with our architecture-conditioned critic
        self.alg.actor_critic.critic = self.go2_agent.critic
        
        # Initialize with first architecture
        hyper_actor_module.change_graph(repeat_sample=True)
        
        # Store for later access
        self.hyper_actor = hyper_actor_module
        
        # Load architecture config for logging
        with open(config_path, 'r') as f:
            self.arch_data = json.load(f)
        self.architectures = self.arch_data['architectures']
        self.meta_batch_size = 4
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [num_critic_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
        
        print(f"🚀 HyperOnPolicyRunner initialized:")
        print(f"  - Policy: {policy_class_name} (replaced with GO2HyperActor)")  
        print(f"  - Algorithm: {algorithm_class_name}")
        print(f"  - Device: {device}")
        print(f"  - Environments: {self.env.num_envs}")
        print(f"  - Steps per env: {self.num_steps_per_env}")
        print(f"  - HyperActor architectures: {len(self.architectures)}")
        print(f"  - Meta batch size: {self.meta_batch_size}")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Learn with HyperPPO support following ManiSkill pattern
        """
        # Initialize writer if logging
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.alg.actor_critic.train()  # switch to train mode

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # HyperPPO: Change architecture at start of each iteration
            print(f"\n🧠 HyperPPO Iteration {it+1}/{tot_iter}")
            self.go2_agent.change_graph(repeat_sample=True)
            print(f"  🔄 Architecture changed and weights regenerated")

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_entropy = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            
            # HyperPPO: Additional weight regenerations during update epochs
            print(f"  ⚡ Weight regenerations during learning: {self.alg.num_learning_epochs * self.alg.num_mini_batches}")

            if self.log_dir is not None:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += learn_time
                self.current_learning_iteration = it

                # Log to tensorboard
                if len(rewbuffer) > 0:
                    self.writer.add_scalar('Train/mean_reward', statistics.mean(rewbuffer), it)
                    self.writer.add_scalar('Train/mean_episode_length', statistics.mean(lenbuffer), it)
                
                self.writer.add_scalar('Loss/value_function', mean_value_loss, it)
                self.writer.add_scalar('Loss/surrogate', mean_surrogate_loss, it)
                self.writer.add_scalar('Loss/entropy', mean_entropy, it)
                self.writer.add_scalar('Policy/learning_rate', self.alg.learning_rate, it)
                self.writer.add_scalar('Perf/total_fps', int(self.tot_timesteps / self.tot_time), it)
                self.writer.add_scalar('Perf/collection_time', collection_time, it)
                self.writer.add_scalar('Perf/learning_time', learn_time, it)
                
                if len(rewbuffer) > 0:
                    print(f"  📊 Mean reward: {statistics.mean(rewbuffer):.2f}")
                    print(f"  📈 Mean episode length: {statistics.mean(lenbuffer):.2f}")

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                    
            ep_infos.clear()
        
        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def save(self, path, infos=None):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        """Load model and training state"""
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """Get policy for inference"""
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference