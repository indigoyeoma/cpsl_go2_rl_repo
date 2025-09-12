# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# HyperPPO Rollout Storage for meta-batch architecture management

import torch
import numpy as np
from rsl_rl.storage.rollout_storage import RolloutStorage

class HyperRolloutStorage(RolloutStorage):
    """
    HyperPPO-specific rollout storage that organizes data by meta-batch architectures
    
    Key differences from standard RolloutStorage:
    1. Groups environments by architecture (meta_batch_size architectures)
    2. Stores architecture descriptors per meta-batch, not per environment
    3. Handles architecture-conditioned data organization
    """
    
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            # HyperPPO: Architecture descriptors (meta_batch_size, descriptor_dim)
            self.arch_descriptors = None
            # HyperPPO: Architecture assignments for environments (num_envs,)
            self.arch_assignments = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, 
                 meta_batch_size=2, arch_descriptor_dim=16, device='cpu'):
        
        # Initialize parent class
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        
        self.meta_batch_size = meta_batch_size
        self.arch_descriptor_dim = arch_descriptor_dim
        
        # HyperPPO specific storage
        # Store architecture descriptors per step (not per environment)
        self.arch_descriptors = torch.zeros(num_transitions_per_env, meta_batch_size, arch_descriptor_dim, device=self.device)
        
        # Store which architecture each environment is using
        self.arch_assignments = torch.zeros(num_transitions_per_env, num_envs, dtype=torch.long, device=self.device)
        
        # HyperRolloutStorage initialization complete

    def add_transitions(self, transition):
        """Add transition data with HyperPPO architecture handling"""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
            
        # Standard data storage
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: 
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, 1))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        # HyperPPO specific: Architecture descriptors and assignments
        if hasattr(transition, 'arch_descriptors') and transition.arch_descriptors is not None:
            if transition.arch_descriptors.shape[0] == self.meta_batch_size:
                # Direct storage: already has meta_batch_size descriptors
                self.arch_descriptors[self.step].copy_(transition.arch_descriptors)
                
                # Create architecture assignments: which env uses which architecture
                envs_per_arch = self.num_envs // self.meta_batch_size
                for arch_idx in range(self.meta_batch_size):
                    start_env = arch_idx * envs_per_arch
                    end_env = min(start_env + envs_per_arch, self.num_envs)
                    self.arch_assignments[self.step, start_env:end_env] = arch_idx
                    
                # Handle remaining environments (assign to first architecture)
                remaining_envs = self.num_envs - (self.meta_batch_size * envs_per_arch)
                if remaining_envs > 0:
                    start_remaining = self.meta_batch_size * envs_per_arch
                    self.arch_assignments[self.step, start_remaining:] = 0
            else:
                # Fallback: extract unique architectures from per-environment descriptors
                # This handles cases where transition has replicated descriptors
                unique_arch_descriptors = []
                envs_per_arch = self.num_envs // self.meta_batch_size
                
                for arch_idx in range(self.meta_batch_size):
                    env_idx = arch_idx * envs_per_arch
                    unique_arch_descriptors.append(transition.arch_descriptors[env_idx])
                
                arch_descriptors_tensor = torch.stack(unique_arch_descriptors)
                self.arch_descriptors[self.step].copy_(arch_descriptors_tensor)
                
                # Create assignments
                for arch_idx in range(self.meta_batch_size):
                    start_env = arch_idx * envs_per_arch
                    end_env = min(start_env + envs_per_arch, self.num_envs)
                    self.arch_assignments[self.step, start_env:end_env] = arch_idx

        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Generate minibatches with architecture-aware sampling that maintains alignment"""
        # CRITICAL: advantages tensor is one step shorter due to [:-1] slicing
        # batch_size must reflect the actual available data
        available_steps = self.num_transitions_per_env - 1  # advantages has one fewer step
        batch_size = self.num_envs * available_steps
        mini_batch_size = batch_size // num_mini_batches
        envs_per_arch = self.num_envs // self.meta_batch_size
        
        
        for epoch in range(num_epochs):
            # CRITICAL: Create indices that maintain architecture-environment alignment
            # We need to ensure that data from the same architecture stays together
            
            # Option 1: Simple sequential minibatches (no randomization within epoch)
            # This maintains perfect alignment but less stochastic
            if True:  # Use sequential minibatches for IsaacGym compatibility
                for i in range(num_mini_batches):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_indices = torch.arange(start, end, device=self.device)
                    
                    # Convert to step and env indices
                    step_indices = batch_indices // self.num_envs
                    env_indices = batch_indices % self.num_envs
                    
                    # Safety bounds checking with clamping
                    # Use available_steps (which is num_transitions_per_env - 1) for advantages tensor
                    step_indices = torch.clamp(step_indices, 0, available_steps - 1)
                    env_indices = torch.clamp(env_indices, 0, self.num_envs - 1)
                    
                    obs_batch = self.observations[step_indices, env_indices]
                    if self.privileged_observations is not None:
                        critic_obs_batch = self.privileged_observations[step_indices, env_indices]
                    else:
                        critic_obs_batch = obs_batch
                    
                    actions_batch = self.actions[step_indices, env_indices]
                    target_values_batch = self.values[step_indices, env_indices].squeeze(-1)
                    returns_batch = self.returns[step_indices, env_indices].squeeze(-1)
                    advantages_batch = self.advantages[step_indices, env_indices].squeeze(-1)
                    old_actions_log_prob_batch = self.actions_log_prob[step_indices, env_indices].squeeze(-1)
                    old_mu_batch = self.mu[step_indices, env_indices]
                    old_sigma_batch = self.sigma[step_indices, env_indices]
                    
                    # Get architecture descriptors based on BOTH step and environment indices
                    # CRITICAL: Use the actual step indices to get the correct architectures
                    arch_indices = torch.clamp(env_indices // envs_per_arch, 0, self.meta_batch_size - 1)
                    
                    # Extract architecture descriptors for each (step, env) pair
                    arch_descriptors_batch = []
                    for i, (step_idx, env_idx) in enumerate(zip(step_indices, env_indices)):
                        step_arch_descriptors = self.arch_descriptors[step_idx.item()]
                        arch_idx = torch.clamp(env_idx // envs_per_arch, 0, self.meta_batch_size - 1)
                        arch_descriptors_batch.append(step_arch_descriptors[arch_idx.item()])
                    
                    arch_descriptors_batch = torch.stack(arch_descriptors_batch)
                    
                    yield obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
                          advantages_batch, returns_batch, old_actions_log_prob_batch, \
                          old_mu_batch, old_sigma_batch, (None, None), None, arch_descriptors_batch
            
            else:
                # Option 2: Architecture-aware shuffling (maintains alignment within architectures)
                # Create shuffled indices that respect architecture boundaries
                indices = []
                for arch_idx in range(self.meta_batch_size):
                    # Get all indices for this architecture
                    arch_start = arch_idx * envs_per_arch
                    arch_end = min((arch_idx + 1) * envs_per_arch, self.num_envs)
                    
                    # Create indices for all timesteps and environments of this architecture
                    arch_indices = []
                    for step in range(self.num_transitions_per_env):
                        for env in range(arch_start, arch_end):
                            arch_indices.append(step * self.num_envs + env)
                    
                    # Shuffle within architecture
                    arch_indices = torch.tensor(arch_indices, device=self.device)
                    arch_indices = arch_indices[torch.randperm(len(arch_indices), device=self.device)]
                    indices.append(arch_indices)
                
                # Interleave architecture indices for balanced minibatches
                all_indices = torch.cat(indices)
                
                for i in range(num_mini_batches):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_indices = all_indices[start:end]
                    
                    # Convert to step and env indices
                    step_indices = batch_indices // self.num_envs
                    env_indices = batch_indices % self.num_envs
                    
                    # Safety bounds checking with clamping
                    # Use available_steps (which is num_transitions_per_env - 1) for advantages tensor
                    step_indices = torch.clamp(step_indices, 0, available_steps - 1)
                    env_indices = torch.clamp(env_indices, 0, self.num_envs - 1)
                    
                    obs_batch = self.observations[step_indices, env_indices]
                    if self.privileged_observations is not None:
                        critic_obs_batch = self.privileged_observations[step_indices, env_indices]
                    else:
                        critic_obs_batch = obs_batch
                    
                    actions_batch = self.actions[step_indices, env_indices]
                    target_values_batch = self.values[step_indices, env_indices].squeeze(-1)
                    returns_batch = self.returns[step_indices, env_indices].squeeze(-1)
                    advantages_batch = self.advantages[step_indices, env_indices].squeeze(-1)
                    old_actions_log_prob_batch = self.actions_log_prob[step_indices, env_indices].squeeze(-1)
                    old_mu_batch = self.mu[step_indices, env_indices]
                    old_sigma_batch = self.sigma[step_indices, env_indices]
                    
                    # Get architecture descriptors
                    first_step_arch_descriptors = self.arch_descriptors[0]
                    arch_indices = torch.clamp(env_indices // envs_per_arch, 0, self.meta_batch_size - 1)
                    arch_descriptors_batch = first_step_arch_descriptors[arch_indices]
                    
                    yield obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
                          advantages_batch, returns_batch, old_actions_log_prob_batch, \
                          old_mu_batch, old_sigma_batch, (None, None), None, arch_descriptors_batch

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Recurrent version - not implemented for HyperPPO"""
        raise NotImplementedError("Recurrent version not implemented for HyperPPO")

    def compute_returns(self, last_values, gamma, lam):
        """Compute returns with proper tensor shapes for HyperPPO"""
        # Ensure last_values has correct shape [num_envs, 1]
        if last_values.dim() == 1:
            last_values = last_values.view(-1, 1)
        
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].float()
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantage = delta + gamma * lam * next_non_terminal * advantage
            self.returns[step] = advantage + self.values[step]
        
        # Compute advantages (normalized)
        self.advantages = self.returns[:-1] - self.values[:-1]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """Get statistics including architecture distribution"""
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([0], dtype=torch.bool), flat_dones.squeeze(-1).bool()))
        trajectory_lengths = (done_indices[1:] != done_indices[:-1]).nonzero(as_tuple=False)[:, 0]
        
        # Architecture usage statistics
        arch_usage = {}
        for arch_idx in range(self.meta_batch_size):
            usage_count = (self.arch_assignments == arch_idx).sum().item()
            arch_usage[f'arch_{arch_idx}_usage'] = usage_count / (self.num_envs * self.num_transitions_per_env)
        
        return trajectory_lengths, arch_usage