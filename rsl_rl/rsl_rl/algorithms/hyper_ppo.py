# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# HyperPPO extension for Graph HyperNetworks

"""
KL Divergence Fix for Continuous Actions with Tanh Squashing

The KL divergence fix required addressing a mathematical inconsistency in how log probabilities were computed for
continuous actions with tanh squashing.

What Was Required:

1. Understanding the Problem
   - PPO requires: ratio = exp(newlogprob - oldlogprob)
   - Our bug: oldlogprob (rollout) and newlogprob (training) used different formulas
   - Result: Artificial KL divergence of ~3.0 instead of ~0

2. The Mathematical Issue
   Tanh-squashed continuous actions require a Jacobian correction because:
   - Raw action u ~ Normal(μ, σ)
   - Final action a = tanh(u)
   - Correct log_prob: log π(a) = log π(u) - log|da/du|
   - Jacobian term: log|da/du| = log(1 - tanh²(u)) = log(1 - a²)

3. The Inconsistency
   Rollout (sampling mode):
   raw_action = dist.sample()
   action = torch.tanh(raw_action)
   log_prob = dist.log_prob(raw_action).sum(1) - torch.log(1 - action.pow(2) + 1e-6).sum(1)
   ✅ Mathematically correct with Jacobian correction

   Training (evaluation mode):
   log_prob = dist.log_prob(action).sum(1)  # Direct on tanh-squashed actions
   ❌ Mathematically wrong - missing Jacobian correction

4. The Fix Required
   Make both phases use the same correct formula:
   # Convert tanh action back to raw action space
   raw_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
   # Apply correct log probability with Jacobian
   log_prob = dist.log_prob(raw_action).sum(1) - torch.log(1 - action.pow(2) + 1e-6).sum(1)

5. Why This Fix Worked
   - Before: Comparing apples (correct log_prob) vs oranges (incorrect log_prob) → High KL
   - After: Comparing apples vs apples (both correct) → Normal KL ≈ 0

6. Key Requirements for the Solution
   1. Mathematical consistency: Same formula for both rollout and training
   2. Numerical stability: Clamping action to avoid atanh overflow
   3. Jacobian correction: Including the log(1 - a²) term for tanh squashing
   4. Proper inverse: Using torch.atanh() to convert back to raw action space

The fix was essentially ensuring that PPO compares like with like - both old and new log probabilities computed using the
same mathematically sound approach for continuous tanh-squashed actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticRMA
from rsl_rl.modules.hyper_actor_critic import HyperPPOActorCritic
from rsl_rl.storage import RolloutStorage
import wandb

class HyperPPO:
    """
    HyperPPO Algorithm with Graph HyperNetwork (GHN) support
    
    Key differences from standard PPO:
    1. Resamples architectures at the start of each training iteration
    2. Generates fresh weights for each minibatch during training
    3. Trains the GHN to generate good weights for any architecture
    """
    actor_critic: HyperPPOActorCritic
    
    def __init__(self,
                 actor_critic,
                 estimator=None,
                 estimator_paras=None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # HyperPPO specific parameters
                 hyper_enabled=True,
                 ratio_clamp_min=0.05,  # Prevent GHN training instability
                 ratio_clamp_max=20.0,
                 ):

        self.device = device
        self.hyper_enabled = hyper_enabled

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        # HyperPPO specific: ratio clamping for GHN stability
        self.ratio_clamp_min = ratio_clamp_min
        self.ratio_clamp_max = ratio_clamp_max

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # Track training iteration for GHN monitoring
        self.training_iteration = 0

        print(f"🚀 HyperPPO initialized with hyper_enabled={hyper_enabled}")
        if self.hyper_enabled:
            print(f"🔧 GHN ratio clamping: [{ratio_clamp_min}, {ratio_clamp_max}]")

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
        
        # CRITICAL: Initialize architectures for first rollout
        if self.hyper_enabled and hasattr(self.actor_critic, 'resample_architectures'):
            print("🎯 Initializing architectures for first rollout...")
            self.actor_critic.resample_architectures()

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        """
        Act with current policy - no architecture changes during rollout
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Get current architecture descriptors for tracking
        arch_descriptors = None
        if self.hyper_enabled and hasattr(self.actor_critic, 'get_current_arch_descriptors'):
            arch_descriptors = self.actor_critic.get_current_arch_descriptors()
            
        # Compute the actions and values - depth now included in observations
        self.transition.actions = self.actor_critic.act(obs, arch_descriptors=arch_descriptors).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, arch_descriptors=arch_descriptors).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # HyperPPO: Store architecture descriptors with transition
        self.transition.arch_descriptors = arch_descriptors
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """
        HyperPPO Update with GHN Training Integration
        
        Key GHN Training Steps:
        1. MAINTAIN: Keep architectures from rollout phase (DO NOT resample)
        2. Minibatch Loop: Generate fresh weights for SAME architectures (GHN training signal)
        3. End: Resample architectures for NEXT rollout iteration
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        
        # CRITICAL: Do NOT resample architectures here - must use rollout architectures!
        # The PPO loss requires consistency between rollout and training architectures
        
        # Optional: Monitor GHN training progress at start of training
        if self.hyper_enabled and self.training_iteration % 50 == 0:
            if hasattr(self.actor_critic, 'monitor_training'):
                print(f"🧠 [Iteration {self.training_iteration}] GHN Training Progress Monitor")
                self.actor_critic.monitor_training()
        
        # HyperPPO Training: Detailed per-epoch, per-minibatch structure
        total_weight_regenerations = 0
        
        # FOR EACH EPOCH (e.g., 3 epochs)
        for epoch in range(self.num_learning_epochs):
            print(f"🏋️ [Iteration {self.training_iteration}] Starting Epoch {epoch + 1}/{self.num_learning_epochs}")
            
            # Get minibatch generator for this epoch
            if self.actor_critic.is_recurrent:
                minibatch_generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, 1)  # 1 epoch at a time
            else:
                minibatch_generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)  # 1 epoch at a time
            
            minibatch_in_epoch = 0
            # FOR EACH MINIBATCH IN THIS EPOCH (e.g., 16 minibatches)  
            for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, arch_descriptors_batch in minibatch_generator:

                minibatch_in_epoch += 1
                total_weight_regenerations += 1
                
                # CRITICAL HYPERPPO STEP 2: Generate fresh weights W_mb_X for SAME architectures
                # This is the KEY to GHN meta-learning - same architectures, fresh weights each minibatch
                if self.hyper_enabled and hasattr(self.actor_critic, 'regenerate_weights'):
                    print(f"⚡ [E{epoch+1}/MB{minibatch_in_epoch}] Regenerating weights #{total_weight_regenerations}")
                    self.actor_critic.regenerate_weights()  # calls change_graph(repeat_sample=True)
                
                # Standard PPO forward pass with fresh weights
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0], arch_descriptors=arch_descriptors_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1], arch_descriptors=arch_descriptors_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL divergence tracking
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)  # Lower minimum for GHN
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # PPO Surrogate loss with GHN-specific ratio clamping
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                
                # CRITICAL: Clamp ratio to prevent GHN training instability
                if self.hyper_enabled:
                    ratio = torch.clamp(ratio, self.ratio_clamp_min, self.ratio_clamp_max)
                
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Total loss: PPO loss trains the GHN to generate good weights
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step: This updates the GHN parameters, not the target network weights!
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
            
            print(f"✅ [Iteration {self.training_iteration}] Epoch {epoch + 1} complete - {minibatch_in_epoch} minibatches processed")
        
        # Training summary
        total_minibatches = self.num_learning_epochs * self.num_mini_batches
        print(f"🎯 [Iteration {self.training_iteration}] Training complete:")
        print(f"   📊 Total minibatches: {total_minibatches}")
        print(f"   ⚡ Weight regenerations: {total_weight_regenerations}")
        print(f"   🔄 Regenerations per architecture: {total_weight_regenerations} times")

        # CRITICAL HYPERPPO STEP 3: Resample architectures for NEXT rollout iteration
        if self.hyper_enabled and hasattr(self.actor_critic, 'resample_architectures'):
            print(f"🔄 [Iteration {self.training_iteration}] Training complete. Resampling architectures for next rollout...")
            self.actor_critic.resample_architectures()  # calls change_graph(repeat_sample=False)

        # Calculate averages across all minibatches
        mean_value_loss /= total_minibatches
        mean_surrogate_loss /= total_minibatches  
        mean_entropy /= total_minibatches
        self.storage.clear()
        
        self.training_iteration += 1
        
        # HyperPPO specific logging with detailed weight regeneration info
        if self.hyper_enabled:
            print(f"🧠 [Iteration {self.training_iteration}] GHN Training Summary:")
            print(f"   📊 Mean Surrogate Loss: {mean_surrogate_loss:.6f}")
            print(f"   📈 Mean Value Loss: {mean_value_loss:.6f}") 
            print(f"   🎲 Mean Entropy: {mean_entropy:.6f}")
            print(f"   🔧 Learning Rate: {self.learning_rate:.2e}")
            print(f"   ⚡ Total Weight Regenerations: {total_weight_regenerations}")
            print(f"   🏗️ Expected (epochs × minibatches): {self.num_learning_epochs} × {self.num_mini_batches} = {total_minibatches}")

        return mean_value_loss, mean_surrogate_loss, mean_entropy