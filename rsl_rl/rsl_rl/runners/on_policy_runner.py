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

import time
import os
from collections import deque
import statistics
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.modules.depth_backbone import DepthAugmentation
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        actor_critic: ActorCriticRMA = ActorCriticRMA(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg).to(self.device)
        estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder (depth_encoder + depth_actor architecture)
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        self.num_parallel_archs = self.depth_encoder_cfg.get("num_parallel_architectures", 1)

        if self.if_depth:
            # Check if using GHN-sampled backbone
            use_ghn = self.depth_encoder_cfg.get("use_ghn_encoder", False)

            # Check for parallel architecture training (GHN training mode)
            if self.num_parallel_archs > 1:
                print(f"=== GHN Training Mode: {self.num_parallel_archs} architectures ===")
                from rsl_rl.modules.depth_encoder_ghn import (
                    sample_depth_encoder_configs,
                    build_depth_backbone,
                    TrainableGHN,
                )

                # Split envs across architectures
                self.envs_per_arch = self.env.num_envs // self.num_parallel_archs
                assert self.env.num_envs % self.num_parallel_archs == 0, \
                    f"num_envs ({self.env.num_envs}) must be divisible by num_parallel_architectures ({self.num_parallel_archs})"
                print(f"  Envs per architecture: {self.envs_per_arch}")

                # Create trainable GHN (will be trained from scratch)
                # hid=128 provides sufficient capacity for the architecture search space
                # (2-4 layers, channels up to 128, kernels 3-7, fc_hidden 64-256)
                self.trainable_ghn = TrainableGHN(
                    max_shape=(128, 128, 7, 7),
                    num_classes=32,  # depth latent dim
                    hid=128,         # increased from 64 for better generalization
                    device=self.device,
                )
                print(f"  GHN params: {sum(p.numel() for p in self.trainable_ghn.parameters()):,}")

                # Sample initial architecture configs
                self.arch_configs = sample_depth_encoder_configs(self.num_parallel_archs)
                print(f"  Sampled {len(self.arch_configs)} architectures:")
                for i, cfg in enumerate(self.arch_configs):
                    print(f"    [{i}] {cfg}")

                # Build raw backbones (GHN will predict weights for these)
                self.depth_backbones = []
                for cfg in self.arch_configs:
                    backbone = build_depth_backbone(cfg).to(self.device)
                    self.depth_backbones.append(backbone)

                # Store build_depth_backbone for rebuilding each iteration
                self._build_depth_backbone = build_depth_backbone

                # Shared depth_actor (takes latent from any architecture)
                self.depth_actor_shared = deepcopy(actor_critic.actor)

                # Shared encoder layers (fuse visual features with proprioception)
                activation = nn.ELU()
                last_activation = nn.Tanh()

                # Proprio fusion: [32 visual + n_proprio] → [32 fused]
                self.combination_mlp_shared = nn.Sequential(
                    nn.Linear(32 + env.cfg.env.n_proprio, 128),
                    activation,
                    nn.Linear(128, 32)
                ).to(self.device)

                # Output: [32 fused] → [34] (32 latent + 2 yaw)
                self.output_mlp_shared = nn.Sequential(
                    nn.Linear(32, 32 + 2),
                    last_activation
                ).to(self.device)

                # Optimizer for GHN + depth_actor + shared encoder MLPs
                self.ghn_optimizer = optim.Adam(
                    list(self.trainable_ghn.parameters()) +
                    list(self.depth_actor_shared.parameters()) +
                    list(self.combination_mlp_shared.parameters()) +
                    list(self.output_mlp_shared.parameters()),
                    lr=self.depth_encoder_cfg["learning_rate"]
                )

                total_backbone_params = sum(sum(p.numel() for p in bb.parameters()) for bb in self.depth_backbones)
                print(f"  Total backbone params (before GHN): {total_backbone_params:,}")
                print(f"  Depth actor params: {sum(p.numel() for p in self.depth_actor_shared.parameters()):,}")

                # For compatibility with PPO class (GHN doesn't use these)
                depth_encoder = None
                depth_actor = None

            elif use_ghn:
                print("Using PPUDA GHN-sampled CNN backbone (single architecture)")
                from rsl_rl.modules.depth_encoder_ghn import (
                    sample_depth_encoder_config,
                    build_depth_backbone,
                    DepthEncoderGHN,
                    BASELINE_CONFIG, DEEP_CONFIG, WIDE_CONFIG, LIGHT_CONFIG,
                )

                # Get config
                ghn_preset = self.depth_encoder_cfg.get("ghn_preset", None)
                ghn_config_dict = self.depth_encoder_cfg.get("ghn_config", None)

                if ghn_preset == 'baseline':
                    ghn_config = BASELINE_CONFIG
                elif ghn_preset == 'deep':
                    ghn_config = DEEP_CONFIG
                elif ghn_preset == 'wide':
                    ghn_config = WIDE_CONFIG
                elif ghn_preset == 'light':
                    ghn_config = LIGHT_CONFIG
                elif ghn_preset == 'random':
                    ghn_config = sample_depth_encoder_config()
                    print(f"Sampled random config: {ghn_config}")
                elif ghn_config_dict is not None:
                    from rsl_rl.modules.depth_encoder_ghn import DepthEncoderConfig
                    ghn_config = DepthEncoderConfig(**ghn_config_dict)
                else:
                    ghn_config = BASELINE_CONFIG

                # Build backbone
                depth_backbone = build_depth_backbone(ghn_config)
                print(f"GHN Backbone: {sum(p.numel() for p in depth_backbone.parameters()):,} params")
                print(f"  Config: {ghn_config}")

                # Optionally initialize with GHN
                if self.depth_encoder_cfg.get("use_ghn_init", True):
                    ghn = DepthEncoderGHN(device=self.device)
                    depth_backbone = ghn(depth_backbone)
                    print("  Initialized with PPUDA GHN")

                # Wrap backbone with SimpleDepthEncoder
                depth_encoder = SimpleDepthEncoder(depth_backbone, env.cfg).to(self.device)
                depth_actor = deepcopy(actor_critic.actor)
            else:
                # Select backbone based on depth resolution (matched to parkour)
                depth_resolution = getattr(env.cfg.depth, 'resized', (87, 58))  # (width, height)
                if depth_resolution == (64, 48):
                    print("Using DepthOnlyFCBackbone48x64 (parkour resolution)")
                    depth_backbone = DepthOnlyFCBackbone48x64(
                        env.cfg.env.n_proprio,
                        self.policy_cfg["scan_encoder_dims"][-1],
                        self.depth_encoder_cfg["hidden_dims"],
                    )
                else:
                    print(f"Using DepthOnlyFCBackbone58x87 (resolution: {depth_resolution})")
                    depth_backbone = DepthOnlyFCBackbone58x87(
                        env.cfg.env.n_proprio,
                        self.policy_cfg["scan_encoder_dims"][-1],
                        self.depth_encoder_cfg["hidden_dims"],
                    )

                # Wrap backbone with SimpleDepthEncoder (adds proprio fusion + yaw prediction)
                depth_encoder = SimpleDepthEncoder(depth_backbone, env.cfg).to(self.device)
                depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None

        # Initialize depth augmentation for sim-to-real transfer
        self.depth_augmentation = None
        if self.if_depth and self.depth_encoder_cfg.get("use_depth_augmentation", False):
            aug_cfg = self.depth_encoder_cfg.get("augmentation", {})
            self.depth_augmentation = DepthAugmentation(
                dropout_prob=aug_cfg.get("dropout_prob", 0.02),
                noise_std=aug_cfg.get("noise_std", 0.03),
                cutout_prob=aug_cfg.get("cutout_prob", 0.2),
                cutout_size_range=aug_cfg.get("cutout_size_range", (5, 15)),
                edge_noise_prob=aug_cfg.get("edge_noise_prob", 0.3),
                edge_noise_std=aug_cfg.get("edge_noise_std", 0.05),
                scale_jitter=aug_cfg.get("depth_scale_range", (0.95, 1.05))[1] - 1.0,  # Convert range to jitter
                flip_prob=aug_cfg.get("flip_prob", 0.0),
                hole_value=aug_cfg.get("hole_value", -0.5),
            ).to(self.device)
            print(f"Depth augmentation enabled: {self.depth_augmentation}")

        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic,
                                  estimator, self.estimator_cfg,
                                  depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Select learning method based on configuration
        if not self.if_depth:
            self.learn = self.learn_RL
        elif self.num_parallel_archs > 1:
            self.learn = self.learn_vision_ghn  # True GHN parallel training
        else:
            self.learn = self.learn_vision
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            # DAgger update: train history encoder to match privileged encoder
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            # Save every save_interval iterations
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        # Open debug log file for simple DAgger
        debug_log_path = os.path.join(self.log_dir, 'dagger_debug.txt') if self.log_dir else None
        debug_log_file = open(debug_log_path, 'w') if debug_log_path else None
        if debug_log_file:
            debug_log_file.write("Simple DAgger Training Debug Log\n")
            debug_log_file.write("=" * 80 + "\n")

        # DIAGNOSTIC: Check depth image values and save to file
        diag_path = os.path.join(self.log_dir, 'diagnostic_output.txt') if self.log_dir else 'diagnostic_output.txt'
        with open(diag_path, 'w') as diag_file:
            diag_file.write("="*60 + "\n")
            diag_file.write("DIAGNOSTIC: Checking depth images and latents\n")
            diag_file.write("="*60 + "\n")
            with torch.no_grad():
                test_depth = infos["depth"]
                diag_file.write(f"Depth image shape: {test_depth.shape}\n")
                diag_file.write(f"Depth image: min={test_depth.min():.4f}, max={test_depth.max():.4f}, mean={test_depth.mean():.4f}, std={test_depth.std():.4f}\n")

                # Check if depth encoder produces reasonable output
                obs_prop_test = obs[:, :self.env.cfg.env.n_proprio].clone()
                obs_prop_test[:, 6:8] = 0
                depth_latent_test = self.alg.depth_encoder(test_depth.clone(), obs_prop_test)
                diag_file.write(f"Depth latent: min={depth_latent_test.min():.4f}, max={depth_latent_test.max():.4f}, mean={depth_latent_test.mean():.4f}, std={depth_latent_test.std():.4f}\n")

                # Compare to scan latent
                scan_latent_test = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                diag_file.write(f"Scan latent: min={scan_latent_test.min():.4f}, max={scan_latent_test.max():.4f}, mean={scan_latent_test.mean():.4f}, std={scan_latent_test.std():.4f}\n")
            diag_file.write("="*60 + "\n")
        print(f"Diagnostic output saved to: {diag_path}")

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] != None:
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    # Apply depth augmentation for sim-to-real transfer (only during training)
                    depth_input = infos["depth"].clone()
                    if self.depth_augmentation is not None:
                        depth_input = self.depth_augmentation(depth_input)
                    depth_latent_and_yaw = self.alg.depth_encoder(depth_input, obs_prop_depth)
                    
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5*depth_latent_and_yaw[:, -2:]
                    
                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])
                
                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                # obs_student[:, 6:8] = yaw.detach()
                obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # DEBUG: Log action stats every 100 iterations (same as GHN)
                if i == 0 and it % 100 == 0 and debug_log_file:
                    action_diff = (actions_teacher - actions_student).abs()
                    # Get teacher's scan_latent for comparison
                    with torch.no_grad():
                        obs_scan = obs[:, self.env.cfg.env.n_proprio:self.env.cfg.env.n_proprio + self.env.cfg.env.n_scan]
                        scan_latent = self.alg.actor_critic.actor.scan_encoder(obs_scan)
                    debug_msg = (
                        f"\n--- Iteration {it} ---\n"
                        f"Teacher actions: mean={actions_teacher.mean().item():.4f}, std={actions_teacher.std().item():.4f}, "
                        f"min={actions_teacher.min().item():.4f}, max={actions_teacher.max().item():.4f}\n"
                        f"Student actions: mean={actions_student.mean().item():.4f}, std={actions_student.std().item():.4f}, "
                        f"min={actions_student.min().item():.4f}, max={actions_student.max().item():.4f}\n"
                        f"Action diff: mean={action_diff.mean().item():.4f}, max={action_diff.max().item():.4f}\n"
                        f"Scan latent (teacher): mean={scan_latent.mean().item():.4f}, std={scan_latent.std().item():.4f}, "
                        f"min={scan_latent.min().item():.4f}, max={scan_latent.max().item():.4f}\n"
                        f"Depth latent (student): mean={depth_latent.mean().item():.4f}, std={depth_latent.std().item():.4f}, "
                        f"min={depth_latent.min().item():.4f}, max={depth_latent.max().item():.4f}\n"
                        f"Yaw pred: mean={yaw.mean().item():.4f}, std={yaw.std().item():.4f}\n"
                        f"Yaw true: mean={obs[:, 6:8].mean().item():.4f}, std={obs[:, 6:8].std().item():.4f}\n"
                        f"Obs scan (first 10): {obs[0, self.env.cfg.env.n_proprio:self.env.cfg.env.n_proprio+10].tolist()}\n"
                        f"Commands: lin_vel_x={self.env.commands[:, 0].mean().item():.4f}, "
                        f"heading={self.env.commands[:, 3].mean().item():.4f}\n"
                        f"Base vel: lin_x={self.env.base_lin_vel[:, 0].mean().item():.4f}, "
                        f"lin_y={self.env.base_lin_vel[:, 1].mean().item():.4f}, "
                        f"ang_z={self.env.base_ang_vel[:, 2].mean().item():.4f}\n"
                    )
                    debug_log_file.write(debug_msg)
                    debug_log_file.flush()
                    print(f"DEBUG iter {it}: teacher={actions_teacher.mean().item():.4f}, student={actions_student.mean().item():.4f}, diff={action_diff.mean().item():.4f}")

                # Student executes its own actions (DAgger)
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

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
            start = stop

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
            depth_actor_loss, yaw_loss = self.alg.update_depth_actor(actions_student_buffer, actions_teacher_buffer, yaw_buffer_student, yaw_buffer_teacher)

            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            # DEBUG: Log rewards and losses every 100 iterations
            if it % 100 == 0 and debug_log_file:
                mean_rew = np.mean(rewbuffer) if len(rewbuffer) > 0 else 0
                mean_len = np.mean(lenbuffer) if len(lenbuffer) > 0 else 0
                debug_msg = (
                    f"Rewards: mean={mean_rew:.4f}, ep_len={mean_len:.1f}\n"
                    f"Losses: depth_actor={depth_actor_loss:.6f}, yaw={yaw_loss:.6f}\n"
                )
                # Add reward breakdown from ep_infos
                if ep_infos:
                    for key in ep_infos[0]:
                        if 'rew' in key:
                            values = [ep_info[key].item() if hasattr(ep_info[key], 'item') else ep_info[key] for ep_info in ep_infos]
                            debug_msg += f"  {key}: {np.mean(values):.4f}\n"
                debug_log_file.write(debug_msg)
                debug_log_file.flush()

            if self.log_dir is not None:
                self.log_vision(locals())
            # Save every save_interval iterations
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        # Close debug log file
        if debug_log_file:
            debug_log_file.write("\n" + "=" * 80 + "\nTraining completed.\n")
            debug_log_file.close()
            print(f"Debug log saved to: {debug_log_path}")

    def learn_vision_ghn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        GHN Training with DAgger - IDENTICAL to learn_vision but uses GHN-predicted backbones.

        Training loop (same as learn_vision):
        1. Student generates actions using depth encoder (GHN-predicted weights)
        2. Teacher provides action labels (privileged)
        3. Student EXECUTES its own actions (DAgger)
        4. Loss = ||student_action - teacher_action||
        5. Backprop through depth encoder and GHN
        """
        from rsl_rl.modules.depth_encoder_ghn import sample_depth_encoder_config

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        # Set training mode for GHN components
        self.trainable_ghn.train()
        for backbone in self.depth_backbones:
            backbone.train()
        self.depth_actor_shared.train()
        self.combination_mlp_shared.train()
        self.output_mlp_shared.train()

        # Track which slices need resampling
        slices_to_resample = set()

        # Track total architectures seen for logging
        total_archs_sampled = self.num_parallel_archs

        # Open debug log file (same as learn_vision)
        debug_log_path = os.path.join(self.log_dir, 'dagger_debug.txt') if self.log_dir else None
        debug_log_file = open(debug_log_path, 'w') if debug_log_path else None
        if debug_log_file:
            debug_log_file.write("GHN DAgger Training Debug Log\n")
            debug_log_file.write("=" * 80 + "\n")

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Resample architectures for slices that had resets
            if len(slices_to_resample) > 0:
                for arch_idx in slices_to_resample:
                    new_config = sample_depth_encoder_config()
                    self.arch_configs[arch_idx] = new_config
                    total_archs_sampled += 1
                slices_to_resample.clear()

            # Build fresh backbones on CPU, then GHN predicts weights and moves to device
            backbones = [self._build_depth_backbone(cfg) for cfg in self.arch_configs]
            self.depth_backbones = self.trainable_ghn.predict_weights(backbones)
            for backbone in self.depth_backbones:
                backbone.train()

            # Buffers for this iteration (same as learn_vision)
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []

            # Collect rollouts - all envs step in parallel (Isaac Gym)
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] is not None:
                    # Get teacher's scandots_latent for comparison (same as learn_vision)
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)

                    # Apply depth augmentation for sim-to-real transfer (only during training)
                    depth_input = infos["depth"].clone()
                    if self.depth_augmentation is not None:
                        depth_input = self.depth_augmentation(depth_input)

                    # Step 1: GHN backbones process depth → [32] visual features
                    depth_visual = torch.zeros(self.env.num_envs, 32, device=self.device)
                    for arch_idx, backbone in enumerate(self.depth_backbones):
                        start_idx = arch_idx * self.envs_per_arch
                        end_idx = (arch_idx + 1) * self.envs_per_arch
                        depth_slice = depth_input[start_idx:end_idx]
                        depth_visual[start_idx:end_idx] = backbone(depth_slice)

                    # Step 2: Get proprioception (zero out yaw for student to predict)
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0

                    # Step 3: Fuse visual + proprio → [32]
                    depth_fused = self.combination_mlp_shared(torch.cat([depth_visual, obs_prop_depth], dim=-1))

                    # Step 4: Output layer → [34] (32 latent + 2 yaw)
                    depth_latent_and_yaw = self.output_mlp_shared(depth_fused)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5 * depth_latent_and_yaw[:, -2:]

                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])

                # Teacher generates actions (privileged, uses scandots_latent)
                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                # Inject predicted yaw into student obs (same as learn_vision)
                obs_student = obs.clone()
                obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())

                # Student generates actions using depth_actor_shared
                actions_student = self.depth_actor_shared(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # DEBUG: Log action stats every 100 iterations (same as learn_vision)
                if i == 0 and it % 100 == 0 and debug_log_file:
                    action_diff = (actions_teacher - actions_student).abs()
                    with torch.no_grad():
                        obs_scan = obs[:, self.env.cfg.env.n_proprio:self.env.cfg.env.n_proprio + self.env.cfg.env.n_scan]
                        scan_latent = self.alg.actor_critic.actor.scan_encoder(obs_scan)
                    debug_msg = (
                        f"\n--- Iteration {it} ---\n"
                        f"Teacher actions: mean={actions_teacher.mean().item():.4f}, std={actions_teacher.std().item():.4f}, "
                        f"min={actions_teacher.min().item():.4f}, max={actions_teacher.max().item():.4f}\n"
                        f"Student actions: mean={actions_student.mean().item():.4f}, std={actions_student.std().item():.4f}, "
                        f"min={actions_student.min().item():.4f}, max={actions_student.max().item():.4f}\n"
                        f"Action diff: mean={action_diff.mean().item():.4f}, max={action_diff.max().item():.4f}\n"
                        f"Scan latent (teacher): mean={scan_latent.mean().item():.4f}, std={scan_latent.std().item():.4f}, "
                        f"min={scan_latent.min().item():.4f}, max={scan_latent.max().item():.4f}\n"
                        f"Depth latent (student): mean={depth_latent.mean().item():.4f}, std={depth_latent.std().item():.4f}, "
                        f"min={depth_latent.min().item():.4f}, max={depth_latent.max().item():.4f}\n"
                        f"Yaw pred: mean={yaw.mean().item():.4f}, std={yaw.std().item():.4f}\n"
                        f"Yaw true: mean={obs[:, 6:8].mean().item():.4f}, std={obs[:, 6:8].std().item():.4f}\n"
                        f"Obs scan (first 10): {obs[0, self.env.cfg.env.n_proprio:self.env.cfg.env.n_proprio+10].tolist()}\n"
                        f"Commands: lin_vel_x={self.env.commands[:, 0].mean().item():.4f}, "
                        f"heading={self.env.commands[:, 3].mean().item():.4f}\n"
                        f"Base vel: lin_x={self.env.base_lin_vel[:, 0].mean().item():.4f}, "
                        f"lin_y={self.env.base_lin_vel[:, 1].mean().item():.4f}, "
                        f"ang_z={self.env.base_ang_vel[:, 2].mean().item():.4f}\n"
                    )
                    debug_log_file.write(debug_msg)
                    debug_log_file.flush()
                    print(f"DEBUG iter {it}: teacher={actions_teacher.mean().item():.4f}, student={actions_student.mean().item():.4f}, diff={action_diff.mean().item():.4f}")

                # DAgger: Student executes its own actions (same as learn_vision)
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                # Logging - mean reward across all envs (same as learn_vision)
                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # Track which slices had episode resets for architecture resampling
                if len(new_ids) > 0:
                    for env_id in new_ids[:, 0].cpu().numpy():
                        arch_idx = env_id // self.envs_per_arch
                        slices_to_resample.add(arch_idx)

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Aggregate buffers (same as learn_vision)
            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)

            # Compute losses (same as learn_vision / update_depth_actor)
            depth_encoder_loss = 0  # Not used, kept for compatibility
            depth_actor_loss = (actions_teacher_buffer.detach() - actions_student_buffer).norm(p=2, dim=1).mean()
            yaw_loss = (yaw_buffer_teacher.detach() - yaw_buffer_student).norm(p=2, dim=1).mean()
            total_loss = depth_actor_loss + yaw_loss

            # Backprop through GHN
            self.ghn_optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients for all trainable parameters
            all_params = (
                list(self.trainable_ghn.parameters()) +
                list(self.depth_actor_shared.parameters()) +
                list(self.combination_mlp_shared.parameters()) +
                list(self.output_mlp_shared.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, self.alg_cfg.get("max_grad_norm", 1.0))
            self.ghn_optimizer.step()

            # Convert to float for logging
            depth_actor_loss = depth_actor_loss.item()
            yaw_loss = yaw_loss.item()

            stop = time.time()
            learn_time = stop - start

            # DEBUG: Log rewards and losses every 100 iterations (same as learn_vision)
            if it % 100 == 0 and debug_log_file:
                mean_rew = np.mean(rewbuffer) if len(rewbuffer) > 0 else 0
                mean_len = np.mean(lenbuffer) if len(lenbuffer) > 0 else 0
                debug_msg = (
                    f"Rewards: mean={mean_rew:.4f}, ep_len={mean_len:.1f}\n"
                    f"Losses: depth_actor={depth_actor_loss:.6f}, yaw={yaw_loss:.6f}\n"
                )
                if ep_infos:
                    for key in ep_infos[0]:
                        if 'rew' in key:
                            values = [ep_info[key].item() if hasattr(ep_info[key], 'item') else ep_info[key] for ep_info in ep_infos]
                            debug_msg += f"  {key}: {np.mean(values):.4f}\n"
                debug_log_file.write(debug_msg)
                debug_log_file.flush()

            # Logging (same format as learn_vision)
            if self.log_dir is not None:
                self.log_vision_ghn(locals(), total_archs_sampled=total_archs_sampled)

            # Save checkpoint
            if it % self.save_interval == 0:
                self.save_ghn(os.path.join(self.log_dir, 'ghn_model_{}.pt'.format(it)))

            ep_infos.clear()
            self.current_learning_iteration = it + 1

        # Close debug log file
        if debug_log_file:
            debug_log_file.write("\n" + "=" * 80 + "\nTraining completed.\n")
            debug_log_file.close()
            print(f"Debug log saved to: {debug_log_path}")

    def log_vision_ghn(self, locs, width=80, pad=35, total_archs_sampled=0):
        """Logging for GHN training with DAgger (same format as log_vision)."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        wandb_dict = {}

        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # Logging (same keys as log_vision for consistency)
        depth_actor_loss = locs['depth_actor_loss']
        yaw_loss = locs['yaw_loss']
        depth_encoder_loss = locs.get('depth_encoder_loss', 0)
        delta_yaw_ok_percentage = locs.get('delta_yaw_ok_percentage', 1.0)

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = delta_yaw_ok_percentage
        wandb_dict['Loss_depth/depth_encoder'] = depth_encoder_loss
        wandb_dict['Loss_depth/depth_actor'] = depth_actor_loss
        wandb_dict['Loss_depth/yaw'] = yaw_loss
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        wandb_dict['GHN/total_archs_sampled'] = total_archs_sampled

        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

        wandb.log(wandb_dict, step=locs['it'])

        # Console output (same format as log_vision)
        str_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} (GHN + DAgger) \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {depth_encoder_loss:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {depth_actor_loss:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {yaw_loss:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {delta_yaw_ok_percentage:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {depth_encoder_loss:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {depth_actor_loss:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {yaw_loss:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {delta_yaw_ok_percentage:.4f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it) if curr_it > 0 else 0
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{f'Num architectures:':>{pad}} {self.num_parallel_archs}\n"""
                       f"""{f'Total archs sampled:':>{pad}} {total_archs_sampled}\n"""
                       f"""{f'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{f'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{f'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{f'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save_ghn(self, path):
        """Save GHN training checkpoint with compatible format."""
        torch.save({
            # GHN-specific - save entire model for easy loading
            'trainable_ghn': self.trainable_ghn,  # Entire model (architecture + weights)
            'ghn_state_dict': self.trainable_ghn.ghn.state_dict(),  # Keep for backwards compat
            'depth_actor_state_dict': self.depth_actor_shared.state_dict(),
            'combination_mlp_state_dict': self.combination_mlp_shared.state_dict(),
            'output_mlp_state_dict': self.output_mlp_shared.state_dict(),
            'ghn_optimizer_state_dict': self.ghn_optimizer.state_dict(),
            'arch_configs': [cfg.to_dict() for cfg in self.arch_configs],
            'iter': self.current_learning_iteration,
            # Standard format (for load() compatibility)
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'infos': None,
        }, path)
        print(f"Saved GHN checkpoint to {path}")

    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = ['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        # Save depth encoder/actor (skip in GHN mode - those use save_ghn instead)
        if self.if_depth and self.num_parallel_archs <= 1:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])

        # Handle GHN mode vs standard depth encoder mode
        if self.if_depth and self.num_parallel_archs > 1:
            # GHN mode: load GHN weights if present, otherwise copy teacher's actor
            if 'ghn_state_dict' in loaded_dict:
                print("GHN mode: Loading GHN and shared components from checkpoint...")
                self.trainable_ghn.ghn.load_state_dict(loaded_dict['ghn_state_dict'])
                self.depth_actor_shared.load_state_dict(loaded_dict['depth_actor_state_dict'])
                # Load shared encoder MLPs if present
                if 'combination_mlp_state_dict' in loaded_dict:
                    self.combination_mlp_shared.load_state_dict(loaded_dict['combination_mlp_state_dict'])
                    self.output_mlp_shared.load_state_dict(loaded_dict['output_mlp_state_dict'])
                if load_optimizer and 'ghn_optimizer_state_dict' in loaded_dict:
                    self.ghn_optimizer.load_state_dict(loaded_dict['ghn_optimizer_state_dict'])
            else:
                print("GHN mode: No GHN weights found, copying teacher actor to depth_actor_shared...")
                self.depth_actor_shared.load_state_dict(self.alg.actor_critic.actor.state_dict())
        elif self.if_depth:
            # Standard mode: load depth_encoder and depth_actor
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
