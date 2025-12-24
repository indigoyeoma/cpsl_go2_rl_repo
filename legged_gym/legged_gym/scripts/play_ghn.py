# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play script for GHN - tests trained GHN by predicting weights for random architectures
# Each environment gets its own unique architecture sampled and initialized by GHN.
# Architecture: backbone → visual features → combination_mlp → output_mlp → (latent + yaw)

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
from legged_gym.utils import webviewer


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()

    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # === GHN Play Settings ===
    # Each env gets its own unique architecture
    num_envs = args.num_envs if hasattr(args, 'num_envs') and args.num_envs is not None else 8

    # Override environment settings
    env_cfg.env.num_envs = num_envs
    env_cfg.depth.camera_num_envs = num_envs  # All envs have cameras
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0.,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.,
        "rough stairs down": 0.,
        "discrete": 0.,
        "stepping stones": 0.0,
        "gaps": 0.,
        "smooth flat": 0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.,
        "large stairs up": 0.,
        "large stairs down": 0.,
        "parkour": 0.0,
        "parkour_hurdle": 0.3,
        "parkour_flat": 0.4,
        "parkour_step": 0.3,
        "parkour_gap": 0.0,
        "demo": 0.0
    }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    env_cfg.terrain.easy_difficulty = True

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # Prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Update num_envs to actual value (in case it was overridden)
    num_envs = env.num_envs

    if args.web:
        web_viewer.setup(env)

    # === Create runner to get actor structure (don't auto-load checkpoint) ===
    print(f"\nCreating runner for actor structure...")
    train_cfg.runner.resume = False  # Don't auto-load - we manually load GHN checkpoint below
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args,
        train_cfg=train_cfg
    )

    # Get teacher's actor as template for depth_actor
    depth_actor = deepcopy(ppo_runner.alg.actor_critic.actor)

    # === Load GHN checkpoint ===
    print(f"\nLooking for GHN checkpoints in: {log_pth}")

    # Find latest GHN checkpoint
    if args.checkpoint == -1:
        ghn_models = [f for f in os.listdir(log_pth) if f.startswith("ghn_model_") and f.endswith(".pt")]
        if ghn_models:
            ghn_models.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ghn_checkpoint_path = os.path.join(log_pth, ghn_models[-1])
        else:
            # Try regular model checkpoints
            models = [f for f in os.listdir(log_pth) if f.startswith("model_") and f.endswith(".pt")]
            if models:
                models.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                ghn_checkpoint_path = os.path.join(log_pth, models[-1])
            else:
                raise FileNotFoundError(f"No checkpoints found in {log_pth}")
    else:
        ghn_checkpoint_path = os.path.join(log_pth, f"ghn_model_{args.checkpoint}.pt")
        if not os.path.exists(ghn_checkpoint_path):
            ghn_checkpoint_path = os.path.join(log_pth, f"model_{args.checkpoint}.pt")

    print(f"Loading GHN checkpoint: {ghn_checkpoint_path}")
    checkpoint = torch.load(ghn_checkpoint_path, map_location=env.device)

    # === Setup GHN and architectures ===
    from rsl_rl.modules.depth_encoder_ghn import (
        TrainableGHN, DepthEncoderConfig, build_depth_backbone, sample_depth_encoder_configs
    )

    # Load entire GHN model (includes architecture + weights)
    trainable_ghn = checkpoint['trainable_ghn']
    trainable_ghn.to(env.device)
    trainable_ghn.eval()
    print("Loaded GHN model from checkpoint")

    # Create shared encoder layers (same as training)
    n_proprio = env.cfg.env.n_proprio
    activation = nn.ELU()
    last_activation = nn.Tanh()

    # Proprio fusion: [32 visual + n_proprio] → [32 fused]
    combination_mlp = nn.Sequential(
        nn.Linear(32 + n_proprio, 128),
        activation,
        nn.Linear(128, 32)
    ).to(env.device)

    # Output: [32 fused] → [34] (32 latent + 2 yaw)
    output_mlp = nn.Sequential(
        nn.Linear(32, 32 + 2),
        last_activation
    ).to(env.device)

    # Load shared encoder weights from checkpoint
    if 'combination_mlp_state_dict' in checkpoint:
        combination_mlp.load_state_dict(checkpoint['combination_mlp_state_dict'])
        output_mlp.load_state_dict(checkpoint['output_mlp_state_dict'])
        print("Loaded combination_mlp and output_mlp weights from checkpoint")
    else:
        print("WARNING: No combination_mlp weights in checkpoint, using random initialization")

    combination_mlp.eval()
    output_mlp.eval()

    # Sample one unique architecture per environment
    print(f"\n=== Sampling {num_envs} unique architectures (one per env) ===")
    arch_configs = sample_depth_encoder_configs(num_envs)
    for i, cfg in enumerate(arch_configs):
        print(f"  Env [{i}]: {cfg}")

    # Build raw backbones and predict weights with GHN
    print("\nPredicting weights with GHN...")
    backbones = [build_depth_backbone(cfg) for cfg in arch_configs]
    backbones = trainable_ghn.predict_weights(backbones)

    # Set to eval mode
    for backbone in backbones:
        backbone.eval()

    # Load depth_actor weights from checkpoint
    if 'depth_actor_state_dict' in checkpoint:
        depth_actor.load_state_dict(checkpoint['depth_actor_state_dict'])
        print("Loaded depth actor weights from checkpoint")
    else:
        print("WARNING: No depth_actor weights in checkpoint")

    depth_actor.eval()

    print(f"\n=== Starting GHN Play: {num_envs} envs, each with unique architecture ===\n")

    # Track rewards per environment (each env has its own architecture)
    env_rewards = [deque(maxlen=100) for _ in range(num_envs)]

    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1] if env.cfg.depth.use_camera else None
    infos["delta_yaw_ok"] = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for i in range(10 * int(env.max_episode_length)):
        # Process depth through environment-specific backbones (one backbone per env)
        if env.cfg.depth.use_camera and infos["depth"] is not None:
            # Step 1: Each env's backbone processes its depth → [32] visual features
            depth_visual = torch.zeros(num_envs, 32, device=env.device)

            for env_idx, backbone in enumerate(backbones):
                depth_single = infos["depth"][env_idx:env_idx+1].clone()

                with torch.no_grad():
                    depth_visual[env_idx:env_idx+1] = backbone(depth_single)

            # Step 2: Get proprioception (zero out yaw for prediction)
            obs_prop_depth = obs[:, :n_proprio].clone()
            obs_prop_depth[:, 6:8] = 0

            # Step 3: Fuse visual + proprio → [32]
            with torch.no_grad():
                depth_fused = combination_mlp(torch.cat([depth_visual, obs_prop_depth], dim=-1))

                # Step 4: Output layer → [34] (32 latent + 2 yaw)
                depth_latent_and_yaw = output_mlp(depth_fused)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = 1.5 * depth_latent_and_yaw[:, -2:]

            # Inject predicted yaw into obs
            obs_student = obs.clone()
            obs_student[infos["delta_yaw_ok"], 6:8] = yaw[infos["delta_yaw_ok"]]
        else:
            depth_latent = None
            obs_student = obs

        # Get actions from depth actor
        with torch.no_grad():
            actions = depth_actor(obs_student.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach())

        if args.web:
            web_viewer.render(fetch_results=True,
                              step_graphics=True,
                              render_all_camera_sensors=True,
                              wait_for_page_load=True)

        # Track rewards per environment
        for env_idx in range(num_envs):
            env_rewards[env_idx].append(rews[env_idx].cpu().item())

        # Print status
        if i % 50 == 0:
            print(f"\n=== Step {i} ===")
            for env_idx in range(num_envs):
                if len(env_rewards[env_idx]) > 0:
                    mean_rew = statistics.mean(env_rewards[env_idx])
                    cfg = arch_configs[env_idx]
                    print(f"  Env [{env_idx}]: mean_reward = {mean_rew:.3f}, arch = layers={cfg.num_layers}, ch={cfg.channels}")

        id = env.lookat_id
        print(f"Robot {id}: time={env.episode_length_buf[id].item()/50:.1f}s, "
              f"vx_cmd={env.commands[id, 0].item():.2f}, "
              f"vx_actual={env.base_lin_vel[id, 0].item():.2f}, "
              f"arch_layers={arch_configs[id].num_layers}")


if __name__ == '__main__':
    args = get_args()
    play(args)
