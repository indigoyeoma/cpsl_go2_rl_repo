#!/usr/bin/env python3

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

os.environ['MESA_VK_DEVICE_SELECT'] = '10de:2231w'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
import json
import random
import argparse
from legged_gym.utils import webviewer

def load_architecture_config(config_path):
    """Load architecture configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def sample_random_architecture(architectures):
    """Sample a random architecture from the pool"""
    return random.choice(architectures)

def get_architecture_by_index(architectures, arch_idx):
    """Get specific architecture by index"""
    if 0 <= arch_idx < len(architectures):
        return architectures[arch_idx]
    else:
        raise ValueError(f"Architecture index {arch_idx} out of range [0, {len(architectures)-1}]")

def hyper_play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Check if this is a HyperPPO task
    is_hyperppo = train_cfg.runner.policy_class_name == "HyperPPOActorCritic"
    if not is_hyperppo:
        print(f"Warning: Task '{args.task}' is not configured for HyperPPO. Policy class: {train_cfg.runner.policy_class_name}")
        print("Consider using regular play.py for non-HyperPPO tasks.")
    
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    
    # Force smaller terrain spacing
    if hasattr(env_cfg.terrain, 'terrain_length'):
        env_cfg.terrain.terrain_length = 2.0
        env_cfg.terrain.terrain_width = 2.0
    
    if hasattr(env_cfg, 'env_spacing'):
        env_cfg.env_spacing = 2.0
    if hasattr(env_cfg.env, 'env_spacing'):
        env_cfg.env.env_spacing = 2.0
        
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Increase camera range and FOV for better visualization
    if hasattr(env_cfg, 'depth'):
        env_cfg.depth.far_clip = 15.0
        env_cfg.depth.horizontal_fov = 120
        env_cfg.depth.angle = [-0.3, 0]
    
    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Enable camera display for depth tasks
    if "depth" in args.task or "visual" in args.task:
        if hasattr(env, 'enable_camera_display'):
            print(f"Number of environments: {env.num_envs}")
            print(f"Terrain size: {env_cfg.terrain.num_rows}x{env_cfg.terrain.num_cols}")
            if hasattr(env_cfg, 'depth'):
                print(f"Camera FOV: {env_cfg.depth.horizontal_fov} degrees")
                print(f"Camera range: {env_cfg.depth.near_clip}m - {env_cfg.depth.far_clip}m")
            
            env.enable_camera_display(show_all=True)
            print("Live camera feed enabled!")

    if args.web:
        web_viewer.setup(env)
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # HyperPPO-specific architecture handling
    if is_hyperppo:
        print("\n=== HyperPPO Architecture Selection ===")
        
        # Load architecture configuration
        architecture_config_path = train_cfg.policy.architecture_config_path
        if not os.path.isabs(architecture_config_path):
            # Make path relative to project root
            architecture_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "..", architecture_config_path)
        
        print(f"Loading architectures from: {architecture_config_path}")
        arch_config = load_architecture_config(architecture_config_path)
        architectures = arch_config['architectures']
        metadata = arch_config['metadata']
        
        print(f"Available architectures: {len(architectures)}")
        print(f"Architecture metadata: {metadata}")
        
        # Architecture selection logic
        if args.arch_idx is not None:
            # Use specific architecture index
            selected_arch = get_architecture_by_index(architectures, args.arch_idx)
            print(f"Selected architecture by index: {args.arch_idx}")
        else:
            # Sample random architecture
            selected_arch = sample_random_architecture(architectures)
            arch_idx = architectures.index(selected_arch)
            print(f"Randomly sampled architecture: {arch_idx}")
        
        print(f"Architecture details:")
        print(f"  CNN layers: {len(selected_arch['cnn_config'])}")
        print(f"  MLP layers: {len(selected_arch['mlp_config'])}")
        print(f"  Total parameters: ~{selected_arch.get('total_params', 'Unknown')}")
        
        # Set the selected architecture in the policy
        policy = ppo_runner.get_inference_policy(device=env.device)
        
        # For HyperPPO, we need to set the specific architecture
        if hasattr(policy, 'actor_critic') and hasattr(policy.actor_critic, 'hyper_actor'):
            print("Setting architecture for HyperPPO policy...")
            
            # Create a single-architecture list for deployment
            deployment_archs = [selected_arch]
            
            # Temporarily override the architecture pool with our selected one
            original_archs = policy.actor_critic.hyper_actor.all_models.copy()
            policy.actor_critic.hyper_actor.all_models = deployment_archs
            policy.actor_critic.hyper_actor.meta_batch_size = 1
            
            # Sample the architecture (will select our only option)
            policy.actor_critic.hyper_actor.change_graph(repeat_sample=False)
            print("Architecture successfully loaded for deployment!")
            
        else:
            print("Warning: Could not find HyperPPO architecture interface in policy")
    
    else:
        # Regular policy loading for non-HyperPPO
        policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Export policy if requested
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    
    # Override command resampling for consistent forward motion
    def fixed_resample_commands(self, env_ids):
        """Fixed forward walking command"""
        self.commands[env_ids, 0] = args.forward_speed  # Use configurable forward speed
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=env.device) * 0.2 - 0.1
        self.commands[env_ids, 2] = 0.0
        
    env._resample_commands = fixed_resample_commands.__get__(env, env.__class__)
    
    # Set initial forward command
    env.commands[:, 0] = args.forward_speed
    env.commands[:, 1] = torch.rand(env.num_envs, device=env.device) * 0.2 - 0.1
    env.commands[:, 2] = 0.0

    print(f"\n=== Starting deployment with forward speed: {args.forward_speed} m/s ===")
    print("Press Ctrl+C to stop...")
    
    try:
        for i in range(10 * int(env.max_episode_length)):
            # Handle different policy types
            if train_cfg.runner.policy_class_name == "VisualActorCritic":
                depth_images = env.depth_image if hasattr(env, 'depth_image') else None
                actions = policy(obs.detach(), depth_images=depth_images)
            elif train_cfg.runner.policy_class_name == "HyperPPOActorCritic":
                # HyperPPO policy with architecture-specific inference
                actions = policy(obs.detach())
            else:
                # Standard policy
                actions = policy(obs.detach())
                
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            if args.web:
                web_viewer.render(fetch_results=True,
                            step_graphics=True,
                            render_all_camera_sensors=True,
                            wait_for_page_load=True)
                            
    except KeyboardInterrupt:
        print("\nStopping deployment...")

    # Cleanup
    if "depth" in args.task or "visual" in args.task:
        if hasattr(env, 'disable_camera_display'):
            env.disable_camera_display()
            print("Camera display closed.")
    
    print("Deployment completed!")

def get_hyper_args():
    """Get command line arguments for HyperPPO deployment"""
    parser = argparse.ArgumentParser(description='HyperPPO Deployment Script')
    parser.add_argument('--task', type=str, default='hyper_go2', 
                       help='Task name (should be HyperPPO compatible)')
    parser.add_argument('--arch_idx', type=int, default=None,
                       help='Specific architecture index to use (if None, random sampling)')
    parser.add_argument('--forward_speed', type=float, default=0.6,
                       help='Forward walking speed in m/s')
    parser.add_argument('--web', action='store_true',
                       help='Enable web viewer')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for architecture sampling')
    parser.add_argument('--headless', action='store_true',
                       help='Run headless (no viewer)')
    parser.add_argument('--num_envs', type=int, default=10,
                       help='Number of environments')
    
    # Add standard legged_gym arguments
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--rl_device', type=str, default="cuda:0", help='RL Device in PyTorch-like syntax')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
    parser.add_argument('--horovod', action='store_true', default=False, help='Use horovod for multi-gpu training')
    parser.add_argument('--rl_train', action='store_true', default=False, help='Run RL training')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from a checkpoint')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment to run or load.')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run.')
    parser.add_argument('--load_run', type=str, default=None, help='Name of the run to load when resume=True.')
    parser.add_argument('--checkpoint', type=int, default=None, help='Saved model checkpoint number.')
    parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of training iterations.')
    
    return parser.parse_args()

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_hyper_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    hyper_play(args)