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

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Check if this is a HyperPPO task
    is_hyperppo = train_cfg.runner.policy_class_name == "HyperPPOActorCritic"
    if is_hyperppo:
        print(f"\n{'='*60}")
        print(f"HyperPPO Task Detected: {args.task}")
        print(f"{'='*60}")

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 3  # Reduce to 3x3 grid for closer spacing
    env_cfg.terrain.num_cols = 3

    # Force smaller terrain spacing to bring robots closer together
    if hasattr(env_cfg.terrain, 'terrain_length'):
        env_cfg.terrain.terrain_length = 2.0  # Reduce from default (usually 8m) to 2m
        env_cfg.terrain.terrain_width = 2.0   # Reduce from default (usually 8m) to 2m

    # Also try setting environment spacing directly
    if hasattr(env_cfg, 'env_spacing'):
        env_cfg.env_spacing = 2.0  # Set environment spacing to 2m
    if hasattr(env_cfg.env, 'env_spacing'):
        env_cfg.env.env_spacing = 2.0  # Alternative location for env spacing
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Increase camera range and FOV to see other robots better
    if hasattr(env_cfg, 'depth'):
        env_cfg.depth.far_clip = 15.0  # Increase from 3.0m to 15.0m
        env_cfg.depth.horizontal_fov = 120  # Increase from 87° to 120° (wider view)
        env_cfg.depth.angle = [-0.3, 0]  # Tilt camera downward to see other robots on ground

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Enable camera display for depth_go2 task in play mode
    if "depth" in args.task and hasattr(env, 'enable_camera_display'):
        # Print environment positions for debugging
        print(f"Number of environments: {env.num_envs}")
        print(f"Terrain size: {env_cfg.terrain.num_rows}x{env_cfg.terrain.num_cols}")
        print(f"Camera FOV: {env_cfg.depth.horizontal_fov} degrees")
        print(f"Camera range: {env_cfg.depth.near_clip}m - {env_cfg.depth.far_clip}m")
        if hasattr(env.cfg.terrain, 'terrain_length'):
            print(f"Terrain spacing: {env.cfg.terrain.terrain_length}m x {env.cfg.terrain.terrain_width}m")

        # Print actual robot positions for debugging
        print("Robot base positions:")
        for i in range(min(5, env.num_envs)):  # Show first 5 robots
            pos = env.root_states[i, :3].cpu().numpy()
            print(f"  Robot {i}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

        env.enable_camera_display(show_all=True)  # Show all 10 cameras simultaneously
        print("Live camera feed enabled for play mode!")

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # HyperPPO-specific architecture handling
    if is_hyperppo:
        print("\n=== HyperPPO Architecture Selection ===")

        # Load architecture configuration (relative to repo root)
        architecture_config_path = "rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth84.json"

        print(f"Loading architectures from: {architecture_config_path}")
        arch_config = load_architecture_config(architecture_config_path)
        architectures = arch_config['architectures']
        metadata = arch_config['metadata']

        print(f"Available architectures: {len(architectures)}")

        # Architecture selection logic
        if hasattr(args, 'arch_idx') and args.arch_idx is not None:
            # Use specific architecture index
            selected_arch = get_architecture_by_index(architectures, args.arch_idx)
            arch_idx = args.arch_idx
            print(f"Using specified architecture: {arch_idx}")
        else:
            # Sample random architecture
            selected_arch = sample_random_architecture(architectures)
            arch_idx = architectures.index(selected_arch)
            print(f"Randomly sampled architecture: {arch_idx}")

        print(f"\nArchitecture Details:")
        print(f"  ID: {arch_idx}")
        print(f"  Name: {selected_arch.get('name', 'N/A')}")
        print(f"  CNN layers: {len(selected_arch['cnn_config'])}")
        print(f"  MLP layers: {len(selected_arch['mlp_config'])}")
        if 'total_params' in selected_arch:
            print(f"  Total parameters: ~{selected_arch['total_params']}")
        print()

        # Set the selected architecture in the HyperActor
        if hasattr(ppo_runner.alg.actor_critic, 'hyper_actor'):
            hyper_actor = ppo_runner.alg.actor_critic.hyper_actor
            print("Setting architecture for HyperPPO policy...")

            # Use set_specific_architecture method to set the architecture
            hyper_actor.set_specific_architecture(arch_idx)
            print(f"✓ Architecture {arch_idx} successfully loaded for deployment!\n")
        else:
            print("Warning: Could not find HyperPPO architecture interface in policy")

    policy = ppo_runner.get_inference_policy(device=env.device)

    # Load estimator
    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        if is_hyperppo:
            # For HyperPPO: Extract the selected architecture with GHN-generated weights
            # and export as standalone JIT model (detached from GHN)
            print(f"\n=== Exporting Architecture {arch_idx} as Standalone Policy ===")

            hyper_actor = ppo_runner.alg.actor_critic.hyper_actor

            # CRITICAL: Verify we're exporting the correct architecture
            if hyper_actor.sampled_indices is None or len(hyper_actor.sampled_indices) == 0:
                raise RuntimeError("No architecture is currently loaded in HyperActor!")

            current_arch_id = hyper_actor.sampled_indices[0]
            if current_arch_id != arch_idx:
                raise RuntimeError(
                    f"Architecture mismatch! Selected arch {arch_idx} but HyperActor has arch {current_arch_id} loaded. "
                    f"This means the architecture was changed after selection."
                )

            print(f'✓ Verified: HyperActor is using architecture {current_arch_id} (matches selected {arch_idx})')

            # Get the current model with GHN-generated weights
            current_model = hyper_actor.current_model[0]

            # Extract state_dict (this gets weight values without computational graph)
            model_state_dict = {k: v.detach().cpu().clone() for k, v in current_model.state_dict().items()}

            # Export the standalone architecture
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            os.makedirs(path, exist_ok=True)

            # Save checkpoint with architecture and weights (overwrite each time)
            model_path = os.path.join(path, 'sampled_architecture.pt')
            torch.save({
                'architecture_id': arch_idx,
                'architecture_config': selected_arch,
                'model_state_dict': model_state_dict,
                'log_std': hyper_actor.log_std[arch_idx].detach().cpu().clone(),
                'metadata': metadata,
            }, model_path)
            print(f'✓ Exported standalone model to: {model_path}')

            # Verify the saved checkpoint by loading it back
            print(f'\n=== Verifying Saved Checkpoint ===')
            saved_checkpoint = torch.load(model_path)

            print(f'Checkpoint keys: {list(saved_checkpoint.keys())}')
            print(f'\nArchitecture ID: {saved_checkpoint["architecture_id"]}')

            print(f'\nArchitecture Config:')
            print(f'  CNN layers: {len(saved_checkpoint["architecture_config"]["cnn_config"])}')
            for i, layer in enumerate(saved_checkpoint["architecture_config"]["cnn_config"]):
                print(f'    Layer {i}: {layer["channels"]} channels, {layer["kernel"]}x{layer["kernel"]} kernel, stride={layer["stride"]}')
            print(f'  State MLP: {saved_checkpoint["architecture_config"]["state_mlp_config"]}')
            print(f'  CNN MLP: {saved_checkpoint["architecture_config"]["cnn_mlp_config"]}')
            print(f'  MLP layers: {saved_checkpoint["architecture_config"]["mlp_config"]}')

            print(f'\nModel State Dict:')
            print(f'  Total parameters: {len(saved_checkpoint["model_state_dict"])}')
            print(f'  Parameter names:')
            for name, param in saved_checkpoint["model_state_dict"].items():
                print(f'    {name}: {list(param.shape)}')

            print(f'\nLog Std shape: {saved_checkpoint["log_std"].shape}')
            print(f'Log Std values: {saved_checkpoint["log_std"].squeeze().numpy()}')

            print(f'\nMetadata:')
            for key, value in saved_checkpoint["metadata"].items():
                if key != 'ghn_config' and key != 'arch_descriptor_config':
                    print(f'  {key}: {value}')

            print(f'\n✓ Checkpoint verification complete!')
        else:
            # Regular policy export for non-HyperPPO
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    # Override command resampling to use fixed forward command
    def fixed_resample_commands(self, env_ids):
        """Fixed forward walking command"""
        self.commands[env_ids, 0] = 0.6  # lin_vel_x = 0.6 m/s forward
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=env.device) * 0.2 - 0.1  # lin_vel_y = small random [-0.1, 0.1]
        self.commands[env_ids, 2] = 0.0  # ang_vel_yaw = 0.0 rad/s

    env._resample_commands = fixed_resample_commands.__get__(env, env.__class__)

    # Set initial forward command
    env.commands[:, 0] = 0.6  # Forward 0.6 m/s
    env.commands[:, 1] = torch.rand(env.num_envs, device=env.device) * 0.2 - 0.1  # Small random sideways [-0.1, 0.1]
    env.commands[:, 2] = 0.0  # No turning

    print("Starting simulation loop...")
    print("Press Ctrl+C to stop...\n")

    for i in range(10*int(env.max_episode_length)):
        # For VisualActorCritic, pass single depth frame
        if train_cfg.runner.policy_class_name == "VisualActorCritic":
            depth_images = env.depth_image if hasattr(env, 'depth_image') else None
            actions = policy(obs.detach(), depth_images=depth_images)
        else:
            # For other policies (including HyperPPO), use the original approach
            actions = policy(obs.detach())

        obs, _, rews, dones, infos = env.step(actions.detach())

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)

    # Cleanup camera display
    if "depth" in args.task and hasattr(env, 'disable_camera_display'):
        env.disable_camera_display()
        print("Camera display closed.")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # Use the standard get_args() from legged_gym
    args = get_args()

    # Add HyperPPO-specific argument if provided via command line
    # You can pass --arch_idx as an extra argument
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == '--arch_idx' and i + 1 < len(sys.argv):
            args.arch_idx = int(sys.argv[i + 1])
            break
    else:
        args.arch_idx = None  # Default to random sampling

    play(args)
