import os
import math
import copy
from types import MethodType

os.environ.setdefault('MESA_VK_DEVICE_SELECT', '10de:2231')
os.environ.setdefault("CUDA_VISIBLE_DEVICES", '0')

import cv2
import numpy as np

import isaacgym  # noqa: F401  (ensure torch loads after isaacgym)
from isaacgym import gymapi
from isaacgym import gymtorch

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 (registers tasks)
from legged_gym.utils import get_args, task_registry


class VisualPolicyWrapper(torch.nn.Module):
    """Wrap state/depth encoders with actor for TorchScript deployment."""

    def __init__(self, actor_critic):
        super().__init__()
        self.base_actor_obs_size = int(actor_critic.base_actor_obs_size)
        self.depth_image_size = int(actor_critic.depth_image_size)
        side = int(math.sqrt(self.depth_image_size))
        if side * side != self.depth_image_size:
            raise ValueError(f"Unsupported depth image size: {self.depth_image_size}")
        self.depth_image_shape = (side, side)
        self.state_encoder = copy.deepcopy(actor_critic.state_encoder).to('cpu')
        self.depth_encoder = copy.deepcopy(actor_critic.depth_encoder).to('cpu')
        self.actor = copy.deepcopy(actor_critic.actor).to('cpu')

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        base_obs = obs[:, :self.base_actor_obs_size]
        depth_flat = obs[:, self.base_actor_obs_size:self.base_actor_obs_size + self.depth_image_size]
        depth_imgs = depth_flat.view(-1, self.depth_image_shape[0], self.depth_image_shape[1])
        depth_dict = {"depth": depth_imgs.unsqueeze(-1)}
        depth_latent = self.depth_encoder(depth_dict)
        state_latent = self.state_encoder(base_obs)
        features = torch.cat([state_latent, depth_latent], dim=-1)
        return self.actor(features)


def export_visual_policy(actor_critic, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    wrapper = VisualPolicyWrapper(actor_critic)
    example = torch.zeros(1, wrapper.base_actor_obs_size + wrapper.depth_image_size)
    scripted = torch.jit.trace(wrapper, example)
    export_path = os.path.join(export_dir, 'policy_depth.pt')
    scripted.save(export_path)
    print('=' * 60)
    print('Exported visual policy as TorchScript to:', export_path)
    print('=' * 60)


def _show_rgb_and_depth(env, env_id=0, win_rgb='RGB', win_depth='Depth'):
    """Render sensors and show live RGB and Depth for a single environment."""
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)

    rgb_tensor = env.gym.get_camera_image_gpu_tensor(
        env.sim, env.envs[env_id], env.cam_handles[env_id], gymapi.IMAGE_COLOR
    )
    depth_tensor = env.gym.get_camera_image_gpu_tensor(
        env.sim, env.envs[env_id], env.cam_handles[env_id], gymapi.IMAGE_DEPTH
    )

    env.gym.end_access_image_tensors(env.sim)

    if rgb_tensor is not None:
        rgb = gymtorch.wrap_tensor(rgb_tensor)
        h, w = env.cfg.depth.original[1], env.cfg.depth.original[0]
        rgb_np = rgb.view(h, w, -1)[:, :, :3].cpu().numpy()
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_rgb, rgb_bgr)

    if depth_tensor is not None:
        depth = gymtorch.wrap_tensor(depth_tensor)
        depth_m = (-depth).clamp(min=env.cfg.depth.near_clip, max=env.cfg.depth.far_clip)
        depth_norm = (depth_m - env.cfg.depth.near_clip) / (env.cfg.depth.far_clip - env.cfg.depth.near_clip)
        depth_img = (depth_norm.cpu().numpy() * 255).astype(np.uint8)
        h, w = env.cfg.depth.original[1], env.cfg.depth.original[0]
        depth_img = depth_img.reshape(h, w)
        depth_color = cv2.applyColorMap(255 - depth_img, cv2.COLORMAP_TURBO)
        cv2.imshow(win_depth, depth_color)


def play_depth(args):
    export_policy = True
    user_fixed_speed = getattr(args, "fixed_speed", None)
    play_steps = getattr(args, "play_steps", None)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    target_envs = args.num_envs if args.num_envs is not None else 1
    target_envs = max(1, target_envs)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, target_envs)

    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.0
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    if hasattr(env_cfg, 'depth'):
        env_cfg.depth.use_camera = True
        env_cfg.depth.original = (84, 84)
        env_cfg.depth.resized = (84, 84)
        env_cfg.depth.update_interval = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    auto_fixed_speed = None
    if user_fixed_speed is not None:
        auto_fixed_speed = user_fixed_speed
    elif hasattr(env, "speed_targets"):
        auto_fixed_speed = 0.6
        print(f"[play_depth] No --fixed_speed provided; defaulting to {auto_fixed_speed:.2f} m/s.")
    elif hasattr(env, "commands"):
        auto_fixed_speed = 0.6
        print(f"[play_depth] No --fixed_speed provided; defaulting to legacy command speed {auto_fixed_speed:.2f} m/s.")

    if auto_fixed_speed is not None:
        if hasattr(env, "speed_targets"):
            original_resample = env._resample_commands

            def fixed_speed_resample(self, env_ids):
                original_resample(env_ids)
                if len(env_ids) == 0:
                    return
                env_ids = env_ids.to(dtype=torch.long, device=self.device)
                self.speed_targets[env_ids] = auto_fixed_speed

            env._resample_commands = MethodType(fixed_speed_resample, env)
            all_envs = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
            env._resample_commands(all_envs)
            env.goal_metrics = None
            print(f"[play_depth] Fixed speed target set to {auto_fixed_speed:.3f} m/s for all environments.")
        elif hasattr(env, "commands"):
            original_resample = env._resample_commands

            def fixed_command_resample(self, env_ids):
                original_resample(env_ids)
                if len(env_ids) == 0:
                    return
                env_ids = env_ids.to(dtype=torch.long, device=self.device)
                self.commands[env_ids, 0] = auto_fixed_speed
                if self.commands.shape[1] > 1:
                    self.commands[env_ids, 1:] = 0.0

            env._resample_commands = MethodType(fixed_command_resample, env)
            all_envs = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
            env._resample_commands(all_envs)
            print(f"[play_depth] Legacy command interface: fixed forward command {auto_fixed_speed:.3f} m/s applied.")

    if hasattr(env.cfg, "commands"):
        env.cfg.commands.resampling_time = 15.0

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = -1
    train_cfg.runner.checkpoint = -1
    setattr(args, "load_run", None)
    setattr(args, "checkpoint", None)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if export_policy:
        export_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_visual_policy(ppo_runner.alg.actor_critic, export_root)

    obs = env.get_observations()
    total_steps = play_steps if play_steps is not None else 10 * int(env.max_episode_length)

    show_camera = not getattr(args, "headless", False)
    if show_camera:
        print("Press 'q' in the depth viewer to exit.")

    for _ in range(total_steps):
        if show_camera and hasattr(env, 'cam_handles') and len(getattr(env, 'cam_handles', [])) > 0:
            _show_rgb_and_depth(env, env_id=0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

    if show_camera:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'depth_obsavoid_go2'

    print(f"Playing depth task: {args.task}")
    print("Policy export: ENABLED (auto)")
    play_depth(args)
