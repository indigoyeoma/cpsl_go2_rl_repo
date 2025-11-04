import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
for path in (REPO_ROOT, PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

ISAAC_GYM_PYTHON = os.path.join(REPO_ROOT, "isaacgym", "python")
if ISAAC_GYM_PYTHON not in sys.path:
    sys.path.insert(0, ISAAC_GYM_PYTHON)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry
import torch
from legged_gym.utils import webviewer


def play(args):
    web = None
    if args.web:
        web = webviewer.WebViewer()

    depth_tasks = {"depth_go2", "depth_obsavoid_go2"}
    if args.task in depth_tasks:
        raise RuntimeError(
            "Depth-based tasks are handled by play_depth.py. "
            "Run `python legged_gym/scripts/play_depth.py --task=depth_obsavoid_go2` instead."
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    target_envs = getattr(args, "num_envs", None)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, target_envs or 1)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if web is not None:
        web.setup(env)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = getattr(args, "load_run", -1) if getattr(args, "load_run", None) is not None else -1
    train_cfg.runner.checkpoint = getattr(args, "checkpoint", -1) if getattr(args, "checkpoint", None) is not None else -1
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    export_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
    export_policy_as_jit(ppo_runner.alg.actor_critic, export_dir)
    print("=" * 60)
    print("Exported policy as jit script to:", export_dir)
    print("=" * 60)

    total_steps = 10 * int(env.max_episode_length)
    for _ in range(total_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        if web is not None:
            web.render(
                fetch_results=True,
                step_graphics=True,
                render_all_camera_sensors=True,
                wait_for_page_load=True,
            )


if __name__ == "__main__":
    args = get_args()
    play(args)
