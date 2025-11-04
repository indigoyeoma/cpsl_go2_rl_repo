import os
import sys
import copy
import math
from datetime import datetime

import numpy as np

# Fix camera buffer creation issues before importing IsaacGym
#MESA_VK_DEVICE_SELECT=list vulkaninfo
#use this code to get the gpu
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:2231'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'  # Make all 4 GPUs visible

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import cv2


def _maybe_close_env(env):
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def _format_run_name(original: str, suffix: str) -> str:
    original = original or ""
    if not original:
        return suffix
    return f"{original}_{suffix}"


def _run_staged_obsavoid_training(args):
    """Two-stage training: flat-surface pretrain then obstacle finetune."""
    pretrain_steps = max(getattr(args, "pretrain_flat_steps", 0) or 0, 0)
    if pretrain_steps <= 0:
        return False

    # Clone registered configs so we can tweak them per stage.
    base_env_cfg, base_train_cfg = task_registry.get_cfgs(args.task)
    env_cfg_stage2 = copy.deepcopy(base_env_cfg)
    train_cfg_stage2 = copy.deepcopy(base_train_cfg)
    env_cfg_stage1 = copy.deepcopy(env_cfg_stage2)
    train_cfg_stage1 = copy.deepcopy(train_cfg_stage2)

    env_stage1 = None
    env_stage2 = None

    # Stage 1: flat terrain with wide command range to focus on gait stability.
    env_cfg_stage1.terrain.mesh_type = "plane"
    if hasattr(env_cfg_stage1.terrain, "curriculum"):
        env_cfg_stage1.terrain.curriculum = False
    if hasattr(env_cfg_stage1.terrain, "selected"):
        env_cfg_stage1.terrain.selected = False
    if hasattr(env_cfg_stage1, "obstacles"):
        env_cfg_stage1.obstacles.num_cubes = 0
    env_cfg_stage1.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg_stage1.commands.ranges.lin_vel_y = [-1.0, 1.0]

    num_steps_per_env = getattr(train_cfg_stage1.runner, "num_steps_per_env", 1) or 1
    pretrain_iters = max(1, math.ceil(pretrain_steps / num_steps_per_env))
    train_cfg_stage1.runner.max_iterations = pretrain_iters
    train_cfg_stage1.runner.experiment_name = _format_run_name(
        getattr(train_cfg_stage1.runner, "experiment_name", "depth_obsavoid_go2"), "flat_pretrain"
    )
    train_cfg_stage1.runner.run_name = _format_run_name(
        getattr(train_cfg_stage1.runner, "run_name", ""), "flat_pretrain"
    )

    original_max_iterations = getattr(args, "max_iterations", None)
    actor_state = None
    optimizer_state = None
    learning_rate = None
    completed_iterations = 0

    try:
        args.max_iterations = train_cfg_stage1.runner.max_iterations

        env_stage1, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg_stage1)
        runner_stage1, train_cfg_stage1 = task_registry.make_alg_runner(
            env=env_stage1, name=args.task, args=args, train_cfg=train_cfg_stage1
        )

        runner_stage1.learn(
            num_learning_iterations=train_cfg_stage1.runner.max_iterations,
            init_at_random_ep_len=True,
        )

        actor_state = copy.deepcopy(runner_stage1.alg.actor_critic.state_dict())
        optimizer_state = copy.deepcopy(runner_stage1.alg.optimizer.state_dict())
        learning_rate = runner_stage1.alg.learning_rate
        completed_iterations = runner_stage1.current_learning_iteration

        _maybe_close_env(env_stage1)
        env_stage1 = None

        # Stage 2: obstacle terrain with task-specific command range.
        if original_max_iterations is not None and original_max_iterations > 0:
            stage2_iters = max(original_max_iterations - completed_iterations, 1)
        else:
            stage2_iters = train_cfg_stage2.runner.max_iterations
        train_cfg_stage2.runner.max_iterations = stage2_iters
        train_cfg_stage2.runner.experiment_name = _format_run_name(
            getattr(train_cfg_stage2.runner, "experiment_name", "depth_obsavoid_go2"), "obstacle_finetune"
        )
        train_cfg_stage2.runner.run_name = _format_run_name(
            getattr(train_cfg_stage2.runner, "run_name", ""), "obstacle_finetune"
        )

        args.max_iterations = stage2_iters if original_max_iterations is not None else original_max_iterations

        env_stage2, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg_stage2)
        runner_stage2, train_cfg_stage2 = task_registry.make_alg_runner(
            env=env_stage2, name=args.task, args=args, train_cfg=train_cfg_stage2
        )

        runner_stage2.alg.actor_critic.load_state_dict(actor_state)
        runner_stage2.alg.optimizer.load_state_dict(optimizer_state)
        runner_stage2.alg.learning_rate = learning_rate
        for group in runner_stage2.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        runner_stage2.current_learning_iteration = completed_iterations

        runner_stage2.learn(
            num_learning_iterations=train_cfg_stage2.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        _maybe_close_env(env_stage1)
        _maybe_close_env(env_stage2)
        args.max_iterations = original_max_iterations
    return True


def train(args):
    if args.task == "depth_obsavoid_go2" and getattr(args, "pretrain_flat_steps", 0) > 0:
        staged = _run_staged_obsavoid_training(args)
        if staged:
            return

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    total_iters = train_cfg.runner.max_iterations
    if getattr(args, 'distill', False):
        distill_iters = getattr(args, 'distill_iterations', None)
        if distill_iters is None:
            distill_iters = total_iters
        distill_iters = max(0, min(distill_iters, total_iters))
        if distill_iters > 0:
            print(f"[Train] Using distillation training loop (teacher-student) for {distill_iters} iterations.")
            ppo_runner.learn_distill(num_learning_iterations=distill_iters, init_at_random_ep_len=True)
        remaining = total_iters - distill_iters
        if remaining > 0:
            print(f"[Train] Continuing with standard PPO for {remaining} iterations.")
            ppo_runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    else:
        ppo_runner.learn(num_learning_iterations=total_iters, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
