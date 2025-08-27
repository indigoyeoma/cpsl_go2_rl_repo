from legged_gym import LEGGED_GYM_ROOT_DIR
import os

os.environ['MESA_VK_DEVICE_SELECT'] = '10de:2231w'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
from legged_gym.utils import webviewer

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Load estimator
    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    for i in range(10*int(env.max_episode_length)):
        # For VisualActorCritic, pass single depth frame
        if train_cfg.runner.policy_class_name == "VisualActorCritic":
            depth_images = env.depth_image if hasattr(env, 'depth_image') else None
            actions = policy(obs.detach(), depth_images=depth_images)
        else:
            # For other policies, use the original approach
            actions = policy(obs.detach())
            
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
