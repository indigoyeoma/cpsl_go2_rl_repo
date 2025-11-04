"""Simple goal visualization drawing helper - call from post_physics_step"""
import numpy as np
from isaacgym import gymutil
from isaacgym.gymutil import gymapi

def draw_goal_pillars(gym, viewer, envs, env_goals, cur_goal_idx, env_idx=0):
    """Draw spheres at goal locations for better visibility.

    Args:
        gym: IsaacGym instance
        viewer: Viewer handle
        envs: List of environment handles
        env_goals: Tensor of goal positions (num_envs, num_goals, 3)
        cur_goal_idx: Tensor of current goal indices (num_envs,)
        env_idx: Which environment to visualize (default 0)
    """
    gym.clear_lines(viewer)

    num_goals = env_goals.shape[1]
    cur_idx = cur_goal_idx[env_idx].item()

    for goal_idx in range(num_goals):
        goal_pos = env_goals[env_idx, goal_idx].cpu().numpy()

        # Color based on status
        if goal_idx < cur_idx:
            color = (0.5, 0.5, 0.5)  # Gray - completed
        elif goal_idx == cur_idx:
            color = (0.0, 1.0, 0.0)  # Green - current
        else:
            color = (0.0, 0.5, 1.0)  # Blue - future

        # Draw sphere at goal position (radius 0.3m for visibility)
        sphere_geom = gymutil.WireframeSphereGeometry(0.3, 12, 12, None, color=color)

        # Create transform using gymapi
        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2] + 0.3)
        sphere_pose.r = gymapi.Quat(0, 0, 0, 1)

        gymutil.draw_lines(sphere_geom, gym, viewer, envs[env_idx], sphere_pose)
