import math
import numpy as np

from legged_gym.utils import terrain_utils


def build_obsavoid_heightfield(
    cfg_terrain,
    num_envs,
    env_spacing,
    cube_size=(0.5, 0.5, 0.5),
    num_cubes=3,
    spawn_area=(-3.0, 3.0, -3.0, 3.0),
    min_distance_robot=2.5,
    min_distance_between=1.5,
    seed=None,
    num_goals=8,
    goal_spawn_area=None,
    goal_min_dist_obstacles=1.5,
):
    """Build a single heightfield that stamps box-like obstacles per env tile.

    Args:
        goal_spawn_area: (min_x, max_x, min_y, max_y) area to place goals. If None, uses spawn_area.
        goal_min_dist_obstacles: Minimum distance goals must be from obstacles [m]

    Returns (height_field_raw:int16 2D array, env_origins: (R,C,3) meters, goals: (R,C,num_goals,3) meters).
    """

    # Use obstacle spawn area for goals if not specified separately
    if goal_spawn_area is None:
        goal_spawn_area = spawn_area
    rng = np.random.default_rng(seed)

    # Derive grid size (rows/cols) dynamically if not provided
    rows_cfg = int(getattr(cfg_terrain, "num_rows", 0)) or None
    cols_cfg = int(getattr(cfg_terrain, "num_cols", 0)) or None

    if rows_cfg is None and cols_cfg is None:
        cols = int(math.ceil(math.sqrt(num_envs)))
        rows = int(math.ceil(num_envs / cols))
    elif rows_cfg is None:
        cols = cols_cfg
        rows = int(math.ceil(num_envs / cols))
    elif cols_cfg is None:
        rows = rows_cfg
        cols = int(math.ceil(num_envs / rows))
    else:
        rows = rows_cfg
        cols = cols_cfg

    hf_hs = float(getattr(cfg_terrain, "horizontal_scale", 0.05))
    vf_hs = float(getattr(cfg_terrain, "vertical_scale", 0.01))

    patch_len_m = float(getattr(cfg_terrain, "terrain_length", env_spacing)) or env_spacing
    patch_wid_m = float(getattr(cfg_terrain, "terrain_width", env_spacing)) or env_spacing
    patch_len_m = max(patch_len_m, env_spacing)
    patch_wid_m = max(patch_wid_m, env_spacing)

    patch_len_px = int(patch_len_m / hf_hs)
    patch_wid_px = int(patch_wid_m / hf_hs)

    border_px = int(getattr(cfg_terrain, "border_size", 0.0) / hf_hs)
    tot_rows = rows * patch_len_px + 2 * border_px
    tot_cols = cols * patch_wid_px + 2 * border_px

    hf = np.zeros((tot_rows, tot_cols), dtype=np.int16)

    # Compute env origins (in meters)
    # Include border offset so tile centers align with the heightfield that has a flat border.
    border_m = float(getattr(cfg_terrain, "border_size", 0.0))
    env_origins = np.zeros((rows, cols, 3), dtype=np.float32)
    goals = np.zeros((rows, cols, num_goals, 3), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            env_origins[i, j, 0] = border_m + (i + 0.5) * patch_len_m
            env_origins[i, j, 1] = border_m + (j + 0.5) * patch_wid_m
            env_origins[i, j, 2] = 0.0

    height_units = max(int(round(cube_size[2] / vf_hs)), 1)
    h_cells = max(int(round(cube_size[0] / hf_hs)), 1)
    w_cells = max(int(round(cube_size[1] / hf_hs)), 1)
    half_h = max(h_cells // 2, 0)
    half_w = max(w_cells // 2, 0)
    obstacle_clearance_px = max(1.0, 0.5 * max(h_cells, w_cells))

    min_robot_px = int(min_distance_robot / hf_hs)
    min_between_px = int(min_distance_between / hf_hs)

    # Keep additional flat safety margin near tile borders
    safety_margin = float(getattr(cfg_terrain, "spawn_safety_margin", 1.0))
    half_len = patch_len_m * 0.5 - safety_margin
    half_wid = patch_wid_m * 0.5 - safety_margin
    min_x_cfg, max_x_cfg, min_y_cfg, max_y_cfg = spawn_area
    spawn_min_x = max(min_x_cfg, -half_len)
    spawn_max_x = min(max_x_cfg, half_len)
    spawn_min_y = max(min_y_cfg, -half_wid)
    spawn_max_y = min(max_y_cfg, half_wid)

    if spawn_min_x >= spawn_max_x or spawn_min_y >= spawn_max_y:
        raise ValueError("Spawn area too small after applying safety margin; adjust spawn_area or safety_margin")

    if isinstance(num_cubes, (list, tuple, np.ndarray)):
        per_env_cubes = list(num_cubes)
        total_tiles = rows * cols
        if len(per_env_cubes) < total_tiles:
            reps = math.ceil(total_tiles / max(len(per_env_cubes), 1))
            per_env_cubes = (per_env_cubes * reps)[:total_tiles]
        else:
            per_env_cubes = per_env_cubes[:total_tiles]
    else:
        total_tiles = rows * cols
        per_env_cubes = [int(num_cubes)] * total_tiles

    env_idx = 0
    for i in range(rows):
        for j in range(cols):
            # Top-left corner of this patch in the global heightfield
            start_x = border_px + i * patch_len_px
            start_y = border_px + j * patch_wid_px

            cubes_this_env = int(per_env_cubes[env_idx])
            env_idx += 1

            placed = []
            attempts = 0
            while len(placed) < cubes_this_env and attempts < 2000:
                attempts += 1
                # Sample obstacle center in meters relative to patch center
                cx_m = rng.uniform(spawn_min_x, spawn_max_x)
                cy_m = rng.uniform(spawn_min_y, spawn_max_y)

                # Convert to pixel indices relative to patch origin
                cx_px = int(patch_len_px / 2 + cx_m / hf_hs)
                cy_px = int(patch_wid_px / 2 + cy_m / hf_hs)

                # Respect min distance from robot (robot at patch center)
                dist_robot = math.hypot(cx_px - patch_len_px // 2, cy_px - patch_wid_px // 2)
                if dist_robot < min_robot_px + obstacle_clearance_px:
                    continue

                # Respect min distance between cubes
                too_close = False
                for (px, py, pr) in placed:
                    min_dist = min_between_px + pr + obstacle_clearance_px
                    if math.hypot(cx_px - px, cy_px - py) < min_dist:
                        too_close = True
                        break
                if too_close:
                    continue

                cx_global = start_x + cx_px
                cy_global = start_y + cy_px

                x1 = max(cx_global - half_h, start_x)
                y1 = max(cy_global - half_w, start_y)
                x2 = min(x1 + h_cells, start_x + patch_len_px)
                y2 = min(y1 + w_cells, start_y + patch_wid_px)

                if x2 <= x1 or y2 <= y1:
                    continue

                hf[x1:x2, y1:y2] = height_units

                placed.append((cx_px, cy_px, obstacle_clearance_px))

            # Generate goals for this environment patch
            # Goals create a path that dodges obstacles within spawn area

            # Use the goal spawn area from config
            min_x_cfg, max_x_cfg, min_y_cfg, max_y_cfg = goal_spawn_area
            goal_x_min = max(min_x_cfg, -half_len)
            goal_x_max = min(max_x_cfg, half_len)
            goal_y_min = max(min_y_cfg, -half_wid)
            goal_y_max = min(max_y_cfg, half_wid)

            # Convert placed obstacles to meter coordinates for easier distance checking
            obstacles_m = []
            for (obs_px, obs_py, _) in placed:
                obs_x_m = (obs_px - patch_len_px / 2) * hf_hs
                obs_y_m = (obs_py - patch_wid_px / 2) * hf_hs
                obstacles_m.append((obs_x_m, obs_y_m))

            # Generate waypoints that navigate around obstacles
            # Variable spacing: 1m for first 4 goals, then 3m
            prev_y = 0.0  # Start from center
            prev_x = 0.0  # Robot starts at patch center (0, 0)
            for goal_idx in range(num_goals):
                # Spacing: 1m for goals 0-3, then 3m for goals 4+
                if goal_idx < 4:
                    spacing = 1.0
                else:
                    spacing = 3.0

                # Place goal at fixed distance from previous position
                goal_x_local = prev_x + spacing

                # Ensure goal stays within spawn area X bounds
                if goal_x_local > goal_x_max:
                    goal_x_local = goal_x_max

                # Try to find valid Y that avoids obstacles and creates smooth path
                best_y = 0.0
                best_score = -float('inf')

                for attempt in range(100):
                    # Sample Y position within spawn area
                    goal_y_local = rng.uniform(goal_y_min, goal_y_max)

                    # Check distance from all obstacles
                    min_obs_dist = float('inf')
                    for (obs_x_m, obs_y_m) in obstacles_m:
                        dist = math.hypot(goal_x_local - obs_x_m, goal_y_local - obs_y_m)
                        min_obs_dist = min(min_obs_dist, dist)

                    # Skip if too close to obstacles
                    if min_obs_dist < goal_min_dist_obstacles:
                        continue

                    # Score: prefer far from obstacles + smooth path (not too far from previous Y)
                    lateral_change = abs(goal_y_local - prev_y)
                    smoothness_penalty = lateral_change / 2.0  # Penalize sharp turns

                    # Score combines obstacle avoidance and path smoothness
                    score = min(min_obs_dist, 5.0) - smoothness_penalty

                    if score > best_score:
                        best_score = score
                        best_y = goal_y_local

                # If no valid position found, stay at previous Y (straight ahead)
                if best_score == -float('inf'):
                    best_y = prev_y

                # Store goal relative to env origin
                goals[i, j, goal_idx] = [goal_x_local, best_y, 0.0]
                prev_y = best_y  # Track for next goal
                prev_x = goal_x_local  # Track X position for next spacing

    return hf, env_origins, goals


def heightfield_to_trimesh(hf, horizontal_scale, vertical_scale, slope_treshold=1.0):
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        hf, horizontal_scale, vertical_scale, slope_treshold
    )
    return vertices, triangles
