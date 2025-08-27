import numpy as np
from numpy.random import choice
from scipy import interpolate

from . import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        
        # Track wall positions for safe robot spawning
        self.wall_positions = []  # List of (x, y, radius) tuples

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        
        # Add boundary walls around the entire grid
        self.add_boundary_walls()
        
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # Check if this is the center patch (for 5x5 grid, center is at [2,2])
            center_row = self.cfg.num_rows // 2
            center_col = self.cfg.num_cols // 2
            is_center_patch = (i == center_row) and (j == center_col)
            
            # Create obstacle terrain for all patches including center
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.08 + 0.25 * difficulty  # Increased step heights: 8cm to 33cm
        discrete_obstacles_height = 0.1 + difficulty * 0.4  # Increased box heights: 10cm to 50cm
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            # Use rectangular walls/columns (increased heights for more challenge)
            num_walls = 7         # More walls for denser obstacle field
            wall_height = 0.5
            wall_min_size = 0.4  # 40cm minimum size (slightly smaller)
            wall_max_size = 0.6  # 60cm maximum size (taller rectangles)
            terrain_utils.walls_only_terrain(terrain, wall_height, wall_min_size, wall_max_size, num_walls, platform_size=0.0)
        elif choice < self.proportions[5]:
            # Individual corridors for each robot
            corridor_width = 3.0  # 3m wide corridors
            corridor_length = 15.0  # 15m long corridors
            wall_height = 0.5
            num_obstacles = 3  # 3 obstacles per corridor
            obstacle_size_range = (0.3, 0.8)  # 30cm to 80cm obstacles
            # Pass self.num_robots as parameter
            terrain_utils.individual_corridors_terrain(terrain, wall_height, corridor_width, corridor_length, num_obstacles, obstacle_size_range, self.num_robots)
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        
        # Collect wall positions from this terrain patch
        if hasattr(terrain, 'wall_positions'):
            for wall_x, wall_y, wall_radius in terrain.wall_positions:
                # Convert to global coordinates
                global_x = env_origin_x + wall_x - self.env_length/2
                global_y = env_origin_y + wall_y - self.env_width/2
                self.wall_positions.append((global_x, global_y, wall_radius))
    
    def add_boundary_walls(self):
        """Add concrete boundary walls around the entire grid to contain robots"""
        if self.type not in ["trimesh", "heightfield"]:
            return
            
        # Wall parameters
        wall_height = int(1.5 / self.cfg.vertical_scale)  # 1.5m tall concrete walls
        wall_thickness = int(0.3 / self.cfg.horizontal_scale)  # 30cm thick walls
        
        # Get grid dimensions
        grid_start = self.border
        grid_end_x = self.border + self.cfg.num_rows * self.length_per_env_pixels
        grid_end_y = self.border + self.cfg.num_cols * self.width_per_env_pixels
        
        # Create boundary walls
        # Top wall (along x-axis)
        self.height_field_raw[grid_start-wall_thickness:grid_start, :] = wall_height
        
        # Bottom wall (along x-axis) 
        self.height_field_raw[grid_end_x:grid_end_x+wall_thickness, :] = wall_height
        
        # Left wall (along y-axis)
        self.height_field_raw[:, grid_start-wall_thickness:grid_start] = wall_height
        
        # Right wall (along y-axis)
        self.height_field_raw[:, grid_end_y:grid_end_y+wall_thickness] = wall_height
        
        # Add corner reinforcements for extra strength
        corner_size = wall_thickness * 2
        
        # Top-left corner
        self.height_field_raw[grid_start-corner_size:grid_start, 
                             grid_start-corner_size:grid_start] = wall_height
        
        # Top-right corner  
        self.height_field_raw[grid_start-corner_size:grid_start,
                             grid_end_y:grid_end_y+corner_size] = wall_height
                             
        # Bottom-left corner
        self.height_field_raw[grid_end_x:grid_end_x+corner_size,
                             grid_start-corner_size:grid_start] = wall_height
                             
        # Bottom-right corner
        self.height_field_raw[grid_end_x:grid_end_x+corner_size,
                             grid_end_y:grid_end_y+corner_size] = wall_height
        
        print(f"Added boundary walls: {1.5}m tall, {0.3}m thick around {self.cfg.num_rows}x{self.cfg.num_cols} grid")

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
