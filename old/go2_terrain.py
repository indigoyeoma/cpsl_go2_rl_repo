"""
GO2-specific wall terrain configurations and utilities.
This module provides specialized wall terrain generation for the GO2 quadruped robot.
"""

import numpy as np
from typing import Optional, Tuple, List
from legged_gym.utils import terrain_utils
from legged_gym.utils.terrain import Terrain


class GO2Terrain(Terrain):
    """
    Specialized terrain class for GO2 robot with wall obstacles
    optimized for quadruped locomotion training.
    """
    
    def __init__(self, cfg, num_robots):
        """
        Initialize GO2-specific terrain.
        
        Args:
            cfg: Terrain configuration object
            num_robots: Number of robots in the environment
        """
        super().__init__(cfg, num_robots)
        
        # GO2-specific wall parameters
        self.go2_obstacle_height_range = (0.08, 0.35)  # 8cm to 35cm wall heights
        
    def make_go2_terrain(self, difficulty: float):
        """
        Create GO2-optimized wall terrain.
        
        Args:
            difficulty: Difficulty level (0.0 to 1.0)
            
        Returns:
            SubTerrain object with generated wall terrain
        """
        terrain = terrain_utils.SubTerrain(
            "go2_terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        # Only use wall-based terrain
        self._create_go2_obstacles(terrain, difficulty)
            
        return terrain
    
    def _create_go2_obstacles(self, terrain, difficulty: float):
        """Create obstacle course for GO2 using walls_only_terrain."""
        # Use the walls_only_terrain function from terrain_utils
        wall_height = self.go2_obstacle_height_range[0] + \
                     (self.go2_obstacle_height_range[1] - self.go2_obstacle_height_range[0]) * difficulty
        
        num_walls = int(3 + 5 * difficulty)  # 3 to 8 walls
        wall_min_size = 0.4 - 0.1 * difficulty  # 40cm to 30cm minimum
        wall_max_size = 0.8 - 0.2 * difficulty  # 80cm to 60cm maximum
        
        # Use the terrain_utils function for consistent wall generation
        terrain_utils.walls_only_terrain(
            terrain,
            wall_height=wall_height,
            min_size=wall_min_size, 
            max_size=wall_max_size,
            num_walls=num_walls,
            platform_size=2.0  # 2m clear platform in center
        )
            


def create_go2_terrain(cfg, num_robots, difficulty: float = 0.5):
    """
    Create wall terrain for GO2 training with fixed difficulty.
    
    Args:
        cfg: Terrain configuration
        num_robots: Number of robots
        difficulty: Difficulty level (0.0 to 1.0)
        
    Returns:
        GO2Terrain object with wall terrain
    """
    terrain = GO2Terrain(cfg, num_robots)
    
    for i in range(cfg.num_rows):
        for j in range(cfg.num_cols):
            # Generate wall terrain with fixed difficulty
            sub_terrain = terrain.make_go2_terrain(difficulty)
                
            terrain.add_terrain_to_map(sub_terrain, i, j)
            
    return terrain


def create_go2_test_terrain(cfg, num_robots, difficulty: float = 0.5):
    """
    Create a single test terrain for GO2 evaluation with walls.
    
    Args:
        cfg: Terrain configuration  
        num_robots: Number of robots
        difficulty: Difficulty level (0.0 to 1.0)
        
    Returns:
        GO2Terrain object with wall test terrain
    """
    terrain = GO2Terrain(cfg, num_robots)
    
    # Fill entire grid with wall terrain
    for i in range(cfg.num_rows):
        for j in range(cfg.num_cols):
            sub_terrain = terrain.make_go2_terrain(difficulty)
            terrain.add_terrain_to_map(sub_terrain, i, j)
            
    return terrain