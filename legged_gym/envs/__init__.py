from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
# from legged_gym.envs.go2.go2_csonfig_hyper import GO2RoughCfg as GO2HyperCfg, GO2RoughCfgPPO as GO2HyperCfgPPO
from legged_gym.envs.go2.go2_env import GO2Robot
from legged_gym.envs.obs_avoid.go2_obsavoid_config import GO2ObsAvoidCfg, GO2ObsAvoidCfgPPO
from legged_gym.envs.obs_avoid.go2_obsavoid_env import GO2ObsAvoidRobot
from legged_gym.envs.go2_visual.go2_config_depth import GO2DepthCfg, GO2DepthCfgPPO
from legged_gym.envs.go2_visual.go2_env_depth import GO2DepthRobot
from legged_gym.envs.go2_visual.go2_config_hyper import GO2HyperCfg, GO2HyperCfgPPO
from legged_gym.envs.go2_visual.go2_env_hyper import GO2HyperRobot
from legged_gym.envs.obs_avoid.go2_config_obsavoid_depth import GO2ObsAvoidDepthCfg, GO2ObsAvoidDepthCfgPPO
from legged_gym.envs.obs_avoid.go2_env_obsavoid_depth import GO2ObsAvoidDepthRobot
# from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
# from legged_gym.envs.h1.h1_env import H1Robot
# from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
# from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
# from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
# from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", GO2Robot, GO2RoughCfg(), GO2RoughCfgPPO())

task_registry.register( "obsavoid_go2", GO2ObsAvoidRobot, GO2ObsAvoidCfg(), GO2ObsAvoidCfgPPO())
task_registry.register( "depth_go2", GO2DepthRobot, GO2DepthCfg(), GO2DepthCfgPPO())
task_registry.register( "depth_obsavoid_go2", GO2ObsAvoidDepthRobot, GO2ObsAvoidDepthCfg(), GO2ObsAvoidDepthCfgPPO())
task_registry.register( "hyper_go2", GO2HyperRobot, GO2HyperCfg(), GO2HyperCfgPPO())
# task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
# task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
# task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
