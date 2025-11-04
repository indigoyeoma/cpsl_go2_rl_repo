from legged_gym.envs.obs_avoid.go2_obsavoid_config import GO2ObsAvoidCfg, GO2ObsAvoidCfgPPO
from legged_gym.envs.go2_visual.go2_config_depth import GO2DepthCfgPPO


class GO2ObsAvoidDepthCfg(GO2ObsAvoidCfg):
    """GO2 obstacle-avoidance configuration extended with the visual pipeline.

    Keeps all task/terrain/reward settings identical to the state-based teacher while
    widening the observation space with the same depth camera setup used by
    ``GO2DepthCfg``. Distillation metadata is provided so the student can load the
    teacher checkpoints automatically.
    """

    class env(GO2ObsAvoidCfg.env):
        num_envs = 196
        num_observations = 48 + 84 * 84  # 7104 state (48) + depth (84x84)
        num_privileged_obs = 48 + 49     # 97 state + terrain samples

    class depth:
        use_camera = True
        camera_num_envs = GO2ObsAvoidCfg.env.num_envs
        original = (424, 240)
        resized = (84, 84)
        horizontal_fov = 86
        near_clip = 0.3
        far_clip = 3.0
        position = [0.3, 0.0, 0.147]
        position_rand = 0.01
        angle = [0.506, 0.0]
        angle_pitch_range_deg = [24.0, 34.0]
        crop_left = 0
        crop_right = 0
        crop_top = 0
        crop_bottom = 0
        dis_noise = 0.01
        update_interval = 5
        buffer_len = 1

    class distillation:
        teacher_cfg_cls = GO2ObsAvoidCfg
        teacher_train_cfg_cls = GO2ObsAvoidCfgPPO
        teacher_experiment = GO2ObsAvoidCfgPPO.runner.experiment_name
        teacher_run = "-1"
        teacher_checkpoint = -1
        resume_run = ""
        resume_checkpoint = -1
        teacher_obs_dim = 48 + 49


class GO2ObsAvoidDepthCfgPPO(GO2DepthCfgPPO):
    class policy(GO2DepthCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(GO2DepthCfgPPO.algorithm):
        learning_rate = 3e-4
        entropy_coef = 0.01
        schedule = "fixed"
        num_mini_batches = 8

    class runner(GO2DepthCfgPPO.runner):
        experiment_name = "obsavoid_depth_student"
        run_name = "vision_student"
        policy_class_name = "VisualActorCritic"
        algorithm_class_name = "PPO"
        max_iterations = 5000
        save_interval = 1000
