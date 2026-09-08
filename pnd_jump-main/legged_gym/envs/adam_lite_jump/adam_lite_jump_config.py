from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_config import (
    AdamLite12dofRoughCfg,
    AdamLite12dofRoughCfgPPO,
)


class AdamLiteJumpCfg(AdamLite12dofRoughCfg):
    """Teacher env: residual jump tracking with future-ref obs (82-dim)."""

    class env(AdamLite12dofRoughCfg.env):
        # v3(58) + future_ref_delta@+5(12) + future_ref_delta@+10(12) = 82
        # Privileged adds lin_vel(3) → 85.
        num_observations = 82
        num_privileged_obs = 85
        num_actions = 12
        episode_length_s = 22

    class terrain(AdamLite12dofRoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(AdamLite12dofRoughCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 1e9

    class motion:
        file = "{LEGGED_GYM_ROOT_DIR}/../adam_lite_dataset/adam_lite/sfu/0005_2FeetJump001.bvh.json"
        rsi = True
        rsi_margin_frames = 50
        rsi_mode = "phase_balanced"
        rsi_phase_edges = [0.0, 0.12, 0.28, 0.42, 0.55, 0.75, 1.0]
        rsi_phase_weights = [0.10, 0.15, 0.25, 0.15, 0.25, 0.10]
        future_frame_offsets = [5, 10]

    class control(AdamLite12dofRoughCfg.control):
        action_scale = 0.35
        # Match Jul14 teacher_ft / working MuJoCo deploy (ankleRoll Kp=0).
        stiffness = {
            "hipPitch": 305.0,
            "hipRoll": 700.0,
            "hipYaw": 405.0,
            "kneePitch": 305.0,
            "anklePitch": 25.0,
            "ankleRoll": 0.0,
        }

    class domain_rand(AdamLite12dofRoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.6]
        randomize_base_mass = True
        added_mass_range = [-0.05, 0.05]
        push_robots = False
        curriculum = False

    class asset(AdamLite12dofRoughCfg.asset):
        terminate_after_contacts_on = ["pelvis", "torso", "shoulder", "elbow"]
        penalize_contacts_on = ["pelvis", "shin", "thigh", "torso", "shoulder", "elbow"]

    class rewards(AdamLite12dofRoughCfg.rewards):
        tracking_sigma = 0.25
        tracking_sigma_vel = 4.0
        tracking_sigma_root = 0.30
        tracking_sigma_height = 0.12
        tracking_sigma_yaw = 0.20
        soft_dof_pos_limit = 0.95
        base_height_target = 0.88
        residual_l2_scale = -0.05
        residual_l2_anneal_steps = 1500

        class scales(AdamLite12dofRoughCfg.rewards.scales):
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = -0.03
            base_height = 0.0
            feet_air_time = 0.0
            feet_swing_height = 0.0
            contact = 0.0
            contact_no_vel = 0.0
            feet_distance = 0.0
            feet_lateral_deviation = 0.0
            ankle_pos = 0.0
            ankle_roll_tracking = 0.0
            action_smoothness = 0.0

            dof_pos_tracking = 2.5
            dof_vel_tracking = 1.2
            root_pos_tracking = 3.5
            root_height_tracking = 2.5
            root_vel_tracking = 1.2
            root_yaw_tracking = 2.0
            orientation = 1.0
            foot_slip = -0.4
            ang_vel_z = -0.05
            action_rate = -0.01
            dof_acc = -2.5e-7
            dof_vel = -5e-4
            collision = -0.5
            alive = 0.15
            residual_l2 = -0.05


class AdamLiteJumpCfgPPO(AdamLite12dofRoughCfgPPO):
    class policy(AdamLite12dofRoughCfgPPO.policy):
        init_noise_std = 0.2
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(AdamLite12dofRoughCfgPPO.algorithm):
        entropy_coef = 0.002
        learning_rate = 1.0e-4
        desired_kl = 0.01
        clip_param = 0.1

    class runner(AdamLite12dofRoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 3000
        run_name = "jump_teacher_future"
        experiment_name = "adam_lite_jump"
        num_steps_per_env = 24
        save_interval = 100
        resume = False
        zero_init_action_head = True
        reset_std_on_load = 0.2
        load_optimizer = False
        use_wandb = False
        wandb_project = "pnd_adam_lite_jump"
        wandb_entity = None
        wandb_tags = ["adam_lite_jump", "teacher", "future_ref", "v4"]

    class noise(AdamLite12dofRoughCfgPPO.noise):
        add_noise = True
        noise_level = 0.3


class AdamLiteJumpCfgPPOFinetune(AdamLiteJumpCfgPPO):
    """PPO fine-tune hyperparams (parent for SFU / height FT)."""

    class policy(AdamLiteJumpCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(AdamLiteJumpCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 5.0e-5
        desired_kl = 0.01
        clip_param = 0.1

    class runner(AdamLiteJumpCfgPPO.runner):
        max_iterations = 3000
        run_name = "jump_teacher_ft"
        save_interval = 100
        resume = True
        zero_init_action_head = False
        force_zero_init_action_head = False
        reset_std_on_load = 0.15
        load_optimizer = False
        wandb_tags = ["adam_lite_jump", "teacher_ft", "lstm", "v4"]


class AdamLiteJumpSfuCfg(AdamLiteJumpCfg):
    """Follow SFU traveling multi-hop (not in-place).

    Same 82-dim obs / motion as Jul14 teacher. Focus:
    - tighter fall cut → less open-loop phase avalanche after tumble
    - stronger orientation / yaw / foot_slip / root tracking for landing + hops
    - mild landing push so recovery is learnable
    """

    class env(AdamLiteJumpCfg.env):
        # rad; default teacher used 1.2 / 1.0
        terminate_pitch = 0.95
        terminate_roll = 0.85
        # meters below reference root z → reset (catches face-plant while phase runs)
        terminate_root_z_below_ref = 0.35

    class domain_rand(AdamLiteJumpCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.7]
        randomize_base_mass = True
        added_mass_range = [-0.06, 0.06]
        push_robots = True
        curriculum = False
        push_interval_s = 1e9
        max_push_vel_xy = 0.12
        landing_push = True
        landing_phase_windows = [[0.22, 0.32], [0.48, 0.58]]
        landing_push_vel_xy = 0.12
        landing_push_prob = 0.25
        action_delay_steps = 1

    class rewards(AdamLiteJumpCfg.rewards):
        tracking_sigma_yaw = 0.15
        tracking_sigma_root = 0.25

        class scales(AdamLiteJumpCfg.rewards.scales):
            root_pos_tracking = 4.0
            root_height_tracking = 2.8
            root_vel_tracking = 1.5
            root_yaw_tracking = 2.8
            orientation = 1.8
            foot_slip = -0.7
            ang_vel_xy = -0.05
            ang_vel_z = -0.08
            action_rate = -0.015
            alive = 0.2


class AdamLiteJumpSfuCfgPPO(AdamLiteJumpCfgPPOFinetune):
    class policy(AdamLiteJumpCfgPPOFinetune.policy):
        init_noise_std = 0.15

    class algorithm(AdamLiteJumpCfgPPOFinetune.algorithm):
        entropy_coef = 0.001
        learning_rate = 5.0e-5

    class runner(AdamLiteJumpCfgPPOFinetune.runner):
        max_iterations = 2000
        run_name = "jump_sfu_track"
        resume = True
        zero_init_action_head = False
        force_zero_init_action_head = False
        reset_std_on_load = 0.15
        load_optimizer = False
        wandb_tags = ["adam_lite_jump", "sfu_track", "landing", "multi_hop"]


class AdamLiteJumpSfuHeightCfg(AdamLiteJumpSfuCfg):
    """Short FT from SFU-3600: only strengthen jump-height tracking.

    Keep SFU terminate / DR / orientation scales unchanged.
    """

    class rewards(AdamLiteJumpSfuCfg.rewards):
        # Tighter height kernel (was 0.12).
        tracking_sigma_height = 0.08

        class scales(AdamLiteJumpSfuCfg.rewards.scales):
            # Was 2.8 on SFU track; only height term is raised.
            root_height_tracking = 4.5
            # Vertical velocity match (new term; xy vel still via root_vel_tracking).
            root_vz_tracking = 1.2


class AdamLiteJumpSfuHeightCfgPPO(AdamLiteJumpCfgPPOFinetune):
    class policy(AdamLiteJumpCfgPPOFinetune.policy):
        init_noise_std = 0.12

    class algorithm(AdamLiteJumpCfgPPOFinetune.algorithm):
        entropy_coef = 0.0008
        learning_rate = 3.0e-5

    class runner(AdamLiteJumpCfgPPOFinetune.runner):
        # Additional iters beyond resume checkpoint (runner uses absolute max).
        max_iterations = 600
        run_name = "jump_sfu_height"
        resume = True
        zero_init_action_head = False
        force_zero_init_action_head = False
        reset_std_on_load = 0.12
        load_optimizer = False
        wandb_tags = ["adam_lite_jump", "sfu_height", "from_3600", "short_ft"]
