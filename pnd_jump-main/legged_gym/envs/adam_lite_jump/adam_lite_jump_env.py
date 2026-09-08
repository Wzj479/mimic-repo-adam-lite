import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_env import AdamLite12dofRobot
from legged_gym.utils.motion_loader import load_adam_lite_jump_motion


class AdamLiteJumpRobot(AdamLite12dofRobot):
    """Motion-tracking jump env (v4/v5).

    - Residual PD: q_des = q_ref + action * action_scale
    - Obs: v3(58) + future ref deltas + optional proprio history
    - Phase-balanced RSI + residual L2 anneal
    """

    # First 58 dims match legacy v3 (teacher can read obs[:, :58]).
    LEGACY_OBS_DIM = 58
    # Per history frame: dof_track_err(12) + dof_vel(12) + action(12)
    HISTORY_FRAME_DIM = 36

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._motion_data = self._load_reference_motion_numpy(cfg)
        self.future_frame_offsets = list(getattr(cfg.motion, "future_frame_offsets", []) or [])
        self.history_len = int(getattr(cfg.motion, "history_len", 0) or 0)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _load_reference_motion_numpy(self, cfg):
        motion_path = cfg.motion.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        target_dt = cfg.control.decimation * cfg.sim.dt
        motion = load_adam_lite_jump_motion(motion_path, target_dt=target_dt)
        print(
            f"[adam_lite_jump] Loaded motion: {motion_path}\n"
            f"  frames={motion['num_frames']}, duration={motion['duration']:.2f}s, "
            f"control_dt={target_dt:.4f}s"
        )
        return motion

    def _init_buffers(self):
        motion = self._motion_data
        self.motion_num_frames = motion["num_frames"]
        self.motion_duration = motion["duration"]
        self.ref_dof_pos = torch.tensor(motion["dof_pos"], dtype=torch.float, device=self.device)
        self.ref_dof_vel = torch.tensor(motion["dof_vel"], dtype=torch.float, device=self.device)
        self.ref_root_pos = torch.tensor(motion["root_pos"], dtype=torch.float, device=self.device)
        self.ref_root_quat = torch.tensor(motion["root_quat"], dtype=torch.float, device=self.device)
        self.ref_root_lin_vel = torch.tensor(motion["root_lin_vel"], dtype=torch.float, device=self.device)
        self.ref_root_ang_vel = torch.tensor(motion["root_ang_vel"], dtype=torch.float, device=self.device)

        super()._init_buffers()
        self.motion_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.ref_dof_pos_curr = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.ref_dof_vel_curr = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.ref_root_pos_curr = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_quat_curr = torch.zeros(self.num_envs, 4, device=self.device)
        self.ref_root_lin_vel_curr = torch.zeros(self.num_envs, 3, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # residual_l2 base captured lazily after _prepare_reward_function multiplies by dt.
        self._residual_l2_base = None
        self._residual_l2_anneal = int(getattr(self.cfg.rewards, "residual_l2_anneal_steps", 1500))

        if self.history_len > 0:
            self.obs_history = torch.zeros(
                self.num_envs,
                self.history_len,
                self.HISTORY_FRAME_DIM,
                device=self.device,
            )
        else:
            self.obs_history = None

        # Action delay: delay ∈ {0, ..., action_delay_steps} sampled per env at reset.
        self.max_action_delay = int(getattr(self.cfg.domain_rand, "action_delay_steps", 0) or 0)
        self.action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Ring of recent actions; index 0 = oldest, -1 = newest (current).
        self.action_delay_buf = torch.zeros(
            self.num_envs,
            self.max_action_delay + 1,
            self.num_actions,
            device=self.device,
        )

        self.landing_push = bool(getattr(self.cfg.domain_rand, "landing_push", False))
        self.landing_phase_windows = list(
            getattr(self.cfg.domain_rand, "landing_phase_windows", []) or []
        )
        self.landing_push_vel_xy = float(
            getattr(self.cfg.domain_rand, "landing_push_vel_xy", 0.15)
        )
        self.landing_push_prob = float(getattr(self.cfg.domain_rand, "landing_push_prob", 0.0))
        # Rising-edge latch so we push at most once per window entry.
        self._in_landing_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        expected = (
            self.LEGACY_OBS_DIM
            + self.num_actions * len(self.future_frame_offsets)
            + self.HISTORY_FRAME_DIM * self.history_len
        )
        if self.num_obs != expected:
            print(
                f"[adam_lite_jump] WARNING: num_obs={self.num_obs} but "
                f"layout expects {expected} (legacy {self.LEGACY_OBS_DIM} + "
                f"{len(self.future_frame_offsets)}*12 future + "
                f"{self.history_len}*{self.HISTORY_FRAME_DIM} hist)"
            )

        self._update_reference_targets(torch.arange(self.num_envs, device=self.device))

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self.p_gains * (self.ref_dof_pos_curr + actions_scaled - self.dof_pos)
            - self.d_gains * self.dof_vel
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _delayed_actions(self):
        """Gather per-env delayed actions from the ring buffer (newest at index -1)."""
        if self.max_action_delay <= 0:
            return self.actions
        # delay=0 → newest; delay=1 → one step back, etc.
        idx = self.max_action_delay - self.action_delay  # in [0, max_delay]
        idx = torch.clamp(idx, 0, self.max_action_delay)
        gather_idx = idx.view(self.num_envs, 1, 1).expand(-1, 1, self.num_actions)
        return torch.gather(self.action_delay_buf, 1, gather_idx).squeeze(1)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        if self.max_action_delay > 0:
            if self.max_action_delay >= 1:
                self.action_delay_buf[:, :-1] = self.action_delay_buf[:, 1:].clone()
            self.action_delay_buf[:, -1] = self.actions
            applied = self._delayed_actions()
        else:
            applied = self.actions

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(applied).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        idx = 0
        noise_vec[idx : idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        noise_vec[idx : idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        noise_vec[idx : idx + 2] = 0.0  # phase
        idx += 2
        noise_vec[idx : idx + 2] = 0.0  # yaw sin/cos
        idx += 2
        noise_vec[idx : idx + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        idx += self.num_actions
        noise_vec[idx : idx + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        idx += self.num_actions
        noise_vec[idx : idx + self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        idx += self.num_actions
        noise_vec[idx : idx + self.num_actions] = 0.0  # last action
        idx += self.num_actions
        # Future ref deltas: light noise
        for _ in self.future_frame_offsets:
            noise_vec[idx : idx + self.num_actions] = (
                0.5 * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            )
            idx += self.num_actions
        # Proprio history: light noise on pos/vel terms, zero on actions
        for _ in range(self.history_len):
            noise_vec[idx : idx + self.num_actions] = (
                0.5 * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            )
            idx += self.num_actions
            noise_vec[idx : idx + self.num_actions] = (
                0.5 * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            )
            idx += self.num_actions
            noise_vec[idx : idx + self.num_actions] = 0.0
            idx += self.num_actions
        return noise_vec

    def _post_physics_step_callback(self):
        self.update_feet_state()
        self.motion_time += 1
        self.phase = self.motion_time.float() / max(self.motion_num_frames - 1, 1)
        self._update_reference_targets(torch.arange(self.num_envs, device=self.device))
        self._anneal_residual_l2()
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

    def _push_robots(self):
        """Landing-phase horizontal impulse (no flight-interval spam)."""
        if not self.cfg.domain_rand.push_robots:
            return
        if self.landing_push and self.landing_phase_windows:
            in_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for lo, hi in self.landing_phase_windows:
                in_window |= (self.phase >= float(lo)) & (self.phase < float(hi))
            entered = in_window & (~self._in_landing_window)
            self._in_landing_window = in_window
            if self.landing_push_prob < 1.0:
                entered = entered & (
                    torch.rand(self.num_envs, device=self.device) < self.landing_push_prob
                )
            push_env_ids = entered.nonzero(as_tuple=False).flatten()
            if len(push_env_ids) == 0:
                return
            max_vel = self.landing_push_vel_xy
            self.root_states[push_env_ids, 7:9] += torch_rand_float(
                -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
            )
            env_ids_int32 = push_env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )
            return
        # Fallback: parent interval-based push.
        super()._push_robots()

    def _anneal_residual_l2(self):
        if "residual_l2" not in self.reward_scales:
            return
        if self._residual_l2_base is None:
            self._residual_l2_base = float(self.reward_scales["residual_l2"])
        # common_step_counter counts env.step calls ≈ policy_iters * num_steps_per_env
        progress = float(self.common_step_counter) / max(self._residual_l2_anneal * 24, 1)
        factor = max(0.0, 1.0 - progress)
        self.reward_scales["residual_l2"] = self._residual_l2_base * factor

    def _update_reference_targets(self, env_ids):
        frame_ids = torch.clamp(self.motion_time[env_ids], 0, self.motion_num_frames - 1)
        self.ref_dof_pos_curr[env_ids] = self.ref_dof_pos[frame_ids]
        self.ref_dof_vel_curr[env_ids] = self.ref_dof_vel[frame_ids]
        self.ref_root_pos_curr[env_ids] = self.ref_root_pos[frame_ids]
        self.ref_root_quat_curr[env_ids] = self.ref_root_quat[frame_ids]
        self.ref_root_lin_vel_curr[env_ids] = self.ref_root_lin_vel[frame_ids]

    def _future_ref_deltas(self, offsets=None):
        """(q_ref[t+k] - q_ref[t]) * scale for each configured offset."""
        chunks = []
        cur = self.ref_dof_pos_curr
        use_offsets = self.future_frame_offsets if offsets is None else offsets
        for offset in use_offsets:
            fut_ids = torch.clamp(self.motion_time + int(offset), 0, self.motion_num_frames - 1)
            fut = self.ref_dof_pos[fut_ids]
            chunks.append((fut - cur) * self.obs_scales.dof_pos)
        return chunks

    def get_teacher_observations(self, future_offsets=(5, 10)):
        """Full 82-dim obs for a future-aware teacher, even if student omits futures."""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        yaw_err = self._yaw_error().unsqueeze(1)
        ref_dof_pos_scaled = (self.ref_dof_pos_curr - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_track_err = (self.dof_pos - self.ref_dof_pos_curr) * self.obs_scales.dof_pos
        parts = (
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            sin_phase,
            cos_phase,
            torch.sin(yaw_err),
            torch.cos(yaw_err),
            ref_dof_pos_scaled,
            dof_track_err,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ) + tuple(self._future_ref_deltas(offsets=list(future_offsets)))
        return torch.cat(parts, dim=-1)

    def _yaw_error(self):
        ref_forward = quat_apply(self.ref_root_quat_curr, self.forward_vec)
        cur_forward = quat_apply(self.base_quat, self.forward_vec)
        ref_yaw = torch.atan2(ref_forward[:, 1], ref_forward[:, 0])
        cur_yaw = torch.atan2(cur_forward[:, 1], cur_forward[:, 0])
        return torch.atan2(torch.sin(ref_yaw - cur_yaw), torch.cos(ref_yaw - cur_yaw))

    def _history_frame(self):
        dof_track_err = (self.dof_pos - self.ref_dof_pos_curr) * self.obs_scales.dof_pos
        return torch.cat(
            (
                dof_track_err,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def _push_history(self):
        if self.obs_history is None:
            return
        # Shift older → newer, write current at the end.
        if self.history_len > 1:
            self.obs_history[:, :-1] = self.obs_history[:, 1:].clone()
        self.obs_history[:, -1] = self._history_frame()

    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        yaw_err = self._yaw_error().unsqueeze(1)
        sin_yaw = torch.sin(yaw_err)
        cos_yaw = torch.cos(yaw_err)
        ref_dof_pos_scaled = (self.ref_dof_pos_curr - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_track_err = (self.dof_pos - self.ref_dof_pos_curr) * self.obs_scales.dof_pos

        # History uses previous steps; push after building obs so current is not duplicated.
        hist_parts = ()
        if self.obs_history is not None:
            hist_parts = (self.obs_history.reshape(self.num_envs, -1),)

        obs = (
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            sin_phase,
            cos_phase,
            sin_yaw,
            cos_yaw,
            ref_dof_pos_scaled,
            dof_track_err,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ) + tuple(self._future_ref_deltas()) + hist_parts

        self.obs_buf = torch.cat(obs, dim=-1)
        self.privileged_obs_buf = torch.cat(
            (self.base_lin_vel * self.obs_scales.lin_vel,) + obs,
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self._push_history()

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        # Configurable so SFU-track FT can cut fallen episodes earlier (less phase avalanche).
        pitch_lim = float(getattr(self.cfg.env, "terminate_pitch", 1.2))
        roll_lim = float(getattr(self.cfg.env, "terminate_roll", 1.0))
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > pitch_lim,
            torch.abs(self.rpy[:, 0]) > roll_lim,
        )
        z_drop = getattr(self.cfg.env, "terminate_root_z_below_ref", None)
        if z_drop is not None:
            target_z = self.ref_root_pos_curr[:, 2] + self.env_origins[:, 2]
            self.reset_buf |= self.root_states[:, 2] < (target_z - float(z_drop))
        self.reset_buf |= self.motion_time >= self.motion_num_frames - 1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_motion_state(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if self.obs_history is not None:
            # Fill with current frame so history starts consistent after reset.
            cur = self._history_frame()[env_ids]
            self.obs_history[env_ids] = cur.unsqueeze(1).expand(-1, self.history_len, -1)
        if self.max_action_delay > 0:
            self.action_delay_buf[env_ids] = 0.0
            self.action_delay[env_ids] = torch.randint(
                0,
                self.max_action_delay + 1,
                (len(env_ids),),
                device=self.device,
            )
        if self.landing_push:
            self._in_landing_window[env_ids] = False

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _sample_phase_balanced_starts(self, n):
        edges = list(getattr(self.cfg.motion, "rsi_phase_edges", [0.0, 1.0]))
        weights = list(getattr(self.cfg.motion, "rsi_phase_weights", [1.0]))
        margin = self.cfg.motion.rsi_margin_frames
        max_start = max(self.motion_num_frames - margin - 1, 1)

        n_buckets = min(len(weights), len(edges) - 1)
        weights = torch.tensor(weights[:n_buckets], dtype=torch.float, device=self.device)
        weights = weights / weights.sum()
        bucket_ids = torch.multinomial(weights, n, replacement=True)

        starts = torch.zeros(n, dtype=torch.long, device=self.device)
        for b in range(n_buckets):
            mask = bucket_ids == b
            if not torch.any(mask):
                continue
            lo = int(edges[b] * max_start)
            hi = max(int(edges[b + 1] * max_start), lo + 1)
            hi = min(hi, max_start)
            lo = min(lo, hi - 1)
            count = int(mask.sum().item())
            starts[mask] = torch.randint(lo, hi, (count,), device=self.device)
        return starts

    def _reset_motion_state(self, env_ids):
        margin = self.cfg.motion.rsi_margin_frames
        max_start = max(self.motion_num_frames - margin - 1, 1)
        mode = str(getattr(self.cfg.motion, "rsi_mode", "uniform"))

        if not self.cfg.motion.rsi:
            start_frames = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        elif mode == "phase_balanced":
            start_frames = self._sample_phase_balanced_starts(len(env_ids))
        else:
            start_frames = torch.randint(0, max_start, (len(env_ids),), device=self.device)

        self.motion_start_time[env_ids] = start_frames
        self.motion_time[env_ids] = start_frames
        self.phase[env_ids] = start_frames.float() / max(self.motion_num_frames - 1, 1)
        self._update_reference_targets(env_ids)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.ref_dof_pos_curr[env_ids]
        self.dof_vel[env_ids] = self.ref_dof_vel_curr[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids, :3] = self.ref_root_pos_curr[env_ids] + self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = self.ref_root_quat_curr[env_ids]
        self.root_states[env_ids, 7:10] = self.ref_root_lin_vel_curr[env_ids]
        self.root_states[env_ids, 10:13] = self.ref_root_ang_vel[self.motion_time[env_ids]]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        return

    def _reward_dof_pos_tracking(self):
        error = torch.sum(torch.square(self.dof_pos - self.ref_dof_pos_curr), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_dof_vel_tracking(self):
        error = torch.sum(torch.square(self.dof_vel - self.ref_dof_vel_curr), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma_vel)

    def _reward_root_pos_tracking(self):
        target_pos = self.ref_root_pos_curr + self.env_origins
        error = torch.sum(torch.square(self.root_states[:, :3] - target_pos), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma_root)

    def _reward_root_height_tracking(self):
        target_z = self.ref_root_pos_curr[:, 2] + self.env_origins[:, 2]
        error = torch.square(self.root_states[:, 2] - target_z)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma_height)

    def _reward_root_vz_tracking(self):
        """Match reference vertical velocity only (jump height / timing)."""
        error = torch.square(self.root_states[:, 9] - self.ref_root_lin_vel_curr[:, 2])
        sigma = float(getattr(self.cfg.rewards, "tracking_sigma_vz", 1.0))
        return torch.exp(-error / sigma)

    def _reward_root_vel_tracking(self):
        error = torch.sum(
            torch.square(self.root_states[:, 7:10] - self.ref_root_lin_vel_curr),
            dim=1,
        )
        return torch.exp(-error / self.cfg.rewards.tracking_sigma_root)

    def _reward_root_yaw_tracking(self):
        yaw_err = self._yaw_error()
        return torch.exp(-torch.square(yaw_err) / self.cfg.rewards.tracking_sigma_yaw)

    def _reward_ang_vel_z(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_foot_slip(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        foot_xy_vel = torch.norm(self.feet_vel[:, :, :2], dim=2)
        slip = torch.square(foot_xy_vel) * contact
        return torch.sum(slip, dim=1)

    def _reward_ankle_roll_tracking(self):
        # ankleRoll_L(5), ankleRoll_R(11): keep lateral foot support aligned with ref.
        err = self.dof_pos[:, [5, 11]] - self.ref_dof_pos_curr[:, [5, 11]]
        return torch.sum(torch.square(err), dim=1)

    def _reward_residual_l2(self):
        return torch.sum(torch.square(self.actions), dim=1)
