import argparse
import csv
import gc
import importlib.util
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

LEGGED_GYM_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _load_motion_loader():
    loader_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "utils", "motion_loader.py")
    spec = importlib.util.spec_from_file_location("motion_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_adam_lite_jump_motion


load_adam_lite_jump_motion = _load_motion_loader()


def get_gravity_orientation(quaternion_wxyz):
    qw, qx, qy, qz = quaternion_wxyz
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_xyzw_to_wxyz(quat_xyzw):
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def yaw_from_quat_wxyz(quaternion_wxyz):
    qw, qx, qy, qz = quaternion_wxyz
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def pitch_from_quat_wxyz(quaternion_wxyz):
    qw, qx, qy, qz = quaternion_wxyz
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    return float(np.arcsin(sinp))


def roll_from_quat_wxyz(quaternion_wxyz):
    qw, qx, qy, qz = quaternion_wxyz
    return float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))


def wrap_to_pi(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def apply_yaw_assist(
    target_dof_pos,
    quat_wxyz,
    omega_body,
    ref_quat_xyzw,
    enable,
    kp,
    kd,
    max_corr,
    left_sign,
    right_sign,
    max_tilt_rad=0.45,
    max_yaw_err_rad=0.6,
):
    if not enable:
        ref_yaw = yaw_from_quat_wxyz(quat_xyzw_to_wxyz(ref_quat_xyzw))
        cur_yaw = yaw_from_quat_wxyz(quat_wxyz)
        return target_dof_pos, wrap_to_pi(ref_yaw - cur_yaw), 0.0

    gravity = get_gravity_orientation(quat_wxyz)
    tilt = float(np.linalg.norm(gravity[:2]))
    ref_yaw = yaw_from_quat_wxyz(quat_xyzw_to_wxyz(ref_quat_xyzw))
    cur_yaw = yaw_from_quat_wxyz(quat_wxyz)
    yaw_err = wrap_to_pi(ref_yaw - cur_yaw)

    if tilt > max_tilt_rad or abs(yaw_err) > max_yaw_err_rad:
        return target_dof_pos, yaw_err, 0.0

    omega_z = float(omega_body[2])
    corr = np.clip(kp * yaw_err - kd * omega_z, -max_corr, max_corr)
    out = target_dof_pos.copy()
    out[2] += left_sign * corr
    out[8] += right_sign * corr
    return out, yaw_err, corr


def build_jump_obs(
    omega,
    gravity_orientation,
    phase,
    ref_dof_pos,
    dof_pos,
    dof_vel,
    action,
    default_angles,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
    yaw_err=0.0,
    obs_version="v3",
    future_ref_deltas=None,
    history_flat=None,
):
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    ref_dof_pos_scaled = (ref_dof_pos - default_angles) * dof_pos_scale

    if obs_version in ("v3", "v4", "v5"):
        dof_track_err = (dof_pos - ref_dof_pos) * dof_pos_scale
        parts = [
            omega * ang_vel_scale,
            gravity_orientation,
            np.array([sin_phase, cos_phase], dtype=np.float32),
            np.array([np.sin(yaw_err), np.cos(yaw_err)], dtype=np.float32),
            ref_dof_pos_scaled,
            dof_track_err,
            dof_vel * dof_vel_scale,
            action,
        ]
        if future_ref_deltas is not None:
            for delta in future_ref_deltas:
                parts.append(np.asarray(delta, dtype=np.float32))
        if history_flat is not None:
            parts.append(np.asarray(history_flat, dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    # Legacy v1/v2 obs (56-dim)
    dof_pos_scaled = (dof_pos - default_angles) * dof_pos_scale
    return np.concatenate(
        [
            omega * ang_vel_scale,
            gravity_orientation,
            np.array([sin_phase, cos_phase], dtype=np.float32),
            ref_dof_pos_scaled,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            action,
        ]
    ).astype(np.float32)


def resolve_path(path):
    return path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)


def print_summary(rows, label=""):
    if not rows:
        print("[sim2sim] no log rows", flush=True)
        return

    z = np.array([r["root_z"] for r in rows])
    ref_z = np.array([r["ref_z"] for r in rows])
    yaw_err = np.array([r["yaw_err_deg"] for r in rows])
    pitch = np.array([r["pitch_deg"] for r in rows])
    z_err = np.array([r["z_err"] for r in rows])
    dof_err = np.array([r["dof_pos_err"] for r in rows])
    vz = np.array([r["root_vz"] for r in rows])

    z0 = z[0]
    jump_vs_stand = float(z.max() - z0)
    jump_vs_crouch = float(z.max() - z.min())
    ref_jump_vs_stand = float(ref_z.max() - ref_z[0])

    print(f"\n========== Jump Summary {label} ==========", flush=True)
    print(f"  samples              : {len(rows)}", flush=True)
    print(f"  root_z start/min/max : {z0:.3f} / {z.min():.3f} / {z.max():.3f} m", flush=True)
    print(f"  jump vs stand        : {jump_vs_stand*100:.1f} cm  (ref {ref_jump_vs_stand*100:.1f} cm)", flush=True)
    print(f"  jump vs crouch       : {jump_vs_crouch*100:.1f} cm", flush=True)
    print(f"  max upward vz        : {vz.max():.3f} m/s", flush=True)
    print(f"  mean |z_err|         : {np.mean(np.abs(z_err))*100:.1f} cm", flush=True)
    print(f"  mean dof_pos_err     : {np.mean(dof_err):.3f} rad", flush=True)
    print(f"  yaw_err start→end    : {yaw_err[0]:+.1f} → {yaw_err[-1]:+.1f} deg", flush=True)
    print(f"  yaw_err mean/absmax  : {yaw_err.mean():+.1f} / {np.max(np.abs(yaw_err)):.1f} deg", flush=True)
    print(f"  pitch mean/min/max   : {pitch.mean():+.1f} / {pitch.min():+.1f} / {pitch.max():+.1f} deg", flush=True)
    print("==========================================\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim for adam_lite_jump")
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="adam_lite_jump.yaml",
        help="Config file under deploy/deploy_mujoco/configs/",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer; advance by simulation time (for logging/debug)",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="CSV path for per-control-step logs (default under logs/)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override simulation_duration (seconds)",
    )
    parser.add_argument(
        "--no-yaw-assist",
        action="store_true",
        help="Force disable yaw assist",
    )
    parser.add_argument(
        "--yaw-signs",
        type=str,
        default="",
        help="Override yaw assist signs, e.g. '1,-1' or '-1,1'",
    )
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="Record offscreen MP4 to this path (implies time-based stepping; no interactive viewer)",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=50.0,
        help="Video FPS when using --record (default: 50 = control rate)",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=1280,
        help="Offscreen render width",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=720,
        help="Offscreen render height",
    )
    args = parser.parse_args()

    config_path = os.path.join(
        LEGGED_GYM_ROOT_DIR, "deploy", "deploy_mujoco", "configs", args.config_file
    )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = resolve_path(config["policy_path"])
    xml_path = resolve_path(config["xml_path"])
    motion_path = resolve_path(config["motion_file"])

    simulation_duration = float(
        args.duration if args.duration is not None else config["simulation_duration"]
    )
    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    control_dt = float(config.get("control_dt", simulation_dt * control_decimation))

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    torque_limit_scale = float(config.get("torque_limit_scale", 0.85))
    effort_limits = np.array(
        config.get(
            "effort_limits",
            [230, 160, 105, 230, 40, 12, 230, 160, 105, 230, 40, 12],
        ),
        dtype=np.float32,
    )
    torque_limits = effort_limits * torque_limit_scale

    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale = float(config["action_scale"])
    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    obs_version = str(config.get("obs_version", "v4"))
    control_mode = str(config.get("control_mode", "residual"))  # residual | default
    future_frame_offsets = list(config.get("future_frame_offsets", []) or [])
    history_len = int(config.get("history_len", 0) or 0)
    history_frame_dim = 36  # track_err12 + vel12 + action12
    obs_history = (
        np.zeros((history_len, history_frame_dim), dtype=np.float32)
        if history_len > 0
        else None
    )
    rsi = bool(config.get("rsi", False))
    rsi_margin_frames = int(config.get("rsi_margin_frames", 50))
    yaw_assist_enable = bool(config.get("yaw_assist_enable", False)) and not args.no_yaw_assist
    yaw_assist_kp = float(config.get("yaw_assist_kp", 0.25))
    yaw_assist_kd = float(config.get("yaw_assist_kd", 0.05))
    yaw_assist_max = float(config.get("yaw_assist_max", 0.10))
    yaw_assist_max_tilt = float(config.get("yaw_assist_max_tilt", 0.40))
    yaw_assist_max_err = float(config.get("yaw_assist_max_err", 0.50))
    yaw_assist_left_sign = float(config.get("yaw_assist_left_sign", 1.0))
    yaw_assist_right_sign = float(config.get("yaw_assist_right_sign", -1.0))
    if args.yaw_signs:
        left_s, right_s = args.yaw_signs.split(",")
        yaw_assist_left_sign = float(left_s)
        yaw_assist_right_sign = float(right_s)
        yaw_assist_enable = True

    motion = load_adam_lite_jump_motion(motion_path, target_dt=control_dt)
    num_frames = motion["num_frames"]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    log_rows = []
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "adam_lite_jump", "sim2sim")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    record_arg = (args.record or "").strip()
    if record_arg.lower() == "auto" or record_arg == ".":
        record_path = os.path.join(log_dir, f"jump_record_{stamp}.mp4")
    elif record_arg:
        record_path = (
            record_arg
            if os.path.isabs(record_arg)
            else os.path.join(log_dir, record_arg)
        )
        if not record_path.endswith(".mp4"):
            record_path = record_path + ".mp4"
    else:
        record_path = ""

    log_csv = args.log_csv
    if not log_csv:
        mode = "record" if record_path else ("headless" if args.headless else "viewer")
        assist = "yawon" if yaw_assist_enable else "yawoff"
        log_csv = os.path.join(log_dir, f"jump_state_{mode}_{assist}_{stamp}.csv")

    def _history_frame(qj, dqj, last_action, ref_dof):
        track_err = (qj - ref_dof) * dof_pos_scale
        return np.concatenate(
            [track_err, dqj * dof_vel_scale, last_action], axis=0
        ).astype(np.float32)

    def build_obs_from_state(frame_idx, last_action):
        qj = d.qpos[7 : 7 + num_actions].copy()
        dqj = d.qvel[6 : 6 + num_actions].copy()
        quat_wxyz = d.qpos[3:7]
        omega = d.qvel[3:6].copy()
        phase = frame_idx / max(num_frames - 1, 1)
        ref_dof = motion["dof_pos"][frame_idx]
        ref_yaw = yaw_from_quat_wxyz(quat_xyzw_to_wxyz(motion["root_quat"][frame_idx]))
        cur_yaw = yaw_from_quat_wxyz(quat_wxyz)
        yaw_err = wrap_to_pi(ref_yaw - cur_yaw)
        future_deltas = None
        if future_frame_offsets:
            future_deltas = []
            for off in future_frame_offsets:
                fut_idx = min(frame_idx + int(off), num_frames - 1)
                delta = (motion["dof_pos"][fut_idx] - ref_dof) * dof_pos_scale
                future_deltas.append(delta.astype(np.float32))
        hist_flat = None if obs_history is None else obs_history.reshape(-1)
        obs_local = build_jump_obs(
            omega,
            get_gravity_orientation(quat_wxyz),
            phase,
            ref_dof,
            qj,
            dqj,
            last_action,
            default_angles,
            ang_vel_scale,
            dof_pos_scale,
            dof_vel_scale,
            yaw_err=yaw_err,
            obs_version=obs_version,
            future_ref_deltas=future_deltas,
            history_flat=hist_flat,
        )
        # Push current after reading history (matches training env).
        if obs_history is not None:
            if history_len > 1:
                obs_history[:-1] = obs_history[1:]
            obs_history[-1] = _history_frame(qj, dqj, last_action, ref_dof)
        return obs_local

    def apply_policy(frame_idx, last_action):
        obs_local = build_obs_from_state(frame_idx, last_action)
        assert obs_local.shape[0] == num_obs, (
            f"obs dim {obs_local.shape[0]} != configured num_obs {num_obs}"
        )
        next_action = (
            policy(torch.from_numpy(obs_local).unsqueeze(0)).detach().numpy().squeeze()
        ).astype(np.float32)
        ref_dof = motion["dof_pos"][frame_idx]
        if control_mode == "residual":
            next_target = ref_dof + next_action * action_scale
        else:
            next_target = next_action * action_scale + default_angles
        next_target, yaw_err, yaw_corr = apply_yaw_assist(
            next_target,
            d.qpos[3:7],
            d.qvel[3:6],
            motion["root_quat"][frame_idx],
            yaw_assist_enable,
            yaw_assist_kp,
            yaw_assist_kd,
            yaw_assist_max,
            yaw_assist_left_sign,
            yaw_assist_right_sign,
            max_tilt_rad=yaw_assist_max_tilt,
            max_yaw_err_rad=yaw_assist_max_err,
        )
        return next_action, next_target, obs_local, yaw_err, yaw_corr

    def log_state(frame_idx, yaw_err, yaw_corr, sim_t):
        quat = d.qpos[3:7]
        root = d.qpos[:3]
        ref_root = motion["root_pos"][frame_idx]
        ref_dof = motion["dof_pos"][frame_idx]
        qj = d.qpos[7 : 7 + num_actions]
        gravity = get_gravity_orientation(quat)
        row = {
            "t": sim_t,
            "frame": frame_idx,
            "root_x": float(root[0]),
            "root_y": float(root[1]),
            "root_z": float(root[2]),
            "ref_x": float(ref_root[0]),
            "ref_y": float(ref_root[1]),
            "ref_z": float(ref_root[2]),
            "z_err": float(root[2] - ref_root[2]),
            "root_vz": float(d.qvel[2]),
            "yaw_deg": np.rad2deg(yaw_from_quat_wxyz(quat)),
            "ref_yaw_deg": np.rad2deg(yaw_from_quat_wxyz(quat_xyzw_to_wxyz(motion["root_quat"][frame_idx]))),
            "yaw_err_deg": np.rad2deg(yaw_err),
            "yaw_corr": float(yaw_corr),
            "pitch_deg": np.rad2deg(pitch_from_quat_wxyz(quat)),
            "roll_deg": np.rad2deg(roll_from_quat_wxyz(quat)),
            "tilt": float(np.linalg.norm(gravity[:2])),
            "dof_pos_err": float(np.linalg.norm(qj - ref_dof)),
            "hipYaw_L": float(qj[2]),
            "hipYaw_R": float(qj[8]),
            "hipPitch_L": float(qj[0]),
            "hipPitch_R": float(qj[6]),
        }
        log_rows.append(row)
        return row

    def reset_to_motion_frame(frame_idx):
        nonlocal action, target_dof_pos, obs

        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
        ref_dof = motion["dof_pos"][frame_idx]
        ref_dof_vel = motion["dof_vel"][frame_idx]
        ref_root_pos = motion["root_pos"][frame_idx]
        ref_root_lin_vel = motion["root_lin_vel"][frame_idx]
        ref_root_ang_vel = motion["root_ang_vel"][frame_idx]
        ref_root_quat_wxyz = quat_xyzw_to_wxyz(motion["root_quat"][frame_idx])

        d.qpos[:3] = ref_root_pos
        d.qpos[3:7] = ref_root_quat_wxyz
        d.qpos[7 : 7 + num_actions] = ref_dof
        d.qvel[:] = 0.0
        d.qvel[:3] = ref_root_lin_vel
        d.qvel[3:6] = ref_root_ang_vel
        d.qvel[6 : 6 + num_actions] = ref_dof_vel
        mujoco.mj_forward(m, d)

        if hasattr(policy, "reset_memory"):
            policy.reset_memory()

        action = np.zeros(num_actions, dtype=np.float32)
        if obs_history is not None:
            qj0 = d.qpos[7 : 7 + num_actions].copy()
            dqj0 = d.qvel[6 : 6 + num_actions].copy()
            cur = _history_frame(qj0, dqj0, action, ref_dof)
            obs_history[:] = cur
        action, target_dof_pos, obs, _, _ = apply_policy(frame_idx, action)
        return frame_idx

    start_frame = (
        0
        if not rsi
        else np.random.randint(0, max(num_frames - rsi_margin_frames - 1, 1))
    )
    motion_time = reset_to_motion_frame(start_frame)
    counter = 0
    last_print_t = -1.0

    print(f"[sim2sim] policy: {policy_path}", flush=True)
    print(
        f"[sim2sim] motion: {motion_path} ({num_frames} frames, {motion['duration']:.2f}s)",
        flush=True,
    )
    print(f"[sim2sim] start frame: {motion_time} (rsi={rsi})", flush=True)
    print(
        f"[sim2sim] yaw_assist={yaw_assist_enable} kp={yaw_assist_kp} "
        f"signs=({yaw_assist_left_sign},{yaw_assist_right_sign})",
        flush=True,
    )
    print(f"[sim2sim] log_csv: {log_csv}", flush=True)
    print(f"[sim2sim] headless={args.headless} duration={simulation_duration}s", flush=True)
    if record_path:
        print(f"[sim2sim] record: {record_path} fps={args.record_fps}", flush=True)

    video_writer = None
    renderer = None
    rec_cam = None
    frames_written = 0
    record_fps = float(args.record_fps)
    frame_interval = 1.0 / max(record_fps, 1.0)
    next_frame_t = 0.0

    def _update_camera():
        pos = np.array(d.qpos[0:3], dtype=np.float64)
        rec_cam.lookat[:] = pos
        rec_cam.distance = 3.2
        rec_cam.elevation = -20.0
        rec_cam.azimuth = 140.0

    def _capture_frame(sim_t):
        nonlocal frames_written, next_frame_t
        if renderer is None or video_writer is None or rec_cam is None:
            return
        if sim_t + 1e-9 < next_frame_t:
            return
        _update_camera()
        renderer.update_scene(d, camera=rec_cam)
        rgb = renderer.render()
        video_writer.append_data(np.asarray(rgb))
        frames_written += 1
        next_frame_t += frame_interval

    def step_once(sim_t):
        nonlocal motion_time, action, target_dof_pos, obs, counter, last_print_t

        tau = pd_control(
            target_dof_pos,
            d.qpos[7 : 7 + num_actions],
            kps,
            np.zeros_like(kds),
            d.qvel[6 : 6 + num_actions],
            kds,
        )
        d.ctrl[:num_actions] = np.clip(tau, -torque_limits, torque_limits)
        mujoco.mj_step(m, d)
        counter += 1
        _capture_frame(sim_t)

        if counter % control_decimation == 0:
            action, target_dof_pos, obs, yaw_err, yaw_corr = apply_policy(
                motion_time, action
            )
            row = log_state(motion_time, yaw_err, yaw_corr, sim_t)

            if sim_t - last_print_t >= 1.0 or last_print_t < 0:
                print(
                    f"[sim2sim] t={sim_t:5.1f}s f={motion_time:4d} "
                    f"z={row['root_z']:.3f}(ref {row['ref_z']:.3f}) "
                    f"dz={row['z_err']*100:+5.1f}cm "
                    f"vz={row['root_vz']:+5.2f} "
                    f"yaw_err={row['yaw_err_deg']:+6.1f}deg "
                    f"pitch={row['pitch_deg']:+5.1f}deg "
                    f"dof_err={row['dof_pos_err']:.2f}",
                    flush=True,
                )
                last_print_t = sim_t

            motion_time += 1
            if motion_time >= num_frames - 1:
                motion_time = reset_to_motion_frame(
                    0
                    if not rsi
                    else np.random.randint(0, max(num_frames - rsi_margin_frames - 1, 1))
                )

    try:
        use_record_loop = bool(record_path) or args.headless
        if record_path:
            import imageio.v2 as imageio

            os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
            # Prefer EGL for headless; set before Renderer if unset.
            os.environ.setdefault("MUJOCO_GL", "egl")
            rw = int(args.record_width)
            rh = int(args.record_height)
            # Enlarge offscreen framebuffer if model defaults are too small.
            m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), rw)
            m.vis.global_.offheight = max(int(m.vis.global_.offheight), rh)
            renderer = mujoco.Renderer(m, height=rh, width=rw)
            rec_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(m, rec_cam)
            video_writer = imageio.get_writer(
                record_path,
                fps=record_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=None,
            )

        if use_record_loop:
            n_steps = int(simulation_duration / simulation_dt)
            for i in range(n_steps):
                step_once(i * simulation_dt)
        else:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                start = time.time()
                while viewer.is_running() and time.time() - start < simulation_duration:
                    step_start = time.time()
                    sim_t = counter * simulation_dt
                    step_once(sim_t)
                    viewer.sync()
                    sleep_dt = m.opt.timestep - (time.time() - step_start)
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)
    finally:
        if video_writer is not None:
            video_writer.close()
            video_writer = None
            print(
                f"[sim2sim] wrote {frames_written} frames -> {record_path}",
                flush=True,
            )
        if renderer is not None:
            renderer.close()
            renderer = None

        if log_rows:
            with open(log_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"[sim2sim] wrote {len(log_rows)} rows -> {log_csv}", flush=True)
            print_summary(log_rows)

        policy = None
        m = None
        d = None
        gc.collect()

    print("[sim2sim] simulation finished", flush=True)


if __name__ == "__main__":
    main()
