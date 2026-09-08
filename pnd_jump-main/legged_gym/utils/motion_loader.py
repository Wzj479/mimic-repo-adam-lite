import json
import os

import numpy as np

LEG_JOINT_NAMES = [
    "hipPitch_Left", "hipRoll_Left", "hipYaw_Left",
    "kneePitch_Left", "anklePitch_Left", "ankleRoll_Left",
    "hipPitch_Right", "hipRoll_Right", "hipYaw_Right",
    "kneePitch_Right", "anklePitch_Right", "ankleRoll_Right",
]


def _get_label_indices(labels, prefix):
    return [i for i, label in enumerate(labels) if label.startswith(prefix)]


def _resample_array(data, src_dt, tgt_dt):
    """Linearly resample [T, D] array from src_dt to tgt_dt."""
    num_frames = data.shape[0]
    src_times = np.arange(num_frames, dtype=np.float64) * src_dt
    duration = src_times[-1]
    tgt_times = np.arange(0.0, duration + 1e-9, tgt_dt)
    resampled = np.zeros((len(tgt_times), data.shape[1]), dtype=np.float32)
    for col in range(data.shape[1]):
        resampled[:, col] = np.interp(tgt_times, src_times, data[:, col])
    return resampled


def _resample_quat(quats, src_dt, tgt_dt):
    """Resample quaternions (x,y,z,w) with normalized linear interpolation."""
    resampled = _resample_array(quats, src_dt, tgt_dt)
    norms = np.linalg.norm(resampled, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return resampled / norms


def _finite_diff_vel(positions, dt):
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) / dt
    vel[0] = vel[1]
    return vel


def load_adam_lite_jump_motion(json_path, target_dt=0.02):
    """Load Adam Lite JSON motion and resample to the control timestep.

    Returns dict with numpy arrays at target_dt:
        dof_pos [T, 12], dof_vel [T, 12],
        root_pos [T, 3], root_quat [T, 4] (xyzw),
        root_lin_vel [T, 3], root_ang_vel [T, 3] (zeros)
    """
    json_path = os.path.expanduser(json_path)
    with open(json_path, "r") as f:
        motion = json.load(f)

    labels = motion["Labels"]
    frames = np.array(motion["Frames"], dtype=np.float32)
    src_dt = float(motion["FrameDuration"])

    dof_indices = [_label_index(labels, f"dof_pos/{name}") for name in LEG_JOINT_NAMES]
    dof_vel_indices = [_label_index(labels, f"dof_vel/{name}") for name in LEG_JOINT_NAMES]
    root_pos_indices = _get_label_indices(labels, "root_pos/")
    root_quat_indices = _get_label_indices(labels, "root_quat/")

    dof_pos = frames[:, dof_indices]
    dof_vel = frames[:, dof_vel_indices]
    root_pos = frames[:, root_pos_indices]
    root_quat = frames[:, root_quat_indices]

    dof_pos = _resample_array(dof_pos, src_dt, target_dt)
    dof_vel = _resample_array(dof_vel, src_dt, target_dt)
    root_pos = _resample_array(root_pos, src_dt, target_dt)
    root_quat = _resample_quat(root_quat, src_dt, target_dt)
    root_lin_vel = _finite_diff_vel(root_pos, target_dt)

    return {
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": np.zeros((len(root_pos), 3), dtype=np.float32),
        "dt": target_dt,
        "duration": len(dof_pos) * target_dt,
        "num_frames": len(dof_pos),
    }


def _label_index(labels, name):
    try:
        return labels.index(name)
    except ValueError as exc:
        raise ValueError(f"Label '{name}' not found in motion file") from exc
