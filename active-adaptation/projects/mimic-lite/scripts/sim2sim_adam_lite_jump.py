#!/usr/bin/env python3
"""Native-MuJoCo sim2sim for the MimicLite Adam Lite jump ONNX.

Uses pnd_jump's 12DOF scene + PD loop, but NOT their 82-dim LSTM observation
or training code. Policy I/O is the exported MimicLite student ONNX
(command=168, policy=246).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import tempfile
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

WORKSPACE = Path("/root/autodl-tmp/mimic-repro-adam-lite")
PND_ROOT = WORKSPACE / "pnd_jump-main"
DEFAULT_XML = PND_ROOT / "resources/robots/adam_lite/scene_12dof.xml"
DEFAULT_MOTION = WORKSPACE / "adam_lite_dataset/adam_lite/sfu/0005_2FeetJump001.bvh.json"
DEFAULT_ONNX = (
    WORKSPACE
    / "active-adaptation/projects/mimic-lite/scripts/exports/AdamLiteJumpTrack"
    / "policy-wlvs4cnd-unknown.onnx"
)

TRAIN_KPS = np.array(
    [305, 700, 405, 305, 25, 1, 305, 700, 405, 305, 25, 1], dtype=np.float32
)
TRAIN_KDS = np.array(
    [6.1, 30, 6.1, 6.1, 3, 0.35, 6.1, 30, 6.1, 6.1, 3, 0.35], dtype=np.float32
)
DEPLOY_KPS = np.array(
    [305, 700, 405, 305, 30, 3, 305, 700, 405, 305, 30, 3], dtype=np.float32
)
DEPLOY_KDS = TRAIN_KDS.copy()
EFFORT_LIMITS = np.array(
    [230, 160, 105, 230, 40, 12, 230, 160, 105, 230, 40, 12], dtype=np.float32
)
DEFAULT_ANGLES = np.array(
    [-0.32, 0.0, -0.18, 0.66, -0.29, 0.0, -0.32, 0.0, 0.18, 0.66, -0.29, 0.0],
    dtype=np.float32,
)
CMD_FUTURE_STEPS = (-8, -4, -2, 0, 1, 2, 3, 4)
HIST_STEPS = (0, 1, 2, 3, 4, 8, 16)
PREV_ACTION_STEPS = 3
ACTION_SCALE = 0.35
NUM_ACTIONS = 12


def _load_motion_fn():
    loader_path = PND_ROOT / "legged_gym/utils/motion_loader.py"
    spec = importlib.util.spec_from_file_location("pnd_motion_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_adam_lite_jump_motion


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_conjugate(q):
    out = q.copy()
    out[1:] *= -1.0
    return out


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q, v):
    q_inv = quat_conjugate(q)
    t = 2.0 * np.cross(q_inv[1:], v)
    return v + q_inv[0] * t + np.cross(q_inv[1:], t)


def matrix_from_quat_wxyz(q):
    r, i, j, k = q
    two_s = 2.0 / max(float(np.dot(q, q)), 1e-8)
    return np.array(
        [
            [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
            [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
            [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)],
        ],
        dtype=np.float32,
    )


def projected_yaw_quat(q, x_axis_xy_threshold=0.1):
    rot = matrix_from_quat_wxyz(q)
    x_axis = rot[:, 0]
    z_axis = rot[:, 2]
    x_xy = x_axis[:2]
    z_xy = z_axis[:2]
    if np.linalg.norm(x_xy) > x_axis_xy_threshold:
        heading = x_xy
    else:
        heading = z_xy if x_axis[2] < 0.0 else -z_xy
    yaw = np.arctan2(heading[1], heading[0])
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def gravity_b(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)
    g[1] = -2.0 * (qz * qy + qw * qx)
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return g


def clip_frame(idx, n):
    return int(np.clip(idx, 0, n - 1))


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp + (0.0 - dq) * kd


def foot_contact_forces(model, data, body_ids):
    forces = np.zeros(len(body_ids), dtype=np.float32)
    cforce = np.zeros(6, dtype=np.float64)
    wanted = {int(b): i for i, b in enumerate(body_ids)}
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, cforce)
        mag = float(np.linalg.norm(cforce[:3]))
        for geom in (con.geom1, con.geom2):
            bid = int(model.geom_bodyid[geom])
            if bid in wanted:
                forces[wanted[bid]] += mag
    return forces


class HistoryBuf:
    def __init__(self, dim: int, max_lag: int):
        self.buf = deque(maxlen=max_lag + 1)
        self.dim = dim

    def reset(self, value: np.ndarray):
        self.buf.clear()
        for _ in range(self.buf.maxlen):
            self.buf.append(value.astype(np.float32).copy())

    def push(self, value: np.ndarray):
        self.buf.append(value.astype(np.float32).copy())

    def gather(self, steps) -> np.ndarray:
        n = len(self.buf)
        parts = [self.buf[n - 1 - s] for s in steps]
        return np.concatenate(parts, axis=0)


def build_command(motion, frame_idx, robot_pos, robot_quat):
    n = motion["num_frames"]
    cur = clip_frame(frame_idx, n)
    ref_pos0 = motion["root_pos"][cur]
    ref_quat0 = quat_xyzw_to_wxyz(motion["root_quat"][cur])
    yaw_q = projected_yaw_quat(ref_quat0)
    anchor = ref_pos0.copy()
    anchor[2] = 0.0

    pos_local = []
    ori6d = []
    joints = []
    robot_q_inv = quat_conjugate(robot_quat)
    for step in CMD_FUTURE_STEPS:
        f = clip_frame(frame_idx + step, n)
        fut_pos = motion["root_pos"][f]
        fut_quat = quat_xyzw_to_wxyz(motion["root_quat"][f])
        pos_local.append(quat_rotate_inverse(yaw_q, fut_pos - anchor))
        rel = quat_mul(robot_q_inv, fut_quat)
        mat = matrix_from_quat_wxyz(rel)
        ori6d.append(mat[:2, :].reshape(-1))
        joints.append(motion["dof_pos"][f])
    return np.concatenate(
        [np.concatenate(pos_local), np.concatenate(ori6d), np.concatenate(joints)]
    ).astype(np.float32)


def build_policy_obs(hist_ang, hist_grav, hist_q, hist_dq, prev_actions):
    return np.concatenate(
        [
            hist_ang.gather(HIST_STEPS),
            hist_grav.gather(HIST_STEPS),
            hist_q.gather(HIST_STEPS),
            hist_dq.gather(HIST_STEPS),
            prev_actions.reshape(-1),
        ]
    ).astype(np.float32)


MIMIC_ROBOT_XML = (
    WORKSPACE
    / "active-adaptation/projects/mimic-lite/mimic_lite/assets/adam_lite/adam_lite_12dof.xml"
)


def load_pnd_scene(xml_path: str) -> mujoco.MjModel:
    """Load pnd 12DOF scene with visual OBJ meshes stripped.

    Current MuJoCo rejects several pnd visual meshes (`volume is too small`).
    Collision geoms and the 12 leg motors are kept so the PD loop matches deploy.
    """
    xml_path = Path(xml_path).resolve()
    robot_xml = xml_path.parent / "adam_lite_12dof.xml"
    lines = []
    for line in robot_xml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("<mesh ") and "file=" in stripped:
            continue
        if stripped.startswith("<texture ") and "file=" in stripped:
            continue
        if stripped.startswith("<material name=\"pnd_logo\""):
            continue
        if "<geom" in stripped and "mesh=" in stripped:
            continue
        if 'material="pnd_logo"' in stripped and "<geom" in stripped:
            continue
        lines.append(line)
    robot_text = "\n".join(lines) + "\n"
    robot_text = robot_text.replace(
        '<compiler angle="radian" meshdir="assets" texturedir="assets"/>',
        '<compiler angle="radian" inertiafromgeom="false"/>',
        1,
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="adam_lite_sim2sim_"))
    patched_robot = tmpdir / "adam_lite_12dof.xml"
    patched_robot.write_text(robot_text, encoding="utf-8")
    scene = tmpdir / "scene_12dof.xml"
    scene.write_text(
        "\n".join(
            [
                '<mujoco model="adam_lite_12dof scene">',
                f'  <include file="{patched_robot}" />',
                '  <statistic center="0 0 1" extent="1.8" />',
                "  <asset>",
                '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />',
                '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />',
                "  </asset>",
                "  <worldbody>",
                '    <light pos="0 0 3.5" dir="0 0 -1" directional="true" />',
                '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />',
                "  </worldbody>",
                '  <option timestep="0.0025"/>',
                "</mujoco>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[sim2sim] patched_scene={scene}", flush=True)
    return mujoco.MjModel.from_xml_path(str(scene))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX))
    p.add_argument("--xml", type=str, default=str(DEFAULT_XML))
    p.add_argument("--motion", type=str, default=str(DEFAULT_MOTION))
    p.add_argument("--pd", choices=("train", "deploy"), default="train")
    p.add_argument("--duration", type=float, default=22.0)
    p.add_argument("--sim-dt", type=float, default=0.0025)
    p.add_argument("--decimation", type=int, default=8)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--record", type=str, default="")
    p.add_argument("--log-csv", type=str, default="")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    load_motion = _load_motion_fn()
    control_dt = args.sim_dt * args.decimation
    motion = load_motion(args.motion, target_dt=control_dt)
    num_frames = int(motion["num_frames"])
    kps = TRAIN_KPS if args.pd == "train" else DEPLOY_KPS
    kds = TRAIN_KDS if args.pd == "train" else DEPLOY_KDS
    torque_limits = EFFORT_LIMITS * 0.85

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    if set(in_names) != {"command", "policy"}:
        raise SystemExit(f"unexpected ONNX inputs {in_names}")

    m = load_pnd_scene(args.xml)
    d = mujoco.MjData(m)
    m.opt.timestep = args.sim_dt
    toe_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("toeLeft", "toeRight")
    ]

    hist_ang = HistoryBuf(3, max(HIST_STEPS))
    hist_grav = HistoryBuf(3, max(HIST_STEPS))
    hist_q = HistoryBuf(NUM_ACTIONS, max(HIST_STEPS))
    hist_dq = HistoryBuf(NUM_ACTIONS, max(HIST_STEPS))
    prev_actions = np.zeros((PREV_ACTION_STEPS, NUM_ACTIONS), dtype=np.float32)

    def reset_to_frame(frame_idx: int):
        frame_idx = clip_frame(frame_idx, num_frames)
        d.qpos[:3] = motion["root_pos"][frame_idx]
        d.qpos[3:7] = quat_xyzw_to_wxyz(motion["root_quat"][frame_idx])
        d.qpos[7 : 7 + NUM_ACTIONS] = motion["dof_pos"][frame_idx]
        d.qvel[:] = 0.0
        d.qvel[:3] = motion["root_lin_vel"][frame_idx]
        d.qvel[6 : 6 + NUM_ACTIONS] = motion["dof_vel"][frame_idx]
        mujoco.mj_forward(m, d)
        qj = d.qpos[7 : 7 + NUM_ACTIONS].copy()
        dqj = d.qvel[6 : 6 + NUM_ACTIONS].copy()
        ang = d.qvel[3:6].copy()
        grav = gravity_b(d.qpos[3:7])
        hist_ang.reset(ang)
        hist_grav.reset(grav)
        hist_q.reset(qj)
        hist_dq.reset(dqj)
        prev_actions[:] = 0.0
        return frame_idx

    def policy_step(frame_idx: int):
        qj = d.qpos[7 : 7 + NUM_ACTIONS].astype(np.float32)
        dqj = d.qvel[6 : 6 + NUM_ACTIONS].astype(np.float32)
        ang = d.qvel[3:6].astype(np.float32)
        grav = gravity_b(d.qpos[3:7])
        hist_ang.push(ang)
        hist_grav.push(grav)
        hist_q.push(qj)
        hist_dq.push(dqj)
        command = build_command(motion, frame_idx, d.qpos[:3], d.qpos[3:7])
        policy = build_policy_obs(hist_ang, hist_grav, hist_q, hist_dq, prev_actions)
        if command.shape != (168,) or policy.shape != (246,):
            raise RuntimeError(f"obs shapes {command.shape} {policy.shape}")
        action, _priv = sess.run(
            None, {"command": command, "policy": policy}
        )
        action = np.asarray(action, dtype=np.float32).reshape(NUM_ACTIONS)
        prev_actions[1:] = prev_actions[:-1]
        prev_actions[0] = action
        ref_dof = motion["dof_pos"][clip_frame(frame_idx, num_frames)]
        target = ref_dof + action * ACTION_SCALE
        return action, target

    frame_idx = reset_to_frame(0)
    action, target = policy_step(frame_idx)
    rows = []
    peak_total = 0.0
    peak_per_foot = 0.0
    max_vz = float(d.qvel[2])
    start_z = float(d.qpos[2])
    peak_z = start_z
    both_air = 0
    longest_air = 0
    n_ctrl = 0
    renderer = None
    writer = None
    record_path = (args.record or "").strip()
    if record_path:
        import imageio.v2 as imageio

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 1280)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), 720)
        renderer = mujoco.Renderer(m, height=720, width=1280)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        writer = imageio.get_writer(
            record_path,
            fps=int(round(1.0 / control_dt)),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )

    n_steps = int(args.duration / args.sim_dt)
    print(
        f"[sim2sim] onnx={args.onnx} pd={args.pd} frames={num_frames} "
        f"dt={args.sim_dt} decimation={args.decimation}",
        flush=True,
    )
    for step in range(n_steps):
        tau = pd_control(
            target,
            d.qpos[7 : 7 + NUM_ACTIONS],
            kps,
            d.qvel[6 : 6 + NUM_ACTIONS],
            kds,
        )
        d.ctrl[:NUM_ACTIONS] = np.clip(tau, -torque_limits, torque_limits)
        mujoco.mj_step(m, d)
        if step % args.decimation != args.decimation - 1:
            continue
        feet = foot_contact_forces(m, d, toe_ids)
        total = float(feet.sum())
        peak_total = max(peak_total, total)
        peak_per_foot = max(peak_per_foot, float(feet.max()))
        z = float(d.qpos[2])
        vz = float(d.qvel[2])
        peak_z = max(peak_z, z)
        max_vz = max(max_vz, vz)
        air = bool((feet < 1.0).all())
        both_air = both_air + 1 if air else 0
        longest_air = max(longest_air, both_air)
        ref_z = float(motion["root_pos"][clip_frame(frame_idx, num_frames)][2])
        rows.append(
            {
                "t": (step + 1) * args.sim_dt,
                "frame": frame_idx,
                "root_z": z,
                "ref_z": ref_z,
                "z_err": z - ref_z,
                "root_vz": vz,
                "grf_sum": total,
                "grf_l": float(feet[0]),
                "grf_r": float(feet[1]),
            }
        )
        if writer is not None:
            cam.lookat[:] = d.qpos[:3]
            cam.distance = 3.2
            cam.elevation = -20.0
            cam.azimuth = 140.0
            renderer.update_scene(d, camera=cam)
            writer.append_data(np.asarray(renderer.render()))
        action, target = policy_step(frame_idx)
        n_ctrl += 1
        if n_ctrl % 50 == 0:
            print(
                f"[sim2sim] t={rows[-1]['t']:5.1f}s z={z:.3f}(ref {ref_z:.3f}) "
                f"vz={vz:+.2f} grf={total:.0f}N",
                flush=True,
            )
        frame_idx += 1
        if frame_idx >= num_frames - 1:
            break

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"[sim2sim] wrote {record_path}", flush=True)

    stats = {
        "onnx": str(args.onnx),
        "pd": args.pd,
        "mean_alive_s": float(n_ctrl * control_dt),
        "peak_jump_m": float(peak_z - start_z),
        "takeoff_vz": float(max_vz),
        "flight_s": float(longest_air * control_dt),
        "peak_total_grf_n": peak_total,
        "peak_per_foot_n": peak_per_foot,
        "final_z": float(d.qpos[2]),
        "n_ctrl": n_ctrl,
    }
    print("[sim2sim] stats", json.dumps(stats, indent=2), flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(stats, indent=2) + "\n")
    if args.log_csv and rows:
        with open(args.log_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        print(f"[sim2sim] wrote {args.log_csv}", flush=True)
    print("[sim2sim] finished", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
