#!/usr/bin/env python3
"""Native-MuJoCo sim2sim for the MimicLite Adam Lite jump ONNX.

Uses pnd_jump's 12DOF scene, but NOT their 82-dim LSTM observation or
training code. Policy I/O is the exported MimicLite student ONNX
(command=168, policy=246).

Modes (harder toward real, no retraining):
  train   H0: mjlab-like position actuators, 0.005 x 4, full effort
  motor   H1: motor + Python PD, 0.0025 x 8, torque x0.85, keep armature
  realish H2: same as motor, XML armature/friction, Euler integrator
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
    / "policy-adam_lite_jump_best.onnx"
)
LEG_JOINT_NAMES = [
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
]
JOINT_CTRLRANGE = {
    "hipPitch_Left": (-2.09, 2.09),
    "hipRoll_Left": (-0.78, 1.57),
    "hipYaw_Left": (-0.78, 0.78),
    "kneePitch_Left": (-0.09, 2.4),
    "anklePitch_Left": (-1.0, 0.35),
    "ankleRoll_Left": (-0.3491, 0.3491),
    "hipPitch_Right": (-2.09, 2.09),
    "hipRoll_Right": (-1.57, 0.78),
    "hipYaw_Right": (-0.78, 0.78),
    "kneePitch_Right": (-0.09, 2.4),
    "anklePitch_Right": (-1.0, 0.35),
    "ankleRoll_Right": (-0.3491, 0.3491),
}

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
    # Match active_adaptation.utils.math.quat_rotate_inverse (wxyz, world gravity).
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v - q[0] * t + np.cross(xyz, t)


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
    return quat_rotate_inverse(
        quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )


def finite_diff_angvel_body(quats_xyzw, dt):
    n = len(quats_xyzw)
    out = np.zeros((n, 3), dtype=np.float32)
    qw = np.stack([quat_xyzw_to_wxyz(q) for q in quats_xyzw])
    for i in range(n):
        j = min(i + 1, n - 1)
        q_rel = quat_mul(quat_conjugate(qw[i]), qw[j])
        if q_rel[0] < 0.0:
            q_rel = -q_rel
        out[i] = (2.0 * q_rel[1:]) / max(dt, 1e-8)
    if n > 1:
        out[-1] = out[-2]
    return out.astype(np.float32)


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


def _position_actuator_xml(name: str, kp: float, kd: float, effort: float) -> str:
    lo, hi = JOINT_CTRLRANGE[name]
    return (
        f'    <position name="{name}" joint="{name}" kp="{kp}" kv="{kd}" '
        f'ctrllimited="true" ctrlrange="{lo} {hi}" forcerange="{-effort} {effort}"/>'
    )


def _motor_actuator_xml(name: str, effort: float) -> str:
    return (
        f'    <motor name="{name}" joint="{name}" gear="1" '
        f'ctrllimited="true" ctrlrange="{-effort} {effort}"/>'
    )


def _paint_collision_geoms(model: mujoco.MjModel) -> None:
    """Training MJCF hides collision geoms (alpha=0) and has no visual meshes."""
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name in ("floor",) or "ground" in name:
            continue
        rgba = model.geom_rgba[i]
        if "Left" in name or "left" in name:
            rgba[:3] = (0.20, 0.55, 0.90)
        elif "Right" in name or "right" in name:
            rgba[:3] = (0.90, 0.40, 0.18)
        elif "pelvis" in name or "torso" in name or "waist" in name:
            rgba[:3] = (0.95, 0.85, 0.25)
        else:
            rgba[:3] = (0.70, 0.72, 0.75)
        rgba[3] = 1.0


def update_track_camera(cam: mujoco.MjvCamera, root_pos) -> None:
    """Third-person chase camera looking at the pelvis, not a body-fixed view."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]) + 0.15)
    cam.distance = 3.6
    cam.elevation = -18.0
    cam.azimuth = 135.0


def load_pnd_scene(
    xml_path: str,
    sim_dt: float,
    kps: np.ndarray,
    kds: np.ndarray,
    *,
    actuator: str,
    integrator: str,
    apply_armature: bool,
) -> mujoco.MjModel:
    """Load the training 12DOF MJCF and inject actuators.

    MimicLite XML has no actuators (mjlab injects them). Native MuJoCo needs
    them in the MJCF. Visual meshes are omitted so current MuJoCo can load.
    """
    del xml_path  # kept for CLI compatibility; training XML is the source of truth
    robot_text = MIMIC_ROBOT_XML.read_text(encoding="utf-8")
    actuator_lines = ["  <actuator>"]
    for i, name in enumerate(LEG_JOINT_NAMES):
        effort = float(EFFORT_LIMITS[i])
        if actuator == "position":
            actuator_lines.append(
                _position_actuator_xml(name, float(kps[i]), float(kds[i]), effort)
            )
        elif actuator == "motor":
            actuator_lines.append(_motor_actuator_xml(name, effort))
        else:
            raise ValueError(f"unknown actuator {actuator}")
    actuator_lines.append("  </actuator>")
    actuator_block = "\n".join(actuator_lines)
    if "</mujoco>" not in robot_text:
        raise RuntimeError(f"unexpected robot xml {MIMIC_ROBOT_XML}")
    robot_text = robot_text.replace("</mujoco>", actuator_block + "\n</mujoco>", 1)
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
                f'  <option timestep="{sim_dt}" integrator="{integrator}"/>',
                "</mujoco>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[sim2sim] patched_scene={scene}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    _paint_collision_geoms(model)
    if apply_armature:
        for name in LEG_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = int(model.jnt_dofadr[jid])
            model.dof_armature[dof] = 0.01
            model.dof_frictionloss[dof] = 0.01
    return model


MODE_PRESETS = {
    "train": {
        "sim_dt": 0.005,
        "decimation": 4,
        "effort_scale": 1.0,
        "pd": "train",
        "actuator": "position",
        "integrator": "implicitfast",
        "apply_armature": True,
    },
    "motor": {
        "sim_dt": 0.0025,
        "decimation": 8,
        "effort_scale": 0.85,
        "pd": "deploy",
        "actuator": "motor",
        "integrator": "implicitfast",
        "apply_armature": True,
    },
    "realish": {
        "sim_dt": 0.0025,
        "decimation": 8,
        "effort_scale": 0.85,
        "pd": "deploy",
        "actuator": "motor",
        "integrator": "Euler",
        "apply_armature": False,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX))
    p.add_argument("--xml", type=str, default=str(DEFAULT_XML))
    p.add_argument("--motion", type=str, default=str(DEFAULT_MOTION))
    p.add_argument(
        "--mode",
        choices=("train", "motor", "realish"),
        default="train",
        help="train=H0 position actuators; motor=H1 motor PD; realish=H2 no extra armature",
    )
    p.add_argument("--pd", choices=("train", "deploy"), default=None)
    p.add_argument("--duration", type=float, default=22.0)
    p.add_argument("--sim-dt", type=float, default=None)
    p.add_argument("--decimation", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--effort-scale", type=float, default=None)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--record", type=str, default="")
    p.add_argument("--record-width", type=int, default=960)
    p.add_argument("--record-height", type=int, default=540)
    p.add_argument("--record-stride", type=int, default=2)
    p.add_argument("--log-csv", type=str, default="")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def apply_mode_defaults(args):
    preset = MODE_PRESETS[args.mode]
    if args.sim_dt is None:
        args.sim_dt = float(preset["sim_dt"])
    if args.decimation is None:
        args.decimation = int(preset["decimation"])
    if args.effort_scale is None:
        args.effort_scale = float(preset["effort_scale"])
    if args.pd is None:
        args.pd = str(preset["pd"])
    args.actuator = str(preset["actuator"])
    args.integrator = str(preset["integrator"])
    args.apply_armature = bool(preset["apply_armature"])
    return args


def main():
    args = apply_mode_defaults(parse_args())
    load_motion = _load_motion_fn()
    control_dt = args.sim_dt * args.decimation
    motion = load_motion(args.motion, target_dt=control_dt)
    motion["root_ang_vel"] = finite_diff_angvel_body(motion["root_quat"], control_dt)
    num_frames = int(motion["num_frames"])
    kps = TRAIN_KPS if args.pd == "train" else DEPLOY_KPS
    kds = TRAIN_KDS if args.pd == "train" else DEPLOY_KDS
    torque_limits = EFFORT_LIMITS * float(args.effort_scale)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    if set(in_names) != {"command", "policy"}:
        raise SystemExit(f"unexpected ONNX inputs {in_names}")

    m = load_pnd_scene(
        args.xml,
        args.sim_dt,
        kps,
        kds,
        actuator=args.actuator,
        integrator=args.integrator,
        apply_armature=args.apply_armature,
    )
    d = mujoco.MjData(m)
    m.opt.timestep = args.sim_dt
    if args.integrator.lower() == "implicitfast":
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if args.actuator == "position" and abs(args.effort_scale - 1.0) > 1e-6:
        for i in range(min(NUM_ACTIONS, m.nu)):
            m.actuator_forcerange[i] *= args.effort_scale
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
        d.qvel[3:6] = motion["root_ang_vel"][frame_idx]
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

    def policy_step(frame_idx: int, push_hist: bool = True):
        qj = d.qpos[7 : 7 + NUM_ACTIONS].astype(np.float32)
        dqj = d.qvel[6 : 6 + NUM_ACTIONS].astype(np.float32)
        ang = d.qvel[3:6].astype(np.float32)
        grav = gravity_b(d.qpos[3:7])
        if push_hist:
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
        return action, target, command, policy

    frame_idx = reset_to_frame(args.start_frame)
    action, target, command0, policy0 = policy_step(frame_idx, push_hist=False)
    prev_target = motion["dof_pos"][clip_frame(frame_idx, num_frames)].copy()
    rows = []
    peak_total = 0.0
    peak_per_foot = 0.0
    max_vz = float(d.qvel[2])
    start_z = float(d.qpos[2])
    peak_z = start_z
    both_air = 0
    longest_air = 0
    n_ctrl = 0
    fall_t = None
    renderer = None
    writer = None
    rec_stride = 1
    record_path = (args.record or "").strip()
    if record_path:
        import imageio.v2 as imageio

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        rec_w = max(320, int(args.record_width))
        rec_h = max(240, int(args.record_height))
        rec_stride = max(1, int(args.record_stride))
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), rec_w)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), rec_h)
        renderer = mujoco.Renderer(m, height=rec_h, width=rec_w)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        update_track_camera(cam, d.qpos[:3])
        record_fps = max(1, int(round(1.0 / (control_dt * rec_stride))))
        writer = imageio.get_writer(
            record_path,
            fps=record_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        print(
            f"[sim2sim] record {rec_w}x{rec_h} stride={rec_stride} fps={record_fps}",
            flush=True,
        )

    n_steps = int(args.duration / args.sim_dt)
    print(
        f"[sim2sim] mode={args.mode} actuator={args.actuator} onnx={args.onnx} "
        f"pd={args.pd} frames={num_frames} dt={args.sim_dt} "
        f"decimation={args.decimation} effort={args.effort_scale} "
        f"armature={args.apply_armature} start={args.start_frame} nu={m.nu} "
        f"cmd0={command0[:6].tolist()} pol0={policy0[:6].tolist()}",
        flush=True,
    )
    for step in range(n_steps):
        # Training delay=1 physics substep: first tick of a control period
        # still applies the previous target (or zero residual at t=0).
        use_target = prev_target if (step % args.decimation == 0) else target
        qj = d.qpos[7 : 7 + NUM_ACTIONS]
        dqj = d.qvel[6 : 6 + NUM_ACTIONS]
        if args.actuator == "position":
            d.ctrl[:NUM_ACTIONS] = use_target
        else:
            tau = pd_control(use_target, qj, kps, dqj, kds)
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
        grav = gravity_b(d.qpos[3:7])
        fallen = z < 0.30 or float(grav[2]) > -0.5
        if fallen and fall_t is None:
            fall_t = (step + 1) * args.sim_dt
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
        if writer is not None and (n_ctrl % rec_stride == 0):
            update_track_camera(cam, d.qpos[:3])
            renderer.update_scene(d, camera=cam)
            writer.append_data(np.asarray(renderer.render()))
        prev_target = target
        action, target, _, _ = policy_step(frame_idx, push_hist=True)
        n_ctrl += 1
        if n_ctrl % 50 == 0:
            print(
                f"[sim2sim] t={rows[-1]['t']:5.1f}s z={z:.3f}(ref {ref_z:.3f}) "
                f"vz={vz:+.2f} grf={total:.0f}N fall={fall_t}",
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
        "mode": args.mode,
        "actuator": args.actuator,
        "pd": args.pd,
        "sim_dt": args.sim_dt,
        "decimation": args.decimation,
        "effort_scale": args.effort_scale,
        "apply_armature": args.apply_armature,
        "integrator": args.integrator,
        "start_frame": args.start_frame,
        "mean_alive_s": float((fall_t if fall_t is not None else n_ctrl * control_dt)),
        "time_to_fall_s": fall_t,
        "peak_jump_m": float(peak_z - start_z),
        "takeoff_vz": float(max_vz),
        "flight_s": float(longest_air * control_dt),
        "peak_total_grf_n": peak_total,
        "peak_per_foot_n": peak_per_foot,
        "final_z": float(d.qpos[2]),
        "n_ctrl": n_ctrl,
        "fallen": fall_t is not None,
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
