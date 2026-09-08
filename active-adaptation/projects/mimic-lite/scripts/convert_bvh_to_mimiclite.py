#!/usr/bin/env python3
"""Convert Adam Lite BVH JSON motion into any4hdmi qpos NPZ + manifest.

Keeps the source frame rate. MimicLite resamples to task FPS at load time.
No GPU required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MIMIC_LITE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = MIMIC_LITE_ROOT.parents[2]
DEFAULT_JSON = (
    WORKSPACE_ROOT / "adam_lite_dataset/adam_lite/sfu/0005_2FeetJump001.bvh.json"
)
DEFAULT_MJCF = MIMIC_LITE_ROOT / "mimic_lite/assets/adam_lite/adam_lite_12dof.xml"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "any4hdmi/output/adam_lite/jump"

FREE_JOINT_NAME = "floating_joint"
ACTUATED_JOINTS = [
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
# Equality-fixed joints in adam_lite_12dof.xml. Elbows stay at the intended
# bent pose (joint range is [-2, 0]); wrists are absent from the JSON.
LOCKED_JOINT_DEFAULTS = {
    "waistRoll": 0.0,
    "waistPitch": 0.0,
    "waistYaw": 0.0,
    "shoulderPitch_Left": 0.0,
    "shoulderRoll_Left": 0.0,
    "shoulderYaw_Left": 0.0,
    "elbow_Left": -1.6,
    "wristYaw_Left": 0.0,
    "shoulderPitch_Right": 0.0,
    "shoulderRoll_Right": 0.0,
    "shoulderYaw_Right": 0.0,
    "elbow_Right": -1.6,
    "wristYaw_Right": 0.0,
}
TOE_GEOMS = ("toeLeft_collision", "toeRight_collision")
FLIGHT_CLEARANCE_M = 0.02
CONTACT_Z_M = 0.03
MIN_JUMP_HEIGHT_M = 0.15
MIN_FLIGHT_SECONDS = 0.15
MIN_CONTACT_SECONDS = 0.5


def qpos_names_from_model(model: mujoco.MjModel, base_joint_name: str) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            if joint_name == base_joint_name:
                names.extend(
                    [
                        "root_tx",
                        "root_ty",
                        "root_tz",
                        "root_qw",
                        "root_qx",
                        "root_qy",
                        "root_qz",
                    ]
                )
            else:
                names.extend(
                    [
                        f"{joint_name}_tx",
                        f"{joint_name}_ty",
                        f"{joint_name}_tz",
                        f"{joint_name}_qw",
                        f"{joint_name}_qx",
                        f"{joint_name}_qy",
                        f"{joint_name}_qz",
                    ]
                )
        elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            names.append(str(joint_name))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            names.extend(
                [
                    f"{joint_name}_qw",
                    f"{joint_name}_qx",
                    f"{joint_name}_qy",
                    f"{joint_name}_qz",
                ]
            )
        else:
            raise ValueError(f"Unsupported joint type {joint_type} for {joint_name}")
    if len(names) != model.nq:
        raise ValueError(f"Expected {model.nq} qpos names, got {len(names)}")
    return names


def hinge_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in MJCF: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def load_bvh_json(path: Path) -> tuple[dict, np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload["Labels"]
    frames = np.asarray(payload["Frames"], dtype=np.float32)
    timestep = float(payload["FrameDuration"])
    if frames.ndim != 2:
        raise ValueError(f"Expected Frames to be rank 2, got {frames.shape}")
    if timestep <= 0:
        raise ValueError(f"Invalid FrameDuration: {timestep}")
    return {"Labels": labels, **{k: v for k, v in payload.items() if k != "Frames"}}, frames, timestep


def label_index(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError as exc:
        raise ValueError(f"Label {name!r} not found in motion JSON") from exc


def json_to_qpos(
    model: mujoco.MjModel,
    labels: list[str],
    frames: np.ndarray,
    *,
    lock_unactuated: bool,
) -> np.ndarray:
    n_frames = frames.shape[0]
    qpos = np.zeros((n_frames, model.nq), dtype=np.float32)

    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT_NAME)
    if free_id < 0:
        raise ValueError(f"Free joint {FREE_JOINT_NAME!r} not found")
    root_adr = int(model.jnt_qposadr[free_id])
    qpos[:, root_adr + 0] = frames[:, label_index(labels, "root_pos/x")]
    qpos[:, root_adr + 1] = frames[:, label_index(labels, "root_pos/y")]
    qpos[:, root_adr + 2] = frames[:, label_index(labels, "root_pos/z")]
    quat_xyzw = frames[
        :,
        [
            label_index(labels, "root_quat/x"),
            label_index(labels, "root_quat/y"),
            label_index(labels, "root_quat/z"),
            label_index(labels, "root_quat/w"),
        ],
    ]
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("root_quat contains a zero quaternion")
    quat_xyzw = quat_xyzw / norms
    qpos[:, root_adr + 3] = quat_xyzw[:, 3]
    qpos[:, root_adr + 4] = quat_xyzw[:, 0]
    qpos[:, root_adr + 5] = quat_xyzw[:, 1]
    qpos[:, root_adr + 6] = quat_xyzw[:, 2]

    for joint_name, value in LOCKED_JOINT_DEFAULTS.items():
        qpos[:, hinge_qpos_adr(model, joint_name)] = value

    filled: list[str] = []
    skipped_locked: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        label = f"dof_pos/{joint_name}"
        if label not in labels:
            continue
        if lock_unactuated and joint_name in LOCKED_JOINT_DEFAULTS:
            skipped_locked.append(joint_name)
            continue
        qpos[:, int(model.jnt_qposadr[joint_id])] = frames[:, labels.index(label)]
        filled.append(joint_name)

    missing_actuated = [name for name in ACTUATED_JOINTS if name not in filled]
    if missing_actuated:
        raise ValueError(f"JSON missing actuated dof_pos: {missing_actuated}")

    print(f"filled hinge joints from JSON ({len(filled)}): {filled}")
    if skipped_locked:
        print(
            "locked unactuated joints to equality defaults "
            f"({len(skipped_locked)}): {skipped_locked}"
        )
    missing_in_json = [
        name
        for name in LOCKED_JOINT_DEFAULTS
        if f"dof_pos/{name}" not in labels
    ]
    if missing_in_json:
        print(f"JSON has no channel; using defaults: {missing_in_json}")
    return qpos


def write_dataset(
    *,
    output_root: Path,
    mjcf_path: Path,
    qpos: np.ndarray,
    qpos_names: list[str],
    timestep: float,
    source: dict,
    motion_stem: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    motions_dir = output_root / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)
    npz_path = motions_dir / f"{motion_stem}.npz"
    np.savez_compressed(npz_path, qpos=np.asarray(qpos, dtype=np.float32))

    total_hours = float(qpos.shape[0] * timestep / 3600.0)
    mjcf_rel = Path(os_relpath(mjcf_path.resolve(), output_root.resolve()))
    payload = {
        "format_version": 2,
        "dataset_name": "adam_lite_jump",
        "mjcf": mjcf_rel.as_posix(),
        "motions_subdir": "motions",
        "timestep": float(timestep),
        "qpos_dim": int(qpos.shape[1]),
        "qpos_names": qpos_names,
        "num_motions": 1,
        "source": source,
        "total_hours": total_hours,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return npz_path


def os_relpath(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def _longest_true_run(flags: np.ndarray) -> int:
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Lowest world-z of a box collision geom (half-extents in model.geom_size)."""
    size = model.geom_size[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    min_z = np.inf
    for sx in (-size[0], size[0]):
        for sy in (-size[1], size[1]):
            for sz in (-size[2], size[2]):
                corner = xpos + xmat @ np.array([sx, sy, sz], dtype=np.float64)
                min_z = min(min_z, float(corner[2]))
    return float(min_z)


def verify_fk(model: mujoco.MjModel, qpos: np.ndarray, timestep: float) -> dict:
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TOE_GEOMS
    ]
    if pelvis_id < 0 or any(i < 0 for i in geom_ids):
        raise ValueError("pelvis or toe collision geoms missing from MJCF")

    n_frames = qpos.shape[0]
    pelvis_z = np.empty(n_frames, dtype=np.float32)
    sole_z = np.empty((n_frames, 2), dtype=np.float32)
    quat = qpos[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1)
    if np.any(np.abs(quat_norm - 1.0) > 1e-2):
        raise ValueError(
            f"root quaternion not unit: min={quat_norm.min():.4f} max={quat_norm.max():.4f}"
        )

    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        pelvis_z[t] = data.xpos[pelvis_id, 2]
        sole_z[t, 0] = geom_min_z(model, data, geom_ids[0])
        sole_z[t, 1] = geom_min_z(model, data, geom_ids[1])

    both_air = (sole_z[:, 0] > FLIGHT_CLEARANCE_M) & (sole_z[:, 1] > FLIGHT_CLEARANCE_M)
    either_contact = (sole_z[:, 0] < CONTACT_Z_M) | (sole_z[:, 1] < CONTACT_Z_M)
    jump_height = float(pelvis_z.max() - pelvis_z[0])
    stats = {
        "frames": n_frames,
        "duration_s": n_frames * timestep,
        "pelvis_z_start": float(pelvis_z[0]),
        "pelvis_z_min": float(pelvis_z.min()),
        "pelvis_z_max": float(pelvis_z.max()),
        "jump_height_m": jump_height,
        "sole_z_min": [float(sole_z[:, 0].min()), float(sole_z[:, 1].min())],
        "sole_z_max": [float(sole_z[:, 0].max()), float(sole_z[:, 1].max())],
        "flight_frames": int(both_air.sum()),
        "flight_seconds": float(both_air.sum() * timestep),
        "longest_flight_s": _longest_true_run(both_air) * timestep,
        "contact_seconds": float(either_contact.sum() * timestep),
    }
    print("FK stats:", json.dumps(stats, indent=2))
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"qpos_dim {qpos.shape[1]} != model.nq {model.nq}")
    if jump_height < MIN_JUMP_HEIGHT_M:
        raise SystemExit(
            f"jump height {jump_height:.3f}m < {MIN_JUMP_HEIGHT_M}m; not a clear jump"
        )
    if stats["longest_flight_s"] < MIN_FLIGHT_SECONDS:
        raise SystemExit(
            f"longest double-foot flight {stats['longest_flight_s']:.3f}s "
            f"< {MIN_FLIGHT_SECONDS}s"
        )
    if stats["contact_seconds"] < MIN_CONTACT_SECONDS:
        raise SystemExit(
            f"feet almost never near ground ({stats['contact_seconds']:.3f}s); "
            "flight check would be meaningless"
        )
    print("FK jump check OK")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lock-unactuated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep waist/arms at equality defaults (12DOF robot cannot track them).",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CPU FK jump-height / double-foot flight checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json.expanduser().resolve()
    mjcf_path = args.mjcf.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"motion JSON not found: {json_path}")
    if not mjcf_path.is_file():
        raise SystemExit(f"MJCF not found: {mjcf_path}")

    print(f"json={json_path}")
    print(f"mjcf={mjcf_path}")
    print(f"output={output_root}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    qpos_names = qpos_names_from_model(model, FREE_JOINT_NAME)
    payload, frames, timestep = load_bvh_json(json_path)
    print(
        f"source frames={len(frames)} timestep={timestep:.6f}s "
        f"duration={len(frames) * timestep:.3f}s nq={model.nq}"
    )
    qpos = json_to_qpos(
        model,
        payload["Labels"],
        frames,
        lock_unactuated=args.lock_unactuated,
    )
    if qpos.shape != (len(frames), model.nq):
        raise SystemExit(f"unexpected qpos shape {qpos.shape}")

    npz_path = write_dataset(
        output_root=output_root,
        mjcf_path=mjcf_path,
        qpos=qpos,
        qpos_names=qpos_names,
        timestep=timestep,
        source={
            "kind": "pndbotics/adam_lite_dataset",
            "json": str(json_path),
            "lock_unactuated": bool(args.lock_unactuated),
            "n_frames": int(qpos.shape[0]),
        },
        motion_stem=json_path.name.replace(".bvh.json", "").replace(".json", ""),
    )
    print(f"wrote {npz_path} shape={qpos.shape}")
    print(f"wrote {output_root / 'manifest.json'}")

    if args.verify:
        verify_fk(model, qpos, timestep)


if __name__ == "__main__":
    main()
