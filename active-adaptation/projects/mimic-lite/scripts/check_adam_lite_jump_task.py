#!/usr/bin/env python3
"""CPU Hydra + reset smoke for Adam Lite jump tracking. No training, no GPU required."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "cfg"


def compose_task_cfg():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import mimic_lite  # noqa: F401  registers assets
    import mimic_lite_learning  # noqa: F401  registers algo configs

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="play",
            overrides=[
                "task=tracking-adam_lite-jump",
                "task.num_envs=4",
                "headless=true",
                "device=cpu",
            ],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def check_composed(cfg) -> None:
    from omegaconf import OmegaConf

    task = cfg.task
    print("task.name", task.name)
    print("robot", task.robot.name)
    print("num_envs", task.num_envs)
    print("residual", task.input.action.residual)
    print("action_scaling", dict(task.input.action.action_scaling))
    print("rsi", task.command.rsi, task.command.rsi_mode, task.command.rsi_margin_frames)
    print("motion", OmegaConf.to_container(task.command.motion_cfgs.jump, resolve=True))
    print("terminations", list(task.termination.keys()))
    if task.name != "AdamLiteJumpTrack":
        raise SystemExit(f"unexpected task name {task.name}")
    if task.robot.name != "adam_lite_12dof":
        raise SystemExit(f"unexpected robot {task.robot.name}")
    if not bool(task.input.action.residual):
        raise SystemExit("residual control is not enabled")
    scale = float(task.input.action.action_scaling[".*"])
    if abs(scale - 0.35) > 1e-6:
        raise SystemExit(f"action_scale {scale} != 0.35")
    if not bool(task.command.rsi) or task.command.rsi_mode != "phase_balanced":
        raise SystemExit("RSI is not phase_balanced")
    motion_path = Path(str(task.command.motion_cfgs.jump.path))
    if not (motion_path / "manifest.json").is_file():
        raise SystemExit(f"jump dataset missing: {motion_path}")
    print("Hydra compose OK")


def check_motion_dataset(cfg) -> None:
    import json

    import numpy as np

    jump = cfg.task.command.motion_cfgs.jump
    dataset_root = Path(str(jump.path))
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    qpos_names = list(manifest["qpos_names"])
    npz_files = sorted((dataset_root / "motions").glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"no npz motions under {dataset_root}")
    qpos = np.load(npz_files[0])["qpos"]
    print("manifest qpos_dim", manifest["qpos_dim"], "npz", qpos.shape, npz_files[0].name)
    missing_joints = [
        name
        for name in cfg.task.shared.tracking_joint_names
        if name not in qpos_names
    ]
    if missing_joints:
        raise SystemExit(f"qpos missing tracking joints: {missing_joints}")
    if int(manifest["qpos_dim"]) != int(qpos.shape[1]):
        raise SystemExit("manifest qpos_dim does not match npz")
    root_z = qpos[:, qpos_names.index("root_tz")]
    print(
        "root_tz start/min/max",
        float(root_z[0]),
        float(root_z.min()),
        float(root_z.max()),
    )
    print("motion dataset OK")


def check_reset(cfg) -> None:
    print(
        "skipping MjlabBackendEnv reset: creating the mjlab/warp env (and any4hdmi FK cache) "
        "is SIGKILL'd on this CPU-only machine. Stage 4 env play needs a GPU."
    )


def main() -> None:
    cfg = compose_task_cfg()
    check_composed(cfg)
    check_motion_dataset(cfg)
    check_reset(cfg)
    print("stage 3 task checks passed")


if __name__ == "__main__":
    main()
