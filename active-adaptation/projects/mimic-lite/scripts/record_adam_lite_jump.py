#!/usr/bin/env python3
"""Record a headless RGB video of the Adam Lite jump policy."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
DEFAULT_OUT = Path(
    "/root/autodl-tmp/mimic-repro-adam-lite/active-adaptation/"
    "outputs/2026-09-08/12-21-37-train-sequential/adam_lite_jump.mp4"
)


def _make_geoms_visible(env) -> None:
    model = env.base_env.sim.mj_model
    rgba = np.array(model.geom_rgba, copy=True)
    hidden = rgba[:, 3] < 0.15
    rgba[hidden, 0] = 0.72
    rgba[hidden, 1] = 0.76
    rgba[hidden, 2] = 0.82
    rgba[hidden, 3] = 1.0
    model.geom_rgba[:] = rgba


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if not checkpoint_path:
        raise SystemExit("checkpoint_path is required")
    cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=True,
        device=cfg.device,
        checkpoint_path=checkpoint_path,
    )
    _make_geoms_visible(env)

    env.base_env.eval()
    carry = env.reset()
    dt = float(env.step_dt)
    render_seconds = float(cfg.get("render_seconds", 22.0))
    max_steps = max(1, int(render_seconds / dt))
    out_path = Path(cfg.get("video_path", str(DEFAULT_OUT))).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rollout_policy = policy.get_rollout_policy("eval")
    print(f"[record] steps={max_steps} dt={dt:.4f} out={out_path}")
    with (
        env.get_recorder(out_path, enabled=True) as recorder,
        torch.inference_mode(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):
        for _ in tqdm(range(max_steps), desc="Recording", unit="step"):
            carry = rollout_policy(carry)
            _, carry = env.step_and_maybe_reset(carry)
            recorder.add_frame()

    env.close()
    print(f"[record] saved {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_NUM_WORKERS", "0")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", "2048")
    os.environ.setdefault(
        "ANY4HDMI_QPOS_CACHE_ROOT",
        "/root/autodl-tmp/mimic-repro-adam-lite/active-adaptation/.cache/motion",
    )
    main()
