#!/usr/bin/env python3
"""Dump first-step MimicLite command/policy obs for sim2sim alignment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)
    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if not checkpoint_path:
        raise SystemExit("checkpoint_path is required")
    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=True,
        device=cfg.device,
        checkpoint_path=checkpoint_path,
    )
    env.base_env.eval()
    td = env.reset()
    with torch.inference_mode():
        td = policy(td)
        td = env.step(td)
    command = env.base_env.command_manager
    cmd = td["command"][0].detach().cpu().numpy()
    pol = td["policy"][0].detach().cpu().numpy()
    print("command_shape", cmd.shape, "policy_shape", pol.shape)
    print("t", int(command.t[0].item()), "motion_len", int(command.motion_len[0].item()))
    print("robot_root", command.robot_root_pos_w[0].detach().cpu().tolist())
    print("ref_root_t", command.ref_root_pos_future_w[0, command.obs_current_step_index].detach().cpu().tolist())
    print("command[:12]", cmd[:12].tolist())
    print("command[24:36]", cmd[24:36].tolist())
    print("command[72:84]", cmd[72:84].tolist())
    print("policy[:9]", pol[:9].tolist())
    print("policy[21:30]", pol[21:30].tolist())
    print("policy[42:54]", pol[42:54].tolist())
    out = {
        "t": int(command.t[0].item()),
        "command": cmd.tolist(),
        "policy": pol.tolist(),
        "robot_root": command.robot_root_pos_w[0].detach().cpu().tolist(),
        "robot_quat": command.robot_root_quat_w[0].detach().cpu().tolist(),
        "joint_pos": env.scene.articulations["robot"].data.joint_pos[0, :12].detach().cpu().tolist(),
        "joint_vel": env.scene.articulations["robot"].data.joint_vel[0, :12].detach().cpu().tolist(),
        "ang_vel_b": env.scene.articulations["robot"].data.root_com_ang_vel_b[0].detach().cpu().tolist(),
        "proj_grav": env.scene.articulations["robot"].data.projected_gravity_b[0].detach().cpu().tolist(),
        "physics_dt": float(getattr(env, "physics_dt", getattr(env.base_env, "physics_dt"))),
        "step_dt": float(env.step_dt),
        "decimation": int(getattr(env, "decimation", getattr(env.base_env, "decimation"))),
    }
    path = Path.cwd() / "dump_jump_obs.json"
    path.write_text(json.dumps(out) + "\n")
    print("wrote", path)
    env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "disabled")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_NUM_WORKERS", "0")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", "2048")
    os.environ.setdefault(
        "ANY4HDMI_QPOS_CACHE_ROOT",
        "/root/autodl-tmp/mimic-repro-adam-lite/active-adaptation/.cache/motion",
    )
    sys.path.insert(0, str(FILE_PATH))
    main()
