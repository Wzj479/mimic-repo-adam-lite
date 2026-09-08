#!/usr/bin/env python3
"""Stage 3 reset + stage 4 open-loop residual PD (action=0). GPU required."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa
from active_adaptation.learning.ppo.common import ACTION_KEY

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
DURATION_S = 12.0
INSTANT_CRASH_S = 2.0
JUMP_HEIGHT_OK_M = 0.10


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(
        cfg.task,
        cfg.algo,
        seed=cfg.seed,
        headless=True,
        device=cfg.device,
        checkpoint_path=None,
    )
    command = env.base_env.command_manager
    print("robot", cfg.task.robot.name)
    print("residual", env.base_env.action_manager.residual)
    print("action_dim", env.base_env.action_manager.action_dim)
    print("tracking_joints", command.tracking_joint_names)
    print("tracking_bodies", command.tracking_body_names)

    env.base_env.eval()
    carry = env.reset()
    print("reset t", command.t.detach().cpu().tolist())
    print("motion_len", command.motion_len.detach().cpu().tolist())
    print("ref_joint_pos", tuple(command.ref_joint_pos.shape))
    if tuple(command.ref_joint_pos.shape)[-1] != 12:
        raise SystemExit(f"expected 12 tracked joints, got {command.ref_joint_pos.shape}")
    if int(command.motion_len.min().item()) <= 1:
        raise SystemExit("motion length looks empty")
    print("stage 3 reset OK")

    rollout_policy = policy.get_rollout_policy("eval")
    dt = float(env.step_dt)
    max_steps = max(1, int(DURATION_S / dt))
    n_envs = int(env.num_envs)
    device = env.device

    robot_z = torch.zeros(n_envs, max_steps, device=device)
    ref_z = torch.zeros(n_envs, max_steps, device=device)
    done_step = torch.full((n_envs,), max_steps, device=device, dtype=torch.long)
    nan_hit = False
    first_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

    print(f"[openloop] action=0 for {DURATION_S:.1f}s ({max_steps} steps), envs={n_envs}, dt={dt:.4f}")
    with torch.inference_mode():
        for step in range(max_steps):
            carry = rollout_policy(carry)
            carry[ACTION_KEY] = torch.zeros_like(carry[ACTION_KEY])
            td, carry = env.step_and_maybe_reset(carry)
            z = command.robot_root_pos_w[:, 2]
            rz = command.ref_root_pos_future_w[:, command.obs_current_step_index, 2]
            robot_z[:, step] = z
            ref_z[:, step] = rz
            if not torch.isfinite(z).all():
                nan_hit = True
                print(f"[openloop] NaN at step {step}")
                break
            done = td["next"]["done"].reshape(n_envs)
            newly_done = done & ~first_done
            if newly_done.any():
                done_step[newly_done] = step
                first_done |= newly_done
            if step % max(1, max_steps // 10) == 0:
                print(
                    f"[openloop] t={step * dt:5.1f}s  "
                    f"z={z.mean().item():.3f}  ref_z={rz.mean().item():.3f}  "
                    f"done={first_done.float().mean().item() * 100:.1f}%"
                )

    alive_s = done_step.float() * dt
    peak = robot_z.max(dim=1).values
    start = robot_z[:, 0]
    jump = (peak - start).mean().item()
    ref_jump = (ref_z.max(dim=1).values - ref_z[:, 0]).mean().item()
    instant = (alive_s < INSTANT_CRASH_S).float().mean().item()
    mean_alive = alive_s.mean().item()
    if instant >= 0.8:
        kind = "openloop_cannot_track"
    elif jump >= JUMP_HEIGHT_OK_M and mean_alive >= 4.0:
        kind = "pd_tracks_jump"
    else:
        kind = "mixed_needs_rl"

    stats = {
        "nan": nan_hit,
        "kind": kind,
        "mean_alive_s": mean_alive,
        "instant_crash_frac": instant,
        "peak_jump_m": jump,
        "ref_peak_jump_m": ref_jump,
        "z_start": float(start.mean().item()),
        "z_peak": float(peak.mean().item()),
        "n_envs": n_envs,
        "duration_s": DURATION_S,
    }
    print("[openloop] stats", json.dumps(stats, indent=2))
    out = Path.cwd() / "openloop_adam_lite_jump.json"
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    env.close()
    if nan_hit:
        raise SystemExit("open-loop produced NaN")
    print(f"stage 4 openloop OK  kind={kind}")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "disabled")
    # Env CUDA is already initialized before motion FK cache; fork workers deadlock.
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_NUM_WORKERS", "0")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", "2048")
    os.environ.setdefault(
        "ANY4HDMI_QPOS_CACHE_ROOT",
        "/root/autodl-tmp/mimic-repro-adam-lite/active-adaptation/.cache/motion",
    )
    main()
