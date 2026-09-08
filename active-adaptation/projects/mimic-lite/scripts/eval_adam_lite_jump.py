#!/usr/bin/env python3
"""Stage 7: jump-metric eval + optional ONNX export for Adam Lite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.wandb import parse_checkpoint_path

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
sys.path.insert(0, str(FILE_PATH))
from play import export_policy  # noqa: E402
HEIGHT_ERR_OK_M = 0.05
TAKEOFF_VZ_OK = 2.0
FLIGHT_OK_S = 0.3
SUCCESS_FLIGHT_S = 0.25
SUCCESS_HEIGHT_ERR_M = 0.12
SUCCESS_ALIVE_FRAC = 0.35


def _foot_contact(sensor, body_ids: torch.Tensor) -> torch.Tensor:
    data = sensor.data
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        return force_history[:, body_ids, -1].norm(dim=-1).gt(1.0)
    current_contact_time = getattr(data, "current_contact_time", None)
    if current_contact_time is not None:
        return current_contact_time[:, body_ids] > 1e-6
    raise RuntimeError("contact sensor has no force_history or current_contact_time")


def _current_foot_force(sensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Per-foot GRF [N, n_feet] from the latest contact-history sample."""
    data = sensor.data
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        return force_history[:, body_ids, -1].norm(dim=-1)
    net = getattr(data, "net_forces_w", None)
    if net is not None:
        return net[:, body_ids].norm(dim=-1)
    raise RuntimeError("contact sensor has no force_history or net_forces_w")


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    aa.init(cfg, auto_rank=True)

    from active_adaptation.envs.utils import find_sensor_bodies
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
    command = env.base_env.command_manager
    asset = env.scene.articulations["robot"]
    contact = env.scene.sensors["contact_forces"]
    foot_ids, _ = find_sensor_bodies(asset, contact, ["toeLeft", "toeRight"])
    foot_ids_t = torch.as_tensor(foot_ids, device=env.device, dtype=torch.long)

    env.base_env.eval()
    carry = env.reset()
    dt = float(env.step_dt)
    motion_len = int(command.motion_len.min().item())
    max_steps = max(1, min(int(cfg.get("eval_steps", motion_len + 10)), motion_len + 25))
    n_envs = int(env.num_envs)
    device = env.device
    root_idx = command.tracking_body_names.index(command.root_body_name)

    robot_z = torch.zeros(n_envs, max_steps, device=device)
    ref_z = torch.zeros(n_envs, max_steps, device=device)
    robot_vz = torch.zeros(n_envs, max_steps, device=device)
    both_air = torch.zeros(n_envs, max_steps, dtype=torch.bool, device=device)
    foot_force_sum = torch.zeros(n_envs, max_steps, device=device)
    foot_force_peak = torch.zeros(n_envs, max_steps, device=device)
    done_step = torch.full((n_envs,), max_steps, device=device, dtype=torch.long)
    first_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
    nan_hit = False

    rollout_policy = policy.get_rollout_policy("eval")
    print(
        f"[eval] envs={n_envs} steps={max_steps} dt={dt:.4f} "
        f"motion_len={motion_len} ckpt={checkpoint_path}"
    )
    with torch.inference_mode(), set_exploration_type(ExplorationType.DETERMINISTIC), VecNorm.freeze():
        for step in range(max_steps):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            z = command.robot_root_pos_w[:, 2]
            rz = command.ref_root_pos_future_w[:, command.obs_current_step_index, 2]
            vz = command.robot_body_lin_vel_w[:, root_idx, 2]
            in_contact = _foot_contact(contact, foot_ids_t)  # [N, n_feet]
            robot_z[:, step] = z
            ref_z[:, step] = rz
            robot_vz[:, step] = vz
            both_air[:, step] = ~in_contact.any(dim=-1)
            per_foot = _current_foot_force(contact, foot_ids_t)
            foot_force_sum[:, step] = per_foot.sum(dim=-1)
            foot_force_peak[:, step] = per_foot.max(dim=-1).values
            if not torch.isfinite(z).all():
                nan_hit = True
                print(f"[eval] NaN at step {step}")
                break
            done = td["next"]["done"].reshape(n_envs)
            newly_done = done & ~first_done
            if newly_done.any():
                done_step[newly_done] = step
                first_done |= newly_done
            if step % max(1, max_steps // 8) == 0:
                print(
                    f"[eval] t={step * dt:5.1f}s  z={z.mean().item():.3f}  "
                    f"ref_z={rz.mean().item():.3f}  done={first_done.float().mean().item() * 100:.1f}%"
                )

    alive_s = done_step.float() * dt
    start_z = robot_z[:, 0]
    peak_z = robot_z.max(dim=1).values
    ref_peak_z = ref_z.max(dim=1).values
    jump_m = peak_z - start_z
    ref_jump_m = ref_peak_z - ref_z[:, 0]
    height_err = (peak_z - ref_peak_z).abs()
    takeoff_vz = robot_vz.max(dim=1).values
    # longest consecutive double-foot flight before first done
    flight_s = torch.zeros(n_envs, device=device)
    peak_total_grf = torch.zeros(n_envs, device=device)
    peak_per_foot = torch.zeros(n_envs, device=device)
    for i in range(n_envs):
        end = int(done_step[i].item())
        air = both_air[i, :end].detach().cpu()
        longest = 0
        cur = 0
        for flag in air.tolist():
            if flag:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        flight_s[i] = longest * dt
        sl = max(end, 1)
        peak_total_grf[i] = foot_force_sum[i, :sl].max()
        peak_per_foot[i] = foot_force_peak[i, :sl].max()

    success = (
        (height_err < SUCCESS_HEIGHT_ERR_M)
        & (flight_s > SUCCESS_FLIGHT_S)
        & (alive_s > SUCCESS_ALIVE_FRAC * motion_len * dt)
        & (takeoff_vz > 1.2)
    )
    stats = {
        "nan": bool(nan_hit),
        "n_envs": n_envs,
        "checkpoint": str(checkpoint_path),
        "mean_alive_s": float(alive_s.mean().item()),
        "peak_jump_m": float(jump_m.mean().item()),
        "ref_peak_jump_m": float(ref_jump_m.mean().item()),
        "height_error_m": float(height_err.mean().item()),
        "height_error_ok_frac": float((height_err < HEIGHT_ERR_OK_M).float().mean().item()),
        "takeoff_vz": float(takeoff_vz.mean().item()),
        "takeoff_vz_ok_frac": float((takeoff_vz > TAKEOFF_VZ_OK).float().mean().item()),
        "flight_s": float(flight_s.mean().item()),
        "flight_ok_frac": float((flight_s > FLIGHT_OK_S).float().mean().item()),
        "peak_total_grf_n": float(peak_total_grf.mean().item()),
        "peak_per_foot_n": float(peak_per_foot.mean().item()),
        "alive_full_clip_frac": float((alive_s >= motion_len * dt * 0.98).float().mean().item()),
        "success_rate": float(success.float().mean().item()),
        "targets": {
            "height_error_m": HEIGHT_ERR_OK_M,
            "takeoff_vz": TAKEOFF_VZ_OK,
            "flight_s": FLIGHT_OK_S,
            "success_rate": 0.95,
            "peak_total_grf_n": None,
            "note": "v1.0 500N is ~1 BW standing; landing peak is observation-only",
        },
        "gates": {
            "height_error": bool(height_err.mean().item() < HEIGHT_ERR_OK_M),
            "takeoff_vz": bool(takeoff_vz.mean().item() > TAKEOFF_VZ_OK),
            "flight": bool(flight_s.mean().item() > FLIGHT_OK_S),
            "success": bool(success.float().mean().item() > 0.95),
        },
    }
    print("[eval] stats", json.dumps(stats, indent=2))
    out = Path.cwd() / "eval_adam_lite_jump.json"
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)

    if bool(cfg.get("export_policy", True)):
        export_policy(cfg, env, policy)

    env.close()
    if nan_hit:
        raise SystemExit("eval produced NaN")
    print("stage 7 eval done")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "disabled")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_NUM_WORKERS", "0")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_BATCH_SIZE", "2048")
    os.environ.setdefault(
        "ANY4HDMI_QPOS_CACHE_ROOT",
        "/root/autodl-tmp/mimic-repro-adam-lite/active-adaptation/.cache/motion",
    )
    main()
