#!/usr/bin/env python3
"""Phase 0: open-loop residual diagnostic (action=0 → pure q_ref PD tracking)."""
from __future__ import annotations

import os
import sys

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed


def quat_yaw(q):
    # q: (N,4) xyzw
    w, x, y, z = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(args):
    duration_s = float(os.environ.get("DIAG_DURATION_S", "12"))
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # Fair reference-PD eval: always start at frame 0 (not phase-balanced RSI).
    env.cfg.motion.rsi = False
    env.reset()

    dt = env.dt
    max_steps = int(duration_s / dt)
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    z0 = env.root_states[:, 2].clone()
    yaw0 = quat_yaw(env.root_states[:, 3:7])
    z_max = z0.clone()
    yaw_abs_max = torch.zeros(env.num_envs, device=env.device)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Time-to-first-fall (steps); -1 if never fell
    first_fall = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    print(
        f"[reference-PD] action=0 for {duration_s}s ({max_steps} steps), "
        f"envs={env.num_envs}, dt={dt:.4f}, rsi=False (start frame 0)"
    )

    for step in range(max_steps):
        _obs, _priv, _rew, dones, _info = env.step(zeros)
        z = env.root_states[:, 2]
        yaw = quat_yaw(env.root_states[:, 3:7])
        yaw_err = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
        z_max = torch.maximum(z_max, z)
        yaw_abs_max = torch.maximum(yaw_abs_max, yaw_err.abs())
        # Count falls only before motion timeout (contact / tilt terminations).
        is_fall = dones.bool() & (~env.time_out_buf.bool())
        newly = is_fall & (first_fall < 0)
        first_fall[newly] = step
        fell |= is_fall

        if step % max(1, max_steps // 10) == 0:
            print(
                f"[reference-PD] t={step * dt:5.1f}s  "
                f"z_mean={z.mean():.3f}  z_max_mean={z_max.mean():.3f}  "
                f"yaw_err_mean={yaw_err.abs().mean() * 180 / 3.1416:.1f}deg  "
                f"fell={fell.float().mean() * 100:.1f}%"
            )

    jump_cm = ((z_max - z0) * 100).mean().item()
    fell_frac = fell.float().mean().item() * 100
    ttf = first_fall[first_fall >= 0].float()
    ttf_s = (ttf.mean() * dt).item() if ttf.numel() > 0 else float("nan")
    print(
        f"[reference-PD] DONE  peak_jump≈{jump_cm:.1f}cm (vs start)  "
        f"max|yaw|≈{yaw_abs_max.mean().item() * 180 / 3.1416:.1f}deg  "
        f"fell={fell_frac:.1f}%  mean_time_to_fall≈{ttf_s:.2f}s"
    )
    if fell_frac > 80 and (ttf.numel() == 0 or ttf_s < 1.5):
        print(
            "[reference-PD] WARN: most envs fall early under pure q_ref PD — "
            "policy residual is necessary; still OK to proceed to teacher distill, "
            "but expect MuJoCo gap if contact/PD differ."
        )
    else:
        print(
            "[reference-PD] If MuJoCo reference-PD is much worse, align kp/kd/torque before BC/RL."
        )


if __name__ == "__main__":
    if not any(a.startswith("--task") for a in sys.argv):
        sys.argv.append("--task=adam_lite_jump")
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")
    args = get_args()
    set_seed(1)
    main(args)
