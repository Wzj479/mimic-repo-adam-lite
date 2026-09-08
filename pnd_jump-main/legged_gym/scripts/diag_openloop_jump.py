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
    env.reset()

    dt = env.dt
    max_steps = int(duration_s / dt)
    zeros = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    z0 = env.root_states[:, 2].clone()
    yaw0 = quat_yaw(env.root_states[:, 3:7])
    z_max = z0.clone()
    yaw_abs_max = torch.zeros(env.num_envs, device=env.device)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    print(
        f"[openloop] action=0 for {duration_s}s ({max_steps} steps), "
        f"envs={env.num_envs}, dt={dt:.4f}"
    )

    for step in range(max_steps):
        _obs, _priv, _rew, dones, _info = env.step(zeros)
        z = env.root_states[:, 2]
        yaw = quat_yaw(env.root_states[:, 3:7])
        yaw_err = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
        z_max = torch.maximum(z_max, z)
        yaw_abs_max = torch.maximum(yaw_abs_max, yaw_err.abs())
        fell |= dones.bool() & (env.episode_length_buf < env.max_episode_length)

        if step % max(1, max_steps // 10) == 0:
            print(
                f"[openloop] t={step * dt:5.1f}s  "
                f"z_mean={z.mean():.3f}  z_max_mean={z_max.mean():.3f}  "
                f"yaw_err_mean={yaw_err.abs().mean() * 180 / 3.1416:.1f}deg  "
                f"fell={fell.float().mean() * 100:.1f}%"
            )

    jump_cm = ((z_max - z0) * 100).mean().item()
    print(
        f"[openloop] DONE  peak_jump≈{jump_cm:.1f}cm (vs start)  "
        f"max|yaw|≈{yaw_abs_max.mean().item() * 180 / 3.1416:.1f}deg  "
        f"fell={fell.float().mean().item() * 100:.1f}%"
    )
    print(
        "[openloop] If MuJoCo open-loop is much worse, align kp/kd/torque before BC/RL."
    )


if __name__ == "__main__":
    if not any(a.startswith("--task") for a in sys.argv):
        sys.argv.append("--task=adam_lite_jump")
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")
    args = get_args()
    set_seed(1)
    main(args)
