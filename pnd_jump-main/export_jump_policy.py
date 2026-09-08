#!/usr/bin/env python3
"""Export adam_lite_jump LSTM JIT without full play visualization."""
import os
import sys

import isaacgym  # noqa: F401  must import before torch in some setups
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, parse_sim_params

LOAD_RUN = os.environ.get("LOAD_RUN", "Jul13_18-20-10_jump_motion_tracking_v3")
CHECKPOINT = os.environ.get("CHECKPOINT", "5800")

sys.argv = [
    "export_jump_policy.py",
    "--task=adam_lite_jump",
    f"--load_run={LOAD_RUN}",
    f"--checkpoint={CHECKPOINT}",
    "--num_envs=1",
    "--headless",
]

args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
env_cfg.env.num_envs = 1
sim_params = {"sim": class_to_dict(env_cfg.sim)}
sim_params = parse_sim_params(args, sim_params)
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
train_cfg.runner.resume = True
ppo_runner, train_cfg = task_registry.make_alg_runner(
    env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
)
path = os.path.join(
    LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies"
)
export_policy_as_jit(ppo_runner.alg.actor_critic, path)
out = os.path.join(path, "policy_lstm_1.pt")
print(f"[export] wrote {out}")
policy = torch.jit.load(out)
y = policy(torch.zeros(1, env_cfg.env.num_observations))
print(f"[export] obs_dim={env_cfg.env.num_observations} action={tuple(y.shape)}")
