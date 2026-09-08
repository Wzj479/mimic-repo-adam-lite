#!/usr/bin/env python3
"""Export adam_lite_jump_sfu_height (82-dim) LSTM JIT → policy_lstm_sfu_height.pt."""
import os
import shutil
import sys

import isaacgym  # noqa: F401
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401
from legged_gym.utils import get_args, task_registry, export_policy_as_jit, get_load_path, class_to_dict
from legged_gym.utils.helpers import update_cfg_from_args, parse_sim_params
from rsl_rl.modules import ActorCriticRecurrent

LOAD_RUN = os.environ.get("LOAD_RUN")
CHECKPOINT = os.environ.get("CHECKPOINT", "4000")
LOAD_RUN = LOAD_RUN or "Jul24_11-41-12_jump_sfu_height"
assert LOAD_RUN, "Set LOAD_RUN"

sys.argv = [
    "export_jump_sfu_height.py",
    "--task=adam_lite_jump_sfu_height",
    f"--load_run={LOAD_RUN}",
    f"--checkpoint={CHECKPOINT}",
    "--num_envs=1",
    "--headless",
]
args = get_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
env_cfg.env.num_envs = 1
_ = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

policy_cfg = class_to_dict(train_cfg.policy)
actor_critic = ActorCriticRecurrent(
    env.num_obs,
    env.num_privileged_obs,
    env.num_actions,
    **policy_cfg,
).to(args.rl_device)

root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
ckpt_path = get_load_path(root, load_run=LOAD_RUN, checkpoint=int(CHECKPOINT))
print(f"[export-sfu-height] loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=args.rl_device)
actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
actor_critic.eval()

out_dir = os.path.join(root, "exported", "policies")
export_policy_as_jit(actor_critic, out_dir)
src = os.path.join(out_dir, "policy_lstm_1.pt")
dst = os.path.join(out_dir, "policy_lstm_sfu_height.pt")
shutil.copy2(src, dst)
jul14 = os.path.join(out_dir, "policy_lstm_jul14_teacher_ft_3000.pt")
if os.path.isfile(jul14):
    shutil.copy2(jul14, src)
    print(f"[export-sfu-height] restored baseline {src}")
# Keep preferred stable SFU-3600 as policy_lstm_sfu.pt if present.
sfu3600 = os.path.join(out_dir, "policy_lstm_sfu_3600.pt")
if os.path.isfile(sfu3600):
    shutil.copy2(sfu3600, os.path.join(out_dir, "policy_lstm_sfu.pt"))
print(f"[export-sfu-height] wrote {dst}")
y = torch.jit.load(dst)(torch.zeros(1, env.num_obs))
print(f"[export-sfu-height] obs_dim={env.num_obs} action={tuple(y.shape)}")
