"""Policy initialization helpers for residual jump training."""
from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_action_mean_head(actor_critic, verbose=True):
    """Zero the final Linear of the actor MLP so mu(o)≈0 (q_des≈q_ref)."""
    actor = actor_critic.actor
    last = None
    if isinstance(actor, nn.Sequential):
        for mod in reversed(list(actor)):
            if isinstance(mod, nn.Linear):
                last = mod
                break
    elif isinstance(actor, nn.Linear):
        last = actor

    if last is None:
        raise RuntimeError("Could not find actor mean Linear layer to zero-init")

    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()

    if verbose:
        print(
            "[policy_init] zeroed actor mean head "
            f"(weight {tuple(last.weight.shape)})"
        )


def set_action_std(actor_critic, std_value, verbose=True):
    """Force exploration std (overrides whatever was in a checkpoint)."""
    if not hasattr(actor_critic, "std"):
        raise RuntimeError("actor_critic has no std parameter")
    with torch.no_grad():
        actor_critic.std[:] = float(std_value)
    if verbose:
        print(
            f"[policy_init] set std={float(std_value):.4f} "
            f"(tensor mean={actor_critic.std.mean().item():.4f})"
        )


def _cfg_get(runner_cfg, name, default=None):
    if isinstance(runner_cfg, dict):
        return runner_cfg.get(name, default)
    return getattr(runner_cfg, name, default)


def apply_policy_init(actor_critic, train_cfg, resumed=False, verbose=True):
    """Apply optional zero-init / std reset from train_cfg.runner flags."""
    runner_cfg = train_cfg.runner if hasattr(train_cfg, "runner") else train_cfg["runner"]

    zero_init = bool(_cfg_get(runner_cfg, "zero_init_action_head", False))
    force_zero = bool(_cfg_get(runner_cfg, "force_zero_init_action_head", False))
    if (zero_init and not resumed) or force_zero:
        zero_init_action_mean_head(actor_critic, verbose=verbose)

    reset_std = _cfg_get(runner_cfg, "reset_std_on_load", None)
    if reset_std is not None:
        set_action_std(actor_critic, reset_std, verbose=verbose)
    elif not resumed:
        policy_cfg = train_cfg.policy if hasattr(train_cfg, "policy") else train_cfg["policy"]
        init_std = _cfg_get(policy_cfg, "init_noise_std", None)
        if init_std is not None:
            set_action_std(actor_critic, init_std, verbose=verbose)

    if verbose and hasattr(actor_critic, "std"):
        print(f"[policy_init] std now: {actor_critic.std.detach().cpu().numpy()}")
