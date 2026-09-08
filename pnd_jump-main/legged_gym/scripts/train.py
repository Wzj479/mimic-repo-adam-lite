import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, create_wandb_logger_from_config, class_to_dict, save_code_snapshot
from legged_gym.utils.wandb_runner import WandbOnPolicyRunner
import torch


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Initialize wandb if enabled
    wandb_logger = None
    if hasattr(train_cfg.runner, 'use_wandb') and train_cfg.runner.use_wandb:
        print("\n" + "="*50)
        print("Initializing Weights & Biases logging...")
        print("="*50 + "\n")
        
        # Convert configs to dict for logging
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(env_cfg)
        
        # Create wandb logger
        wandb_logger = create_wandb_logger_from_config(train_cfg_dict, env_cfg_dict)
        
        # Replace the runner with WandbOnPolicyRunner if wandb is enabled
        if wandb_logger and wandb_logger.enabled:
            from rsl_rl.env import VecEnv
            # Create new runner with wandb integration
            ppo_runner = WandbOnPolicyRunner(
                env=env,
                train_cfg_dict=train_cfg_dict,
                log_dir=ppo_runner.log_dir,
                device=ppo_runner.device,
                wandb_logger=wandb_logger
            )
            
            # Save code snapshot for wandb runs (already done in task_registry for regular runs)
            # This ensures wandb runs also have code snapshots
            # Note: Code snapshot was already saved in task_registry.make_alg_runner
            # Load checkpoint if resuming
            if train_cfg.runner.resume:
                from legged_gym.utils import get_load_path
                resume_path = get_load_path(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'logs', train_cfg.runner.experiment_name),
                    load_run=train_cfg.runner.load_run,
                    checkpoint=train_cfg.runner.checkpoint
                )
                print(f"Loading model from: {resume_path}")
                ppo_runner.load(resume_path)
            
            print("\n" + "="*50)
            print("WandB logging enabled!")
            try:
                import wandb
                print(f"View run at: {wandb.run.url}")
            except Exception:
                print(f"View project at: https://wandb.ai/{wandb_logger.entity or ''}/{wandb_logger.project}")
            print("="*50 + "\n")
    
    # Start training
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
