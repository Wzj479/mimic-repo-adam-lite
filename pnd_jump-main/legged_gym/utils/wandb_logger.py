"""
Weights & Biases integration for PND RL Gym
This module provides utilities for logging training metrics to wandb
"""

import os
import torch
import wandb
from typing import Dict, Any, Optional
from datetime import datetime


class WandbLogger:
    """
    Wrapper class for Weights & Biases logging
    Integrates with rsl-rl's OnPolicyRunner
    """
    
    def __init__(
        self,
        project: str = "pnd_rl_gym",
        entity: Optional[str] = None,
        experiment_name: str = "test",
        run_name: str = "",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        resume: bool = False,
    ):
        """
        Initialize wandb logger
        
        Args:
            project: wandb project name
            entity: wandb entity (team/username)
            experiment_name: name of the experiment
            run_name: specific name for this run
            config: configuration dictionary to log
            tags: list of tags for the run
            resume: whether resuming from checkpoint
        """
        self.enabled = True
        self.project = project
        self.entity = entity
        
        # Check if wandb is logged in
        if not self._check_wandb_login():
            print("\n" + "="*70)
            print("ERROR: WandB is not logged in!")
            print("="*70)
            print("\nTo use WandB, please login first:")
            print("\n  1. Run: wandb login")
            print("  2. Get your API key from: https://wandb.ai/authorize")
            print("  3. Paste the API key when prompted")
            print("\nAlternatively, set your API key directly:")
            print("  wandb login <your_api_key>")
            print("\nOr to run without WandB, use:")
            print("  WANDB_MODE=disabled python legged_gym/scripts/train.py --task=adam_lite")
            print("="*70 + "\n")
            self.enabled = False
            return
        
        # Build a timestamped run name like Oct08_17-16-23_experiment_run
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  # e.g., Oct08_17-16-23
        if run_name:
            full_run_name = f"{timestamp}_{experiment_name}_{run_name}"
        else:
            full_run_name = f"{timestamp}_{experiment_name}"
            
        # Initialize wandb
        try:
            wandb.init(
                project=project,
                entity=entity,
                name=full_run_name,
                config=config,
                tags=tags or [],
                resume="allow" if resume else None,
                sync_tensorboard=True,  # Also sync tensorboard logs if they exist
            )
            print(f"[WandB] ✓ Successfully initialized logging to project '{project}'")
            print(f"[WandB] ✓ Run name: '{full_run_name}'")
            # Use non-deprecated run URL
            print(f"[WandB] ✓ View run at: {wandb.run.url}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to initialize wandb: {e}")
            print("[WandB] Continuing without wandb logging")
            self.enabled = False
    
    def _check_wandb_login(self) -> bool:
        """
        Check if wandb is logged in
        
        Returns:
            True if logged in, False otherwise
        """
        try:
            # Try to get the API key
            api_key = wandb.api.api_key
            if api_key:
                return True
        except:
            pass
        
        # Check if WANDB_API_KEY environment variable is set
        if os.environ.get('WANDB_API_KEY'):
            return True
        
        # Check if wandb is in offline mode (which is okay)
        if os.environ.get('WANDB_MODE') == 'offline':
            return True
            
        return False
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb
        
        Args:
            data: dictionary of metrics to log
            step: training iteration/step number
        """
        if not self.enabled:
            return
            
        try:
            wandb.log(data, step=step)
        except Exception as e:
            print(f"[WandB] Warning: Failed to log metrics: {e}")
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """
        Watch model for gradient and parameter logging
        
        Args:
            model: PyTorch model to watch
            log_freq: frequency of logging (every N steps)
        """
        if not self.enabled:
            return
            
        try:
            wandb.watch(model, log="all", log_freq=log_freq)
            print("[WandB] Watching model gradients and parameters")
        except Exception as e:
            print(f"[WandB] Warning: Failed to watch model: {e}")
    
    def log_video(self, video_path: str, name: str = "video", step: Optional[int] = None):
        """
        Log video to wandb
        
        Args:
            video_path: path to video file
            name: name for the video in wandb
            step: training iteration/step number
        """
        if not self.enabled:
            return
            
        try:
            wandb.log({name: wandb.Video(video_path)}, step=step)
        except Exception as e:
            print(f"[WandB] Warning: Failed to log video: {e}")
    
    def save_model(self, model_path: str, aliases: Optional[list] = None):
        """
        Save model checkpoint to wandb
        
        Args:
            model_path: path to model checkpoint file
            aliases: list of aliases for the artifact (e.g., ["latest", "best"])
        """
        if not self.enabled:
            return
            
        try:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="Model checkpoint"
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact, aliases=aliases or ["latest"])
            print(f"[WandB] Saved model checkpoint: {model_path}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to save model: {e}")
    
    def finish(self):
        """Finish the wandb run"""
        if not self.enabled:
            return
            
        try:
            wandb.finish()
            print("[WandB] Finished logging")
        except Exception as e:
            print(f"[WandB] Warning: Failed to finish wandb: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'enabled') and self.enabled:
            try:
                wandb.finish()
            except:
                pass


def create_wandb_logger_from_config(train_cfg: Dict[str, Any], env_cfg: Optional[Dict[str, Any]] = None) -> Optional[WandbLogger]:
    """
    Create a WandbLogger from training configuration
    
    Args:
        train_cfg: training configuration dictionary
        env_cfg: environment configuration dictionary (optional)
        
    Returns:
        WandbLogger instance if enabled, None otherwise
    """
    runner_cfg = train_cfg.get('runner', {})
    
    # Check if wandb is enabled
    if not runner_cfg.get('use_wandb', False):
        print("[WandB] Logging disabled in config")
        return None
    
    # Prepare config to log
    config = {
        'train_config': train_cfg,
    }
    if env_cfg is not None:
        config['env_config'] = env_cfg
    
    # Create logger
    logger = WandbLogger(
        project=runner_cfg.get('wandb_project', 'pnd_rl_gym'),
        entity=runner_cfg.get('wandb_entity', None),
        experiment_name=runner_cfg.get('experiment_name', 'test'),
        run_name=runner_cfg.get('run_name', ''),
        config=config,
        tags=runner_cfg.get('wandb_tags', []),
        resume=runner_cfg.get('resume', False),
    )
    
    return logger

