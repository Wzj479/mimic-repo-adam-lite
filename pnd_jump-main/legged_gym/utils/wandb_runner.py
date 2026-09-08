"""
Enhanced OnPolicyRunner with WandB integration
This module wraps rsl-rl's OnPolicyRunner to add wandb logging capabilities
"""

from rsl_rl.runners import OnPolicyRunner
from .wandb_logger import WandbLogger
from typing import Optional, Dict, Any
import os


class WandbOnPolicyRunner(OnPolicyRunner):
    """
    Extended OnPolicyRunner that integrates with WandB for logging
    Inherits from rsl-rl's OnPolicyRunner and adds wandb logging hooks
    """
    
    def __init__(self, env, train_cfg_dict, log_dir, device='cpu', wandb_logger: Optional[WandbLogger] = None):
        """
        Initialize the runner with wandb logger
        
        Args:
            env: vectorized environment
            train_cfg_dict: training configuration dictionary
            log_dir: directory for saving logs and models
            device: device to run on
            wandb_logger: WandbLogger instance for logging to wandb
        """
        super().__init__(env, train_cfg_dict, log_dir, device)
        self.wandb_logger = wandb_logger
        
        # Watch the model if wandb is enabled
        if self.wandb_logger and self.wandb_logger.enabled:
            self.wandb_logger.watch_model(self.alg.actor_critic, log_freq=100)
    
    def log(self, locs, width=80, pad=35):
        """
        Override log method to add wandb logging alongside tensorboard
        
        Args:
            locs: local variables from the training loop
            width: width of the log output
            pad: padding for the log output
        """
        # Call parent log method for tensorboard logging
        super().log(locs, width, pad)
        
        # Additionally log to wandb
        if self.wandb_logger and self.wandb_logger.enabled:
            self._log_to_wandb(locs)
    
    def save(self, path, infos=None):
        """
        Override save to also save checkpoints to wandb
        
        Args:
            path: path to save the model
            infos: additional information to save
        """
        # Call parent save method
        super().save(path, infos)
        
        # Also save to wandb at save intervals
        if self.wandb_logger and self.wandb_logger.enabled:
            # Only save every save_interval or final model
            if self.current_learning_iteration % self.save_interval == 0:
                self.wandb_logger.save_model(path, aliases=["latest"])
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Override learn to ensure wandb is properly closed at the end
        
        Args:
            num_learning_iterations: number of iterations to train
            init_at_random_ep_len: whether to initialize at random episode length
        """
        # Call parent learn method - it will use our overridden log method
        super().learn(num_learning_iterations, init_at_random_ep_len)
        
        # Finish wandb logging
        if self.wandb_logger and self.wandb_logger.enabled:
            # Save final model to wandb with special alias
            final_model_path = os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt')
            if os.path.exists(final_model_path):
                self.wandb_logger.save_model(final_model_path, aliases=["final", "latest"])
            self.wandb_logger.finish()
    
    def _log_to_wandb(self, locs: Dict[str, Any]):
        """
        Log metrics to wandb
        
        Args:
            locs: local variables from the learn loop
        """
        if not self.wandb_logger or not self.wandb_logger.enabled:
            return
        
        import numpy as np
        
        # Get iteration from locs or use current_learning_iteration
        iteration = locs.get('it', self.current_learning_iteration)
        
        wandb_dict = {}
        
        # Training metrics
        if locs.get('mean_value_loss') is not None:
            wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        if locs.get('mean_surrogate_loss') is not None:
            wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        
        # Learning rate
        if hasattr(self.alg, 'learning_rate'):
            wandb_dict['Train/learning_rate'] = self.alg.learning_rate
        
        # Timing
        if locs.get('collection_time') is not None:
            wandb_dict['Time/collection'] = locs['collection_time']
        if locs.get('learn_time') is not None:
            wandb_dict['Time/learn'] = locs['learn_time']
        
        # Episode statistics
        rewbuffer = locs.get('rewbuffer', [])
        lenbuffer = locs.get('lenbuffer', [])
        
        if len(rewbuffer) > 0:
            wandb_dict['Train/mean_reward'] = np.mean(rewbuffer)
            wandb_dict['Train/mean_episode_length'] = np.mean(lenbuffer)
            wandb_dict['Train/mean_reward/time'] = np.mean(rewbuffer) / max(np.mean(lenbuffer), 1)
        
        # Episode info
        ep_infos = locs.get('ep_infos', [])
        if len(ep_infos) > 0:
            for key in ep_infos[0]:
                infotensor = [ep_info[key] for ep_info in ep_infos]
                try:
                    # Convert tensors to cpu if needed
                    if hasattr(infotensor[0], 'cpu'):
                        infotensor = [x.cpu().item() if hasattr(x, 'item') else x.cpu().numpy() for x in infotensor]
                    value = np.mean(infotensor)
                    wandb_dict[f'Episode/{key}'] = value
                except Exception as e:
                    # Skip if conversion fails
                    pass
        
        # Environment info
        infos = locs.get('infos', {})
        if isinstance(infos, dict):
            for key, value in infos.items():
                if key != 'episode':
                    try:
                        if hasattr(value, 'mean'):
                            wandb_dict[f'Info/{key}'] = value.mean().item()
                        elif isinstance(value, (int, float)):
                            wandb_dict[f'Info/{key}'] = value
                    except:
                        pass
        
        # Log all metrics
        if wandb_dict:
            self.wandb_logger.log(wandb_dict, step=iteration)

