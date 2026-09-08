# WandB Integration for pndbotics rl Gym

This document describes the Weights & Biases (WandB) integration for experiment tracking and visualization.

## Quick Start

### 1. Install and Login

```bash
# Install dependencies
pip install -e .

# Login to WandB
wandb login
```

### 2. Train with WandB

```bash
# Enable WandB with command line flag
python legged_gym/scripts/train.py --task=adam_lite --wandb

# Specify project name
python legged_gym/scripts/train.py --task=adam_lite --wandb --wandb_project=my_project

# Full example
python legged_gym/scripts/train.py \
    --task=adam_lite \
    --wandb \
    --wandb_project=pndbotics_experiments \
    --experiment_name=rough_terrain \
    --run_name=baseline \
    --num_envs=4096 \
    --max_iterations=2000 \
    --headless
```

## Configuration

### Command Line Arguments

- `--wandb`: Enable WandB logging
- `--wandb_project <name>`: Set WandB project name
- `--wandb_entity <name>`: Set WandB entity (team/username)

### Config File

Add to your robot config class:

```python
class MyRobotCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # ... other settings ...
        
        # WandB configuration
        use_wandb = True
        wandb_project = 'my_project'
        wandb_entity = None  # optional
        wandb_tags = ['experiment', 'baseline']  # optional
```

## What Gets Logged

### Training Metrics
- `Loss/value_function` - Value function loss
- `Loss/surrogate` - Surrogate loss
- `Train/learning_rate` - Learning rate
- `Train/mean_reward` - Mean reward per episode
- `Train/mean_episode_length` - Mean episode length

### Time Statistics
- `Time/collection` - Data collection time
- `Time/learn` - Learning time

### Episode Info
- `Episode/*` - Various episode statistics

### Model Checkpoints
- Models are automatically saved to WandB Artifacts

## Testing the Integration

Run the test script to verify WandB is properly configured:

```bash
python scripts/test_wandb.py
```

## Disabling WandB

### Temporarily disable
```bash
WANDB_MODE=disabled python legged_gym/scripts/train.py --task=adam_lite
```

### Use offline mode
```bash
WANDB_MODE=offline python legged_gym/scripts/train.py --task=adam_lite --wandb
```

## Implementation Details

### Files Modified
- `setup.py` - Added wandb dependency
- `legged_gym/envs/base/legged_robot_config.py` - Added WandB config options
- `legged_gym/utils/helpers.py` - Added command line argument parsing
- `legged_gym/scripts/train.py` - Integrated WandB into training loop
- `legged_gym/utils/__init__.py` - Exported WandB utilities

### Files Created
- `legged_gym/utils/wandb_logger.py` - WandB logger wrapper class
- `legged_gym/utils/wandb_runner.py` - Extended OnPolicyRunner with WandB support
- `scripts/test_wandb.py` - Test script for verifying installation

### Architecture

```
train.py
    ↓
WandbOnPolicyRunner (extends rsl_rl.OnPolicyRunner)
    ↓
WandbLogger
    ↓
WandB API
```

The integration is designed to:
- Be non-intrusive (original code works without WandB)
- Coexist with TensorBoard logging
- Handle errors gracefully (continues training if WandB fails)
- Support both command line and config file configuration

## Troubleshooting

### Not logged in
```bash
wandb login --relogin
```

### Network issues
```bash
export WANDB_MODE=offline
# Train, then later sync:
wandb sync wandb/offline-run-xxx
```

### Disable WandB
```bash
# Don't use --wandb flag
# Or set in config:
use_wandb = False
```

## Resources

- WandB Documentation: https://docs.wandb.ai/
- WandB Python API: https://github.com/wandb/wandb
- rsl-rl Library: https://github.com/leggedrobotics/rsl_rl

## Features

✅ Easy enable/disable via command line or config  
✅ Automatic logging of training metrics  
✅ Model checkpoint versioning with Artifacts  
✅ Coexists with TensorBoard  
✅ Graceful error handling  
✅ Support for resume training  
✅ Offline mode support  
✅ Team collaboration features  

## Example Workflow

```bash
# 1. Login (first time only)
wandb login

# 2. Start training
python legged_gym/scripts/train.py --task=adam_lite --wandb --wandb_project=adam_lite_locomotion

# 3. View results in browser (link shown in console)

# 4. Resume if needed
python legged_gym/scripts/train.py --task=adam_lite --wandb --resume
```

## Summary

WandB integration provides powerful experiment tracking and visualization capabilities for your reinforcement learning training. It's designed to be easy to use while remaining flexible and non-intrusive to existing workflows.

