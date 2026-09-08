from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_config import AdamLite12dofRoughCfg, AdamLite12dofRoughCfgPPO
from legged_gym.envs.adam_lite_12dof.adam_lite_12dof_env import AdamLite12dofRobot
from legged_gym.envs.adam_lite_jump.adam_lite_jump_config import (
    AdamLiteJumpCfg,
    AdamLiteJumpCfgPPO,
    AdamLiteJumpCfgPPOFinetune,
    AdamLiteJumpSfuCfg,
    AdamLiteJumpSfuCfgPPO,
    AdamLiteJumpSfuHeightCfg,
    AdamLiteJumpSfuHeightCfgPPO,
)
from legged_gym.envs.adam_lite_jump.adam_lite_jump_env import AdamLiteJumpRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "adam_lite_12dof", AdamLite12dofRobot, AdamLite12dofRoughCfg(), AdamLite12dofRoughCfgPPO())
task_registry.register( "adam_lite_jump", AdamLiteJumpRobot, AdamLiteJumpCfg(), AdamLiteJumpCfgPPO())
# Same env, lower lr / entropy for fine-tunes that resume teacher weights.
task_registry.register(
    "adam_lite_jump_bc_ft",
    AdamLiteJumpRobot,
    AdamLiteJumpCfg(),
    AdamLiteJumpCfgPPOFinetune(),
)
# SFU traveling multi-hop track: landing / orientation / early-fall cut.
task_registry.register(
    "adam_lite_jump_sfu",
    AdamLiteJumpRobot,
    AdamLiteJumpSfuCfg(),
    AdamLiteJumpSfuCfgPPO(),
)
# Short height FT from SFU-3600: only root_height / root_vz rewards raised.
task_registry.register(
    "adam_lite_jump_sfu_height",
    AdamLiteJumpRobot,
    AdamLiteJumpSfuHeightCfg(),
    AdamLiteJumpSfuHeightCfgPPO(),
)
