#!/usr/bin/env bash
# Native-MuJoCo sim2sim with the current best MimicLite jump policy.
# Best weight: sequential PPO-ROA finetune checkpoint_1000
# (policy-adam_lite_jump_best.onnx). Do not use landing-FT checkpoint_1600.
#
# Default: H0 train-matched physics + deploy ankle PD.
# Harder (no retraining):  bash sim2sim_jump_best.sh --mode motor
#                          bash sim2sim_jump_best.sh --mode realish
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/active-adaptation/venv/mjlab/.venv/bin/python"
ONNX="$ROOT/active-adaptation/projects/mimic-lite/scripts/exports/AdamLiteJumpTrack/policy-adam_lite_jump_best.onnx"
cd "$ROOT/active-adaptation"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-disabled}"
exec "$PY" projects/mimic-lite/scripts/sim2sim_adam_lite_jump.py \
  --onnx "$ONNX" \
  --pd deploy \
  "$@"
