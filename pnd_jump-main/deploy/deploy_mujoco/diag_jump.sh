#!/usr/bin/env bash
# Headless MuJoCo jump diagnostics (no viewer).
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-pnd_jump}"
DURATION="${DURATION:-12}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "$(dirname "$0")"
python deploy_mujoco_jump.py adam_lite_jump.yaml \
  --headless \
  --duration "${DURATION}" \
  --no-yaw-assist
