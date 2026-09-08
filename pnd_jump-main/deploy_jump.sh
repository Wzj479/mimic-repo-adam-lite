#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
# Default to v1 peak checkpoint (stronger dof tracking than v2-2800).
LOAD_RUN="${LOAD_RUN:-Jul11_18-47-58_jump_motion_tracking}"
CHECKPOINT="${CHECKPOINT:-7400}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"

echo "Step 1/2: export JIT policy from Isaac Gym checkpoint (${LOAD_RUN}/${CHECKPOINT})"
LOAD_RUN="${LOAD_RUN}" CHECKPOINT="${CHECKPOINT}" bash play_jump.sh

echo "Step 2/2: run MuJoCo sim2sim"
cd deploy/deploy_mujoco
python deploy_mujoco_jump.py adam_lite_jump.yaml
