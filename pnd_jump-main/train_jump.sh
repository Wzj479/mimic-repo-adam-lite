#!/usr/bin/env bash
# Fresh PPO with zero action-head init + future-ref obs + phase RSI (no teacher).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
LOG_FILE="${LOG_FILE:-logs/adam_lite_jump_train_v4.log}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
mkdir -p logs

echo "Training adam_lite_jump v4 (zero-init + future ref + phase RSI)"
echo "  num_envs=${NUM_ENVS} max_iterations=${MAX_ITERATIONS}"

python legged_gym/scripts/train.py \
    --task=adam_lite_jump \
    --headless \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    2>&1 | tee "${LOG_FILE}"
