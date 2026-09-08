#!/usr/bin/env bash
# Resume adam_lite_jump v3 from a checkpoint until remaining iterations complete.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
LOAD_RUN="${LOAD_RUN:-Jul13_16-48-29_jump_motion_tracking_v3}"
CHECKPOINT="${CHECKPOINT:-2800}"
# rsl_rl runs: range(current_iter, current_iter + max_iterations)
# From 2800 -> 9000 needs 6200 more iterations.
MAX_ITERATIONS="${MAX_ITERATIONS:-6200}"
NUM_ENVS="${NUM_ENVS:-2048}"
LOG_FILE="${LOG_FILE:-logs/adam_lite_jump_train_v3_resume.log}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
mkdir -p logs

echo "Resuming adam_lite_jump v3"
echo "  load_run=${LOAD_RUN}"
echo "  checkpoint=${CHECKPOINT}"
echo "  additional_iterations=${MAX_ITERATIONS}  (target total ~$((CHECKPOINT + MAX_ITERATIONS)))"
echo "  num_envs=${NUM_ENVS}"
echo "  log_file=${LOG_FILE}"

python legged_gym/scripts/train.py \
    --task=adam_lite_jump \
    --headless \
    --resume \
    --load_run="${LOAD_RUN}" \
    --checkpoint="${CHECKPOINT}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    2>&1 | tee -a "${LOG_FILE}"
