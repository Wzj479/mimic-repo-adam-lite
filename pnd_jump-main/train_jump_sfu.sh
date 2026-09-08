#!/usr/bin/env bash
# SFU multi-hop tracking finetune (not in-place).
# Same 82-dim obs as Jul14; stronger landing/orientation + earlier fall reset.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
LOAD_RUN="${LOAD_RUN:-Jul14_13-31-50_jump_teacher_ft}"
CHECKPOINT="${CHECKPOINT:-3000}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
LOG_FILE="${LOG_FILE:-logs/adam_lite_jump_sfu.log}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
mkdir -p logs

echo "[sfu] finetune adam_lite_jump_sfu from ${LOAD_RUN}/${CHECKPOINT}"
echo "  goal=follow SFU traveling hops  num_envs=${NUM_ENVS} max_iterations=${MAX_ITERATIONS}"

python legged_gym/scripts/train.py \
    --task=adam_lite_jump_sfu \
    --headless \
    --resume \
    --load_run="${LOAD_RUN}" \
    --checkpoint="${CHECKPOINT}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[sfu] done. Export / deploy:"
echo "  LOAD_RUN=<new_run> CHECKPOINT=<iter> bash export_jump_sfu.sh"
echo "  python deploy/deploy_mujoco/deploy_mujoco_jump.py adam_lite_jump_sfu.yaml"
echo "  Metrics to watch: mean |z_err|, yaw/pitch at land, multi-hop survival (not XY-vs-origin)."
