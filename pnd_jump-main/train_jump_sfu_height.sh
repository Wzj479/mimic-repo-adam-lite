#!/usr/bin/env bash
# Short FT from SFU-3600: only strengthen jump-height rewards.
# Keep terminate / DR / orientation from adam_lite_jump_sfu.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
LOAD_RUN="${LOAD_RUN:-Jul23_16-17-14_jump_sfu_track}"
CHECKPOINT="${CHECKPOINT:-3600}"
NUM_ENVS="${NUM_ENVS:-2048}"
# Additional iterations beyond resume ckpt (OnPolicyRunner: start + max_iterations).
# 3600 + 600 → ends at 4200.
MAX_ITERATIONS="${MAX_ITERATIONS:-600}"
LOG_FILE="${LOG_FILE:-logs/adam_lite_jump_sfu_height.log}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
mkdir -p logs

echo "[sfu_height] FT from ${LOAD_RUN}/${CHECKPOINT} +${MAX_ITERATIONS} iters"
echo "  only height rewards raised; stability settings unchanged"

python legged_gym/scripts/train.py \
    --task=adam_lite_jump_sfu_height \
    --headless \
    --resume \
    --load_run="${LOAD_RUN}" \
    --checkpoint="${CHECKPOINT}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[sfu_height] done. Prefer sim2sim sweep before picking ckpt (4000 beat 4200)."
echo "  LOAD_RUN=<new_run> CHECKPOINT=4000 bash export_jump_sfu_height.sh"
echo "  bash sim2sim_jump_best.sh"
