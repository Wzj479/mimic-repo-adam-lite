#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
# v3 peak checkpoint (mean reward ~96 around iter 5927)
LOAD_RUN="${LOAD_RUN:-Jul13_18-20-10_jump_motion_tracking_v3}"
CHECKPOINT="${CHECKPOINT:-5800}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"

python legged_gym/scripts/play.py \
    --task=adam_lite_jump \
    --load_run="${LOAD_RUN}" \
    --checkpoint="${CHECKPOINT}"
