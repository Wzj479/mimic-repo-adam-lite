#!/usr/bin/env bash
# Phase 0: Isaac reference-PD diagnostic (action=0 → q_des=q_ref).
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
NUM_ENVS="${NUM_ENVS:-64}"
DIAG_DURATION_S="${DIAG_DURATION_S:-12}"

export DIAG_DURATION_S

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"

python legged_gym/scripts/diag_reference_pd.py \
    --task=adam_lite_jump \
    --headless \
    --num_envs="${NUM_ENVS}"
