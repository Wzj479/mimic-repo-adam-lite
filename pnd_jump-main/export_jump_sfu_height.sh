#!/usr/bin/env bash
# Export SFU-height JIT → policy_lstm_sfu_height.pt
# Preferred (sim2sim best): Jul24_11-41-12_jump_sfu_height / 4000
# Final train iter was 4200 but 4000 wins on MuJoCo (height↑ + stable).
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-pnd_jump}"
LOAD_RUN="${LOAD_RUN:-Jul24_11-41-12_jump_sfu_height}"
CHECKPOINT="${CHECKPOINT:-4000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
export LOAD_RUN CHECKPOINT
python export_jump_sfu_height.py
