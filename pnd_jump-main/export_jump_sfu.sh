#!/usr/bin/env bash
# Export SFU track JIT → policy_lstm_sfu.pt
# Preferred: Jul23_16-17-14_jump_sfu_track / 3600
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-pnd_jump}"
LOAD_RUN="${LOAD_RUN:-Jul23_16-17-14_jump_sfu_track}"
CHECKPOINT="${CHECKPOINT:-3600}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
export LOAD_RUN CHECKPOINT
python export_jump_sfu.py
