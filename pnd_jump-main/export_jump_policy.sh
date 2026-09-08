#!/usr/bin/env bash
# Fast-export LSTM JIT from a checkpoint without full play visualization.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
# Legacy teacher export → policy_lstm_1.pt (Jul14 teacher FT).
# Current recommended deploy is SFU-height 4000; see export_jump_sfu_height.sh.
export LOAD_RUN="${LOAD_RUN:-Jul14_13-31-50_jump_teacher_ft}"
export CHECKPOINT="${CHECKPOINT:-3000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")"
python export_jump_policy.py
