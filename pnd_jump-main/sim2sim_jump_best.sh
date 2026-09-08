#!/usr/bin/env bash
# Run MuJoCo sim2sim with the current recommended policy (SFU-height @ 4000).
# Usage:
#   bash sim2sim_jump_best.sh              # viewer
#   bash sim2sim_jump_best.sh --headless
#   bash sim2sim_jump_best.sh --record auto
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pnd_jump}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
POL_DIR="${ROOT}/logs/adam_lite_jump/exported/policies"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export DISPLAY="${DISPLAY:-:1}"

# Prefer named snapshot; fall back to generic height export.
if [[ -f "${POL_DIR}/policy_lstm_sfu_height_4000.pt" ]]; then
  cp -f "${POL_DIR}/policy_lstm_sfu_height_4000.pt" "${POL_DIR}/policy_lstm_sfu_height.pt"
fi

cd "${ROOT}/deploy/deploy_mujoco"
echo "[sim2sim_best] policy=${POL_DIR}/policy_lstm_sfu_height.pt"
echo "[sim2sim_best] config=adam_lite_jump_sfu_height.yaml"
python deploy_mujoco_jump.py adam_lite_jump_sfu_height.yaml "$@"
