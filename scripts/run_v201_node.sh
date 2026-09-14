#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action}"
RANK="${2:-0}"
NODES="${3:-2}"
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK="$RANK" NUM_NODES="$NODES" \
    GPU_LIST=0,1,2,3,4,5,6,7 RUN_UNIT_TESTS=0 \
    bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh "$ACTION"
