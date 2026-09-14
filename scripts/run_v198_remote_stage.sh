#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action required}"
RANK="${2:-0}"
NODES="${3:-1}"
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
NODE_RANK="$RANK" NUM_NODES="$NODES" GPU_LIST=0,1,2,3,4,5,6,7 \
    bash scripts/run_v198_long60_evaluation.sh "$ACTION"
