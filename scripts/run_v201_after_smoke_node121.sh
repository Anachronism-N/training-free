#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
SMOKE_PID="${1:-4181021}"
while kill -0 "$SMOKE_PID" 2>/dev/null; do sleep 60; done
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=1 RUN_UNIT_TESTS=0 \
    bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh audit-smoke
mkdir -p "$ROOT/runs/v201_head_phase_horizon_sf_screen/status"
touch "$ROOT/runs/v201_head_phase_horizon_sf_screen/status/smoke_ready"
METHODS=sf_native,landmark_all_recent,landmark_all_coverage,landmark_static_top10,landmark_horizon_top10,landmark_horizon_shift_top10,retrieval_all_recent \
    PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=1 \
    GPU_LIST=0,1,2,3,4,5,6,7 RUN_UNIT_TESTS=0 \
    bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh generate32
echo "[v201-node121] assigned methods complete $(date)"
