#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt METHODS=retrieval_head_only NODE_RANK=0 \
    NUM_NODES=1 GPU_LIST=4,5,6,7 RUN_UNIT_TESTS=0 \
    bash scripts/run_v190_head_phase_causal_screen_32gpu.sh generate32
