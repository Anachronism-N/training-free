#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action}"
RANK="${2:-0}"
NODES="${3:-4}"
GPUS="${4:-0,1,2,3,4,5,6,7}"
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
checkpoint=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
[[ -s /tmp/self_forcing_dmd.pt ]] && checkpoint=/tmp/self_forcing_dmd.pt
case "$ACTION" in
    vbench-*)
        NODE_RANK="$RANK" NUM_NODES="$NODES" GPU_LIST="$GPUS" \
            bash scripts/run_v207_vbench_long.sh "${ACTION#vbench-}"
        ;;
    *)
        PF_CHECKPOINT="$checkpoint" NODE_RANK="$RANK" NUM_NODES="$NODES" \
            GPU_LIST="$GPUS" RUN_UNIT_TESTS=0 \
            bash scripts/run_v207_context_budget_phase_screen_32gpu.sh "$ACTION"
        ;;
esac
