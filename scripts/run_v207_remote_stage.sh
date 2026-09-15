#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action}"
RANK="${2:-0}"
NODES="${3:-3}"
GPUS="${4:-0,1,2,3,4,5,6,7}"
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
OUT="$ROOT/runs/v207_horizon_seed_replication"
METHODS_SET=sf_native,landmark_all_recent,landmark_all_coverage,landmark_static_top10,landmark_horizon_top10,landmark_horizon_shift_top10,retrieval_all_recent,retrieval_all_coverage,retrieval_static_top10,retrieval_horizon_top10,retrieval_horizon_shift_top10
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
checkpoint=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
[[ -s /tmp/self_forcing_dmd.pt ]] && checkpoint=/tmp/self_forcing_dmd.pt
case "$ACTION" in
    vbench-*)
        STAGE="${ACTION#vbench-}"
        V201_OUT_ROOT="$OUT" RUN_ROOT="$OUT/screen32" \
            NODE_RANK="$RANK" NUM_NODES="$NODES" GPU_LIST="$GPUS" \
            bash scripts/run_v201_vbench_long.sh "$STAGE"
        ;;
    *)
        V201_OUT_ROOT="$OUT" V201_EXPLORATORY_SEED=1 SEED=20700 \
            METHODS="$METHODS_SET" PF_CHECKPOINT="$checkpoint" \
            NODE_RANK="$RANK" NUM_NODES="$NODES" GPU_LIST="$GPUS" \
            RUN_UNIT_TESTS=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh "$ACTION"
        ;;
esac
