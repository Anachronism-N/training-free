#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"; PF="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"; PROMPTS="$ROOT/prompts/aba_controlled_3.txt"
run(){ local gpu=$1 id=$2 tag=$3; shift 3; env CUDA_VISIBLE_DEVICES=$gpu PROMPTS="$PROMPTS" FRAMES=120 SEED=0 STRENGTH=0 GATE_LAMBDA=0 RECENT_FRAMES=21 METHOD_TAG=$tag RUN_ID=$id "$@" bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"; }
COMMON=(CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=1 MEMORY_STORAGE_MODE=archive MEMORY_ARCHIVE_MAX_FRAMES=24 MEMORY_ARCHIVE_POLICY=coverage MEMORY_TOP_K_FRAMES=3 MEMORY_RECENT_EXCLUDE_FRAMES=4 MEMORY_FUSION_MODE=convex MEMORY_READOUT_MODE=all MEMORY_GATE=0.075 MEMORY_CONFIDENCE=0.25 MEMORY_TEMPERATURE=0.3 MEMORY_LAYER_START=15 MEMORY_LAYER_END=21 MEMORY_WARMUP_BLOCKS=6 MEMORY_HEAD_ROUTING=static MEMORY_HEAD_LABELS="" MEMORY_POSITION_MODE=none MEMORY_MIN_RETRIEVAL_MARGIN=0.0)
run 0 20260720_v59_pf pf CONFIG_PATH="$PF" STRUCTURED_MEMORY_ENABLE=0 &
run 1 20260720_v59_correct correct "${COMMON[@]}" MEMORY_SELECTION_POLICY=query MEMORY_CONTROL_MODE=normal &
# At A2 start, newest eligible historical frames come from scene B.
run 2 20260720_v59_wrong_b wrong_b "${COMMON[@]}" MEMORY_SELECTION_POLICY=newest MEMORY_CONTROL_MODE=normal &
run 3 20260720_v59_shuffled_v shuffled_v "${COMMON[@]}" MEMORY_SELECTION_POLICY=query MEMORY_CONTROL_MODE=shuffled_v &
run 4 20260720_v59_abstain abstain "${COMMON[@]}" MEMORY_SELECTION_POLICY=query MEMORY_CONTROL_MODE=abstain &
status=0; for p in $(jobs -p); do wait "$p" || status=1; done; exit $status
