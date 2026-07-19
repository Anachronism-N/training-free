#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$ROOT/prompts/review_motion_scene_2.txt"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local tag="$3"
    local labels="$4"
    local gate="$5"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES=120 \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        STRUCTURED_MEMORY_ENABLE=1 \
        MEMORY_STORAGE_MODE=archive \
        MEMORY_ARCHIVE_MAX_FRAMES=64 \
        MEMORY_TOP_K_FRAMES=1 \
        MEMORY_RECENT_EXCLUDE_FRAMES=4 \
        MEMORY_SELECTION_POLICY=query \
        MEMORY_FUSION_MODE=convex \
        MEMORY_READOUT_MODE=clean_only \
        MEMORY_GATE="$gate" \
        MEMORY_CONFIDENCE=0.10 \
        MEMORY_HEAD_LABELS="$labels" \
        MEMORY_LAYER_START=15 \
        MEMORY_LAYER_END=21 \
        RUN_ID="$run_id" \
        METHOD_TAG="$tag" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260720_v44_stable1_30s stable1_30s 1 0.20 &
pid_stable1=$!
run_variant 1 20260720_v44_stable12_30s stable12_30s 1,2 0.20 &
pid_stable12=$!
run_variant 2 20260720_v44_stable1_g30_30s stable1_g30_30s 1 0.30 &
pid_stable1_g30=$!

status=0
for pid in "$pid_stable1" "$pid_stable12" "$pid_stable1_g30"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
