#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$ROOT/prompts/review_motion_scene_2.txt"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local tag="$3"
    shift 3
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES=120 \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        RUN_ID="$run_id" \
        METHOD_TAG="$tag" \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260719_v43_pf_30s pf_30s \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_pf=$!

run_variant 1 20260719_v43_conf_30s conf_30s \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=compressed \
    MEMORY_READOUT_MODE=clean_only \
    MEMORY_GATE=0.10 \
    MEMORY_CONFIDENCE=0.10 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 &
pid_conf=$!

run_variant 2 20260719_v43_archive_query_30s archive_query_30s \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=archive \
    MEMORY_ARCHIVE_MAX_FRAMES=64 \
    MEMORY_TOP_K_FRAMES=1 \
    MEMORY_RECENT_EXCLUDE_FRAMES=4 \
    MEMORY_SELECTION_POLICY=query \
    MEMORY_FUSION_MODE=convex \
    MEMORY_READOUT_MODE=clean_only \
    MEMORY_GATE=0.20 \
    MEMORY_CONFIDENCE=0.10 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 &
pid_query=$!

run_variant 3 20260719_v43_archive_lag5_30s archive_lag5_30s \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=archive \
    MEMORY_ARCHIVE_MAX_FRAMES=64 \
    MEMORY_TOP_K_FRAMES=1 \
    MEMORY_RECENT_EXCLUDE_FRAMES=4 \
    MEMORY_SELECTION_POLICY=newest \
    MEMORY_FUSION_MODE=convex \
    MEMORY_READOUT_MODE=clean_only \
    MEMORY_GATE=0.20 \
    MEMORY_CONFIDENCE=0.10 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 &
pid_lag=$!

status=0
for pid in "$pid_pf" "$pid_conf" "$pid_query" "$pid_lag"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
