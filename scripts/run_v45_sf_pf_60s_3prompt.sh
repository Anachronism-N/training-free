#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
SF_CONFIG="$ROOT/third_party/Self-Forcing/configs/self_forcing_dmd.yaml"
PROMPTS="$ROOT/prompts/review_3_prompts.txt"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local tag="$3"
    shift 3
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES=240 \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        RUN_ID="$run_id" \
        METHOD_TAG="$tag" \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260720_v45_sf_native_60s sf_native_60s \
    CONFIG_PATH="$SF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_sf=$!

run_variant 1 20260720_v45_sf_pf_60s sf_pf_60s \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_pf=$!

run_variant 2 20260720_v45_sf_pf_ours_60s sf_pf_ours_60s \
    CONFIG_PATH="$PF_CONFIG" \
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
    MEMORY_HEAD_LABELS=1,2 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 &
pid_ours=$!

status=0
for pid in "$pid_sf" "$pid_pf" "$pid_ours"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
