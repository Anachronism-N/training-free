#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$ROOT/prompts/review_artifact_2.txt"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local method_tag="$3"
    local enabled="$4"
    local readout_mode="$5"
    local gate="$6"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES=120 \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        STRUCTURED_MEMORY_ENABLE="$enabled" \
        MEMORY_VALUE_MODE=full \
        MEMORY_READOUT_MODE="$readout_mode" \
        MEMORY_BUDGET=4 \
        MEMORY_SPATIAL_STRIDE=4 \
        MEMORY_LOCAL_DISTANCE=0.08 \
        MEMORY_CORE_WEIGHT=0.5 \
        MEMORY_GATE="$gate" \
        MEMORY_TEMPERATURE=0.1 \
        MEMORY_CONFIDENCE=0.1 \
        MEMORY_LAYER_START=15 \
        MEMORY_LAYER_END=21 \
        RUN_ID="$run_id" \
        METHOD_TAG="$method_tag" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260719_v42_pf_30s pf_30s 0 all 0 &
pid_pf=$!
run_variant 1 20260719_v42_all005_30s all005_30s 1 all 0.05 &
pid_all=$!
run_variant 2 20260719_v42_clean010_30s clean010_30s 1 clean_only 0.10 &
pid_clean=$!

status=0
for pid in "$pid_pf" "$pid_all" "$pid_clean"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
