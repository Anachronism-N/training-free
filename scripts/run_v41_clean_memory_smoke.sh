#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="${PROMPTS:-$ROOT/prompts/review_artifact_2.txt}"
FRAMES="${FRAMES:-36}"

run_variant() {
    local gpu="$1"
    local gate="$2"
    local tag="clean${gate//./}"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES="$FRAMES" \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        STRUCTURED_MEMORY_ENABLE=1 \
        MEMORY_VALUE_MODE=full \
        MEMORY_READOUT_MODE=clean_only \
        MEMORY_BUDGET=4 \
        MEMORY_SPATIAL_STRIDE=4 \
        MEMORY_LOCAL_DISTANCE=0.08 \
        MEMORY_CORE_WEIGHT=0.5 \
        MEMORY_GATE="$gate" \
        MEMORY_TEMPERATURE=0.1 \
        MEMORY_CONFIDENCE=0.1 \
        MEMORY_LAYER_START=15 \
        MEMORY_LAYER_END=21 \
        RUN_ID="20260719_v41_${tag}" \
        METHOD_TAG="$tag" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 0.05 &
pid_005=$!
run_variant 1 0.10 &
pid_010=$!
run_variant 2 0.20 &
pid_020=$!

status=0
for pid in "$pid_005" "$pid_010" "$pid_020"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
