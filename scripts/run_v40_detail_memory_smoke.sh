#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="${PROMPTS:-$ROOT/prompts/smoke_identity_motion.txt}"
FRAMES="${FRAMES:-36}"

run_variant() {
    local gpu="$1"
    local gate="$2"
    local tag="detail${gate//./}"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES="$FRAMES" \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        STRUCTURED_MEMORY_ENABLE=1 \
        MEMORY_VALUE_MODE=spatial_detail \
        MEMORY_BUDGET=4 \
        MEMORY_SPATIAL_STRIDE=4 \
        MEMORY_LOCAL_DISTANCE=0.08 \
        MEMORY_CORE_WEIGHT=0.5 \
        MEMORY_GATE="$gate" \
        MEMORY_TEMPERATURE=0.1 \
        MEMORY_CONFIDENCE=0.1 \
        MEMORY_LAYER_START=15 \
        MEMORY_LAYER_END=21 \
        RUN_ID="20260719_v40_${tag}" \
        METHOD_TAG="$tag" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 0.03 &
pid_003=$!
run_variant 1 0.05 &
pid_005=$!
run_variant 2 0.08 &
pid_008=$!

status=0
for pid in "$pid_003" "$pid_005" "$pid_008"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
