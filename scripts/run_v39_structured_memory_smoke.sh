#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="${PROMPTS:-$ROOT/prompts/smoke_identity_motion.txt}"
FRAMES="${FRAMES:-36}"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local method_tag="$3"
    local memory_enabled="$4"
    local memory_gate="$5"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES="$FRAMES" \
        SEED=0 \
        STRENGTH=0 \
        GATE_LAMBDA=0 \
        STRUCTURED_MEMORY_ENABLE="$memory_enabled" \
        MEMORY_BUDGET=4 \
        MEMORY_SPATIAL_STRIDE=4 \
        MEMORY_LOCAL_DISTANCE=0.08 \
        MEMORY_CORE_WEIGHT=0.5 \
        MEMORY_GATE="$memory_gate" \
        MEMORY_TEMPERATURE=0.1 \
        MEMORY_CONFIDENCE=0.1 \
        MEMORY_LAYER_START=15 \
        MEMORY_LAYER_END=21 \
        RUN_ID="$run_id" \
        METHOD_TAG="$method_tag" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260719_v39_pf36 pf36 0 0 &
pid_pf=$!
run_variant 1 20260719_v39_mem002 mem002 1 0.02 &
pid_mem002=$!
run_variant 2 20260719_v39_mem005 mem005 1 0.05 &
pid_mem005=$!

status=0
for pid in "$pid_pf" "$pid_mem002" "$pid_mem005"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
