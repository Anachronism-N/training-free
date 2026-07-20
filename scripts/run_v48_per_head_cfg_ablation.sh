#!/usr/bin/env bash
set -euo pipefail

# v4.8: Per-head CFG + targeted ablation for dynamic head routing validation
#
# Key experiments:
# A4: Memory + adaptive routing + per-head CFG (full method)
# A5: Memory + NO routing (all heads equal) — isolate routing contribution
# A6: Memory + RANDOM routing — control for routing itself

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
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
        RECENT_FRAMES=21 \
        METHOD_TAG="$tag" \
        RUN_ID="$run_id" \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

COMMON_MEM=(
    CONFIG_PATH="$PF_CONFIG"
    STRUCTURED_MEMORY_ENABLE=1
    MEMORY_STORAGE_MODE=archive
    MEMORY_ARCHIVE_MAX_FRAMES=64
    MEMORY_TOP_K_FRAMES=3
    MEMORY_RECENT_EXCLUDE_FRAMES=4
    MEMORY_SELECTION_POLICY=query
    MEMORY_FUSION_MODE=convex
    MEMORY_READOUT_MODE=all
    MEMORY_GATE=0.15
    MEMORY_CONFIDENCE=0.25
    MEMORY_TEMPERATURE=0.3
    MEMORY_LAYER_START=15
    MEMORY_LAYER_END=21
    MEMORY_WARMUP_BLOCKS=6
)

# A4: Full method — adaptive routing + per-head CFG — GPU 0
run_variant 0 20260720_v48_full full \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=confidence_adaptive \
    MEMORY_ROUTING_SHARPNESS=5.0 \
    PER_HEAD_CFG_ENABLED=1 \
    PER_HEAD_CFG_MIN_SCALE=1.0 \
    PER_HEAD_CFG_MAX_SCALE=5.0 &
pid_a4=$!

# A5: No routing — all heads get equal memory access — GPU 1
# This isolates the contribution of routing itself
run_variant 1 20260720_v48_no_routing no_routing \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=static \
    MEMORY_HEAD_LABELS="" \
    PER_HEAD_CFG_ENABLED=0 &
pid_a5=$!

# A6: Static PF routing + per-head CFG — GPU 2
# Tests if per-head CFG works with PF labels instead of adaptive routing
run_variant 2 20260720_v48_static_cfg static_cfg \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=static \
    MEMORY_HEAD_LABELS=1,2 \
    PER_HEAD_CFG_ENABLED=1 \
    PER_HEAD_CFG_MIN_SCALE=1.0 \
    PER_HEAD_CFG_MAX_SCALE=5.0 &
pid_a6=$!

status=0
for pid in "$pid_a4" "$pid_a5" "$pid_a6"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
