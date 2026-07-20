#!/usr/bin/env bash
set -euo pipefail

# New direction exploration on GPU 4-7
# Based on docs/49 analysis: don't compete with PF on head classification.
# Instead, test orthogonal contributions: retrieval + per-head CFG.
#
# B0: SF native (no PF, no memory) — pure baseline
# B1: PF + archive retrieval, ALL heads, no routing, no per-head CFG
#     (simplest possible retrieval — does retrieval alone help?)
# B2: PF + archive retrieval + per-head CFG, ALL heads
#     (does per-head CFG add value without routing?)
# B3: PF + archive retrieval + per-head CFG + dynamic global CFG
#     (full guidance modulation)

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
SF_CONFIG="$ROOT/third_party/Self-Forcing/configs/self_forcing_dmd.yaml"
PROMPTS="$ROOT/prompts/review_3_prompts.txt"
FRAMES=240

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local tag="$3"
    shift 3
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        FRAMES="$FRAMES" \
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
    MEMORY_HEAD_ROUTING=static
    MEMORY_HEAD_LABELS=""
)

# B0: SF native — GPU 4
run_variant 4 20260720_v48_sf_native_60s sf_native \
    CONFIG_PATH="$SF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_b0=$!

# B1: PF + retrieval, all heads, no CFG modulation — GPU 5
run_variant 5 20260720_v48_retrieval_only retrieval_only \
    "${COMMON_MEM[@]}" \
    PER_HEAD_CFG_ENABLED=0 \
    DYNAMIC_CFG_ENABLED=0 &
pid_b1=$!

# B2: PF + retrieval + per-head CFG, all heads — GPU 6
run_variant 6 20260720_v48_retrieval_cfg retrieval_cfg \
    "${COMMON_MEM[@]}" \
    PER_HEAD_CFG_ENABLED=1 \
    PER_HEAD_CFG_MIN_SCALE=1.0 \
    PER_HEAD_CFG_MAX_SCALE=5.0 \
    DYNAMIC_CFG_ENABLED=0 &
pid_b2=$!

# B3: PF + retrieval + per-head CFG + dynamic global CFG — GPU 7
run_variant 7 20260720_v48_full_guidance full_guidance \
    "${COMMON_MEM[@]}" \
    PER_HEAD_CFG_ENABLED=1 \
    PER_HEAD_CFG_MIN_SCALE=1.0 \
    PER_HEAD_CFG_MAX_SCALE=5.0 \
    DYNAMIC_CFG_ENABLED=1 \
    DYNAMIC_CFG_MIN_SCALE=1.0 \
    DYNAMIC_CFG_MAX_SCALE=4.0 &
pid_b3=$!

echo "Waiting for all 4 new-direction variants..."
status=0
for pid in "$pid_b0" "$pid_b1" "$pid_b2" "$pid_b3"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "New direction experiments done. Status: $status"
exit "$status"
