#!/usr/bin/env bash
set -euo pipefail

# 32-prompt ablation on GPU 0-3
# 4 variants × 32 prompts × 120 frames
# A0: PF baseline
# A1: Memory + static PF routing
# A2: Memory + confidence-adaptive routing
# A4: Memory + adaptive routing + per-head CFG (full method)

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
PROMPTS="$ROOT/third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt"
FRAMES=120

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
)

# A0: PF baseline — GPU 0
run_variant 0 20260720_v48_32p_pf pf_32p \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_a0=$!

# A1: Memory + static PF routing — GPU 1
run_variant 1 20260720_v48_32p_static static_32p \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=static \
    MEMORY_HEAD_LABELS=1,2 \
    PER_HEAD_CFG_ENABLED=0 &
pid_a1=$!

# A2: Memory + confidence-adaptive routing — GPU 2
run_variant 2 20260720_v48_32p_adaptive adaptive_32p \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=confidence_adaptive \
    MEMORY_ROUTING_SHARPNESS=5.0 \
    PER_HEAD_CFG_ENABLED=0 &
pid_a2=$!

# A4: Full method (adaptive + per-head CFG) — GPU 3
run_variant 3 20260720_v48_32p_full full_32p \
    "${COMMON_MEM[@]}" \
    MEMORY_HEAD_ROUTING=confidence_adaptive \
    MEMORY_ROUTING_SHARPNESS=5.0 \
    PER_HEAD_CFG_ENABLED=1 \
    PER_HEAD_CFG_MIN_SCALE=1.0 \
    PER_HEAD_CFG_MAX_SCALE=5.0 &
pid_a4=$!

echo "Waiting for all 4 variants to complete..."
status=0
for pid in "$pid_a0" "$pid_a1" "$pid_a2" "$pid_a4"; do
    if ! wait "$pid"; then
        status=1
        echo "WARNING: a variant failed (pid=$pid)"
    fi
done
echo "All variants done. Status: $status"
exit "$status"
