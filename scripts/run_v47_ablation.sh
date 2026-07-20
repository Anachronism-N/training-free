#!/usr/bin/env bash
set -euo pipefail

# v4.7: Dynamic CFG + Confidence-Adaptive Head Routing + Ablation
#
# Key innovations that differentiate from PF:
# 1. Dynamic CFG: memory confidence modulates guidance scale
#    - High confidence → lower CFG (let memory provide structure)
#    - Low confidence → higher CFG (prompt-driven generation)
# 2. Confidence-Adaptive Head Routing (replaces PF static labels):
#    - Per-head memory access decided by retrieval confidence, not offline labels
#    - Sigmoid soft mask: heads with high confidence get memory, low don't
# 3. Ablation matrix to verify each component's contribution

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

# === ABLATION MATRIX ===
# Each variant tests one component's contribution.

# A0: PF baseline (no memory) — GPU 0
run_variant 0 20260720_v47_pf_baseline pf_baseline \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_a0=$!

# A1: Memory + static head routing (PF labels {1,2}) — GPU 1
# This is v4.6, used as "memory with PF routing" reference
run_variant 1 20260720_v47_mem_static mem_static \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=archive \
    MEMORY_ARCHIVE_MAX_FRAMES=64 \
    MEMORY_TOP_K_FRAMES=3 \
    MEMORY_RECENT_EXCLUDE_FRAMES=4 \
    MEMORY_SELECTION_POLICY=query \
    MEMORY_FUSION_MODE=convex \
    MEMORY_READOUT_MODE=all \
    MEMORY_GATE=0.15 \
    MEMORY_CONFIDENCE=0.25 \
    MEMORY_TEMPERATURE=0.3 \
    MEMORY_HEAD_LABELS=1,2 \
    MEMORY_HEAD_ROUTING=static \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 \
    MEMORY_WARMUP_BLOCKS=6 &
pid_a1=$!

# A2: Memory + confidence-adaptive routing (our innovation) — GPU 2
# Replaces PF static labels with dynamic per-head confidence routing
run_variant 2 20260720_v47_mem_adaptive mem_adaptive \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=archive \
    MEMORY_ARCHIVE_MAX_FRAMES=64 \
    MEMORY_TOP_K_FRAMES=3 \
    MEMORY_RECENT_EXCLUDE_FRAMES=4 \
    MEMORY_SELECTION_POLICY=query \
    MEMORY_FUSION_MODE=convex \
    MEMORY_READOUT_MODE=all \
    MEMORY_GATE=0.15 \
    MEMORY_CONFIDENCE=0.25 \
    MEMORY_TEMPERATURE=0.3 \
    MEMORY_HEAD_ROUTING=confidence_adaptive \
    MEMORY_ROUTING_SHARPNESS=5.0 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 \
    MEMORY_WARMUP_BLOCKS=6 &
pid_a2=$!

status=0
for pid in "$pid_a0" "$pid_a1" "$pid_a2"; do
    if ! wait "$pid"; then
        status=1
    fi
done

# A3: Memory + adaptive routing + dynamic CFG — GPU 0 (after A0 finishes)
run_variant 0 20260720_v47_mem_adaptive_dyn mem_adaptive_dyn \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_STORAGE_MODE=archive \
    MEMORY_ARCHIVE_MAX_FRAMES=64 \
    MEMORY_TOP_K_FRAMES=3 \
    MEMORY_RECENT_EXCLUDE_FRAMES=4 \
    MEMORY_SELECTION_POLICY=query \
    MEMORY_FUSION_MODE=convex \
    MEMORY_READOUT_MODE=all \
    MEMORY_GATE=0.15 \
    MEMORY_CONFIDENCE=0.25 \
    MEMORY_TEMPERATURE=0.3 \
    MEMORY_HEAD_ROUTING=confidence_adaptive \
    MEMORY_ROUTING_SHARPNESS=5.0 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 \
    MEMORY_WARMUP_BLOCKS=6 \
    DYNAMIC_CFG_ENABLED=1 \
    DYNAMIC_CFG_MIN_SCALE=1.0 \
    DYNAMIC_CFG_MAX_SCALE=4.0 &
pid_a3=$!

wait "$pid_a3" || status=1
exit "$status"
