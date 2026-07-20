#!/usr/bin/env bash
set -euo pipefail

# v4.6: Optimized 60s comparison — SF / PF / Ours
# Key improvements over v4.5:
# 1. Preservation-aware archive (always keep frame 0 = identity anchor)
# 2. Warmup ramp (3 blocks) to avoid first-few-seconds discontinuity
# 3. Softer retrieval: top_k=3, temperature=0.3 (less argmax-like)
# 4. Higher confidence threshold: 0.25 (suppress bad retrievals in high motion)
# 5. Memory on all passes (not just clean) — direct denoising influence

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
        RECENT_FRAMES=21 \
        METHOD_TAG="$tag" \
        RUN_ID="$run_id" \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

# Variant 0: SF native (GPU 0)
run_variant 0 20260720_v46_sf_native_60s sf_native_60s \
    CONFIG_PATH="$SF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_sf=$!

# Variant 1: PF baseline (GPU 1)
run_variant 1 20260720_v46_pf_60s pf_60s \
    CONFIG_PATH="$PF_CONFIG" \
    STRUCTURED_MEMORY_ENABLE=0 &
pid_pf=$!

# Variant 2: PF + Ours v4.6 (GPU 2)
run_variant 2 20260720_v46_ours_60s ours_60s \
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
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 \
    MEMORY_WARMUP_BLOCKS=3 &
pid_ours=$!

status=0
for pid in "$pid_sf" "$pid_pf" "$pid_ours"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
