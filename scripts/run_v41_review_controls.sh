#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$ROOT/prompts/review_artifact_2.txt"

env \
    CUDA_VISIBLE_DEVICES=3 \
    PROMPTS="$PROMPTS" \
    FRAMES=36 \
    SEED=0 \
    STRENGTH=0 \
    GATE_LAMBDA=0 \
    STRUCTURED_MEMORY_ENABLE=0 \
    RUN_ID=20260719_v41_pf_control \
    METHOD_TAG=pf_control \
    bash "$ROOT/scripts/run_v35_pf_value_refresh.sh" &
pid_pf=$!

env \
    CUDA_VISIBLE_DEVICES=4 \
    PROMPTS="$PROMPTS" \
    FRAMES=36 \
    SEED=0 \
    STRENGTH=0 \
    GATE_LAMBDA=0 \
    STRUCTURED_MEMORY_ENABLE=1 \
    MEMORY_VALUE_MODE=full \
    MEMORY_READOUT_MODE=all \
    MEMORY_GATE=0.05 \
    MEMORY_CONFIDENCE=0.1 \
    MEMORY_LAYER_START=15 \
    MEMORY_LAYER_END=21 \
    RUN_ID=20260719_v41_all005 \
    METHOD_TAG=all005 \
    bash "$ROOT/scripts/run_v35_pf_value_refresh.sh" &
pid_all=$!

status=0
for pid in "$pid_pf" "$pid_all"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
