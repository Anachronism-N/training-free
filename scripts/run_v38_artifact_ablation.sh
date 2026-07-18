#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PROMPTS="$ROOT/prompts/review_artifact_2.txt"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local method_tag="$3"
    shift 3
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PROMPTS="$PROMPTS" \
        RUN_ID="$run_id" \
        METHOD_TAG="$method_tag" \
        GATE_LAMBDA=3.0 \
        MOMENT_MODE=variance_only \
        TARGET_FRAMES=8 \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260719_v38_overlap overlap \
    TRANSITION_LAMBDA=0.0 MAX_STD_RATIO=0.0 &
pid_overlap=$!

run_variant 1 20260719_v38_bounded bounded \
    TRANSITION_LAMBDA=0.0 MAX_STD_RATIO=1.5 &
pid_bounded=$!

run_variant 2 20260719_v38_confidence confidence \
    TRANSITION_LAMBDA=2.0 MAX_STD_RATIO=1.5 &
pid_confidence=$!

status=0
for pid in "$pid_overlap" "$pid_bounded" "$pid_confidence"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
