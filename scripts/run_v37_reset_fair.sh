#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"

run_variant() {
    local gpu="$1"
    local run_id="$2"
    local method_tag="$3"
    shift 3
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        RUN_ID="$run_id" \
        METHOD_TAG="$method_tag" \
        "$@" \
        bash "$ROOT/scripts/run_v35_pf_value_refresh.sh"
}

run_variant 0 20260718_v37_reset_pf pf_reset \
    STRENGTH=0.0 GATE_LAMBDA=0.0 MOMENT_MODE=full &
pid_pf=$!

run_variant 1 20260718_v37_reset_middle middle_reset \
    STRENGTH=0.5 GATE_LAMBDA=3.0 MOMENT_MODE=full \
    LAYER_START=10 LAYER_END=20 &
pid_middle=$!

run_variant 2 20260718_v37_reset_variance variance_reset \
    STRENGTH=0.5 GATE_LAMBDA=3.0 MOMENT_MODE=variance_only &
pid_variance=$!

status=0
for pid in "$pid_pf" "$pid_middle" "$pid_variance"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
