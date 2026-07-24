#!/usr/bin/env bash
# Resumable v93 generation and metric queue for one 16-GPU node.
# Usage: nohup bash scripts/run_v93_moviebench_10h.sh > runs/v93_10h.log 2>&1 &
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_HEAD32="${RUN_HEAD32:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
MAIN_READY=1
HEAD_READY=1
STATUS=0

run_stage() {
    local name="$1"
    shift
    echo "[v93-queue] start $name $(date -Iseconds)"
    if "$@"; then
        echo "[v93-queue] done $name $(date -Iseconds)"
        return 0
    fi
    local code=$?
    echo "[v93-queue] failed $name code=$code $(date -Iseconds)"
    return "$code"
}

if [[ "$RUN_MAIN" == "1" ]]; then
    run_stage main-generation \
        bash "$ROOT/scripts/run_v93_moviebench_main_16gpu.sh" || {
        MAIN_READY=0
        STATUS=1
    }
fi

if [[ "$RUN_HEAD32" == "1" ]]; then
    run_stage head32-generation \
        bash "$ROOT/scripts/run_v93_moviebench_head32_16gpu.sh" || {
        HEAD_READY=0
        STATUS=1
    }
fi

if [[ "$RUN_METRICS" == "1" && "$MAIN_READY" == "1" ]]; then
    run_stage main-metrics \
        bash "$ROOT/scripts/postprocess_v93_moviebench.sh" main || STATUS=1
fi

if [[ "$RUN_METRICS" == "1" && "$HEAD_READY" == "1" ]]; then
    run_stage head32-metrics \
        bash "$ROOT/scripts/postprocess_v93_moviebench.sh" head32 || STATUS=1
fi

echo "[v93-queue] complete status=$STATUS $(date -Iseconds)"
exit "$STATUS"
