#!/usr/bin/env bash
# End-to-end v97 profiling, 16-cell generation, and metrics.
set -euo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
export REPO_ROOT="$ROOT"

START_SECONDS="$SECONDS"
echo "[v97-10h] start=$(date --iso-8601=seconds) commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "[v97-10h] profile=$RUN_PROFILE generation=$RUN_GENERATION metrics=$RUN_METRICS"

if [[ "$RUN_PROFILE" == "1" ]]; then
    bash "$ROOT/scripts/run_v97_qk_head_profile_16gpu.sh"
fi
if [[ "$RUN_GENERATION" == "1" ]]; then
    bash "$ROOT/scripts/run_v97_threshold_pf_merge_16gpu.sh"
fi
if [[ "$RUN_METRICS" == "1" ]]; then
    bash "$ROOT/scripts/postprocess_v97_threshold_pf_merge.sh"
fi

echo "[v97-10h] complete=$(date --iso-8601=seconds) elapsed_seconds=$((SECONDS-START_SECONDS))"
