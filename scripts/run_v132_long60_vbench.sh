#!/usr/bin/env bash
# Evaluate the paired 60-second SF/main comparison with VBench-Long.
set -euo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "split" && "$ACTION" != "preflight" && \
      "$ACTION" != "eval" && "$ACTION" != "collect" ]]; then
    echo "usage: bash scripts/run_v132_long60_vbench.sh split|preflight|eval|collect"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
export COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/runs/v132_main_60s_comparison}"
export VBENCH_EXPECTED_EXPERIMENT="v132_main_60s_comparison"
export VBENCH_EXPECTED_METHOD_COUNT=2
export VBENCH_EXPECTED_NUM_OUTPUT_FRAMES=240
export V129_METRIC_PROFILE="${V132_METRIC_PROFILE:-core}"

exec bash "$ROOT/scripts/run_v129_vbench_long.sh" "$ACTION"
