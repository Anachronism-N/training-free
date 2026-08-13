#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in prepare|split|preflight|eval|resume-missing|status|collect|decision) ;;
*) echo "usage: bash scripts/run_v178_vbench_long.sh {prepare|split|preflight|eval|resume-missing|status|collect|decision}"; exit 2;; esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v178_rccp_holdout_generation}"
PROVISIONAL_COUNT="${PROVISIONAL_COUNT:-0}"
if [[ "$PROVISIONAL_COUNT" -ne 0 ]]; then
    [[ "$PROVISIONAL_COUNT" -ge 1 && "$PROVISIONAL_COUNT" -lt 32 ]] || {
        echo "[error] PROVISIONAL_COUNT must be 0 or in [1,31]"
        exit 2
    }
    DEFAULT_COMPARISON_ROOT="$RUN_ROOT/provisional_$(printf '%02d' "$PROVISIONAL_COUNT")/vbench_comparison"
    DEFAULT_PARTS_ROOT="$RUN_ROOT/provisional_$(printf '%02d' "$PROVISIONAL_COUNT")/metrics/vbench_long_parts"
    DEFAULT_SUMMARY_ROOT="$RUN_ROOT/provisional_$(printf '%02d' "$PROVISIONAL_COUNT")/metrics"
    DEFAULT_ANALYSIS_ROOT="$RUN_ROOT/provisional_$(printf '%02d' "$PROVISIONAL_COUNT")/analysis"
else
    DEFAULT_COMPARISON_ROOT="$RUN_ROOT/vbench_comparison"
    DEFAULT_PARTS_ROOT="$RUN_ROOT/metrics/vbench_long_parts"
    DEFAULT_SUMMARY_ROOT="$RUN_ROOT/metrics"
    DEFAULT_ANALYSIS_ROOT="$RUN_ROOT/analysis"
fi
COMPARISON_ROOT="${COMPARISON_ROOT:-$DEFAULT_COMPARISON_ROOT}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$DEFAULT_PARTS_ROOT}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$DEFAULT_SUMMARY_ROOT}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$DEFAULT_ANALYSIS_ROOT}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}" NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if [[ "$ACTION" == "decision" ]]; then
    [[ "$PROVISIONAL_COUNT" -eq 0 ]] || {
        echo "[error] provisional v178 metrics cannot make a membership decision"
        exit 4
    }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v178_paired_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[error] missing paired result: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
print(f"decision={payload['decision']}")
print(f"membership_hypothesis_gate={payload['membership_hypothesis_gate']}")
for name, passed in payload.get("gate_checks", {}).items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
if (
    payload.get("membership_hypothesis_gate") is not True
    or payload.get("decision")
    != "advance_rccp_membership_to_broader_generation"
    or payload.get("failed_gate_checks")
):
    raise SystemExit(3)
PY
    exit $?
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    PREPARE_ARGS=(--run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT")
    if [[ "$PROVISIONAL_COUNT" -ne 0 ]]; then
        PREPARE_ARGS+=(--provisional-count "$PROVISIONAL_COUNT")
    fi
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v178_vbench_comparison.py" "${PREPARE_ARGS[@]}"
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"; conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V178_SPLIT_WORKERS:-2}" --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"; conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
TORCH_HUB_DIR="${V178_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V178_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && "${V178_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" --torch-hub-dir "$TORCH_HUB_DIR" \
        --runtime-home "$RUNTIME_HOME"
fi
PYTHON_ACTION="$ACTION"; [[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V178_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME")
fi
"$PYTHON_BIN" "$ROOT/scripts/run_v178_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary --analysis-stem v178_vbench_analysis \
    --summary-title "v178 RCCP Holdout VBench-Long" "${EXTRA_ARGS[@]}"
if [[ "$ACTION" == "collect" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v178_paired_metrics.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v178_paired_metrics.json"
fi
