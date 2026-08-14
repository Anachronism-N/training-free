#!/usr/bin/env bash
# Incremental core-9 evaluation and automatic 2x2 attribution for v179.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in prepare|split|preflight|eval|resume-missing|status|collect|decision) ;;
*)
    echo "usage: bash scripts/run_v179_vbench_long.sh {prepare|split|preflight|eval|resume-missing|status|collect|decision}"
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v179_rccp_head_attribution}"
V178_ROOT="${V178_OUT_ROOT:-$ROOT/runs/v178_rccp_holdout_generation}"
PROVISIONAL_COUNT="${PROVISIONAL_COUNT:-0}"
if [[ "$PROVISIONAL_COUNT" -ne 0 ]]; then
    [[ "$PROVISIONAL_COUNT" -ge 1 && "$PROVISIONAL_COUNT" -lt 32 ]] || {
        echo "[error] PROVISIONAL_COUNT must be 0 or in [1,31]"
        exit 2
    }
    SCOPE="provisional_$(printf '%02d' "$PROVISIONAL_COUNT")"
    DEFAULT_COMPARISON_ROOT="$RUN_ROOT/$SCOPE/vbench_comparison"
    DEFAULT_PARTS_ROOT="$RUN_ROOT/$SCOPE/metrics/vbench_long_parts"
    DEFAULT_SUMMARY_ROOT="$RUN_ROOT/$SCOPE/metrics"
    DEFAULT_ANALYSIS_ROOT="$RUN_ROOT/$SCOPE/analysis"
    DEFAULT_V178_PAIRED="$V178_ROOT/$SCOPE/analysis/v178_paired_metrics.json"
else
    DEFAULT_COMPARISON_ROOT="$RUN_ROOT/vbench_comparison"
    DEFAULT_PARTS_ROOT="$RUN_ROOT/metrics/vbench_long_parts"
    DEFAULT_SUMMARY_ROOT="$RUN_ROOT/metrics"
    DEFAULT_ANALYSIS_ROOT="$RUN_ROOT/analysis"
    DEFAULT_V178_PAIRED="$V178_ROOT/analysis/v178_paired_metrics.json"
fi
COMPARISON_ROOT="${COMPARISON_ROOT:-$DEFAULT_COMPARISON_ROOT}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$DEFAULT_PARTS_ROOT}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$DEFAULT_SUMMARY_ROOT}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$DEFAULT_ANALYSIS_ROOT}"
V178_PAIRED="${V178_PAIRED:-$DEFAULT_V178_PAIRED}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if [[ "$ACTION" == "decision" ]]; then
    [[ "$PROVISIONAL_COUNT" -eq 0 ]] || {
        echo "[error] provisional v179 metrics cannot make an attribution decision"
        exit 4
    }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v179_head_attribution.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[error] missing attribution result: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("provisional") is not False
    or payload.get("attribution_decision_allowed") is not True
):
    raise SystemExit("[error] v179 result is not a formal attribution scope")
print(f"decision={payload['decision']}")
print(f"top1_directional_positive={payload['top1_directional_positive']}")
print(f"remainder_directional_positive={payload['remainder_directional_positive']}")
print(f"top1_confirmatory_positive={payload['top1_confirmatory_positive']}")
print(f"remainder_confirmatory_positive={payload['remainder_confirmatory_positive']}")
for metric, row in payload["contribution_share"].items():
    print(
        f"{metric}: total={row['matched_total_mean_delta']:.6f} "
        f"top1={row['top1_shapley_mean_delta']:.6f} "
        f"remainder={row['remainder_shapley_mean_delta']:.6f}"
    )
PY
    exit $?
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    PREPARE_ARGS=(--run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT")
    if [[ "$PROVISIONAL_COUNT" -ne 0 ]]; then
        PREPARE_ARGS+=(
            --provisional-count "$PROVISIONAL_COUNT"
            --v178-paired "$V178_PAIRED"
        )
    fi
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v179_vbench_comparison.py" "${PREPARE_ARGS[@]}"
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V179_SPLIT_WORKERS:-2}" --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V179_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V179_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && "${V179_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" --torch-hub-dir "$TORCH_HUB_DIR" \
        --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V179_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(
        --local-models --torch-hub-dir "$TORCH_HUB_DIR"
        --runtime-home "$RUNTIME_HOME"
    )
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v179_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_incremental_summary \
    --analysis-stem v179_vbench_incremental_analysis \
    --summary-title "v179 Incremental VBench-Long" "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v179_head_attribution.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_incremental_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v179_head_attribution.json"
fi
