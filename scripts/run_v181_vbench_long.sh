#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in prepare|split|preflight|eval|resume-missing|status|collect|decision) ;;
*)
    echo "usage: bash scripts/run_v181_vbench_long.sh ACTION"
    echo "actions: prepare split preflight eval resume-missing status collect decision"
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCOPE="${SCOPE:-long60_seed0}"
case "$SCOPE" in long60_seed0|long60_seed10000_64) ;;
*) echo "[error] unsupported v181 scope: $SCOPE"; exit 2 ;;
esac
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v181_rccp_long_stress}"
SCOPE_ROOT="$RUN_ROOT/scopes/$SCOPE"
COMPARISON_ROOT="${COMPARISON_ROOT:-$SCOPE_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$SCOPE_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$SCOPE_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$SCOPE_ROOT/analysis}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi
if [[ "$ACTION" == "resume-missing" && \
      ( "$NODE_RANK" != "0" || "$NUM_NODES" != "1" ) ]]; then
    echo "[error] resume-missing requires NODE_RANK=0 NUM_NODES=1"
    exit 2
fi

if [[ "$ACTION" == "decision" ]]; then
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v181_long_stress_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[error] missing paired result: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
print(f"scope={payload['scope']}")
print(f"decision={payload['decision']}")
for key in (
    "quality_identity_gate",
    "identity_motion_gate",
    "late_identity_gate",
    "dynamic_nonregression_gate",
):
    print(f"{key}={'PASS' if payload.get(key) else 'FAIL'}")
accepted = {
    "long_horizon_quality_identity_motion_confirmed",
    "long_horizon_quality_identity_confirmed",
    "long_horizon_identity_motion_confirmed",
}
if payload.get("decision") not in accepted:
    raise SystemExit(3)
PY
    exit $?
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v181_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" \
        --scope "$SCOPE"
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V181_SPLIT_WORKERS:-2}" --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V181_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V181_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V181_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" --torch-hub-dir "$TORCH_HUB_DIR" \
        --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V181_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(
        --local-models --torch-hub-dir "$TORCH_HUB_DIR"
        --runtime-home "$RUNTIME_HOME"
    )
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v181_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem "v181_${SCOPE}_vbench_analysis" \
    --summary-title "v181 $SCOPE VBench-Long" "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v181_long_stress_metrics.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v181_long_stress_metrics.json"
fi
