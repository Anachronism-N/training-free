#!/usr/bin/env bash
# Quantify recovered v181 media without promoting it to formal classifier evidence.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|collect|decision) ;;
    *)
        echo "usage: bash scripts/run_v185_recovered_long60_vbench.sh ACTION"
        echo "actions: prepare split preflight eval resume-missing status collect decision"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RUN_ROOT="${V185_SOURCE_RUN_ROOT:-$ROOT/runs/v181_rccp_long_stress}"
OUT_ROOT="${V185_OUT_ROOT:-$ROOT/runs/v185_recovered_v181_long60}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$OUT_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$OUT_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$OUT_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$OUT_ROOT/analysis}"
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

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v185_recovered_long60_comparison.py" \
        --run-root "$SOURCE_RUN_ROOT" --comparison-root "$COMPARISON_ROOT"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v185_recovered_long60_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"verdict={payload['verdict']}")
print("evidence_grade=" + payload["evidence_grade"])
print("formal_classifier_claim_eligible=false")
print("manual_review_required=false")
print("review_queue=" + ",".join(
    f"p{row['prompt_index']}" for row in payload["targeted_review"]
))
PY
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V185_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V185_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V185_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V185_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V185_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v185_recovered_long60.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v185_recovered_long60_vbench_analysis \
    --summary-title "v185 Recovered Long60 VBench-Long (exploratory)" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v185_recovered_long60.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v185_recovered_long60_metrics.json"
fi
