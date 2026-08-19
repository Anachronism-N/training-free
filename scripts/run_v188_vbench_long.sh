#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
SCOPE="${2:-${V188_SCOPE:-}}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|collect|decision) ;;
    aggregate|efficiency|prefix-audit) ;;
    *)
        echo "usage: bash scripts/run_v188_vbench_long.sh ACTION [SCOPE]"
        echo "scope actions: prepare split preflight eval resume-missing status collect decision"
        echo "scopes: replica64_seed20000 long60_seed10000_32 mechanism32_seed10000"
        echo "global actions: aggregate efficiency prefix-audit"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_BASE="${V188_OUT_ROOT:-$ROOT/runs/v188_robustness_matrix}"
INPUT_MANIFEST="${V188_INPUT_MANIFEST:-$OUT_BASE/inputs/manifest.json}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

V187_ROOT="${V187_OUT_ROOT:-$ROOT/runs/v187_unseen128_confirmation}"
V187_COMPARISON_ROOT="${V187_COMPARISON_ROOT:-$V187_ROOT/confirm128/vbench_comparison}"
V187_SUMMARY="${V187_SUMMARY:-$V187_ROOT/confirm128/metrics/vbench_core9_summary.json}"
V187_PARTS_ROOT="${V187_PARTS_ROOT:-$V187_ROOT/confirm128/metrics/vbench_long_parts}"

if [[ "$ACTION" != "aggregate" && "$ACTION" != "efficiency" && "$ACTION" != "prefix-audit" ]]; then
    case "$SCOPE" in
        replica64_seed20000|long60_seed10000_32|mechanism32_seed10000) ;;
        *) echo "[error] a valid v188 scope is required"; exit 2 ;;
    esac
fi
if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi
if [[ "$ACTION" == "resume-missing" && \
      ( "$NODE_RANK" != "0" || "$NUM_NODES" != "1" ) ]]; then
    echo "[error] resume-missing requires NODE_RANK=0 NUM_NODES=1"
    exit 2
fi

if [[ "$ACTION" == "aggregate" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] aggregate requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/aggregate_v188_robustness_decision.py" \
        --replica "$OUT_BASE/replica64_seed20000/analysis/v188_replica64_seed20000_paired.json" \
        --long60 "$OUT_BASE/long60_seed10000_32/analysis/v188_long60_seed10000_32_paired.json" \
        --mechanism "$OUT_BASE/mechanism32_seed10000/analysis/v188_mechanism32_seed10000_paired.json" \
        --output "$OUT_BASE/analysis/v188_robustness_decision.json"
    exit $?
fi

if [[ "$ACTION" == "efficiency" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] efficiency requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v188_efficiency.py" \
        --input-manifest "$INPUT_MANIFEST" --run-base "$OUT_BASE" \
        --output "$OUT_BASE/analysis/v188_efficiency.json"
    exit $?
fi

if [[ "$ACTION" == "prefix-audit" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prefix-audit requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/audit_v188_long60_prefix.py" \
        --input-manifest "$INPUT_MANIFEST" --run-base "$OUT_BASE" \
        --output "$OUT_BASE/long60_seed10000_32/analysis/v188_long60_prefix.json"
    exit $?
fi

RUN_ROOT="$OUT_BASE/$SCOPE"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v188_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" \
        --input-manifest "$INPUT_MANIFEST" --scope "$SCOPE"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v188_${SCOPE}_paired.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("replication_confirmed", "long_horizon_confirmed", "phase_specificity_supported"):
    if key in payload:
        print(f"[v188-decision] scope={payload['scope']} {key}={str(payload[key]).lower()}")
print("manual_review_required=" + str(payload["manual_review_required"]).lower())
print("review_queue=" + ",".join(
    f"p{row['prompt_index']}" for row in payload["targeted_review_queue"]
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
        --workers "${V188_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V188_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V188_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V188_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V188_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v188_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem "v188_${SCOPE}_vbench_analysis" \
    --summary-title "v188 $SCOPE VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    REFERENCE_ARGS=()
    if [[ "$SCOPE" == "replica64_seed20000" ]]; then
        REFERENCE_ARGS+=(--v187-comparison-root "$V187_COMPARISON_ROOT")
        REFERENCE_ARGS+=(--v187-summary "$V187_SUMMARY")
        REFERENCE_ARGS+=(--v187-parts-root "$V187_PARTS_ROOT")
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v188_robustness_matrix.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v188_${SCOPE}_paired.json" \
        "${REFERENCE_ARGS[@]}"
fi
