#!/usr/bin/env bash
# Audit and evaluate the completed v180 videos without treating the placeholder
# v178 result as a formal membership gate.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
logs|audit|prepare|split|preflight|eval|resume-missing|status|collect|decision|package) ;;
*)
    echo "usage: bash scripts/run_v183_v180_recovery.sh ACTION"
    echo "actions: logs audit prepare split preflight eval resume-missing status collect decision package"
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${V180_RUN_ROOT:-$ROOT/runs/v180_rccp_fresh128}"
RECOVERY_ROOT="${V183_RECOVERY_ROOT:-$RUN_ROOT/recovery_v183}"
INPUT_MANIFEST="$RUN_ROOT/inputs/manifest.json"
COMPARISON_ROOT="$RECOVERY_ROOT/vbench_comparison"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="$RECOVERY_ROOT/metrics/vbench_long_parts"
SUMMARY_ROOT="$RECOVERY_ROOT/metrics"
ANALYSIS_ROOT="$RECOVERY_ROOT/analysis"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"
}

if [[ "$ACTION" == "logs" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/audit_v183_v180_recovery.py" logs \
        --run-root "$RUN_ROOT" --recovery-root "$RECOVERY_ROOT" \
        --input-manifest "$INPUT_MANIFEST"
    exit $?
fi

if [[ "$ACTION" == "audit" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] audit requires node 0"; exit 2; }
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/audit_v183_v180_recovery.py" full \
        --run-root "$RUN_ROOT" --recovery-root "$RECOVERY_ROOT" \
        --input-manifest "$INPUT_MANIFEST"
    exit $?
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v183_v180_recovery_vbench.py" \
        --recovery-root "$RECOVERY_ROOT" --comparison-root "$COMPARISON_ROOT"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v183_v180_recovery_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[error] missing paired result: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
print(f"recommendation={payload['recommendation']}")
print(f"formal_rccp_membership_claim_allowed={payload['formal_rccp_membership_claim_allowed']}")
for key in (
    "end_to_end_directional_nonregression",
    "strict5_increment_directional_nonregression",
    "all_head_coverage_directional_nonregression",
    "strong_strict5_exploratory_signal",
):
    print(f"{key}={'PASS' if payload.get(key) else 'FAIL'}")
print(payload["claim_boundary"])
PY
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    target="$RECOVERY_ROOT/v183_v180_recovery_diagnostics.tar.gz"
    tar -C "$RECOVERY_ROOT" -czf "$target" \
        audits contracts published_manifest.json metrics analysis
    echo "$target"
    exit 0
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

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V183_SPLIT_WORKERS:-2}" --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V183_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V183_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V183_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" --torch-hub-dir "$TORCH_HUB_DIR" \
        --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V183_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v183_v180_recovery_vbench.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary --analysis-stem v183_vbench_analysis \
    --summary-title "v183 Recovered v180 VBench-Long" "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v183_v180_recovery_metrics.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --output "$ANALYSIS_ROOT/v183_v180_recovery_metrics.json"
fi
