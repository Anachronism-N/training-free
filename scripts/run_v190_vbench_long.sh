#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision) ;;
    *)
        echo "usage: bash scripts/run_v190_vbench_long.sh {prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v190_head_phase_causal_screen/screen32}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V190_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local published="$RUN_ROOT/published_manifest.json"
    [[ -s "$published" ]] || {
        echo "[error] missing audited screen manifest: $published"; exit 2;
    }
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$published" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("ok") is not True or payload.get("scope") != "screen32":
    raise SystemExit("v190 screen must pass audit before temporal diagnostics")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -gt 1 ]] || {
        echo "[error] v190 temporal diagnostics found too few methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 32 --max-width "${V190_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V190_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V190_TEMPORAL_WORKERS:-8}"
}

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v190_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v190_head_phase_causal_screen.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v190-decision] {payload['recommendation']}")
print(f"selected={payload['selected_for_fresh128']}")
print(f"manual_review_required={str(payload['manual_review_required']).lower()}")
dynamic = payload["metric_validity"]["dynamic_degree"]
print(f"dynamic_informative={str(dynamic['informative']).lower()}")
print(f"dynamic_ceiling_only={str(dynamic['ceiling_nonregression_only']).lower()}")
for row in payload["targeted_review_queue"]:
    print(f"review={row['candidate']}:p{row['prompt_index']}")
PY
    exit $?
fi

if [[ "$ACTION" == "temporal" ]]; then
    compute_temporal
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v174_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V190_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V190_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V190_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V190_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V190_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v174_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v190_vbench_analysis \
    --summary-title "v190 Head x Phase Causal Screen (holdout32)" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" ]]; then
        compute_temporal
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v190_head_phase_causal_screen.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --temporal-csv "$TEMPORAL_CSV" \
        --output "$ANALYSIS_ROOT/v190_head_phase_causal_screen.json"
fi
