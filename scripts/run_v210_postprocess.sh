#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
  prepare|split|preflight|eval|resume-missing|status|collect|temporal|bind-temporal|analyze|postprocess|decision) ;;
  *)
    echo "usage: bash scripts/run_v210_postprocess.sh {prepare|split|preflight|eval|resume-missing|status|collect|temporal|bind-temporal|analyze|postprocess|decision}" >&2
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATION_ROOT="${V210_GENERATION_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v210_runs/v210_lphc_789d8604_sixnode}"
EVALUATION_COMMIT="${V210_EVALUATION_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
SCREEN_ROOT="${V210_SCREEN_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v210_runs/v210_eval_${EVALUATION_COMMIT:0:8}_screen8}"
COMPARISON_ROOT="${V210_COMPARISON_ROOT:-$SCREEN_ROOT/vbench_comparison}"
METRICS_ROOT="${V210_METRICS_ROOT:-$SCREEN_ROOT/metrics}"
PARTS_ROOT="${V210_PARTS_ROOT:-$METRICS_ROOT/vbench_long_parts}"
ANALYSIS_ROOT="${V210_ANALYSIS_ROOT:-$SCREEN_ROOT/analysis}"

VBENCH_ROOT="${VBENCH_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench}"
SHARED_MODEL_ROOT="${V210_SHARED_MODEL_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$SHARED_MODEL_ROOT/vbench_cache}"
TORCH_HUB_DIR="${V210_TORCH_HUB_DIR:-$SHARED_MODEL_ROOT/_model_cache/torch_hub}"
RUNTIME_HOME="${V210_RUNTIME_HOME:-$SHARED_MODEL_ROOT/_model_cache/dreamsim_home}"
ACTIVATE_SH="${ACTIVATE_SH:-/apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-6}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if [[ ! "$EVALUATION_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[error] V210_EVALUATION_COMMIT must be a full git commit" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$EVALUATION_COMMIT" ]]; then
  echo "[error] checkout HEAD does not match V210_EVALUATION_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "[error] v210 postprocessing requires an exact clean evaluation checkout" >&2
  exit 2
fi
if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "[error] require 0 <= NODE_RANK < NUM_NODES" >&2
  exit 2
fi
if [[ "$ACTION" == "resume-missing" && ( "$NODE_RANK" != "0" || "$NUM_NODES" != "1" ) ]]; then
  echo "[error] resume-missing requires NODE_RANK=0 NUM_NODES=1" >&2
  exit 2
fi
if [[ "$ACTION" =~ ^(prepare|collect|temporal|bind-temporal|analyze|postprocess|decision)$ && "$NODE_RANK" != "0" ]]; then
  echo "[error] $ACTION requires NODE_RANK=0" >&2
  exit 2
fi

activate_longlive() {
  # This is the project-required activation entry point on every authorized node.
  source "$ACTIVATE_SH" "$CONDA_ENV"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

run_vbench() {
  local mode="$1"
  "$PYTHON_BIN" "$ROOT/scripts/run_v210_vbench.py" "$mode" \
    --comparison-root "$COMPARISON_ROOT" \
    --evaluation-root "$ROOT" \
    --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" \
    --summary-root "$METRICS_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" \
    --num-nodes "$NUM_NODES" \
    --gpu-list "$GPU_LIST" \
    --local-models \
    --torch-hub-dir "$TORCH_HUB_DIR" \
    --runtime-home "$RUNTIME_HOME" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v210_vbench_analysis \
    --summary-title "v210 LPHC Screen8 VBench-Long Core-9"
}

run_temporal() {
  local inputs=()
  local methods=(
    sf_fifo21
    sf_sink1_21
    sf_fifo25
    lphc_e1_a002_correct
    lphc_e1_a005_correct
    lphc_e1_a010_correct
    lphc_e2_a010_correct
    lphc_full_a010_correct
  )
  local method
  for method in "${methods[@]}"; do
    inputs+=("$COMPARISON_ROOT/published/$method")
  done
  "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "${inputs[@]}" \
    --output "$METRICS_ROOT/temporal_diagnostics.csv" \
    --expected-videos 8 \
    --max-width 256 \
    --frame-step 2 \
    --workers "${V210_TEMPORAL_WORKERS:-8}"
}

bind_temporal() {
  "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
    --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
    --temporal-csv "$METRICS_ROOT/temporal_diagnostics.csv" \
    --output "$METRICS_ROOT/temporal_diagnostics.contract.json"
}

run_analysis() {
  "$PYTHON_BIN" "$ROOT/scripts/analyze_v210_lphc_screen.py" \
    --screen-root "$SCREEN_ROOT"
}

case "$ACTION" in
  prepare)
    activate_longlive
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v210_vbench_comparison.py" \
      --run-root "$GENERATION_ROOT" \
      --comparison-root "$COMPARISON_ROOT" \
      --repo-root "$ROOT" \
      --vbench-root "$VBENCH_ROOT"
    ;;
  split)
    activate_longlive
    "$PYTHON_BIN" "$ROOT/scripts/run_v210_vbench.py" split \
      --comparison-root "$COMPARISON_ROOT" \
      --evaluation-root "$ROOT" \
      --vbench-root "$VBENCH_ROOT" \
      --workers "${V210_SPLIT_WORKERS:-2}" \
      --node-rank "$NODE_RANK" \
      --num-nodes "$NUM_NODES"
    ;;
  preflight|eval|status|collect)
    activate_longlive
    run_vbench "$ACTION"
    ;;
  resume-missing)
    activate_longlive
    run_vbench eval-missing
    ;;
  temporal)
    activate_longlive
    run_temporal
    ;;
  bind-temporal)
    activate_longlive
    bind_temporal
    ;;
  analyze)
    activate_longlive
    run_analysis
    ;;
  postprocess)
    activate_longlive
    run_vbench collect
    run_temporal
    bind_temporal
    run_analysis
    ;;
  decision)
    activate_longlive
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v210_lphc_screen.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
selected = payload.get("selected_for_matched_random_extension")
count = payload.get("selection_count")
if count not in (0, 1) or (count == 0) != (selected is None):
    raise SystemExit("invalid v210 selection contract")
print(json.dumps({
    "recommendation": payload.get("recommendation"),
    "selection_count": count,
    "selected_for_matched_random_extension": selected,
}, sort_keys=True))
PY
    ;;
esac
