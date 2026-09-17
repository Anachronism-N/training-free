#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
  prepare|split|preflight|eval|resume-missing|status|collect|temporal|bind-temporal|analyze|postprocess|decision) ;;
  *)
    echo "usage: bash scripts/run_v211_postprocess.sh {prepare|split|preflight|eval|resume-missing|status|collect|temporal|bind-temporal|analyze|postprocess|decision}" >&2
    exit 2
    ;;
esac

ROOT="${V211_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${V211_PYTHON_BIN:-python}"
GENERATION_ROOT="${V211_GENERATION_ROOT:?V211_GENERATION_ROOT is required}"
GENERATION_COMMIT="${V211_GENERATION_COMMIT:?V211_GENERATION_COMMIT is required}"
EVALUATION_COMMIT="${V211_EVALUATION_COMMIT:?V211_EVALUATION_COMMIT is required}"
NODE_ADDRESS="${V211_NODE_ADDRESS:?V211_NODE_ADDRESS is required}"
SCREEN_ROOT="${V211_SCREEN_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v211_runs/v211_eval_${EVALUATION_COMMIT:0:8}_screen8}"
COMPARISON_ROOT="${V211_COMPARISON_ROOT:-$SCREEN_ROOT/vbench_comparison}"
METRICS_ROOT="${V211_METRICS_ROOT:-$SCREEN_ROOT/metrics}"
PARTS_ROOT="${V211_PARTS_ROOT:-$METRICS_ROOT/vbench_long_parts}"
ANALYSIS_ROOT="${V211_ANALYSIS_ROOT:-$SCREEN_ROOT/analysis}"

VBENCH_ROOT="${V211_VBENCH_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench}"
SHARED_MODEL_ROOT="${V211_SHARED_MODEL_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs}"
VBENCH_CACHE_DIR="${V211_VBENCH_CACHE_DIR:-$SHARED_MODEL_ROOT/vbench_cache}"
TORCH_HUB_DIR="${V211_TORCH_HUB_DIR:-$SHARED_MODEL_ROOT/_model_cache/torch_hub}"
RUNTIME_HOME="${V211_RUNTIME_HOME:-$SHARED_MODEL_ROOT/_model_cache/dreamsim_home}"
ACTIVATE_SH="${V211_ACTIVATE_SH:-/apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh}"
CONDA_ENV="${V211_CONDA_ENV:-longlive}"
NODE_RANK="${V211_NODE_RANK:-0}"
NUM_NODES="${V211_NUM_NODES:-6}"
GPU_LIST="${V211_GPU_LIST:-0,1,2,3,4,5,6,7}"

case "${GENERATION_ROOT,,}:${SCREEN_ROOT,,}:${COMPARISON_ROOT,,}:${METRICS_ROOT,,}:${PARTS_ROOT,,}:${ANALYSIS_ROOT,,}" in
  *v209*|*v210*)
    echo "[error] v211 generation/comparison roots must not reference v209/v210 artifacts" >&2
    exit 2
    ;;
esac

if [[ "$GENERATION_COMMIT" != "0cde4689ee4c2dc0d29b4aa386720e96dd51a5b6" ]]; then
  echo "[error] V211_GENERATION_COMMIT must equal the frozen generation SHA" >&2
  exit 2
fi
if [[ ! "$EVALUATION_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[error] V211_EVALUATION_COMMIT must be a full git commit" >&2
  exit 2
fi
if [[ "$EVALUATION_COMMIT" == "$GENERATION_COMMIT" ]]; then
  echo "[error] V211_EVALUATION_COMMIT must be distinct from generation" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$EVALUATION_COMMIT" ]]; then
  echo "[error] checkout HEAD does not match V211_EVALUATION_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "[error] v211 postprocessing requires an exact clean evaluation checkout" >&2
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
  "$PYTHON_BIN" "$ROOT/scripts/run_v211_vbench.py" "$mode" \
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
    --analysis-stem v211_vbench_analysis \
    --summary-title "v211 LPHC Screen8 VBench-Long Core-9"
}

run_temporal() {
  local inputs=()
  local methods=(
    sf_fifo21
    sf_sink1_21
    lphc_sink1_e1_a002_r4
    lphc_sink1_e1_a010_r4
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
    --workers "${V211_TEMPORAL_WORKERS:-8}"
}

bind_temporal() {
  "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
    --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
    --temporal-csv "$METRICS_ROOT/temporal_diagnostics.csv" \
    --output "$METRICS_ROOT/temporal_diagnostics.contract.json"
}

run_analysis() {
  "$PYTHON_BIN" "$ROOT/scripts/analyze_v211_lphc_screen.py" \
    --screen-root "$SCREEN_ROOT"
}

case "$ACTION" in
  prepare)
    activate_longlive
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v211_vbench_comparison.py" \
      --run-root "$GENERATION_ROOT" \
      --generation-commit "$GENERATION_COMMIT" \
      --comparison-root "$COMPARISON_ROOT" \
      --repo-root "$ROOT" \
      --vbench-root "$VBENCH_ROOT" \
      --node-rank "$NODE_RANK" \
      --num-nodes "$NUM_NODES" \
      --node-address "$NODE_ADDRESS"
    ;;
  split)
    activate_longlive
    "$PYTHON_BIN" "$ROOT/scripts/run_v211_vbench.py" split \
      --comparison-root "$COMPARISON_ROOT" \
      --evaluation-root "$ROOT" \
      --vbench-root "$VBENCH_ROOT" \
      --workers "${V211_SPLIT_WORKERS:-2}" \
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
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v211_lphc_screen.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
selected = payload.get("selected_candidate")
count = payload.get("selection_count")
if count not in (0, 1) or (count == 0) != (selected is None):
    raise SystemExit("invalid v211 selection contract")
print(json.dumps({
    "recommendation": payload.get("recommendation"),
    "selection_count": count,
    "selected_candidate": selected,
}, sort_keys=True))
PY
    ;;
esac
