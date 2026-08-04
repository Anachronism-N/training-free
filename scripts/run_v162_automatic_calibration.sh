#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "${ACTION}" in
  history-preflight|prepare|split|preflight|eval|resume-missing|status|collect|calibrate|prepare-review) ;;
  *)
    echo "usage: $0 {history-preflight|prepare|split|preflight|eval|resume-missing|status|collect|calibrate|prepare-review}" >&2
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SOURCE_RUN_ROOT="${V162_SOURCE_RUN_ROOT:-${ROOT}/runs/v161_state_matched_motion_moviebench16/full8}"
V162_ROOT="${V162_ROOT:-${ROOT}/runs/v162_automatic_calibration}"
COMPARISON_ROOT="${V162_COMPARISON_ROOT:-${V162_ROOT}/vbench_comparison}"
PARTS_ROOT="${V162_PARTS_ROOT:-${V162_ROOT}/vbench_parts}"
SUMMARY_ROOT="${V162_SUMMARY_ROOT:-${V162_ROOT}/metrics}"
ANALYSIS_ROOT="${V162_ANALYSIS_ROOT:-${V162_ROOT}/analysis}"
V157_PARTS_ROOT="${V157_PARTS_ROOT:-${ROOT}/runs/v157_layer_gated_moviebench16/full8/metrics/vbench_long_parts}"
VBENCH_ROOT="${VBENCH_ROOT:-${ROOT}/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${ROOT}/runs/vbench_cache}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "[error] require 0 <= NODE_RANK < NUM_NODES" >&2
  exit 2
fi

require_rank_zero() {
  if [[ "${NODE_RANK}" != "0" ]]; then
    echo "[error] ${ACTION} requires NODE_RANK=0" >&2
    exit 2
  fi
}

activate_runtime() {
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

if [[ "${ACTION}" == "history-preflight" ]]; then
  require_rank_zero
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v162_metric_human_calibration.py" \
    --v157-parts-root "${V157_PARTS_ROOT}" \
    --history-preflight
  exit $?
fi

if [[ "${ACTION}" == "prepare" ]]; then
  require_rank_zero
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v162_vbench_comparison.py" \
    --run-root "${SOURCE_RUN_ROOT}" \
    --comparison-root "${COMPARISON_ROOT}"
  exit $?
fi

if [[ "${ACTION}" == "split" ]]; then
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v162_vbench_splits.py" \
    --comparison-root "${COMPARISON_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --workers "${V162_SPLIT_WORKERS:-2}" \
    --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}"
  exit $?
fi

if [[ "${ACTION}" == "calibrate" ]]; then
  require_rank_zero
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v162_metric_human_calibration.py" \
    --v157-parts-root "${V157_PARTS_ROOT}" \
    --v162-parts-root "${PARTS_ROOT}" \
    --output "${ANALYSIS_ROOT}/v162_metric_human_calibration.json"
  exit $?
fi

if [[ "${ACTION}" == "prepare-review" ]]; then
  require_rank_zero
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v162_minimal_review.py" \
    --run-root "${SOURCE_RUN_ROOT}" \
    --calibration-report "${ANALYSIS_ROOT}/v162_metric_human_calibration.json" \
    --output-root "${V162_ROOT}/minimal_review"
  exit $?
fi

if [[ "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ]]; then
  activate_runtime
fi

TORCH_HUB_DIR="${V162_TORCH_HUB_DIR:-${ROOT}/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V162_RUNTIME_HOME:-${ROOT}/runs/_model_cache/dreamsim_home}"
if [[ ( "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ) && \
      "${V162_LOCAL_MODELS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_local_cache.py" \
    --vbench-cache "${VBENCH_CACHE_DIR}" \
    --torch-hub-dir "${TORCH_HUB_DIR}" \
    --runtime-home "${RUNTIME_HOME}"
fi

PYTHON_ACTION="${ACTION}"
RUN_NUM_NODES="${NUM_NODES}"
RUN_NODE_RANK="${NODE_RANK}"
if [[ "${ACTION}" == "resume-missing" ]]; then
  require_rank_zero
  PYTHON_ACTION="eval-missing"
  RUN_NUM_NODES=1
  RUN_NODE_RANK=0
fi
if [[ "${ACTION}" == "collect" ]]; then
  require_rank_zero
fi

EXTRA_ARGS=()
if [[ "${V162_LOCAL_MODELS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--local-models)
  EXTRA_ARGS+=(--torch-hub-dir "${TORCH_HUB_DIR}")
  EXTRA_ARGS+=(--runtime-home "${RUNTIME_HOME}")
fi

"${PYTHON_BIN}" "${ROOT}/scripts/run_v162_vbench_long.py" "${PYTHON_ACTION}" \
  --comparison-root "${COMPARISON_ROOT}" \
  --vbench-root "${VBENCH_ROOT}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" \
  --parts-root "${PARTS_ROOT}" \
  --summary-root "${SUMMARY_ROOT}" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --node-rank "${RUN_NODE_RANK}" --num-nodes "${RUN_NUM_NODES}" \
  --gpu-list "${GPU_LIST}" \
  --summary-stem "vbench_core9_summary" \
  --analysis-stem "v162_vbench_analysis" \
  --summary-title "v162 Automatic Calibration VBench-Long Core-9" \
  "${EXTRA_ARGS[@]}"
