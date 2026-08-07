#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
if [[ "${ACTION}" != "prepare" && "${ACTION}" != "split" && \
      "${ACTION}" != "preflight" && "${ACTION}" != "eval" && \
      "${ACTION}" != "resume-missing" && "${ACTION}" != "status" && \
      "${ACTION}" != "collect" ]]; then
  echo "usage: $0 {prepare|split|preflight|eval|resume-missing|status|collect}" >&2
  exit 2
fi

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v164_direction_freshness_moviebench16/full8}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${RUN_ROOT}/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-${ROOT}/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${ROOT}/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-${RUN_ROOT}/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RUN_ROOT}/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RUN_ROOT}/analysis}"
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

if [[ "${ACTION}" == "prepare" ]]; then
  [[ "${NODE_RANK}" == "0" ]] || {
    echo "prepare requires rank 0" >&2
    exit 2
  }
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v164_vbench_comparison.py" \
    --run-root "${RUN_ROOT}" \
    --comparison-root "${COMPARISON_ROOT}"
  exit $?
fi

if [[ "${ACTION}" == "split" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v164_vbench_splits.py" \
    --comparison-root "${COMPARISON_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --workers "${V164_SPLIT_WORKERS:-2}" \
    --node-rank "${NODE_RANK}" \
    --num-nodes "${NUM_NODES}"
  exit $?
fi

if [[ "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V164_TORCH_HUB_DIR:-${ROOT}/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V164_RUNTIME_HOME:-${ROOT}/runs/_model_cache/dreamsim_home}"
if [[ ( "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ) && \
      "${V164_LOCAL_MODELS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_local_cache.py" \
    --vbench-cache "${VBENCH_CACHE_DIR}" \
    --torch-hub-dir "${TORCH_HUB_DIR}" \
    --runtime-home "${RUNTIME_HOME}"
fi

PYTHON_ACTION="${ACTION}"
if [[ "${ACTION}" == "resume-missing" ]]; then
  PYTHON_ACTION="eval-missing"
fi

EXTRA_ARGS=()
if [[ "${V164_LOCAL_MODELS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--local-models)
  EXTRA_ARGS+=(--torch-hub-dir "${TORCH_HUB_DIR}")
  EXTRA_ARGS+=(--runtime-home "${RUNTIME_HOME}")
fi

"${PYTHON_BIN}" "${ROOT}/scripts/run_v164_vbench_long.py" "${PYTHON_ACTION}" \
  --comparison-root "${COMPARISON_ROOT}" \
  --vbench-root "${VBENCH_ROOT}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" \
  --parts-root "${PARTS_ROOT}" \
  --summary-root "${SUMMARY_ROOT}" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --node-rank "${NODE_RANK}" \
  --num-nodes "${NUM_NODES}" \
  --gpu-list "${GPU_LIST}" \
  --summary-stem "vbench_core9_summary" \
  --analysis-stem "v164_vbench_analysis" \
  --summary-title "v164 Direction/Freshness VBench-Long Core-9" \
  "${EXTRA_ARGS[@]}"
