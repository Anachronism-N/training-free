#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "${ACTION}" in
  prepare|split|preflight|eval|resume-missing|status|collect|collect-core) ;;
  *)
    echo "usage: $0 {prepare|split|preflight|eval|resume-missing|status|collect|collect-core}" >&2
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v158_interleaved_budget_moviebench16/full8}"
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
  [[ "${NODE_RANK}" == "0" ]] || { echo "prepare requires rank 0" >&2; exit 2; }
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v158_vbench_comparison.py" \
    --run-root "${RUN_ROOT}" --comparison-root "${COMPARISON_ROOT}"
  exit $?
fi

if [[ "${ACTION}" == "split" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v158_vbench_splits.py" \
    --comparison-root "${COMPARISON_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --workers "${V158_SPLIT_WORKERS:-2}" \
    --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}"
  exit $?
fi

if [[ "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V158_TORCH_HUB_DIR:-${ROOT}/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V158_RUNTIME_HOME:-${ROOT}/runs/_model_cache/dreamsim_home}"
if [[ ( "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ) && \
      "${V158_LOCAL_MODELS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_local_cache.py" \
    --vbench-cache "${VBENCH_CACHE_DIR}" \
    --torch-hub-dir "${TORCH_HUB_DIR}" \
    --runtime-home "${RUNTIME_HOME}"
fi

PYTHON_ACTION="${ACTION}"
DIMENSIONS="${V158_VBENCH_DIMENSIONS:-}"
SUMMARY_STEM="vbench_long_summary"
ANALYSIS_STEM="v158_vbench_analysis"
SUMMARY_TITLE="v158 Interleaved Budget VBench-Long Summary"
if [[ "${ACTION}" == "resume-missing" ]]; then
  PYTHON_ACTION="eval-missing"
fi
if [[ "${ACTION}" == "collect-core" ]]; then
  PYTHON_ACTION="collect"
  DIMENSIONS="${V158_CORE_DIMENSIONS:-subject_consistency,background_consistency,temporal_flickering,motion_smoothness,overall_consistency,dynamic_degree,aesthetic_quality,imaging_quality,temporal_style}"
  SUMMARY_STEM="vbench_core9_summary"
  ANALYSIS_STEM="v158_vbench_core9_analysis"
  SUMMARY_TITLE="v158 Interleaved Budget VBench-Long Core-9 Summary"
fi

EXTRA_ARGS=()
if [[ -n "${DIMENSIONS}" ]]; then
  EXTRA_ARGS+=(--dimensions "${DIMENSIONS}")
fi
if [[ "${V158_LOCAL_MODELS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--local-models)
  EXTRA_ARGS+=(--torch-hub-dir "${TORCH_HUB_DIR}")
  EXTRA_ARGS+=(--runtime-home "${RUNTIME_HOME}")
fi

"${PYTHON_BIN}" "${ROOT}/scripts/run_v158_vbench_long.py" "${PYTHON_ACTION}" \
  --comparison-root "${COMPARISON_ROOT}" \
  --vbench-root "${VBENCH_ROOT}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" \
  --parts-root "${PARTS_ROOT}" \
  --summary-root "${SUMMARY_ROOT}" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}" \
  --gpu-list "${GPU_LIST}" \
  --summary-stem "${SUMMARY_STEM}" \
  --analysis-stem "${ANALYSIS_STEM}" \
  --summary-title "${SUMMARY_TITLE}" \
  "${EXTRA_ARGS[@]}"

if [[ "${ACTION}" == "collect" || "${ACTION}" == "collect-core" ]]; then
  PAPER_TABLE_ROOT="${SUMMARY_ROOT}/paper_table"
  if [[ "${ACTION}" == "collect-core" ]]; then
    PAPER_TABLE_ROOT="${SUMMARY_ROOT}/paper_table_core9"
  fi
  "${PYTHON_BIN}" "${ROOT}/scripts/build_v129_paper_table.py" \
    --summary-json "${SUMMARY_ROOT}/${SUMMARY_STEM}.json" \
    --comparison-manifest "${COMPARISON_ROOT}/comparison_manifest.json" \
    --vbench-root "${VBENCH_ROOT}" \
    --output-root "${PAPER_TABLE_ROOT}" \
    --title "${SUMMARY_TITLE} paper table"
fi
