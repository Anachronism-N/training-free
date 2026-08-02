#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
if [[ "${ACTION}" != "prepare" && "${ACTION}" != "split" && \
      "${ACTION}" != "preflight" && "${ACTION}" != "eval" && \
      "${ACTION}" != "collect" ]]; then
  echo "usage: $0 {prepare|split|preflight|eval|collect}" >&2
  exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v155_profile_aligned_moviebench16/full7}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${RUN_ROOT}/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-${ROOT}/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HOME}/.cache/vbench}"
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
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_comparison.py" \
    --run-root "${RUN_ROOT}" --comparison-root "${COMPARISON_ROOT}"
  exit $?
fi

if [[ "${ACTION}" == "split" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_splits.py" \
    --comparison-root "${COMPARISON_ROOT}" \
    --vbench-root "${VBENCH_ROOT}" \
    --workers "${V155_SPLIT_WORKERS:-2}" \
    --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}"
  exit $?
fi

if [[ "${ACTION}" == "eval" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

"${PYTHON_BIN}" "${ROOT}/scripts/run_v155_vbench_long.py" "${ACTION}" \
  --comparison-root "${COMPARISON_ROOT}" \
  --vbench-root "${VBENCH_ROOT}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" \
  --parts-root "${PARTS_ROOT}" \
  --summary-root "${SUMMARY_ROOT}" \
  --analysis-root "${ANALYSIS_ROOT}" \
  --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}" \
  --gpu-list "${GPU_LIST}"

if [[ "${ACTION}" == "collect" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/build_v129_paper_table.py" \
    --summary-json "${SUMMARY_ROOT}/vbench_long_summary.json" \
    --comparison-manifest "${COMPARISON_ROOT}/comparison_manifest.json" \
    --vbench-root "${VBENCH_ROOT}" \
    --output-root "${SUMMARY_ROOT}/paper_table" \
    --title "v155 profile-aligned VBench-Long paper table"
fi
