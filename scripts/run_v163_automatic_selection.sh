#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "${ACTION}" in
  mechanism|temporal|comprehensive|prepare|split|preflight|eval|resume-missing|status|collect|select|prepare-review) ;;
  *)
    echo "usage: $0 {mechanism|temporal|comprehensive|prepare|split|preflight|eval|resume-missing|status|collect|select|prepare-review}" >&2
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${V163_RUN_ROOT:-${ROOT}/runs/v163_recency_regularized_state_motion_moviebench16/full8}"
METRICS="${V163_METRICS_ROOT:-${RUN_ROOT}/automated_selection}"
COMPARISON_ROOT="${V163_COMPARISON_ROOT:-${RUN_ROOT}/vbench_comparison}"
PARTS_ROOT="${V163_PARTS_ROOT:-${RUN_ROOT}/vbench_parts}"
SUMMARY_ROOT="${V163_SUMMARY_ROOT:-${METRICS}/vbench_metrics}"
ANALYSIS_ROOT="${V163_ANALYSIS_ROOT:-${METRICS}}"
CALIBRATION_REPORT="${V162_CALIBRATION_REPORT:-${ROOT}/runs/v162_automatic_calibration/analysis/v162_metric_human_calibration.json}"
PROMPTS="${PROMPTS:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.txt}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.json}"
VBENCH_ROOT="${VBENCH_ROOT:-${ROOT}/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${ROOT}/runs/vbench_cache}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5}"
mkdir -p "${METRICS}/comprehensive_parts" "${METRICS}/logs"

METHODS=(
  sf_native
  ours_middle10_reservoir2_stateage12motionpair1
  ours_middle10_reservoir2_statebalancedmotionpair1
  ours_middle10_reservoir2_statemotionpair1_reference
  ours_middle10_reservoir2_freshmotionpair1_reference
  ours_middle10_reservoir4_reference
)

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

if [[ "${ACTION}" == "mechanism" ]]; then
  require_rank_zero
  "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v163_recency_trace.py" \
    --trace-dir "${RUN_ROOT}/traces" \
    --output "${METRICS}/recency_trace.json"
  exit $?
fi

if [[ "${ACTION}" == "temporal" ]]; then
  require_rank_zero
  inputs=()
  for method in "${METHODS[@]}"; do
    inputs+=("${RUN_ROOT}/published_indexed/${method}")
  done
  "${PYTHON_BIN}" "${ROOT}/scripts/compute_temporal_jump_diagnostic.py" \
    "${inputs[@]}" \
    --output "${METRICS}/temporal_diagnostics.csv" \
    --expected-videos 16 --max-width 256 --frame-step 2 \
    --workers "${TEMPORAL_WORKERS:-8}"
  exit $?
fi

if [[ "${ACTION}" == "comprehensive" ]]; then
  require_rank_zero
  IFS=',' read -r -a gpus <<< "${EVAL_GPUS}"
  if (( ${#gpus[@]} < ${#METHODS[@]} )); then
    echo "EVAL_GPUS must provide at least ${#METHODS[@]} GPUs" >&2
    exit 2
  fi
  pids=()
  for index in "${!METHODS[@]}"; do
    method="${METHODS[$index]}"
    gpu="${gpus[$index]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      "${ROOT}/scripts/evaluate_comprehensive.py" \
      --video_dirs "${RUN_ROOT}/published_indexed/${method}" \
      --prompts "${PROMPTS}" \
      --output "${METRICS}/comprehensive_parts/${method}.json" \
      --gpu 0 --sample_frames 64 --batch_size 8 \
      >"${METRICS}/logs/comprehensive_${method}.log" 2>&1 &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
  done
  if (( failures > 0 )); then
    echo "${failures} comprehensive workers failed" >&2
    exit 1
  fi
  parts=()
  for method in "${METHODS[@]}"; do
    parts+=("${METRICS}/comprehensive_parts/${method}.json")
  done
  "${PYTHON_BIN}" "${ROOT}/scripts/merge_comprehensive_results.py" \
    "${parts[@]}" --output "${METRICS}/comprehensive.json" \
    --expected-methods "${METHODS[@]}" --expected-videos 16
  exit $?
fi

if [[ "${ACTION}" == "prepare" ]]; then
  require_rank_zero
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v163_vbench_comparison.py" \
    --run-root "${RUN_ROOT}" --comparison-root "${COMPARISON_ROOT}" \
    --prompt-manifest "${PROMPT_MANIFEST}"
  exit $?
fi

if [[ "${ACTION}" == "split" ]]; then
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v163_vbench_splits.py" \
    --comparison-root "${COMPARISON_ROOT}" --vbench-root "${VBENCH_ROOT}" \
    --workers "${V163_SPLIT_WORKERS:-2}" \
    --node-rank "${NODE_RANK}" --num-nodes "${NUM_NODES}"
  exit $?
fi

if [[ "${ACTION}" == "select" ]]; then
  require_rank_zero
  activate_runtime
  "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v163_automatic_selection.py" \
    --vbench-parts-root "${PARTS_ROOT}" \
    --temporal-csv "${METRICS}/temporal_diagnostics.csv" \
    --comprehensive-json "${METRICS}/comprehensive.json" \
    --trace-report "${METRICS}/recency_trace.json" \
    --calibration-report "${CALIBRATION_REPORT}" \
    --output "${METRICS}/automatic_selection.json"
  exit $?
fi

if [[ "${ACTION}" == "prepare-review" ]]; then
  require_rank_zero
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v163_minimal_review.py" \
    --run-root "${RUN_ROOT}" \
    --selection-report "${METRICS}/automatic_selection.json" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --output-root "${RUN_ROOT}/minimal_review"
  exit $?
fi

if [[ "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ]]; then
  activate_runtime
fi
TORCH_HUB_DIR="${V163_TORCH_HUB_DIR:-${ROOT}/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V163_RUNTIME_HOME:-${ROOT}/runs/_model_cache/dreamsim_home}"
if [[ ( "${ACTION}" == "eval" || "${ACTION}" == "resume-missing" ) && \
      "${V163_LOCAL_MODELS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v155_vbench_local_cache.py" \
    --vbench-cache "${VBENCH_CACHE_DIR}" \
    --torch-hub-dir "${TORCH_HUB_DIR}" --runtime-home "${RUNTIME_HOME}"
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
if [[ "${ACTION}" == "collect" || "${ACTION}" == "status" ]]; then
  require_rank_zero
fi
EXTRA_ARGS=()
if [[ "${V163_LOCAL_MODELS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--local-models)
  EXTRA_ARGS+=(--torch-hub-dir "${TORCH_HUB_DIR}")
  EXTRA_ARGS+=(--runtime-home "${RUNTIME_HOME}")
fi

"${PYTHON_BIN}" "${ROOT}/scripts/run_v163_vbench_long.py" "${PYTHON_ACTION}" \
  --comparison-root "${COMPARISON_ROOT}" --vbench-root "${VBENCH_ROOT}" \
  --vbench-cache "${VBENCH_CACHE_DIR}" --parts-root "${PARTS_ROOT}" \
  --summary-root "${SUMMARY_ROOT}" --analysis-root "${ANALYSIS_ROOT}" \
  --node-rank "${RUN_NODE_RANK}" --num-nodes "${RUN_NUM_NODES}" \
  --gpu-list "${GPU_LIST}" --summary-stem "vbench_core9_summary" \
  --analysis-stem "v163_vbench_analysis" \
  --summary-title "v163 Recency-Regularized VBench-Long Core-9" \
  "${EXTRA_ARGS[@]}"
