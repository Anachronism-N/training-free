#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-screen}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v164_direction_freshness_moviebench16/full8}"
PROMPTS="${PROMPTS:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.txt}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.json}"
METRICS="${METRICS:-${RUN_ROOT}/automated_screen}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5}"
mkdir -p "${METRICS}/comprehensive_parts" "${METRICS}/logs"

METHODS=(
  sf_native
  ours_middle10_reservoir2_directionmatch1
  ours_middle10_reservoir2_directionfresh1
  ours_middle10_reservoir2_statemotionpair1_reference
  ours_middle10_reservoir4_reference
  ours_all_recent8_reference
)

run_mechanism() {
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/analyze_v164_direction_freshness_trace.py" \
    --trace-dir "${RUN_ROOT}/traces" \
    --output "${METRICS}/direction_freshness_trace.json"
}

run_temporal() {
  inputs=()
  for method in "${METHODS[@]}"; do
    inputs+=("${RUN_ROOT}/published_indexed/${method}")
  done
  "${PYTHON_BIN}" "${ROOT}/scripts/compute_temporal_jump_diagnostic.py" \
    "${inputs[@]}" \
    --output "${METRICS}/temporal_diagnostics.csv" \
    --expected-videos 16 \
    --max-width 256 \
    --frame-step 2 \
    --workers "${TEMPORAL_WORKERS:-8}"
}

run_comprehensive() {
  IFS=',' read -r -a gpus <<< "${EVAL_GPUS}"
  if (( ${#gpus[@]} < ${#METHODS[@]} )); then
    echo "EVAL_GPUS must provide at least ${#METHODS[@]} GPUs" >&2
    return 2
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
      --gpu 0 \
      --sample_frames 64 \
      --batch_size 8 \
      >"${METRICS}/logs/comprehensive_${method}.log" 2>&1 &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "${failures} comprehensive workers failed" >&2
    return 1
  fi
  parts=()
  for method in "${METHODS[@]}"; do
    parts+=("${METRICS}/comprehensive_parts/${method}.json")
  done
  "${PYTHON_BIN}" "${ROOT}/scripts/merge_comprehensive_results.py" \
    "${parts[@]}" \
    --output "${METRICS}/comprehensive.json" \
    --expected-methods "${METHODS[@]}" \
    --expected-videos 16
}

run_screen() {
  "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v164_automated_screen.py" \
    --temporal-csv "${METRICS}/temporal_diagnostics.csv" \
    --comprehensive-json "${METRICS}/comprehensive.json" \
    --prompt-manifest "${PROMPT_MANIFEST}" \
    --published-manifest "${RUN_ROOT}/published_manifest.json" \
    --output-dir "${METRICS}"
}

case "${ACTION}" in
  mechanism)
    run_mechanism
    ;;
  temporal)
    run_temporal
    ;;
  comprehensive)
    run_comprehensive
    ;;
  screen)
    run_screen
    ;;
  all)
    run_mechanism
    run_temporal
    run_comprehensive
    run_screen
    echo "automatic screen complete; do not review videos until metrics are analyzed"
    ;;
  *)
    echo "usage: $0 {mechanism|temporal|comprehensive|screen|all}" >&2
    exit 2
    ;;
esac
