#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "${ACTION}" in
  prepare|analyze) ;;
  *)
    echo "usage: $0 {prepare|analyze}" >&2
    exit 2
    ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v157_layer_gated_moviebench16/full8}"
REVIEW_ROOT="${REVIEW_ROOT:-${RUN_ROOT}/metric_screened_review64}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${RUN_ROOT}/analysis}"

if [[ "${ACTION}" == "prepare" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v157_metric_screened_review.py" \
    --run-root "${RUN_ROOT}" \
    --output-root "${REVIEW_ROOT}"
  exit $?
fi

"${PYTHON_BIN}" "${ROOT}/scripts/analyze_v157_metric_screened_review.py" \
  --run-root "${RUN_ROOT}" \
  --review-sheet \
    "${REVIEW_ROOT}/reviewer/v157_metric_screened_review.csv" \
  --blind-key \
    "${REVIEW_ROOT}/private/v157_metric_screened_blind_key.json" \
  --output-root "${ANALYSIS_ROOT}"
