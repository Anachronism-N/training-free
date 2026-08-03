#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-prepare}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/v159_motion_coherent_reservoir_moviebench16/full8}"
REVIEW_ROOT="${REVIEW_ROOT:-${RUN_ROOT}/blind_review64}"

case "${ACTION}" in
  prepare)
    "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v159_blind_review.py" \
      --run-root "${RUN_ROOT}" --output-root "${REVIEW_ROOT}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v159_blind_review.py" \
      --review-sheet "${REVIEW_ROOT}/reviewer/v159_review_sheet.csv" \
      --blind-key "${REVIEW_ROOT}/private/v159_blind_key.json" \
      --output-root "${RUN_ROOT}/analysis"
    ;;
  *)
    echo "usage: $0 {prepare|analyze}" >&2
    exit 2
    ;;
esac
