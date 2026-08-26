#!/usr/bin/env bash
# Run and package the zero-GPU v197 Head x Phase structure audit.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    analyze|show|package) ;;
    *)
        echo "usage: bash scripts/run_v197_head_phase_structure.sh {analyze|show|package}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V189_ROOT="${V189_OUT_ROOT:-$ROOT/runs/v189_structured_head_phase_profile}"
OUT_ROOT="${V197_OUT_ROOT:-$ROOT/runs/v197_head_phase_structure}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
V189_ANALYSIS="$V189_ROOT/analysis/analysis.json"
CELL_SCORES="$V189_ROOT/analysis/cell_scores.csv"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
}

run_analysis() {
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v197_head_phase_structure.py" \
        --v189-analysis "$V189_ANALYSIS" --cell-scores "$CELL_SCORES" \
        --output-dir "$OUT_ROOT/analysis"
}

case "$ACTION" in
    analyze)
        run_analysis
        ;;
    show)
        run_analysis
        cat "$OUT_ROOT/analysis/analysis.md"
        ;;
    package)
        run_analysis
        archive="$OUT_ROOT/v197_small_artifacts.tar.gz"
        tar -czf "$archive" -C "$OUT_ROOT" analysis
        echo "[v197-package] $archive"
        ;;
esac
