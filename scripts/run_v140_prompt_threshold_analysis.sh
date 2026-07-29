#!/usr/bin/env bash
# CPU-only held-out threshold audit over frozen v134/v136 prompt profiles.
set -euo pipefail

ACTION="${1:-all}"
case "$ACTION" in
    analyze|status|all)
        ;;
    *)
        echo "usage: bash scripts/run_v140_prompt_threshold_analysis.sh [analyze|status|all]"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
V136_ROOT="${V136_OUT_DIR:-$ROOT/runs/v134_head_discovery/analysis_multi_axis_v136}"
INPUT="${V140_PROMPT_JOB_AXES:-$V136_ROOT/head_prompt_job_axes.csv}"
OUTPUT="${V140_OUT_DIR:-$ROOT/runs/v134_head_discovery/analysis_prompt_threshold_v140}"
EXPECTED_JOBS="${EXPECTED_JOBS:-128}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
}

status() {
    if [[ -f "$INPUT" ]]; then
        echo "[v140-status] input=ready path=$INPUT"
    else
        echo "[v140-status] input=missing path=$INPUT"
        return 1
    fi
}

analyze() {
    status
    activate_env
    python "$ROOT/scripts/analyze_v140_prompt_threshold_robustness.py" \
        --prompt-job-axes "$INPUT" \
        --output-dir "$OUTPUT" \
        --expected-jobs "$EXPECTED_JOBS"
    echo "[v140] wrote $OUTPUT/threshold_summary.md"
}

case "$ACTION" in
    status)
        status
        ;;
    analyze|all)
        analyze
        ;;
esac
