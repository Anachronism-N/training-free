#!/usr/bin/env bash
# CPU-only multi-axis analysis of frozen v134 head profiles.
set -euo pipefail

ACTION="${1:-all}"
case "$ACTION" in
    analyze|package|status|all)
        ;;
    *)
        echo "usage: bash scripts/run_v136_multi_axis_analysis.sh [analyze|package|status|all]"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
V134_ROOT="${V134_OUT_ROOT:-$ROOT/runs/v134_head_discovery}"
OBS_DIR="$V134_ROOT/profiles/observational"
CF_DIR="$V134_ROOT/profiles/counterfactual"
OUT_DIR="${V136_OUT_DIR:-$V134_ROOT/analysis_multi_axis_v136}"
PACKAGE_DIR="${V136_PACKAGE_DIR:-$ROOT/docs/results/v136_multi_axis_head_discovery}"
EXPECTED_COUNT="${EXPECTED_COUNT:-128}"
EXPECTED_STATES="${EXPECTED_STATES:-27}"
RECENT_FRAMES="${RECENT_FRAMES:-4}"
BOOTSTRAP_ROUNDS="${BOOTSTRAP_ROUNDS:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2026}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
}

count_profiles() {
    local directory="$1"
    if [[ ! -d "$directory" ]]; then
        printf '0\n'
        return
    fi
    find "$directory" -maxdepth 1 -type f -name '*.pt' | wc -l
}

audit_inputs() {
    local observational counterfactual
    observational="$(count_profiles "$OBS_DIR")"
    counterfactual="$(count_profiles "$CF_DIR")"
    echo "[v136-status] observational=$observational/$EXPECTED_COUNT counterfactual=$counterfactual/$EXPECTED_COUNT"
    [[ "$observational" -eq "$EXPECTED_COUNT" ]] || {
        echo "[error] incomplete observational profiles"
        exit 1
    }
    [[ "$counterfactual" -eq "$EXPECTED_COUNT" ]] || {
        echo "[error] incomplete counterfactual profiles"
        exit 1
    }
}

analyze() {
    audit_inputs
    activate_env
    python "$ROOT/scripts/analyze_v136_multi_axis_head_discovery.py" \
        --observational-dir "$OBS_DIR" \
        --counterfactual-dir "$CF_DIR" \
        --output-dir "$OUT_DIR" \
        --recent-frames "$RECENT_FRAMES" \
        --expected-count "$EXPECTED_COUNT" \
        --expected-states "$EXPECTED_STATES" \
        --bootstrap-rounds "$BOOTSTRAP_ROUNDS" \
        --bootstrap-seed "$BOOTSTRAP_SEED"
}

package() {
    activate_env
    python "$ROOT/scripts/package_v136_multi_axis_results.py" \
        --analysis-dir "$OUT_DIR" \
        --output-dir "$PACKAGE_DIR"
}

case "$ACTION" in
    analyze)
        analyze
        ;;
    package)
        package
        ;;
    status)
        audit_inputs
        ;;
    all)
        analyze
        package
        ;;
esac
