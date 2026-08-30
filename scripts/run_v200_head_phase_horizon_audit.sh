#!/usr/bin/env bash
# Zero-GPU cross-fit audit of AR-horizon structure in v189 profiles.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    preflight|analyze|show|package) ;;
    *)
        echo "usage: bash scripts/run_v200_head_phase_horizon_audit.sh ACTION"
        echo "actions: preflight analyze show package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V189_ROOT="${V189_OUT_ROOT:-$ROOT/runs/v189_structured_head_phase_profile}"
MANIFEST="${V189_MANIFEST:-$V189_ROOT/inputs/manifest.json}"
V189_ANALYSIS="${V189_ANALYSIS:-$V189_ROOT/analysis/analysis.json}"
V189_AUDIT="${V189_PROFILE_AUDIT:-$V189_ROOT/profile_audit.json}"
PROFILE_ROOT="${V189_PROFILE_ROOT:-$V189_ROOT/profiles}"
OUT_ROOT="${V200_OUT_ROOT:-$ROOT/runs/v200_head_phase_horizon_audit}"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
REPORT="$ANALYSIS_ROOT/analysis.json"

CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
BOOTSTRAP_SAMPLES="${V200_BOOTSTRAP_SAMPLES:-5000}"
PERMUTATIONS="${V200_PERMUTATIONS:-5000}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
}

preflight() {
    activate_env
    for path in "$MANIFEST" "$V189_ANALYSIS" "$V189_AUDIT" \
        "$PROFILE_ROOT/landmark" "$PROFILE_ROOT/retrieval"; do
        [[ -e "$path" ]] || {
            echo "[error] missing $path; finish v189 audit/analyze first"
            exit 2
        }
    done
    python "$ROOT/scripts/prepare_v189_structured_head_phase_profile.py" verify \
        --manifest "$MANIFEST"
    python - "$V189_ANALYSIS" "$V189_AUDIT" <<'PY'
import json, sys
from pathlib import Path
analysis, audit = [json.loads(Path(value).read_text(encoding="utf-8")) for value in sys.argv[1:]]
assert analysis.get("experiment") == "v189_structured_head_phase_profile"
assert analysis.get("manual_review_required") is False
assert audit.get("experiment") == "v189_structured_head_phase_profile_audit"
assert audit.get("ok") is True
print("[v200-preflight] v189_sources=valid zero_gpu=true")
PY
    if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q tests/test_v200_head_phase_horizon.py)
    fi
}

analyze() {
    preflight
    python "$ROOT/scripts/analyze_v200_head_phase_horizon.py" \
        --manifest "$MANIFEST" --v189-analysis "$V189_ANALYSIS" \
        --v189-profile-audit "$V189_AUDIT" --profile-root "$PROFILE_ROOT" \
        --output-dir "$ANALYSIS_ROOT" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --permutations "$PERMUTATIONS"
}

show() {
    [[ -s "$REPORT" ]] || {
        echo "[error] missing $REPORT; run analyze"
        exit 2
    }
    python - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"recommendation={report['recommendation']}")
print("generation_candidates=" + ",".join(report["generation_candidates"]))
for operator, row in report["operators"].items():
    primary = next(item for item in row["selector_tests"] if item["fraction"] == 0.10)
    print(
        f"{operator}:gate={str(row['horizon_conditioning_gate']).lower()} "
        f"delta={primary['paired_delta_mean']:.6f} "
        f"ci_lower={primary['paired_delta_ci95'][0]:.6f} "
        f"time_p={primary['time_assignment_permutation_p']:.4g}"
    )
print("manual_review_required=false")
PY
}

package() {
    [[ -s "$REPORT" ]] || {
        echo "[error] missing $REPORT; run analyze"
        exit 2
    }
    local archive="$OUT_ROOT/v200_head_phase_horizon_small_artifacts.tar.gz"
    tar -czf "$archive" -C "$OUT_ROOT" analysis
    echo "$archive"
}

case "$ACTION" in
    preflight) preflight ;;
    analyze) analyze ;;
    show) show ;;
    package) package ;;
esac
