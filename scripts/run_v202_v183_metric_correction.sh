#!/usr/bin/env bash
# Recompute v183 evidence after the corrected torchvision-RAFT rerun.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    analyze|show|package) ;;
    *)
        echo "usage: bash scripts/run_v202_v183_metric_correction.sh {analyze|show|package}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECOVERY_ROOT="${V183_RECOVERY_ROOT:-$ROOT/runs/v180_rccp_fresh128/recovery_v183}"
COMPARISON_ROOT="$RECOVERY_ROOT/vbench_comparison"
SUMMARY="$RECOVERY_ROOT/metrics/vbench_core9_summary.json"
PARTS_ROOT="$RECOVERY_ROOT/metrics/vbench_long_parts"
OUTPUT="$RECOVERY_ROOT/analysis/v202_v183_corrected_evidence.json"
PYTHON_BIN="${PYTHON_BIN:-python}"

analyze() {
    for path in "$COMPARISON_ROOT/comparison_manifest.json" "$SUMMARY" "$PARTS_ROOT"; do
        [[ -e "$path" ]] || { echo "[error] missing v183 artifact: $path"; exit 2; }
    done
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v202_v183_metric_correction.py" \
        --comparison-root "$COMPARISON_ROOT" --summary "$SUMMARY" \
        --parts-root "$PARTS_ROOT" --output "$OUTPUT"
}

show() {
    [[ -s "$OUTPUT" ]] || { echo "[error] run analyze first: $OUTPUT"; exit 2; }
    "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v202-decision] {payload['recommendation']}")
sf = payload["corrected_aggregate_scores"]["sf_native"]["official_quality_score"]
for method, row in payload["corrected_aggregate_scores"].items():
    quality = row["official_quality_score"]
    print(f"{method}: corrected_quality={quality:.6f} delta_vs_sf={quality-sf:+.6f}")
print("dynamic_degree_informative=false")
print("manual_review_required=false")
PY
}

package() {
    [[ -s "$OUTPUT" && -s "${OUTPUT%.json}.md" ]] || {
        echo "[error] run analyze first: $OUTPUT"; exit 2;
    }
    local archive="$RECOVERY_ROOT/v202_v183_metric_correction.tar.gz"
    tar -C "$RECOVERY_ROOT" -czf "$archive" \
        analysis/v202_v183_corrected_evidence.json \
        analysis/v202_v183_corrected_evidence.md \
        metrics/vbench_long_parts/sf_native/dynamic_degree/v129_dynamic_degree_eval_results.json \
        metrics/vbench_long_parts/rccp_matched/dynamic_degree/v129_dynamic_degree_eval_results.json \
        metrics/vbench_long_parts/all_recent/dynamic_degree/v129_dynamic_degree_eval_results.json \
        metrics/vbench_long_parts/all_coverage/dynamic_degree/v129_dynamic_degree_eval_results.json
    echo "$archive"
}

case "$ACTION" in
    analyze) analyze ;;
    show) show ;;
    package) package ;;
esac
