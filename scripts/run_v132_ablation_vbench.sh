#!/usr/bin/env bash
# Reuse the audited v129 VBench sharding implementation for a v132 comparison.
set -euo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "split" && "$ACTION" != "preflight" && \
      "$ACTION" != "eval" && "$ACTION" != "collect" ]]; then
    echo "usage: bash scripts/run_v132_ablation_vbench.sh split|preflight|eval|collect"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
: "${COMPARISON_ROOT:?set COMPARISON_ROOT to the assembled v132 comparison}"
MANIFEST="$COMPARISON_ROOT/comparison_manifest.json"
[[ -s "$MANIFEST" ]] || {
    echo "[error] missing comparison manifest: $MANIFEST"
    exit 2
}

METHOD_COUNT="$(
    python - "$MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["experiment"] == "v132_binary_memory_ablation_comparison_30s"
print(len(payload["methods"]))
PY
)"
export VBENCH_EXPECTED_EXPERIMENT="v132_binary_memory_ablation_comparison_30s"
export VBENCH_EXPECTED_METHOD_COUNT="$METHOD_COUNT"
export V129_METRIC_PROFILE="${V132_METRIC_PROFILE:-core}"

exec bash "$ROOT/scripts/run_v129_vbench_long.sh" "$ACTION"
