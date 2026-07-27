#!/usr/bin/env bash
# Run the repository's eight-metric diagnostic suite on audited v116 videos.
set -uo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "eval" && "$ACTION" != "collect" ]]; then
    echo "usage: bash scripts/run_v116_aux_metrics.sh eval|collect"
    exit 2
fi
if [[ "${V115_PROMOTION_APPROVED:-0}" != "1" ]]; then
    echo "[blocked] set V115_PROMOTION_APPROVED=1 after one-video review"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
METHODS_CSV="${V116_METHODS:-prototype_motion1,snapshot_motion1,control_landmark_recent,control_all_recent8}"
METHOD_SET_ID="$(
    python - "$METHODS_CSV" <<'PY'
import hashlib
import sys

keys = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
print(f"m{len(keys)}_{hashlib.sha256(','.join(keys).encode()).hexdigest()[:12]}")
PY
)"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v116_role_memory_diverse16/$METHOD_SET_ID}"
MANIFEST="$RUN_ROOT/published_manifest.json"
METRICS_ROOT="${METRICS_ROOT:-$RUN_ROOT/metrics/auxiliary}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi

[[ -s "$MANIFEST" ]] || {
    echo "[error] missing audited v116 manifest: $MANIFEST"
    exit 2
}
mapfile -t META < <(
    python - "$MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("ok") or int(payload.get("prompt_count", 0)) != 16:
    raise SystemExit("invalid v116 published manifest")
print(payload["prompt_file"])
for row in payload["methods"]:
    print(row["key"])
PY
)
PROMPTS="${META[0]}"
METHODS=("${META[@]:1}")
[[ -s "$PROMPTS" ]] || { echo "[error] missing prompt subset: $PROMPTS"; exit 2; }

if [[ "$ACTION" == "collect" ]]; then
    python - "$METRICS_ROOT" "$RUN_ROOT/metrics" "${METHODS[@]}" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
methods = sys.argv[3:]
rows = []
failures = []
for method in methods:
    path = root / method / "results.json"
    if not path.is_file():
        failures.append(f"{method}: missing {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("per_method", {}).get(method)
    if not isinstance(aggregate, dict):
        failures.append(f"{method}: missing per_method aggregate")
        continue
    rows.append({"method": method, **aggregate})
if failures:
    raise SystemExit("\n".join(failures))
keys = sorted({key for row in rows for key in row if key != "method"})
output.mkdir(parents=True, exist_ok=True)
(output / "auxiliary_summary.json").write_text(
    json.dumps({"methods": rows}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with (output / "auxiliary_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", *keys])
    writer.writeheader()
    writer.writerows(rows)
lines = ["# v116 auxiliary metrics", "", "| method | " + " | ".join(keys) + " |",
         "|---|" + "|".join("---:" for _ in keys) + "|"]
for row in rows:
    values = []
    for key in keys:
        value = row.get(key, "")
        values.append(f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else str(value))
    lines.append(f"| {row['method']} | " + " | ".join(values) + " |")
(output / "auxiliary_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output / "auxiliary_summary.md")
PY
    exit $?
fi

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -gt 0 ]] || {
    echo "[error] GPU_LIST must contain at least one GPU id"
    exit 2
}
mkdir -p "$METRICS_ROOT"

PIDS=()
LOCAL_SLOT=0
for index in "${!METHODS[@]}"; do
    if (( index % NUM_NODES != NODE_RANK )); then
        continue
    fi
    method="${METHODS[$index]}"
    gpu="${GPUS[$((LOCAL_SLOT % ${#GPUS[@]}))]}"
    gpu="${gpu//[[:space:]]/}"
    output="$METRICS_ROOT/$method"
    mkdir -p "$output"
    if [[ -s "$output/results.json" ]]; then
        echo "[skip] method=$method results already exist"
        LOCAL_SLOT=$((LOCAL_SLOT + 1))
        continue
    fi
    echo "[launch] node=$NODE_RANK gpu=$gpu method=$method"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python "$ROOT/scripts/evaluate_comprehensive.py" \
            --video_dirs "$RUN_ROOT/published_indexed/$method" \
            --prompts "$PROMPTS" \
            --output "$output/results.json" \
            --gpu 0 \
            --sample_frames "$SAMPLE_FRAMES" \
            --batch_size "$BATCH_SIZE"
    ) >"$output/run.log" 2>&1 &
    PIDS+=("$!")
    LOCAL_SLOT=$((LOCAL_SLOT + 1))
done

STATUS=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done
echo "[complete] node=$NODE_RANK jobs=${#PIDS[@]} status=$STATUS"
exit "$STATUS"
