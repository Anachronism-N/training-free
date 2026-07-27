#!/usr/bin/env bash
# Evaluate an audited v120 MovieBench-32 method set with VBench-Long.
set -uo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "eval" && "$ACTION" != "collect" ]]; then
    echo "usage: bash scripts/run_v120_vbench_long.sh eval|collect"
    exit 2
fi
if [[ "${V119_PROMOTION_APPROVED:-0}" != "1" ]]; then
    echo "[blocked] set V119_PROMOTION_APPROVED=1 after one-video review"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CANDIDATES_CSV="${V120_CANDIDATES:-landmark_motion1}"
METHOD_SET_ID="$(
    python - "$CANDIDATES_CSV" <<'PY'
import hashlib
import sys

keys = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
if not 1 <= len(keys) <= 2 or len(keys) != len(set(keys)):
    raise SystemExit("V120_CANDIDATES must contain one or two unique keys")
print(f"ours{len(keys)}_{hashlib.sha256(','.join(keys).encode()).hexdigest()[:12]}")
PY
)" || exit 2
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v120_moviebench32_main/$METHOD_SET_ID}"
MANIFEST="$RUN_ROOT/published_manifest.json"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
METRICS_ROOT="${METRICS_ROOT:-$RUN_ROOT/metrics/vbench_long}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi

[[ -s "$MANIFEST" ]] || {
    echo "[error] missing audited v120 manifest: $MANIFEST"
    exit 2
}
mapfile -t METHODS < <(
    python - "$MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("ok"):
    raise SystemExit("published manifest is not successful")
if int(payload.get("prompt_count", 0)) != 32:
    raise SystemExit("published manifest is not MovieBench-32")
for row in payload["methods"]:
    print(row["key"])
PY
)
[[ "${#METHODS[@]}" -ge 3 ]] || {
    echo "[error] manifest must contain SF, PF, and at least one ours method"
    exit 2
}

DIMS=(
    subject_consistency
    background_consistency
    aesthetic_quality
    imaging_quality
    motion_smoothness
    dynamic_degree
)

if [[ "$ACTION" == "collect" ]]; then
    mkdir -p "$RUN_ROOT/metrics"
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS_ROOT" \
        --methods "${METHODS[@]}" \
        --dimensions "${DIMS[@]}" \
        --output-json "$RUN_ROOT/metrics/vbench_long_summary.json" \
        --output-csv "$RUN_ROOT/metrics/vbench_long_summary.csv" \
        --output-md "$RUN_ROOT/metrics/vbench_long_summary.md"
    exit $?
fi

EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
[[ -f "$EVAL" ]] || { echo "[error] missing evaluator: $EVAL"; exit 2; }
[[ -f "$INFO" ]] || { echo "[error] missing metadata: $INFO"; exit 2; }
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
    [[ -n "$gpu" ]] || { echo "[error] empty GPU id"; exit 2; }
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
        cd "$VBENCH_ROOT" || exit 2
        python "$EVAL" \
            --videos_path "$RUN_ROOT/published/$method" \
            --dimension "${DIMS[@]}" \
            --mode long_custom_input --dev_flag \
            --num_of_samples_per_prompt 1 \
            --output_path "$output" --full_json_dir "$INFO"
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
