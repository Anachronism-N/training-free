#!/usr/bin/env bash
# VBench-Long, temporal safety, and camera-motion selection for v199.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|split|eval|resume-missing|status|collect|temporal|camera-compute|camera-status|camera-collect|decision|package) ;;
    *)
        echo "usage: bash scripts/run_v199_vbench_long.sh {prepare|preflight|split|eval|resume-missing|status|collect|temporal|camera-compute|camera-status|camera-collect|decision|package}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${V199_OUT_ROOT:-$ROOT/runs/v199_retrieval_storage_attribution}"
COMPARISON_ROOT="$OUT_ROOT/vbench_comparison"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/third_party/VBench-Long}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/_model_cache/vbench_long}"
PARTS_ROOT="$OUT_ROOT/vbench_long_parts"
SUMMARY_ROOT="$OUT_ROOT/metrics"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
TEMPORAL_CSV="$SUMMARY_ROOT/temporal_diagnostics.csv"
TEMPORAL_CONTRACT="$SUMMARY_ROOT/temporal_diagnostics.contract.json"
REPORT="$ANALYSIS_ROOT/v199_retrieval_storage.json"
CAMERA_ROOT="$OUT_ROOT/camera_motion"
CAMERA_SHARED_ROOT="$CAMERA_ROOT/shared"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CANDIDATES=(retrieval_archive4 retrieval_archive8 retrieval_archive12)

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
}

compute_temporal() {
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -eq 4 ]] || {
        echo "[error] v199 temporal diagnostics require four methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 32 --max-width "${V199_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V199_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V199_TEMPORAL_WORKERS:-16}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

camera_report() {
    echo "$CAMERA_ROOT/$1/analysis/v193_camera_motion.json"
}

analyze_capacity() {
    local -a camera_args=()
    local candidate report
    for candidate in "${CANDIDATES[@]}"; do
        report="$(camera_report "$candidate")"
        if [[ -s "$report" ]]; then
            camera_args+=(--camera-motion-report "$candidate=$report")
        fi
    done
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v199_retrieval_storage.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" --output "$REPORT" \
        "${camera_args[@]}"
}

run_camera_shared() {
    local camera_action="$1"
    TARGET=custom \
    V193_SOURCE_RUN_ROOT="$OUT_ROOT" \
    COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
    QUALITY_REPORT="$REPORT" \
    CANDIDATE=retrieval_archive4 CONTROLS=all_recent \
    V193_OUT_ROOT="$CAMERA_SHARED_ROOT" \
    V193_WORKERS="${V199_CAMERA_WORKERS:-8}" \
    V193_MAX_WIDTH="${V199_CAMERA_WIDTH:-256}" \
    V193_FRAME_STEP="${V199_CAMERA_FRAME_STEP:-8}" \
    NODE_RANK="$NODE_RANK" NUM_NODES="$NUM_NODES" \
    CONDA_SH="$CONDA_SH" CONDA_ENV="$CONDA_ENV" PYTHON_BIN="$PYTHON_BIN" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$camera_action"
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v199_vbench_comparison.py" \
        --run-root "$OUT_ROOT" --comparison-root "$COMPARISON_ROOT"
    exit $?
fi

if [[ "$ACTION" == "preflight" ]]; then
    activate_env
    "$PYTHON_BIN" -m pytest -q \
        "$ROOT/tests/test_v199_retrieval_storage_attribution.py" \
        "$ROOT/tests/test_v199_retrieval_storage_evaluation.py"
    "$PYTHON_BIN" "$ROOT/scripts/run_v199_vbench_long.py" preflight \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
        --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST"
    exit $?
fi

if [[ "$ACTION" == "temporal" ]]; then
    activate_env
    compute_temporal
    exit $?
fi

if [[ "$ACTION" == "camera-compute" ]]; then
    [[ -s "$REPORT" ]] || { echo "[error] run collect before camera-compute"; exit 2; }
    run_camera_shared compute
    exit 0
fi

if [[ "$ACTION" == "camera-status" ]]; then
    run_camera_shared status
    exit 0
fi

if [[ "$ACTION" == "camera-collect" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] camera-collect requires node 0"; exit 2; }
    run_camera_shared collect
    activate_env
    for candidate in "${CANDIDATES[@]}"; do
        output="$(camera_report "$candidate")"
        mkdir -p "$(dirname "$output")"
        "$PYTHON_BIN" "$ROOT/scripts/analyze_v193_camera_motion.py" \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --motion-csv "$CAMERA_SHARED_ROOT/metrics/camera_compensated_motion.csv" \
            --motion-contract "$CAMERA_SHARED_ROOT/metrics/camera_compensated_motion.contract.json" \
            --candidate "$candidate" --controls all_recent \
            --quality-report "$REPORT" --output "$output"
    done
    analyze_capacity
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    for candidate in "${CANDIDATES[@]}"; do
        [[ -s "$(camera_report "$candidate")" ]] || {
            echo "[error] missing camera report for $candidate"; exit 2;
        }
    done
    activate_env
    analyze_capacity
    "$PYTHON_BIN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("[v199-decision] recommendation=" + payload["recommendation"])
print("selected_method=" + str(payload["selected_method"]))
print("selected_archive_capacity=" + str(payload["selected_archive_capacity"]))
print("equal_total_storage_retrieval_supported=" + str(payload["equal_total_storage_retrieval_supported"]).lower())
print("manual_review_required=" + str(payload["manual_review_required_for_decision"]).lower())
for row in payload["targeted_debug_queue"]:
    print(f"debug=p{row['prompt_index']}:source{row['source_index']}:flags={','.join(row['automatic_flags'])}")
PY
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
archive = root / "v199_evaluation_evidence.zip"
allowed = {".json", ".md", ".csv", ".txt"}
files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed and path.stat().st_size <= 8 * 1024 * 1024 and "vbench_long_parts" not in path.parts and "raw" not in path.parts and "published" not in path.parts]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in sorted(files): handle.write(path, path.relative_to(root))
payload = {"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": [str(path.relative_to(root)) for path in sorted(files)]}
(root / "evidence_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[v199-package] files={len(files)} archive={archive}")
PY
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V199_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" || \
      "$ACTION" == "status" || "$ACTION" == "collect" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V199_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V199_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V199_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V199_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v199_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary --analysis-stem v199_vbench_analysis \
    --summary-title "v199 Retrieval Storage Attribution VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    analyze_capacity
fi
