#!/usr/bin/env bash
# Protocol-safe, camera-compensated motion audit over existing v183 videos.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    protocol-audit|correct-quality|preflight|motion-compute|motion-status|motion-collect|motion-analyze|show|package) ;;
    *)
        echo "usage: bash scripts/run_v204_existing_video_motion_audit.sh ACTION"
        echo "actions: protocol-audit correct-quality preflight motion-compute motion-status motion-collect motion-analyze show package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V183_ROOT="${V183_RECOVERY_ROOT:-$ROOT/runs/v180_rccp_fresh128/recovery_v183}"
V184_ROOT="${V184_ROOT:-$ROOT/runs/v184_retrieval_128}"
V185_ROOT="${V185_ROOT:-$ROOT/runs/v185_pf_baseline}"
V187_ROOT="${V187_ROOT:-$ROOT/runs/v187_hybrid_retrieval}"
OUT_ROOT="${V204_OUT_ROOT:-$ROOT/runs/v204_existing_video_motion_audit}"
MANIFEST="$V183_ROOT/vbench_comparison/comparison_manifest.json"
QUALITY_REPORT="$V183_ROOT/analysis/v202_v183_corrected_evidence.json"
PROTOCOL_REPORT="$OUT_ROOT/analysis/continuous_flow_protocol_audit.json"
PARTS_DIR="$OUT_ROOT/motion/parts"
MERGED_CSV="$OUT_ROOT/motion/camera_compensated_motion.csv"
MERGED_CONTRACT="$OUT_ROOT/motion/camera_compensated_motion.contract.json"
MOTION_REPORT="$OUT_ROOT/analysis/v204_sf_motion.json"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
WORKERS="${V204_WORKERS:-8}"
MAX_WIDTH="${V204_MAX_WIDTH:-256}"
FRAME_STEP="${V204_FRAME_STEP:-8}"
PART_CSV="$PARTS_DIR/part_$(printf '%02d' "$NODE_RANK")_of_$(printf '%02d' "$NUM_NODES").csv"
PART_CONTRACT="${PART_CSV%.csv}.contract.json"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
}

protocol_audit() {
    activate_env
    mkdir -p "$OUT_ROOT/analysis"
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v204_continuous_flow_protocol.py" \
        --method all_recent \
            "$V183_ROOT/vbench_comparison/comparison_manifest.json" \
            "$V183_ROOT/metrics/vbench_long_parts/all_recent/dynamic_degree/cont/v129_dynamic_degree_eval_results.json" \
        --method all_coverage_retrieval \
            "$V184_ROOT/vbench_comparison/comparison_manifest.json" \
            "$V184_ROOT/metrics/vbench_long_parts/all_coverage_retrieval/dynamic_degree/cont/v129_dynamic_degree_eval_results.json" \
        --method pf_native \
            "$V185_ROOT/vbench_comparison/comparison_manifest.json" \
            "$V185_ROOT/metrics/vbench_long_parts/pf_native/dynamic_degree/cont/v129_dynamic_degree_eval_results.json" \
        --method pf_hybrid_retrieval \
            "$V187_ROOT/vbench_comparison/comparison_manifest.json" \
            "$V187_ROOT/metrics/vbench_long_parts/pf_hybrid_retrieval/dynamic_degree/cont/v129_dynamic_degree_eval_results.json" \
        --comparison all_coverage_retrieval all_recent \
        --comparison pf_hybrid_retrieval pf_native \
        --output "$PROTOCOL_REPORT"
}

correct_quality() {
    activate_env
    PYTHON_BIN="$PYTHON_BIN" V183_RECOVERY_ROOT="$V183_ROOT" \
        bash "$ROOT/scripts/run_v202_v183_metric_correction.sh" analyze
}

preflight() {
    activate_env
    for path in "$MANIFEST" "$ROOT/scripts/compute_v193_camera_motion.py" \
        "$ROOT/scripts/analyze_v193_camera_motion.py"; do
        [[ -s "$path" ]] || { echo "[error] missing v204 input: $path"; exit 2; }
    done
    if [[ "$NODE_RANK" == "0" ]]; then
        (cd "$ROOT" && "$PYTHON_BIN" -m pytest -q \
            tests/test_v193_camera_motion.py \
            tests/test_v204_motion_protocol.py)
    fi
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import sys
from pathlib import Path
from compute_v193_camera_motion import load_video_grid
manifest, grid = load_video_grid(Path(sys.argv[1]))
methods = [str(row["key"]) for row in manifest["methods"]]
expected = ["sf_native", "rccp_matched", "all_recent", "all_coverage"]
if methods != expected or len(grid) != 512:
    raise SystemExit(f"unexpected v183 grid methods={methods} videos={len(grid)}")
print(f"[v204-preflight] prompts=128 methods={len(methods)} videos={len(grid)}")
PY
}

motion_compute() {
    preflight
    mkdir -p "$PARTS_DIR"
    "$PYTHON_BIN" "$ROOT/scripts/compute_v193_camera_motion.py" compute \
        --comparison-manifest "$MANIFEST" \
        --output "$PART_CSV" --contract "$PART_CONTRACT" \
        --max-width "$MAX_WIDTH" --frame-step "$FRAME_STEP" \
        --workers "$WORKERS" --shard-index "$NODE_RANK" --num-shards "$NUM_NODES"
}

motion_status() {
    activate_env
    "$PYTHON_BIN" - "$PARTS_DIR" "$NUM_NODES" <<'PY'
import csv, json, sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
complete = rows = 0
for shard in range(count):
    csv_path = root / f"part_{shard:02d}_of_{count:02d}.csv"
    contract_path = csv_path.with_suffix(".contract.json")
    if not csv_path.is_file() or not contract_path.is_file():
        print(f"[v204-status] shard={shard} missing")
        continue
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        observed = sum(1 for _ in csv.DictReader(handle))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ok = observed == int(contract.get("row_count", -1))
    print(f"[v204-status] shard={shard} rows={observed} ok={str(ok).lower()}")
    complete += int(ok)
    rows += observed if ok else 0
print(f"[v204-status] complete_shards={complete}/{count} rows={rows}/512")
PY
}

motion_collect() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] motion-collect requires NODE_RANK=0"; exit 2; }
    preflight
    mkdir -p "$OUT_ROOT/motion"
    "$PYTHON_BIN" "$ROOT/scripts/compute_v193_camera_motion.py" merge \
        --comparison-manifest "$MANIFEST" --parts-dir "$PARTS_DIR" \
        --output "$MERGED_CSV" --contract "$MERGED_CONTRACT" \
        --expected-shards "$NUM_NODES"
}

motion_analyze() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] motion-analyze requires NODE_RANK=0"; exit 2; }
    activate_env
    [[ -s "$MERGED_CSV" && -s "$MERGED_CONTRACT" ]] || {
        echo "[error] collect camera-compensated motion first"
        exit 2
    }
    correct_quality
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v204_sf_motion.py" \
        --comparison-manifest "$MANIFEST" \
        --motion-csv "$MERGED_CSV" --motion-contract "$MERGED_CONTRACT" \
        --corrected-quality-report "$QUALITY_REPORT" --output "$MOTION_REPORT"
}

show_results() {
    [[ -s "${PROTOCOL_REPORT%.json}.md" ]] || {
        echo "[error] run protocol-audit first"
        exit 2
    }
    cat "${PROTOCOL_REPORT%.json}.md"
    if [[ -s "${MOTION_REPORT%.json}.md" ]]; then
        cat "${MOTION_REPORT%.json}.md"
    else
        echo "[v204-show] camera-compensated SF audit is pending"
    fi
}

package_results() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires NODE_RANK=0"; exit 2; }
    for path in "$PROTOCOL_REPORT" "${PROTOCOL_REPORT%.json}.md" \
        "$MERGED_CSV" "$MERGED_CONTRACT" "$MOTION_REPORT" "${MOTION_REPORT%.json}.md"; do
        [[ -s "$path" ]] || { echo "[error] missing v204 artifact: $path"; exit 2; }
    done
    local archive="$OUT_ROOT/v204_existing_video_motion_small_artifacts.tar.gz"
    tar -C "$OUT_ROOT" -czf "$archive" analysis motion/camera_compensated_motion.csv \
        motion/camera_compensated_motion.contract.json
    echo "$archive"
}

case "$ACTION" in
    protocol-audit) protocol_audit ;;
    correct-quality) correct_quality ;;
    preflight) preflight ;;
    motion-compute) motion_compute ;;
    motion-status) motion_status ;;
    motion-collect) motion_collect ;;
    motion-analyze) motion_analyze ;;
    show) show_results ;;
    package) package_results ;;
esac
