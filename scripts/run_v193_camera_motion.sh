#!/usr/bin/env bash
# Camera-compensated motion diagnostics over existing v191/v192 videos.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    preflight|compute|status|collect|analyze|all-local) ;;
    *)
        echo "usage: bash scripts/run_v193_camera_motion.sh {preflight|compute|status|collect|analyze|all-local}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET="${TARGET:-v191_confirm128}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
WORKERS="${V193_WORKERS:-8}"
MAX_WIDTH="${V193_MAX_WIDTH:-256}"
FRAME_STEP="${V193_FRAME_STEP:-8}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$TARGET" in
    v191_confirm128)
        RUN_ROOT="${V193_SOURCE_RUN_ROOT:-$ROOT/runs/v191_head_phase_confirmation/confirm128}"
        ;;
    v192_seed2026_30s_128)
        RUN_ROOT="${V193_SOURCE_RUN_ROOT:-$ROOT/runs/v192_head_phase_robustness/scopes/seed2026_30s_128}"
        ;;
    v192_long60_seed10000_32)
        RUN_ROOT="${V193_SOURCE_RUN_ROOT:-$ROOT/runs/v192_head_phase_robustness/scopes/long60_seed10000_32}"
        ;;
    custom)
        RUN_ROOT="${V193_SOURCE_RUN_ROOT:?V193_SOURCE_RUN_ROOT is required for TARGET=custom}"
        ;;
    *)
        echo "[error] unsupported TARGET=$TARGET"
        exit 2
        ;;
esac

COMPARISON_MANIFEST="${COMPARISON_MANIFEST:-$RUN_ROOT/vbench_comparison/comparison_manifest.json}"
QUALITY_REPORT="${QUALITY_REPORT:-}"
if [[ -z "$QUALITY_REPORT" ]]; then
    case "$TARGET" in
        v191_confirm128)
            QUALITY_REPORT="$RUN_ROOT/analysis/v191_head_phase_confirmation.json"
            ;;
        v192_seed2026_30s_128|v192_long60_seed10000_32)
            QUALITY_REPORT="$RUN_ROOT/analysis/v192_scope_analysis.json"
            ;;
        custom) QUALITY_REPORT="" ;;
    esac
fi
CANDIDATE="${CANDIDATE:-head_phase_joint}"
CONTROLS="${CONTROLS:-all_recent,sf_native}"
OUT_ROOT="${V193_OUT_ROOT:-$ROOT/runs/v193_camera_motion/$TARGET}"
PARTS_DIR="$OUT_ROOT/parts"
PART_CSV="$PARTS_DIR/part_$(printf '%02d' "$NODE_RANK")_of_$(printf '%02d' "$NUM_NODES").csv"
PART_CONTRACT="${PART_CSV%.csv}.contract.json"
MERGED_CSV="$OUT_ROOT/metrics/camera_compensated_motion.csv"
MERGED_CONTRACT="$OUT_ROOT/metrics/camera_compensated_motion.contract.json"
ANALYSIS="$OUT_ROOT/analysis/v193_camera_motion.json"

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

preflight() {
    [[ -s "$COMPARISON_MANIFEST" ]] || {
        echo "[error] missing comparison manifest: $COMPARISON_MANIFEST"
        echo "[error] finish the corresponding v191/v192 VBench prepare step first"
        exit 2
    }
    activate_env
    "$PYTHON_BIN" - "$COMPARISON_MANIFEST" "$CANDIDATE" "$CONTROLS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
methods = [str(row.get("key", "")) for row in payload.get("methods") or ()]
required = [sys.argv[2], *[value for value in sys.argv[3].split(",") if value]]
if len(set(methods)) != len(methods) or not methods or int(payload.get("prompt_count", -1)) <= 0:
    raise SystemExit("invalid comparison manifest method grid")
missing = [method for method in required if method not in methods]
if missing:
    raise SystemExit(f"candidate/control methods absent from manifest: {missing}")
print(f"[v193-preflight] experiment={payload.get('experiment')} prompts={payload['prompt_count']} methods={methods}")
PY
}

compute() {
    preflight
    mkdir -p "$PARTS_DIR"
    "$PYTHON_BIN" "$ROOT/scripts/compute_v193_camera_motion.py" compute \
        --comparison-manifest "$COMPARISON_MANIFEST" \
        --output "$PART_CSV" --contract "$PART_CONTRACT" \
        --max-width "$MAX_WIDTH" --frame-step "$FRAME_STEP" \
        --workers "$WORKERS" --shard-index "$NODE_RANK" --num-shards "$NUM_NODES"
}

status() {
    "$PYTHON_BIN" - "$PARTS_DIR" "$NUM_NODES" <<'PY'
import csv, json, sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
complete = 0
rows = 0
for shard in range(count):
    csv_path = root / f"part_{shard:02d}_of_{count:02d}.csv"
    contract_path = csv_path.with_suffix(".contract.json")
    if not csv_path.is_file() or not contract_path.is_file():
        print(f"[v193-status] shard={shard} missing")
        continue
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        observed = sum(1 for _ in csv.DictReader(handle))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ok = observed == int(contract.get("row_count", -1))
    print(f"[v193-status] shard={shard} rows={observed} ok={str(ok).lower()}")
    complete += int(ok)
    rows += observed if ok else 0
print(f"[v193-status] complete_shards={complete}/{count} rows={rows}")
PY
}

collect() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] collect requires NODE_RANK=0"; exit 2; }
    preflight
    "$PYTHON_BIN" "$ROOT/scripts/compute_v193_camera_motion.py" merge \
        --comparison-manifest "$COMPARISON_MANIFEST" --parts-dir "$PARTS_DIR" \
        --output "$MERGED_CSV" --contract "$MERGED_CONTRACT" \
        --expected-shards "$NUM_NODES"
}

analyze() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] analyze requires NODE_RANK=0"; exit 2; }
    preflight
    [[ -s "$MERGED_CSV" && -s "$MERGED_CONTRACT" ]] || {
        echo "[error] merged v193 diagnostics are absent; run collect"
        exit 2
    }
    local -a quality_args=()
    if [[ -n "$QUALITY_REPORT" && -s "$QUALITY_REPORT" ]]; then
        quality_args+=(--quality-report "$QUALITY_REPORT")
    else
        echo "[v193-analysis] paired quality report unavailable; motion result cannot be promoted"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v193_camera_motion.py" \
        --comparison-manifest "$COMPARISON_MANIFEST" \
        --motion-csv "$MERGED_CSV" --motion-contract "$MERGED_CONTRACT" \
        --candidate "$CANDIDATE" --controls "$CONTROLS" \
        --output "$ANALYSIS" "${quality_args[@]}"
    "$PYTHON_BIN" - "$ANALYSIS" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v193-decision] {payload['recommendation']}")
print("measurement_calibration_pass=" + str(payload["measurement_calibration_pass"]).lower())
print("directional_all_controls=" + str(payload["directional_local_motion_signal_against_all_controls"]).lower())
print("strong_all_controls=" + str(payload["strong_local_motion_signal_against_all_controls"]).lower())
print("manual_review_required=" + str(payload["manual_review_required"]).lower())
for row in payload["targeted_review_queue"]:
    print(f"review=p{row['prompt_index']}:priority={row['priority']:.4f}")
PY
}

case "$ACTION" in
    preflight) preflight ;;
    compute) compute ;;
    status) status ;;
    collect) collect ;;
    analyze) analyze ;;
    all-local)
        [[ "$NUM_NODES" == "1" && "$NODE_RANK" == "0" ]] || {
            echo "[error] all-local requires NUM_NODES=1 NODE_RANK=0"; exit 2;
        }
        compute
        collect
        analyze
        ;;
esac
