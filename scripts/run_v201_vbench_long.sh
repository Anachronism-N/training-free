#!/usr/bin/env bash
# VBench-Long and automatic temporal decision for the v201 holdout screen.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision|package|motion-compute|motion-status|motion-collect|motion-analyze) ;;
    *)
        echo "usage: bash scripts/run_v201_vbench_long.sh ACTION"
        echo "actions: prepare split preflight eval resume-missing status temporal collect decision package motion-compute motion-status motion-collect motion-analyze"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v201_head_phase_horizon_sf_screen/screen32}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V201_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V201_TEMPORAL_CONTRACT:-$RUN_ROOT/metrics/temporal_diagnostics.contract.json}"
REPORT="$ANALYSIS_ROOT/v201_head_phase_horizon.json"
MOTION_ROOT="${V201_MOTION_ROOT:-$RUN_ROOT/motion_v203}"
MOTION_CSV="$MOTION_ROOT/metrics/camera_compensated_motion.csv"
MOTION_CONTRACT="$MOTION_ROOT/metrics/camera_compensated_motion.contract.json"
MOTION_REPORT="$ANALYSIS_ROOT/v203_v201_continuous_motion.json"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || {
        echo "[error] missing v201 VBench manifest; run prepare"; exit 2;
    }
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("experiment") != "v201_head_phase_horizon_causal_vbench_screen32":
    raise SystemExit("wrong v201 VBench comparison manifest")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -ge 5 ]] || {
        echo "[error] v201 temporal diagnostics found too few methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 32 --max-width "${V201_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V201_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V201_TEMPORAL_WORKERS:-8}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

motion_candidate() {
    "$PYTHON_BIN" - "$COMPARISON_ROOT/comparison_manifest.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("methods") or ():
    if row.get("role") == "primary_head_phase_horizon":
        print(f"{row['key']}|sf_native,{row['operator']}_all_recent")
        break
else:
    raise SystemExit("no v201 horizon candidate in comparison manifest")
PY
}

delegate_motion() {
    local action="$1" descriptor candidate controls
    [[ -s "$COMPARISON_ROOT/comparison_manifest.json" ]] || {
        echo "[error] prepare v201 VBench comparison first"; exit 2;
    }
    descriptor="$(motion_candidate)"
    candidate="${descriptor%%|*}"
    controls="${descriptor#*|}"
    TARGET=custom V193_SOURCE_RUN_ROOT="$RUN_ROOT" \
        COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
        QUALITY_REPORT="$REPORT" CANDIDATE="$candidate" CONTROLS="$controls" \
        V193_OUT_ROOT="$MOTION_ROOT" NODE_RANK="$NODE_RANK" \
        NUM_NODES="$NUM_NODES" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$action"
}

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v201_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v201-decision] {payload['recommendation']}")
print("selected=" + ",".join(payload["selected_for_fresh128"]))
print("sf_interval_supported=" + ",".join(payload["sf_interval_supported_candidates"]))
print("mechanism_supported=" + ",".join(payload["mechanism_supported_candidates"]))
print("directional_only=" + ",".join(payload["directional_only_candidates"]))
print("manual_review_required=false")
for row in payload["targeted_debug_queue"]:
    print(f"optional_review=p{row['prompt_index']}")
PY
    exit $?
fi

if [[ "$ACTION" == "temporal" ]]; then
    compute_temporal
    exit $?
fi

if [[ "$ACTION" == "motion-compute" ]]; then
    delegate_motion compute
    exit $?
fi

if [[ "$ACTION" == "motion-status" ]]; then
    delegate_motion status
    exit $?
fi

if [[ "$ACTION" == "motion-collect" ]]; then
    delegate_motion collect
    exit $?
fi

if [[ "$ACTION" == "motion-analyze" ]]; then
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] motion-analyze requires node 0"; exit 2;
    }
    for path in "$REPORT" "$MOTION_CSV" "$MOTION_CONTRACT"; do
        [[ -s "$path" ]] || { echo "[error] missing v201 motion input: $path"; exit 2; }
    done
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v203_v201_continuous_motion.py" \
        --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
        --motion-csv "$MOTION_CSV" --motion-contract "$MOTION_CONTRACT" \
        --quality-report "$REPORT" --output "$MOTION_REPORT"
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    [[ -s "$REPORT" ]] || { echo "[error] missing $REPORT"; exit 2; }
    archive="$RUN_ROOT/v201_evaluation_small_artifacts.tar.gz"
    optional=()
    if [[ "$MOTION_ROOT" == "$RUN_ROOT/"* && -d "$MOTION_ROOT" ]]; then
        optional+=("${MOTION_ROOT#"$RUN_ROOT/"}")
    fi
    tar -C "$RUN_ROOT" -czf "$archive" \
        vbench_comparison/comparison_manifest.json \
        metrics/vbench_core9_summary.json metrics/vbench_core9_summary.md \
        metrics/temporal_diagnostics.csv \
        metrics/temporal_diagnostics.contract.json analysis "${optional[@]}"
    echo "$archive"
    exit 0
fi

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v174_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V201_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

TORCH_HUB_DIR="${V201_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V201_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V201_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V201_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v201_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v201_vbench_analysis \
    --summary-title "v201 Head x Phase x AR-Horizon Causal Screen" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v201_head_phase_horizon.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" \
        --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" \
        --output "$REPORT"
fi
