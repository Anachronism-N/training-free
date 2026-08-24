#!/usr/bin/env bash
# VBench, temporal, and camera-compensated motion evaluation for v194.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|\
    camera-compute|camera-status|camera-collect|decision) ;;
    *)
        echo "usage: bash scripts/run_v194_vbench_long.sh ACTION"
        echo "actions: prepare split preflight eval resume-missing status temporal collect camera-compute camera-status camera-collect decision"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_BASE="${V194_OUT_ROOT:-$ROOT/runs/v194_cf_checkpoint_transfer}"
INPUT_MANIFEST="${V194_INPUT_MANIFEST:-$OUT_BASE/inputs/manifest.json}"
RUN_ROOT="${RUN_ROOT:-$OUT_BASE/transfer64}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V194_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V194_TEMPORAL_CONTRACT:-$RUN_ROOT/metrics/temporal_diagnostics.contract.json}"
CAMERA_ROOT="${V194_CAMERA_ROOT:-$RUN_ROOT/camera_motion}"
CAMERA_REPORT="$CAMERA_ROOT/analysis/v193_camera_motion.json"
CORE_REPORT="$ANALYSIS_ROOT/v194_checkpoint_transfer.json"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi
if [[ "$ACTION" == "resume-missing" && \
      ( "$NODE_RANK" != "0" || "$NUM_NODES" != "1" ) ]]; then
    echo "[error] resume-missing requires NODE_RANK=0 NUM_NODES=1"
    exit 2
fi

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
}

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || {
        echo "[error] missing v194 comparison manifest; run prepare"; exit 2;
    }
    activate_env
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("experiment") != "v194_causal_checkpoint_transfer_vbench":
    raise SystemExit("wrong v194 comparison manifest")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -eq 3 ]] || {
        echo "[error] v194 temporal diagnostics require three methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 64 --max-width "${V194_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V194_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V194_TEMPORAL_WORKERS:-8}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

analyze_transfer() {
    local -a camera_args=()
    if [[ -s "$CAMERA_REPORT" ]]; then
        camera_args+=(--camera-motion-report "$CAMERA_REPORT")
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v194_cf_checkpoint_transfer.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" \
        --output "$CORE_REPORT" "${camera_args[@]}"
}

run_camera() {
    local camera_action="$1"
    TARGET=custom \
    V193_SOURCE_RUN_ROOT="$RUN_ROOT" \
    COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
    QUALITY_REPORT="$CORE_REPORT" \
    CANDIDATE=cf_head_phase_transfer \
    CONTROLS=cf_all_recent_9ffe,cf_native_21 \
    V193_OUT_ROOT="$CAMERA_ROOT" \
    NODE_RANK="$NODE_RANK" NUM_NODES="$NUM_NODES" \
    CONDA_SH="$CONDA_SH" CONDA_ENV="$CONDA_ENV" PYTHON_BIN="$PYTHON_BIN" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$camera_action"
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v194_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" \
        --input-manifest "$INPUT_MANIFEST"
    exit $?
fi

if [[ "$ACTION" == "temporal" ]]; then
    compute_temporal
    exit $?
fi

if [[ "$ACTION" == "camera-compute" ]]; then
    run_camera compute
    exit $?
fi

if [[ "$ACTION" == "camera-status" ]]; then
    run_camera status
    exit $?
fi

if [[ "$ACTION" == "camera-collect" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] camera-collect requires node 0"; exit 2; }
    [[ -s "$CORE_REPORT" ]] || { echo "[error] run collect before camera-collect"; exit 2; }
    run_camera collect
    run_camera analyze
    activate_env
    analyze_transfer
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    [[ -s "$CAMERA_REPORT" ]] || {
        echo "[error] camera-motion report absent; run camera-collect"; exit 2;
    }
    activate_env
    analyze_transfer
    "$PYTHON_BIN" - "$CORE_REPORT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v194-decision] {payload['recommendation']}")
print("transfer_confirmed=" + str(payload["cross_checkpoint_transfer_confirmed"]).lower())
print("motion_claim=" + str(payload["motion_improvement_claim_supported"]).lower())
print("manual_review_required=" + str(payload["manual_review_required_for_recommendation"]).lower())
for row in payload["targeted_review_queue"]:
    print(f"review=p{row['prompt_index']}:source{row['source_index']}:priority={row['priority']:.4f}")
PY
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V194_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" || \
      "$ACTION" == "preflight" || "$ACTION" == "status" || \
      "$ACTION" == "collect" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V194_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V194_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V194_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V194_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v194_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v194_vbench_analysis \
    --summary-title "v194 Causal Checkpoint Transfer VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    analyze_transfer
fi
