#!/usr/bin/env bash
# VBench-Long and paired confirmatory analysis for v208 main30 or long60.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision|package|motion-compute|motion-status|motion-collect|motion-analyze) ;;
    *)
        echo "usage: SCOPE=main30|long60 bash scripts/run_v208_vbench_long.sh ACTION"
        echo "actions: prepare split preflight eval resume-missing status temporal collect decision package motion-compute motion-status motion-collect motion-analyze"
        exit 2
        ;;
esac

SCOPE="${SCOPE:-main30}"
[[ "$SCOPE" == "main30" || "$SCOPE" == "long60" ]] || {
    echo "[error] SCOPE must be main30 or long60"; exit 2;
}
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v208_paper_confirmation/$SCOPE}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V208_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V208_TEMPORAL_CONTRACT:-$RUN_ROOT/metrics/temporal_diagnostics.contract.json}"
REPORT="$ANALYSIS_ROOT/v208_${SCOPE}_confirmation.json"
MOTION_ROOT="${V208_MOTION_ROOT:-$RUN_ROOT/motion_camera_compensated}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"; exit 2
fi

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
}

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] temporal requires node 0"; exit 2; }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || { echo "[error] run prepare first"; exit 2; }
    activate_env
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" "$SCOPE" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("scope") != sys.argv[2]:
    raise SystemExit("wrong v208 scope")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -eq 3 ]] || { echo "[error] expected three methods"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 128 --max-width "${V208_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V208_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V208_TEMPORAL_WORKERS:-16}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

delegate_motion() {
    local action="$1"
    [[ -s "$REPORT" ]] || { echo "[error] run collect before camera motion"; exit 2; }
    TARGET=custom V193_SOURCE_RUN_ROOT="$RUN_ROOT" \
        COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
        QUALITY_REPORT="${REPORT%.json}_all128_quality_context.json" CANDIDATE=ours CONTROLS=sf_native,recent21 \
        V193_OUT_ROOT="$MOTION_ROOT" NODE_RANK="$NODE_RANK" NUM_NODES="$NUM_NODES" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$action"
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v208_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" --scope "$SCOPE"
    exit $?
fi
if [[ "$ACTION" == "temporal" ]]; then compute_temporal; exit $?; fi
if [[ "$ACTION" == "motion-compute" ]]; then delegate_motion compute; exit $?; fi
if [[ "$ACTION" == "motion-status" ]]; then delegate_motion status; exit $?; fi
if [[ "$ACTION" == "motion-collect" ]]; then delegate_motion collect; exit $?; fi
if [[ "$ACTION" == "motion-analyze" ]]; then delegate_motion analyze; exit $?; fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v208-decision] {payload['recommendation']}")
print(f"paper_confirmation_pass={payload['paper_confirmation_pass']}")
print(f"primary_cohort={payload['primary_cohort']}")
print(f"sf_noninferiority={payload['sf_noninferiority']['pass']}")
print(f"recent21_noninferiority={payload['recent21_noninferiority']['pass']}")
print(f"sf_positive={payload['sf_positive_support']['pass']}")
print(f"recent21_positive={payload['recent21_positive_support']['pass']}")
print("manual_review_required=false")
PY
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    [[ -s "$REPORT" ]] || { echo "[error] missing $REPORT"; exit 2; }
    archive="$RUN_ROOT/v208_${SCOPE}_evaluation_artifacts.tar.gz"
    optional=()
    if [[ "$MOTION_ROOT" == "$RUN_ROOT/"* && -d "$MOTION_ROOT" ]]; then
        optional+=("${MOTION_ROOT#"$RUN_ROOT/"}")
    fi
    tar -C "$RUN_ROOT" -czf "$archive" \
        vbench_comparison/comparison_manifest.json \
        metrics/vbench_core9_summary.json metrics/vbench_core9_summary.md \
        metrics/temporal_diagnostics.csv metrics/temporal_diagnostics.contract.json \
        analysis "${optional[@]}"
    echo "$archive"
    exit 0
fi

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v208_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V208_SPLIT_WORKERS:-4}" --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" || "$ACTION" == "collect" ]]; then activate_env; fi
TORCH_HUB_DIR="${V208_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V208_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && "${V208_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" --torch-hub-dir "$TORCH_HUB_DIR" \
        --runtime-home "$RUNTIME_HOME"
fi
PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V208_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v208_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
    --vbench-cache "$VBENCH_CACHE_DIR" --parts-root "$PARTS_ROOT" \
    --summary-root "$SUMMARY_ROOT" --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary --analysis-stem "v208_${SCOPE}_vbench_analysis" \
    --summary-title "v208 ${SCOPE} Paper Confirmation" "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v208_paper_confirmation.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" --output "$REPORT"
fi
