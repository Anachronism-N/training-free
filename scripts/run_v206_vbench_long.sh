#!/usr/bin/env bash
# VBench-Long and temporal robustness evaluation for one v206 scope.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision|motion-compute|motion-status|motion-collect|motion-analyze) ;;
    *)
        echo "usage: bash scripts/run_v206_vbench_long.sh ACTION"
        echo "actions: prepare split preflight eval resume-missing status temporal collect decision motion-compute motion-status motion-collect motion-analyze"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_BASE="${V206_OUT_ROOT:-$ROOT/runs/v206_horizon_robustness}"
INPUT_MANIFEST="${V206_INPUT_MANIFEST:-$OUT_BASE/inputs/manifest.json}"
SCOPE="${SCOPE:-seed20600_30s_128}"
ALL_SCOPES="seed20600_30s_128,long60_seed20500_32"
case ",$ALL_SCOPES," in
    *",$SCOPE,"*) ;;
    *) echo "[error] unsupported v206 scope: $SCOPE"; exit 2 ;;
esac
RUN_ROOT="${RUN_ROOT:-$OUT_BASE/scopes/$SCOPE}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V206_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V206_TEMPORAL_CONTRACT:-$RUN_ROOT/metrics/temporal_diagnostics.contract.json}"
MOTION_ROOT="${V206_MOTION_ROOT:-$RUN_ROOT/motion}"
MOTION_CSV="$MOTION_ROOT/metrics/camera_compensated_motion.csv"
MOTION_CONTRACT="$MOTION_ROOT/metrics/camera_compensated_motion.contract.json"
MOTION_REPORT="$ANALYSIS_ROOT/v206_continuous_motion.json"
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

scope_prompt_count() {
    "$PYTHON_BIN" - "$INPUT_MANIFEST" "$SCOPE" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = next(item for item in payload["scopes"] if item["key"] == sys.argv[2])
print(row["prompt_count"])
PY
}

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || {
        echo "[error] missing v206 VBench manifest; run prepare"; exit 2;
    }
    activate_env
    local prompt_count
    prompt_count="$(scope_prompt_count)"
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" "$SCOPE" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("experiment") != "v206_horizon_robustness_vbench"
    or payload.get("confirmatory") is not True
    or payload.get("scope") != sys.argv[2]
):
    raise SystemExit("wrong v206 VBench comparison manifest")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -ge 2 && "${#video_dirs[@]}" -le 3 ]] || {
        echo "[error] v206 requires SF plus one or two candidates"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos "$prompt_count" --max-width "${V206_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V206_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V206_TEMPORAL_WORKERS:-8}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

motion_candidate() {
    "$PYTHON_BIN" - "$COMPARISON_ROOT/comparison_manifest.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidates = payload.get("confirmed_v205_candidates") or []
if not candidates:
    raise SystemExit("v206 comparison has no candidate")
print(candidates[0])
PY
}

delegate_motion() {
    local action="$1" candidate
    [[ -s "$COMPARISON_ROOT/comparison_manifest.json" ]] || {
        echo "[error] prepare v206 VBench comparison first"; exit 2;
    }
    candidate="$(motion_candidate)"
    TARGET=custom V193_SOURCE_RUN_ROOT="$RUN_ROOT" \
        COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
        QUALITY_REPORT="$ANALYSIS_ROOT/v206_scope_analysis.json" \
        CANDIDATE="$candidate" CONTROLS=sf_native \
        V193_OUT_ROOT="$MOTION_ROOT" NODE_RANK="$NODE_RANK" \
        NUM_NODES="$NUM_NODES" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$action"
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v206_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" \
        --input-manifest "$INPUT_MANIFEST" --scope "$SCOPE"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v206_horizon_robustness.py" combine \
        --input-manifest "$INPUT_MANIFEST" \
        --seed-report "$OUT_BASE/scopes/seed20600_30s_128/analysis/v206_scope_analysis.json" \
        --long-report "$OUT_BASE/scopes/long60_seed20500_32/analysis/v206_scope_analysis.json" \
        --seed-motion-report "$OUT_BASE/scopes/seed20600_30s_128/analysis/v206_continuous_motion.json" \
        --long-motion-report "$OUT_BASE/scopes/long60_seed20500_32/analysis/v206_continuous_motion.json" \
        --output "$OUT_BASE/analysis/v206_horizon_robustness.json"
    "$PYTHON_BIN" - "$OUT_BASE/analysis/v206_horizon_robustness.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v206-decision] {payload['recommendation']}")
print("robust=" + ",".join(payload["robust_candidates"]))
print("manual_review_required=false")
for row in payload["targeted_review_queue"]:
    print(f"optional_review={row['scope']}:p{row['prompt_index']}:source{row['source_index']}")
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
    for path in "$ANALYSIS_ROOT/v206_scope_analysis.json" "$MOTION_CSV" "$MOTION_CONTRACT"; do
        [[ -s "$path" ]] || { echo "[error] missing v206 motion input: $path"; exit 2; }
    done
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v206_continuous_motion.py" \
        --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
        --motion-csv "$MOTION_CSV" --motion-contract "$MOTION_CONTRACT" \
        --quality-report "$ANALYSIS_ROOT/v206_scope_analysis.json" \
        --output "$MOTION_REPORT"
    exit $?
fi
if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V206_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi
if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" || \
      "$ACTION" == "preflight" || "$ACTION" == "status" || \
      "$ACTION" == "collect" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V206_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V206_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V206_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V206_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v206_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem "v206_${SCOPE}_vbench_analysis" \
    --summary-title "v206 ${SCOPE} VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v206_horizon_robustness.py" scope \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" \
        --output "$ANALYSIS_ROOT/v206_scope_analysis.json"
fi
