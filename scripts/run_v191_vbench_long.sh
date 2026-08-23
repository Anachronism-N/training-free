#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision) ;;
    *)
        echo "usage: bash scripts/run_v191_vbench_long.sh {prepare|split|preflight|eval|resume-missing|status|temporal|collect|decision}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_BASE="${V191_OUT_ROOT:-$ROOT/runs/v191_head_phase_confirmation}"
RUN_ROOT="${RUN_ROOT:-$OUT_BASE/confirm128}"
INPUT_MANIFEST="${V191_INPUT_MANIFEST:-$OUT_BASE/inputs/manifest.json}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$RUN_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$RUN_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$RUN_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RUN_ROOT/analysis}"
TEMPORAL_CSV="${V191_TEMPORAL_CSV:-$RUN_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V191_TEMPORAL_CONTRACT:-$RUN_ROOT/metrics/temporal_diagnostics.contract.json}"
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
}

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || {
        echo "[error] missing v191 VBench manifest: $comparison; run prepare"; exit 2;
    }
    activate_env
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("experiment") != "v191_unseen128_head_phase_vbench"
    or payload.get("confirmatory") is not True
    or int(payload.get("prompt_count", -1)) != 128
):
    raise SystemExit("wrong v191 VBench comparison manifest")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -eq 3 ]] || {
        echo "[error] v191 temporal diagnostics require exactly three methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 128 --max-width "${V191_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V191_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V191_TEMPORAL_WORKERS:-8}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v191_vbench_comparison.py" \
        --run-root "$RUN_ROOT" --comparison-root "$COMPARISON_ROOT" \
        --input-manifest "$INPUT_MANIFEST"
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$ANALYSIS_ROOT/v191_head_phase_confirmation.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v191-decision] {payload['recommendation']}")
print("confirmed=" + str(payload["head_phase_effect_confirmed"]).lower())
print("operator=" + payload["selected_operator"])
print("motion_claim=" + str(payload["motion_improvement_claim_supported"]).lower())
print("manual_review_required=" + str(payload["manual_review_required_for_recommendation"]).lower())
for row in payload["targeted_review_queue"]:
    print(f"review=p{row['prompt_index']}:source{row['source_index']}")
PY
    exit $?
fi

if [[ "$ACTION" == "temporal" ]]; then
    compute_temporal
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v174_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V191_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V191_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V191_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V191_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V191_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models)
    EXTRA_ARGS+=(--torch-hub-dir "$TORCH_HUB_DIR")
    EXTRA_ARGS+=(--runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v174_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v191_vbench_analysis \
    --summary-title "v191 Unseen-128 Head x Phase VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v191_head_phase_confirmation.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" \
        --output "$ANALYSIS_ROOT/v191_head_phase_confirmation.json"
fi
