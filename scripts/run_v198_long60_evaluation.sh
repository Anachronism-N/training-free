#!/usr/bin/env bash
# Audit and evaluate the already-generated v181/v186 60-second videos.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    audit|audit-all|audit-status|prepare|split|preflight|eval|resume-missing|status|\
    temporal|collect|camera-compute|camera-status|camera-collect|decision|package) ;;
    *)
        echo "usage: bash scripts/run_v198_long60_evaluation.sh ACTION"
        echo "actions: audit audit-all audit-status prepare split preflight eval resume-missing status temporal collect camera-compute camera-status camera-collect decision package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V181_ROOT="${V181_ROOT:-$ROOT/runs/v181_rccp_long_stress}"
V186_ROOT="${V186_ROOT:-$ROOT/runs/v186_long60_comparison}"
OUT_ROOT="${V198_OUT_ROOT:-$ROOT/runs/v198_audited_long60}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$OUT_ROOT/vbench_comparison}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
PARTS_ROOT="${PARTS_ROOT:-$OUT_ROOT/metrics/vbench_long_parts}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$OUT_ROOT/metrics}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$OUT_ROOT/analysis}"
TEMPORAL_CSV="${V198_TEMPORAL_CSV:-$OUT_ROOT/metrics/temporal_diagnostics.csv}"
TEMPORAL_CONTRACT="${V198_TEMPORAL_CONTRACT:-$OUT_ROOT/metrics/temporal_diagnostics.contract.json}"
CORE_REPORT="$ANALYSIS_ROOT/v198_long60_operator.json"
CAMERA_ROOT="${V198_CAMERA_ROOT:-$OUT_ROOT/camera_motion}"
CAMERA_REPORT="$CAMERA_ROOT/analysis/v193_camera_motion.json"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
METHODS=(sf_native all_recent pf_native all_coverage_retrieval)

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

audit_one() {
    local method="$1"
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/audit_v198_long60_inputs.py" audit-method \
        --repo-root "$ROOT" --v181-root "$V181_ROOT" --v186-root "$V186_ROOT" \
        --output-root "$OUT_ROOT" --method "$method" \
        --workers "${V198_AUDIT_WORKERS:-8}"
}

compute_temporal() {
    [[ "$NODE_RANK" == "0" ]] || {
        echo "[error] temporal diagnostics require node 0"; exit 2;
    }
    local comparison="$COMPARISON_ROOT/comparison_manifest.json"
    [[ -s "$comparison" ]] || {
        echo "[error] missing v198 comparison manifest; run prepare"; exit 2;
    }
    activate_env
    mapfile -t video_dirs < <(
        "$PYTHON_BIN" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("experiment") != "v198_audited_long60_operator_comparison":
    raise SystemExit("wrong v198 comparison manifest")
for row in payload["methods"]:
    print(row["video_dir"])
PY
    )
    [[ "${#video_dirs[@]}" -eq 4 ]] || {
        echo "[error] v198 temporal diagnostics require four methods"; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${video_dirs[@]}" --output "$TEMPORAL_CSV" \
        --expected-videos 128 --max-width "${V198_TEMPORAL_WIDTH:-256}" \
        --frame-step "${V198_TEMPORAL_FRAME_STEP:-8}" \
        --workers "${V198_TEMPORAL_WORKERS:-16}"
    "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
        --comparison-manifest "$comparison" --temporal-csv "$TEMPORAL_CSV" \
        --output "$TEMPORAL_CONTRACT"
}

analyze_operator() {
    local -a camera_args=()
    if [[ -s "$CAMERA_REPORT" ]]; then
        camera_args+=(--camera-motion-report "$CAMERA_REPORT")
    fi
    "$PYTHON_BIN" "$ROOT/scripts/analyze_v198_long60_operator.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --summary "$SUMMARY_ROOT/vbench_core9_summary.json" \
        --parts-root "$PARTS_ROOT" --temporal-csv "$TEMPORAL_CSV" \
        --temporal-contract "$TEMPORAL_CONTRACT" \
        --output "$CORE_REPORT" "${camera_args[@]}"
}

run_camera() {
    local camera_action="$1"
    TARGET=custom \
    V193_SOURCE_RUN_ROOT="$OUT_ROOT" \
    COMPARISON_MANIFEST="$COMPARISON_ROOT/comparison_manifest.json" \
    QUALITY_REPORT="$CORE_REPORT" \
    CANDIDATE=all_coverage_retrieval \
    CONTROLS=all_recent,sf_native \
    V193_OUT_ROOT="$CAMERA_ROOT" \
    V193_WORKERS="${V198_CAMERA_WORKERS:-8}" \
    V193_MAX_WIDTH="${V198_CAMERA_WIDTH:-256}" \
    V193_FRAME_STEP="${V198_CAMERA_FRAME_STEP:-8}" \
    NODE_RANK="$NODE_RANK" NUM_NODES="$NUM_NODES" \
    CONDA_SH="$CONDA_SH" CONDA_ENV="$CONDA_ENV" PYTHON_BIN="$PYTHON_BIN" \
        bash "$ROOT/scripts/run_v193_camera_motion.sh" "$camera_action"
}

if [[ "$ACTION" == "audit" ]]; then
    method="${V198_AUDIT_METHOD:-}"
    if [[ -z "$method" ]]; then
        [[ "$NUM_NODES" == "4" ]] || {
            echo "[error] set V198_AUDIT_METHOD or use NUM_NODES=4"; exit 2;
        }
        method="${METHODS[$NODE_RANK]}"
    fi
    audit_one "$method"
    exit $?
fi

if [[ "$ACTION" == "audit-all" ]]; then
    [[ "$NODE_RANK" == "0" && "$NUM_NODES" == "1" ]] || {
        echo "[error] audit-all requires NODE_RANK=0 NUM_NODES=1"; exit 2;
    }
    declare -a pids=()
    for method in "${METHODS[@]}"; do
        audit_one "$method" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] one or more v198 audits failed"; exit 1; }
    exit 0
fi

if [[ "$ACTION" == "audit-status" ]]; then
    "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
methods = ("sf_native", "all_recent", "pf_native", "all_coverage_retrieval")
for method in methods:
    path = root / "audits" / f"{method}.json"
    if not path.is_file():
        print(f"[v198-audit-status] {method}: missing")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[v198-audit-status] {method}: ok={str(payload.get('ok') is True).lower()} videos={len(payload.get('videos') or ())}")
PY
    exit $?
fi

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    prompt_args=()
    if [[ -n "${V198_PROMPT_FILE:-}" ]]; then
        prompt_args+=(--prompt-file "$V198_PROMPT_FILE")
    fi
    "$PYTHON_BIN" "$ROOT/scripts/audit_v198_long60_inputs.py" finalize \
        --repo-root "$ROOT" --v181-root "$V181_ROOT" --v186-root "$V186_ROOT" \
        --output-root "$OUT_ROOT" --comparison-root "$COMPARISON_ROOT" \
        "${prompt_args[@]}"
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
    analyze_operator
    exit $?
fi

if [[ "$ACTION" == "decision" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] decision requires node 0"; exit 2; }
    [[ -s "$CAMERA_REPORT" ]] || {
        echo "[error] camera-motion report absent; run camera-collect"; exit 2;
    }
    activate_env
    analyze_operator
    "$PYTHON_BIN" - "$CORE_REPORT" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"[v198-decision] {payload['recommendation']}")
print("candidate_promising=" + str(payload["candidate_promising"]).lower())
print("paper_claim_ready=" + str(payload["paper_claim_ready"]).lower())
print("same_runtime_confirmation_required=" + str(payload["same_runtime_all_recent_confirmation_required"]).lower())
print("manual_review_required=" + str(payload["manual_review_required_for_automatic_decision"]).lower())
for row in payload["targeted_review_queue"]:
    print(f"review=p{row['prompt_index']}:source{row['source_index']}:priority={row['priority']:.4f}")
PY
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    activate_env
    "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
archive = root / "v198_small_artifacts.zip"
allowed = {".json", ".md", ".csv", ".txt"}
files = [
    path for path in root.rglob("*")
    if path.is_file() and path.suffix.lower() in allowed
    and path.stat().st_size <= 8 * 1024 * 1024
    and "vbench_long_parts" not in path.parts
]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in sorted(files):
        handle.write(path, path.relative_to(root))
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
manifest = {"archive": str(archive), "sha256": digest, "files": [str(p.relative_to(root)) for p in sorted(files)]}
(root / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[v198-package] files={len(files)} archive={archive} sha256={digest}")
PY
    exit $?
fi

if [[ "$ACTION" == "split" ]]; then
    activate_env
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v175_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" --vbench-root "$VBENCH_ROOT" \
        --workers "${V198_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES"
    exit $?
fi

if [[ "$ACTION" == "eval" || "$ACTION" == "resume-missing" || \
      "$ACTION" == "preflight" || "$ACTION" == "status" || \
      "$ACTION" == "collect" ]]; then
    activate_env
fi

TORCH_HUB_DIR="${V198_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
RUNTIME_HOME="${V198_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
if [[ ( "$ACTION" == "eval" || "$ACTION" == "resume-missing" ) && \
      "${V198_LOCAL_MODELS:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
        --vbench-cache "$VBENCH_CACHE_DIR" \
        --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME"
fi

PYTHON_ACTION="$ACTION"
[[ "$ACTION" == "resume-missing" ]] && PYTHON_ACTION="eval-missing"
EXTRA_ARGS=()
if [[ "${V198_LOCAL_MODELS:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--local-models --torch-hub-dir "$TORCH_HUB_DIR" --runtime-home "$RUNTIME_HOME")
fi

"$PYTHON_BIN" "$ROOT/scripts/run_v198_vbench_long.py" "$PYTHON_ACTION" \
    --comparison-root "$COMPARISON_ROOT" \
    --vbench-root "$VBENCH_ROOT" --vbench-cache "$VBENCH_CACHE_DIR" \
    --parts-root "$PARTS_ROOT" --summary-root "$SUMMARY_ROOT" \
    --analysis-root "$ANALYSIS_ROOT" \
    --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" \
    --summary-stem vbench_core9_summary \
    --analysis-stem v198_vbench_analysis \
    --summary-title "v198 Audited 60-Second VBench-Long" \
    "${EXTRA_ARGS[@]}"

if [[ "$ACTION" == "collect" ]]; then
    if [[ ! -s "$TEMPORAL_CSV" || ! -s "$TEMPORAL_CONTRACT" ]]; then
        compute_temporal
    else
        "$PYTHON_BIN" "$ROOT/scripts/bind_temporal_diagnostics.py" verify \
            --comparison-manifest "$COMPARISON_ROOT/comparison_manifest.json" \
            --temporal-csv "$TEMPORAL_CSV" --output "$TEMPORAL_CONTRACT"
    fi
    analyze_operator
fi
