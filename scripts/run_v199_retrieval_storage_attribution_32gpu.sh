#!/usr/bin/env bash
# v199: paired 60-second Retrieval archive-capacity attribution on 32 prompts.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|audit-smoke|generate32|status|audit|package) ;;
    *)
        echo "usage: bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh {prepare|preflight|smoke|audit-smoke|generate32|status|audit|package}"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
OUT_ROOT="${V199_OUT_ROOT:-$ROOT/runs/v199_retrieval_storage_attribution}"
INPUT_ROOT="$OUT_ROOT/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
SOURCE_PROMPTS="${V199_SOURCE_PROMPTS:-$ROOT/runs/v181_rccp_long_stress/inputs/prompts/long60_seed0.txt}"
V198_DECISION="${V199_V198_DECISION:-$ROOT/runs/v198_audited_long60/analysis/v198_long60_operator.json}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PROMPT_COUNT=32
FRAMES=240
METHODS=(all_recent retrieval_archive4 retrieval_archive8 retrieval_archive12)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

scrub_experiment_env() {
    local key
    while IFS='=' read -r key _; do
        case "$key" in
            LIFECACHE_*|HEAD_ROLE_*|STRUCTURED_MEMORY_*|COMMIT_FORCING_*|\
            SCENE_TRANSITION_*|CACHE_COMPAT_*|PYRAMIDKV_*) unset "$key" ;;
        esac
    done < <(env)
}

method_archive() {
    case "$1" in
        all_recent) echo 0 ;;
        retrieval_archive4) echo 4 ;;
        retrieval_archive8) echo 8 ;;
        retrieval_archive12) echo 12 ;;
        *) echo "unknown v199 method: $1" >&2; return 2 ;;
    esac
}

method_map() {
    if [[ "$1" == "all_recent" ]]; then
        echo "$INPUT_ROOT/maps/all_recent.csv"
    else
        echo "$INPUT_ROOT/maps/all_coverage.csv"
    fi
}

assert_authorized() {
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("generation_authorized") is not True:
    gate = payload.get("upstream_v198") or {}
    raise SystemExit(
        "[error] v199 generation is gated by v198; "
        f"recommendation={gate.get('recommendation')} reason={gate.get('reason')}"
    )
print("[v199-gate] PASS recommendation=" + payload["upstream_v198"]["recommendation"])
PY
}

run_one() {
    local scope="$1" method="$2" rank="$3" gpu="$4" count="$5"
    local archive map raw_dir log trace
    archive="$(method_archive "$method")"
    map="$(method_map "$method")"
    raw_dir="$OUT_ROOT/$scope/raw/$method"
    log="$OUT_ROOT/$scope/logs/$method/shard$(printf '%02d' "$rank").log"
    trace="$OUT_ROOT/$scope/traces/$method/shard$(printf '%02d' "$rank").policy.jsonl"
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$trace")"
    local -a archive_args=()
    if [[ "$archive" -gt 0 ]]; then
        archive_args+=(--pyramidkv_semantic_retrieval_archive_capacity "$archive")
    fi
    (
        cd "$PF"
        scrub_experiment_env
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
        export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        export PYRAMIDKV_POLICY_TRACE_PATH="$trace"
        export PYRAMIDKV_POLICY_TRACE_LAYERS="0,10,20,29"
        export PYRAMIDKV_POLICY_TRACE_HEADS="0,6,11"
        export PYRAMIDKV_POLICY_TRACE_STRIDE="10"
        export PYRAMIDKV_POLICY_TRACE_MAX_RECORDS="4000"
        "$PYTHON_BIN" inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$CHECKPOINT" \
            --data_path "$INPUT_ROOT/prompts/moviegen_long60_stride4_32.txt" \
            --output_folder "$raw_dir" --num_output_frames "$FRAMES" \
            --seed 0 --num_samples 1 --use_ema --save_with_index \
            --reseed_per_prompt --skip_existing --end_idx "$count" \
            --prompt_stride "$count" --prompt_offset "$rank" \
            --pyramidkv_head_config_path "$map" \
            --pyramidkv_cache_compatibility_policy \
            --pyramidkv_cache_compatibility_coverage_policy retrieval \
            "${archive_args[@]}"
    ) >"$log" 2>&1
}

run_worker() {
    local scope="$1" rank="$2" gpu="$3" count="$4" method
    for method in "${METHODS[@]}"; do
        run_one "$scope" "$method" "$rank" "$gpu" "$count"
    done
}

if [[ "$ACTION" == "prepare" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] prepare requires node 0"; exit 2; }
    args=(prepare --repo-root "$ROOT" --source-prompts "$SOURCE_PROMPTS" --output-root "$INPUT_ROOT")
    [[ -s "$V198_DECISION" ]] && args+=(--v198-decision "$V198_DECISION")
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v199_retrieval_storage_attribution.py" "${args[@]}"
    exit $?
fi

if [[ "$ACTION" == "preflight" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] preflight requires node 0"; exit 2; }
    "$PYTHON_BIN" "$ROOT/scripts/prepare_v199_retrieval_storage_attribution.py" verify --manifest "$MANIFEST"
    git -C "$ROOT" diff --quiet -- "third_party/Pyramid-Forcing/inference.py" \
        "third_party/Pyramid-Forcing/pipeline" "third_party/Pyramid-Forcing/pyramidkv" || {
        echo "[error] tracked v199 runtime has unstaged changes"; exit 2;
    }
    git -C "$ROOT" diff --cached --quiet -- "third_party/Pyramid-Forcing/inference.py" \
        "third_party/Pyramid-Forcing/pipeline" "third_party/Pyramid-Forcing/pyramidkv" || {
        echo "[error] tracked v199 runtime has staged changes"; exit 2;
    }
    "$PYTHON_BIN" -m pytest -q \
        "$ROOT/tests/test_v173_cache_compatibility.py" \
        "$ROOT/tests/test_v199_retrieval_storage_attribution.py"
    echo "[v199-preflight] PASS"
    exit 0
fi

if [[ "$ACTION" == "smoke" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] smoke requires node 0"; exit 2; }
    activate_env
    declare -a pids=()
    for slot in "${!METHODS[@]}"; do
        [[ "$slot" -lt "${#GPUS[@]}" ]] || { echo "[error] smoke requires four GPUs"; exit 2; }
        run_one smoke "${METHODS[$slot]}" 0 "${GPUS[$slot]}" 1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] v199 smoke failed"; exit 1; }
    echo "[v199-smoke] generation complete; run audit-smoke"
    exit 0
fi

if [[ "$ACTION" == "audit-smoke" ]]; then
    "$PYTHON_BIN" - "$OUT_ROOT/smoke" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
methods = {"all_recent": 0, "retrieval_archive4": 4, "retrieval_archive8": 8, "retrieval_archive12": 12}
for method, archive in methods.items():
    videos = list((root / "raw" / method).glob("0-0_ema.mp4"))
    log = (root / "logs" / method / "shard00.log").read_text(encoding="utf-8", errors="replace")
    trace = root / "traces" / method / "shard00.policy.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(videos) == 1 and rows and all(row.get("cache_contract_pass") is True for row in rows)
    assert max(int(row.get("sink_frame_count", 0)) + int(row.get("union_frame_count", 0)) + int(row.get("recent_frame_count", 0)) for row in rows) <= 9
    marker = "[SemanticRetrievalArchive]"
    assert (marker in log) == (archive > 0)
    if archive:
        states = [strategy.get("state") or {} for row in rows for strategy in row.get("strategies") or () if strategy.get("name") == "SemanticRetrievalStrategy"]
        assert states and all(int(state["archive_capacity"]) == archive and int(state["capacity"]) == 4 for state in states)
print("[v199-smoke] PASS methods=4 prompts=1 read_budget<=9")
PY
    exit $?
fi

if [[ "$ACTION" == "generate32" ]]; then
    [[ "$WORLD_SHARDS" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] generate32 requires NUM_NODES*GPUS_PER_NODE=32"; exit 2;
    }
    [[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || exit 2
    activate_env
    assert_authorized
    declare -a pids=()
    for slot in "${!GPUS[@]}"; do
        rank=$((NODE_RANK * GPUS_PER_NODE + slot))
        run_worker . "$rank" "${GPUS[$slot]}" "$PROMPT_COUNT" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] v199 generation failed on node=$NODE_RANK"; exit 1; }
    exit 0
fi

if [[ "$ACTION" == "status" ]]; then
    "$PYTHON_BIN" - "$OUT_ROOT" "$PROMPT_COUNT" <<'PY'
import sys
from pathlib import Path
root, count = Path(sys.argv[1]), int(sys.argv[2])
for method in ("all_recent", "retrieval_archive4", "retrieval_archive8", "retrieval_archive12"):
    observed = {int(path.name.split("-", 1)[0]) for path in (root / "raw" / method).glob("*-0_ema.mp4") if path.name.split("-", 1)[0].isdigit()}
    missing = sorted(set(range(count)) - observed)
    print(f"[v199-status] {method}: videos={len(observed)}/{count} missing={missing[:12]}")
PY
    exit $?
fi

if [[ "$ACTION" == "audit" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] audit requires node 0"; exit 2; }
    activate_env
    for method in "${METHODS[@]}"; do
        "$PYTHON_BIN" "$ROOT/scripts/audit_v199_retrieval_storage.py" method \
            --run-root "$OUT_ROOT" --input-manifest "$MANIFEST" --method "$method"
    done
    "$PYTHON_BIN" "$ROOT/scripts/audit_v199_retrieval_storage.py" finalize \
        --run-root "$OUT_ROOT" --input-manifest "$MANIFEST"
    exit $?
fi

if [[ "$ACTION" == "package" ]]; then
    [[ "$NODE_RANK" == "0" ]] || { echo "[error] package requires node 0"; exit 2; }
    "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
archive = root / "v199_generation_evidence.zip"
files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv", ".txt"} and path.stat().st_size <= 8 * 1024 * 1024 and "raw" not in path.parts and "published" not in path.parts]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in sorted(files): handle.write(path, path.relative_to(root))
payload = {"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": [str(path.relative_to(root)) for path in sorted(files)]}
(root / "generation_evidence_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[v199-package] files={len(files)} archive={archive}")
PY
fi
