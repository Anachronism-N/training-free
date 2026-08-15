#!/usr/bin/env bash
# Frozen-head screen of random versus structured Coverage memories.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate16|audit-smoke|audit-screen|trace-smoke|trace-screen|status|package) ;;
    *)
        echo "usage: bash scripts/run_v182_structured_coverage_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate16 audit-smoke audit-screen"
        echo "         trace-smoke trace-screen status package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V182_SOURCE_PROMPTS:-$ROOT/prompts/moviegen_128_qwen_v154_diverse16.txt}"
STRICT_MAP="${V182_STRICT_MAP:-$ROOT/configs/head_maps/v177_strict5_coverage.csv}"
OUT_BASE="${V182_OUT_ROOT:-$ROOT/runs/v182_structured_coverage}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/prompts/moviegen_diverse16.txt"

ALL_METHODS="all_recent,strict5_reservoir,strict5_landmark,strict5_prototype,strict5_retrieval"
METHODS="${METHODS:-$ALL_METHODS}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
SMOKE_PROMPT_INDEX="${V182_SMOKE_PROMPT_INDEX:-3}"
FORCE="${FORCE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 0 ]] || {
    echo "[error] v182 is frozen at 120 latent frames and seed 0"
    exit 2
}
[[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt 16 ]] || {
    echo "[error] V182_SMOKE_PROMPT_INDEX must be within [0, 15]"
    exit 2
}

IFS=',' read -r -a REQUESTED_METHODS <<<"$METHODS"
declare -A METHOD_SEEN=()
for method in "${REQUESTED_METHODS[@]}"; do
    case ",$ALL_METHODS," in
        *",$method,"*) ;;
        *) echo "[error] unsupported v182 method: $method"; exit 2 ;;
    esac
    [[ -z "${METHOD_SEEN[$method]:-}" ]] || {
        echo "[error] duplicate v182 method: $method"
        exit 2
    }
    METHOD_SEEN[$method]=1
done

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    python "$ROOT/scripts/prepare_v182_structured_coverage.py" prepare \
        --source-prompts "$SOURCE_PROMPTS" \
        --strict-map "$STRICT_MAP" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$MANIFEST" "$PROMPTS"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v182_structured_coverage.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v173_cache_compatibility.py \
            tests/test_v182_structured_coverage.py)
    fi
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["methods"][sys.argv[2]][sys.argv[3]])
PY
}

configure_runtime() {
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
    export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
    export PYRAMIDKV_POLICY_TRACE_LAYERS=0,5,6,8,23
    export PYRAMIDKV_POLICY_TRACE_HEADS=2,3,6,10
    export PYRAMIDKV_POLICY_TRACE_STRIDE="${V182_TRACE_STRIDE:-2}"
    export PYRAMIDKV_POLICY_TRACE_MAX_RECORDS="${V182_TRACE_MAX_RECORDS:-20000}"
    export PYRAMIDKV_ROLE_EVENT_TRACE_LAYERS=0,5,6,8,23
    export PYRAMIDKV_ROLE_EVENT_DEBUG=0
    unset CACHE_COMPAT_PROFILE CACHE_COMPAT_PROFILE_CONTRACT
}

shard_complete() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
    return 0
}

run_shard() {
    local scope_root="$1" method="$2" prompt_count="$3" rank="$4" stride="$5" gpu="$6"
    local raw_dir="$scope_root/raw/$method"
    local shard_name="shard$(printf '%02d' "$rank")"
    local log="$scope_root/logs/$method/$shard_name.log"
    local marker="$scope_root/status/$method/$shard_name.done"
    local policy_trace="$scope_root/traces/$method/$shard_name.policy.jsonl"
    local role_trace="$scope_root/traces/$method/$shard_name.role.jsonl"
    local head_map coverage_policy
    head_map="$(manifest_value "$method" head_map)"
    coverage_policy="$(manifest_value "$method" coverage_policy)"

    if [[ "$rank" -ge "$prompt_count" ]]; then
        return
    fi
    if [[ "$FORCE" == "1" ]]; then
        local index
        for ((index=rank; index<prompt_count; index+=stride)); do
            rm -f "$raw_dir/${index}-0_ema.mp4"
        done
        rm -f "$marker" "$policy_trace" "$role_trace"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v182-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")" \
        "$(dirname "$policy_trace")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYRAMIDKV_POLICY_TRACE_PATH="$policy_trace"
        export PYRAMIDKV_ROLE_EVENT_TRACE_PATH="$role_trace"
        python inference.py \
            --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$raw_dir" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index --reseed_per_prompt --skip_existing \
            --end_idx "$prompt_count" --prompt_stride "$stride" --prompt_offset "$rank" \
            --pyramidkv_head_config_path "$head_map" \
            --pyramidkv_cache_compatibility_policy \
            --pyramidkv_cache_compatibility_coverage_policy "$coverage_policy"
    ) >"$log" 2>&1
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    configure_runtime
    [[ "$GPUS_PER_NODE" -ge "${#REQUESTED_METHODS[@]}" ]] || {
        echo "[error] smoke needs at least one GPU per requested method"
        exit 2
    }
    local scope_root="$OUT_BASE/smoke"
    local -a pids=()
    for slot in "${!REQUESTED_METHODS[@]}"; do
        run_shard "$scope_root" "${REQUESTED_METHODS[$slot]}" 16 \
            "$SMOKE_PROMPT_INDEX" 16 "${GPUS[$slot]}" &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] v182 smoke failed"; exit 1; }
}

generate16() {
    preflight
    configure_runtime
    local scope_root="$OUT_BASE/screen16"
    for method in "${REQUESTED_METHODS[@]}"; do
        echo "[v182-method] method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            local rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            if [[ "$rank" -lt 16 ]]; then
                run_shard "$scope_root" "$method" 16 "$rank" "$WORLD_SHARDS" \
                    "${GPUS[$slot]}" &
                pids+=("$!")
            fi
        done
        local failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v182 method=$method failed on node=$NODE_RANK"
            exit 1
        }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v182_structured_coverage.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" --scope "$scope" \
        --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
}

trace_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] trace analysis requires node 0"; exit 2; }
    python "$ROOT/scripts/analyze_v182_coverage_traces.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --output "$OUT_BASE/$scope/analysis/coverage_trace_summary.json"
}

status() {
    python - "$OUT_BASE" "$ALL_METHODS" "$SMOKE_PROMPT_INDEX" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = sys.argv[2].split(",")
smoke_index = int(sys.argv[3])
for scope, expected in (("smoke", {smoke_index}), ("screen16", set(range(16)))):
    print(f"[{scope}]")
    for method in methods:
        raw = root / scope / "raw" / method
        observed = {
            int(path.name.split("-", 1)[0]) for path in raw.glob("*-0_ema.mp4")
        }
        logs = list((root / scope / "logs" / method).glob("*.log"))
        failed = sum(
            "Traceback (most recent call last)" in path.read_text(
                encoding="utf-8", errors="replace"
            )
            for path in logs
        )
        print(
            f"{method}: videos={len(observed)}/{len(expected)} "
            f"missing={sorted(expected - observed)} logs={len(logs)} failures={failed}"
        )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local package_path="$OUT_BASE/v182_structured_coverage_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$package_path" \
        inputs smoke/contracts smoke/audits smoke/analysis \
        screen16/contracts screen16/audits screen16/analysis \
        screen16/metrics screen16/published_manifest.json \
        smoke/published_manifest.json smoke/logs screen16/logs
    echo "$package_path"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    generate16) generate16 ;;
    audit-smoke) audit_scope smoke ;;
    audit-screen) audit_scope screen16 ;;
    trace-smoke) trace_scope smoke ;;
    trace-screen) trace_scope screen16 ;;
    status) status ;;
    package) package ;;
esac
