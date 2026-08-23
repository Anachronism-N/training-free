#!/usr/bin/env bash
# Conditional unseen-128 confirmation for a frozen v190 Head x Phase map.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate128|status|audit-smoke|audit-confirm|package) ;;
    *)
        echo "usage: bash scripts/run_v191_head_phase_confirmation_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate128 status audit-smoke audit-confirm package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
V190_ROOT="${V190_OUT_ROOT:-$ROOT/runs/v190_head_phase_causal_screen}"
V190_INPUT="${V190_INPUT_MANIFEST:-$V190_ROOT/inputs/manifest.json}"
V190_RUN="${V190_RUN_ROOT:-$V190_ROOT/screen32}"
V190_PUBLISHED="${V190_PUBLISHED:-$V190_RUN/published_manifest.json}"
V190_COMPARISON="${V190_COMPARISON_MANIFEST:-$V190_RUN/vbench_comparison/comparison_manifest.json}"
V190_DECISION="${V190_DECISION:-$V190_RUN/analysis/v190_head_phase_causal_screen.json}"
FRESH_PROMPT_MANIFEST="${V191_FRESH_PROMPT_MANIFEST:-$ROOT/runs/v180_rccp_fresh128/inputs/manifest.json}"
OUT_BASE="${V191_OUT_ROOT:-$ROOT/runs/v191_head_phase_confirmation}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/prompts/moviegen_unseen_0128_0255.txt"

ALL_METHODS="sf_native,all_recent,head_phase_joint"
METHODS="${METHODS:-$ALL_METHODS}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-10000}"
SMOKE_PROMPT_INDEX="${V191_SMOKE_PROMPT_INDEX:-7}"
FORCE="${FORCE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
IFS=',' read -r -a REQUESTED_METHODS <<<"$METHODS"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"; exit 2;
}
[[ "$FRAMES" -eq 120 && "$SEED" -eq 10000 ]] || {
    echo "[error] v191 is frozen at 120 latent frames and seed 10000"; exit 2;
}
[[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt 128 ]] || {
    echo "[error] V191_SMOKE_PROMPT_INDEX must be within [0,127]"; exit 2;
}
declare -A METHOD_SEEN=()
for method in "${REQUESTED_METHODS[@]}"; do
    case ",$ALL_METHODS," in
        *",$method,"*) ;;
        *) echo "[error] unsupported v191 method: $method"; exit 2 ;;
    esac
    [[ -z "${METHOD_SEEN[$method]:-}" ]] || {
        echo "[error] duplicate v191 method: $method"; exit 2;
    }
    METHOD_SEEN[$method]=1
done

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/src:$ROOT:$PF:$SF:${PYTHONPATH:-}"
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

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v191_head_phase_confirmation.py" prepare \
        --v190-decision "$V190_DECISION" \
        --v190-input-manifest "$V190_INPUT" \
        --v190-published "$V190_PUBLISHED" \
        --v190-comparison-manifest "$V190_COMPARISON" \
        --fresh-prompt-manifest "$FRESH_PROMPT_MANIFEST" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" "$CHECKPOINT" \
        "$MANIFEST" "$PROMPTS"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v191_head_phase_confirmation.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v189_structured_head_phase.py \
            tests/test_v191_head_phase_confirmation.py)
    fi
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["methods"][sys.argv[2]][sys.argv[3]])
PY
}

shard_complete() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
    return 0
}

clear_shard() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        rm -f "$raw_dir/${index}-0_ema.mp4"
    done
}

run_shard() {
    local scope_root="$1" method="$2" prompt_count="$3" rank="$4"
    local stride="$5" gpu="$6" full_trace="$7"
    local raw_dir="$scope_root/raw/$method"
    local shard_name="shard$(printf '%03d' "$rank")"
    local log="$scope_root/logs/$method/$shard_name.log"
    local marker="$scope_root/status/$method/$shard_name.done"
    local trace="$scope_root/traces/$method/$shard_name.schedule.jsonl"

    [[ "$rank" -lt "$prompt_count" ]] || return
    if [[ "$FORCE" == "1" ]]; then
        clear_shard "$raw_dir" "$prompt_count" "$rank" "$stride"
        rm -f "$marker" "$trace"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v191-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")" \
        "$(dirname "$trace")"

    if [[ "$method" == "sf_native" ]]; then
        (
            cd "$SF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
            export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" \
                --prompt_offset "$rank"
        ) >"$log" 2>&1
    else
        local operator history_policy phase_map head_map
        operator="$(manifest_value "$method" operator)"
        history_policy="$(manifest_value "$method" history_policy)"
        phase_map="$(manifest_value "$method" head_phase_map)"
        head_map="$(manifest_value "$method" head_bank_map)"
        (
            cd "$PF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
            export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
            if [[ "$full_trace" == "1" ]]; then
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS="$(seq -s, 0 29)"
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS="$(seq -s, 0 11)"
            else
                unset PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH
                unset PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS
                unset PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS
            fi
            export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
            export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
            python inference.py \
                --config_path "$PF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" \
                --prompt_offset "$rank" \
                --pyramidkv_head_config_path "$head_map" \
                --pyramidkv_history_polarity \
                --pyramidkv_history_support_policy "$history_policy" \
                --pyramidkv_history_suppress_policy "$history_policy" \
                --pyramidkv_cache_compatibility_denoise_schedule head_phase \
                --pyramidkv_cache_compatibility_denoise_coverage_policy "$operator" \
                --pyramidkv_cache_compatibility_head_phase_map "$phase_map"
        ) >"$log" 2>&1
    fi
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    local scope_root="$OUT_BASE/smoke"
    local cursor=0 method slot pid failed
    while [[ "$cursor" -lt "${#REQUESTED_METHODS[@]}" ]]; do
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            local index=$((cursor + slot))
            [[ "$index" -lt "${#REQUESTED_METHODS[@]}" ]] || break
            method="${REQUESTED_METHODS[$index]}"
            run_shard "$scope_root" "$method" 128 "$SMOKE_PROMPT_INDEX" 128 \
                "${GPUS[$slot]}" 1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v191 smoke failed"; exit 1; }
        cursor=$((cursor + GPUS_PER_NODE))
    done
}

generate128() {
    preflight
    [[ "$NUM_NODES" -eq 4 && "$GPUS_PER_NODE" -eq 8 && "$WORLD_SHARDS" -eq 32 ]] || {
        echo "[error] v191 generate128 is frozen to 4 nodes x 8 GPUs"; exit 2;
    }
    local scope_root="$OUT_BASE/confirm128"
    local count="${#REQUESTED_METHODS[@]}"
    local rotation=$((NODE_RANK % count))
    local offset method slot rank pid failed trace
    for ((offset=0; offset<count; offset++)); do
        method="${REQUESTED_METHODS[$(((rotation + offset) % count))]}"
        echo "[v191-method] method=$method node=$NODE_RANK order_slot=$offset"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            trace=0
            [[ "$rank" -eq 0 ]] && trace=1
            run_shard "$scope_root" "$method" 128 "$rank" "$WORLD_SHARDS" \
                "${GPUS[$slot]}" "$trace" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v191 method=$method failed on node=$NODE_RANK"; exit 1;
        }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v191_head_phase_confirmation.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --scope "$scope" --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
}

status() {
    python - "$OUT_BASE" "$ALL_METHODS" "$SMOKE_PROMPT_INDEX" <<'PY'
import sys
from pathlib import Path
root, methods, smoke = Path(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
for scope, expected in (("smoke", {smoke}), ("confirm128", set(range(128)))):
    print(f"[{scope}]")
    for method in methods:
        raw = root / scope / "raw" / method
        observed = {
            int(path.name.split("-", 1)[0])
            for path in raw.glob("*-0_ema.mp4")
            if path.name.split("-", 1)[0].isdigit()
        }
        logs = list((root / scope / "logs" / method).glob("*.log"))
        failures = sum(
            "Traceback (most recent call last)" in path.read_text(
                encoding="utf-8", errors="replace"
            )
            for path in logs
        )
        print(
            f"{method}: videos={len(observed)}/{len(expected)} "
            f"missing={sorted(expected-observed)} logs={len(logs)} failures={failures}"
        )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_BASE/v191_head_phase_confirmation_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$target" \
        inputs smoke/contracts smoke/audits smoke/published_manifest.json \
        smoke/logs smoke/traces confirm128/contracts confirm128/audits \
        confirm128/published_manifest.json confirm128/metrics confirm128/analysis \
        confirm128/logs confirm128/traces
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    generate128) generate128 ;;
    status) status ;;
    audit-smoke) audit_scope smoke ;;
    audit-confirm) audit_scope confirm128 ;;
    package) package ;;
esac
