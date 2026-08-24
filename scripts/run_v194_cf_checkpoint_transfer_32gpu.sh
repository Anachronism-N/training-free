#!/usr/bin/env bash
# Conditional no-refit transfer of the v192 route to the Causal-Forcing checkpoint.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate|status|audit-smoke|audit|package) ;;
    *)
        echo "usage: bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate status audit-smoke audit package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CHECKPOINT="${CF_CHECKPOINT:-$ROOT/../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt}"
V192_ROOT="${V192_OUT_ROOT:-$ROOT/runs/v192_head_phase_robustness}"
V192_INPUT="${V192_INPUT_MANIFEST:-$V192_ROOT/inputs/manifest.json}"
V192_DECISION="${V192_DECISION:-$V192_ROOT/analysis/v192_head_phase_robustness.json}"
OUT_BASE="${V194_OUT_ROOT:-$ROOT/runs/v194_cf_checkpoint_transfer}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
RUN_ROOT="$OUT_BASE/transfer64"

ALL_METHODS="cf_native_21,cf_all_recent_9ffe,cf_head_phase_transfer"
METHODS="${METHODS:-$ALL_METHODS}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SMOKE_PROMPT_INDEX="${V194_SMOKE_PROMPT_INDEX:-0}"
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
declare -A METHOD_SEEN=()
for method in "${REQUESTED_METHODS[@]}"; do
    case ",$ALL_METHODS," in
        *",$method,"*) ;;
        *) echo "[error] unsupported v194 method: $method"; exit 2 ;;
    esac
    [[ -z "${METHOD_SEEN[$method]:-}" ]] || {
        echo "[error] duplicate v194 method: $method"; exit 2;
    }
    METHOD_SEEN[$method]=1
done

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

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v194_cf_checkpoint_transfer.py" prepare \
        --v192-decision "$V192_DECISION" \
        --v192-input-manifest "$V192_INPUT" \
        --pf-runtime-root "$PF" \
        --causal-checkpoint "$CHECKPOINT" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$PF" "$CHECKPOINT" "$V192_INPUT" "$V192_DECISION" "$MANIFEST"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare on node 0"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v194_cf_checkpoint_transfer.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v192_head_phase_robustness.py \
            tests/test_v194_cf_checkpoint_transfer.py)
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

top_value() {
    python - "$MANIFEST" "$1" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

shard_complete() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        [[ -s "$raw_dir/${index}-0_regular.mp4" ]] || return 1
    done
    return 0
}

clear_shard() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        rm -f "$raw_dir/${index}-0_regular.mp4"
    done
}

run_shard() {
    local target_root="$1" method="$2" rank="$3" stride="$4" gpu="$5" full_trace="$6"
    local prompt_count frames seed prompts checkpoint checkpoint_sha config
    local raw_dir shard_name log marker trace state_key local_window
    prompt_count="$(top_value prompt_count)"
    frames="$(top_value num_output_frames)"
    seed="$(top_value seed)"
    prompts="$(top_value prompt_file)"
    checkpoint="$(top_value checkpoint.path)"
    checkpoint_sha="$(top_value checkpoint.sha256)"
    state_key="$(manifest_value "$method" checkpoint_state_key)"
    local_window="$(manifest_value "$method" model_local_attn_size)"
    config="$(manifest_value "$method" config)"
    raw_dir="$target_root/raw/$method"
    shard_name="shard$(printf '%03d' "$rank")"
    log="$target_root/logs/$method/$shard_name.log"
    marker="$target_root/status/$method/$shard_name.done"
    trace="$target_root/traces/$method/$shard_name.schedule.jsonl"

    [[ "$rank" -lt "$prompt_count" ]] || return
    if [[ "$FORCE" == "1" ]]; then
        clear_shard "$raw_dir" "$prompt_count" "$rank" "$stride"
        rm -f "$marker" "$trace"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v194-skip] method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")" \
        "$(dirname "$trace")"

    (
        cd "$PF"
        scrub_experiment_env
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
        export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
        export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
        printf '[V194RuntimeContract] method=%s checkpoint_sha256=%s state_key=%s use_ema=false local_attn_size=%s seed=%s reseed_per_prompt=true\n' \
            "$method" "$checkpoint_sha" "$state_key" "$local_window" "$seed"
        local -a args=(
            python inference.py
            --config_path "$config" --checkpoint_path "$checkpoint"
            --checkpoint_state_key "$state_key"
            --model_local_attn_size "$local_window"
            --data_path "$prompts" --output_folder "$raw_dir"
            --num_output_frames "$frames" --seed "$seed" --num_samples 1
            --save_with_index --reseed_per_prompt --skip_existing
            --end_idx "$prompt_count" --prompt_stride "$stride"
            --prompt_offset "$rank"
        )
        if [[ "$method" != "cf_native_21" ]]; then
            local operator history_policy phase_map head_map
            operator="$(manifest_value "$method" operator)"
            history_policy="$(manifest_value "$method" history_policy)"
            phase_map="$(manifest_value "$method" head_phase_map)"
            head_map="$(manifest_value "$method" head_bank_map)"
            export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
            export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
            if [[ "$full_trace" == "1" ]]; then
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS="$(seq -s, 0 29)"
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS="$(seq -s, 0 11)"
            fi
            args+=(
                --pyramidkv_head_config_path "$head_map"
                --pyramidkv_history_polarity
                --pyramidkv_history_support_policy "$history_policy"
                --pyramidkv_history_suppress_policy "$history_policy"
                --pyramidkv_cache_compatibility_denoise_schedule head_phase
                --pyramidkv_cache_compatibility_denoise_coverage_policy "$operator"
                --pyramidkv_cache_compatibility_head_phase_map "$phase_map"
            )
        fi
        "${args[@]}"
    ) >"$log" 2>&1
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    local prompt_count cursor method slot pid failed
    prompt_count="$(top_value prompt_count)"
    [[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt "$prompt_count" ]] || {
        echo "[error] V194_SMOKE_PROMPT_INDEX is outside transfer64"; exit 2;
    }
    cursor=0
    while [[ "$cursor" -lt "${#REQUESTED_METHODS[@]}" ]]; do
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            local index=$((cursor + slot))
            [[ "$index" -lt "${#REQUESTED_METHODS[@]}" ]] || break
            method="${REQUESTED_METHODS[$index]}"
            run_shard "$OUT_BASE/smoke" "$method" "$SMOKE_PROMPT_INDEX" \
                "$prompt_count" "${GPUS[$slot]}" 1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v194 smoke failed"; exit 1; }
        cursor=$((cursor + GPUS_PER_NODE))
    done
}

generate() {
    preflight
    [[ "$NUM_NODES" -eq 4 && "$GPUS_PER_NODE" -eq 8 && "$WORLD_SHARDS" -eq 32 ]] || {
        echo "[error] v194 generation is frozen to 4 nodes x 8 GPUs"; exit 2;
    }
    local count="${#REQUESTED_METHODS[@]}"
    local rotation=$((NODE_RANK % count))
    local offset method slot rank pid failed trace
    for ((offset=0; offset<count; offset++)); do
        method="${REQUESTED_METHODS[$(((rotation + offset) % count))]}"
        echo "[v194-method] method=$method node=$NODE_RANK order_slot=$offset"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            trace=0
            [[ "$rank" -eq 0 ]] && trace=1
            run_shard "$RUN_ROOT" "$method" "$rank" "$WORLD_SHARDS" \
                "${GPUS[$slot]}" "$trace" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v194 method=$method failed on node=$NODE_RANK"; exit 1;
        }
    done
}

audit_run() {
    local target_root="$1" smoke="$2"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    local -a args=(--run-root "$target_root" --input-manifest "$MANIFEST")
    if [[ "$smoke" == "1" ]]; then
        args+=(--smoke-prompt-index "$SMOKE_PROMPT_INDEX")
    fi
    python "$ROOT/scripts/audit_v194_cf_checkpoint_transfer.py" "${args[@]}"
}

status() {
    python - "$RUN_ROOT" "$MANIFEST" "$ALL_METHODS" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
count = int(manifest["prompt_count"])
for method in sys.argv[3].split(","):
    raw = root / "raw" / method
    observed = {
        int(path.name.split("-", 1)[0])
        for path in raw.glob("*-0_regular.mp4")
        if path.name.split("-", 1)[0].isdigit()
    }
    logs = list((root / "logs" / method).glob("*.log"))
    failures = sum(
        "Traceback (most recent call last)" in path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in logs
    )
    print(
        f"{method}: videos={len(observed)}/{count} "
        f"missing={len(set(range(count))-observed)} logs={len(logs)} failures={failures}"
    )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_BASE/v194_cf_checkpoint_transfer_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$target" inputs transfer64/contracts transfer64/audits \
        transfer64/published_manifest.json transfer64/metrics transfer64/analysis \
        transfer64/logs transfer64/traces
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    generate) generate ;;
    status) status ;;
    audit-smoke) audit_run "$OUT_BASE/smoke" 1 ;;
    audit) audit_run "$RUN_ROOT" 0 ;;
    package) package ;;
esac
