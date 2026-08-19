#!/usr/bin/env bash
# Post-v187 seed replication, 60-second persistence, and phase counterfactuals.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate-replica|generate-long|generate-mechanism|status|audit-smoke|audit-replica|audit-long|audit-mechanism|package) ;;
    *)
        echo "usage: bash scripts/run_v188_robustness_matrix_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate-replica generate-long generate-mechanism status audit-smoke audit-replica audit-long audit-mechanism package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
V187_ROOT="${V187_OUT_ROOT:-$ROOT/runs/v187_unseen128_confirmation}"
V187_DECISION="${V187_DECISION:-$V187_ROOT/confirm128/analysis/v187_unseen128_confirmation.json}"
V187_INPUT="${V187_INPUT_MANIFEST:-$V187_ROOT/inputs/manifest.json}"
V187_PUBLISHED="${V187_PUBLISHED:-$V187_ROOT/confirm128/published_manifest.json}"
OUT_BASE="${V188_OUT_ROOT:-$ROOT/runs/v188_robustness_matrix}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"

REPLICA_SCOPE="replica64_seed20000"
LONG_SCOPE="long60_seed10000_32"
MECHANISM_SCOPE="mechanism32_seed10000"
BASE_METHODS="sf_native,all_recent,phase_reservoir,phase_deterministic"
COUNTERFACTUAL_METHODS="phase_deterministic,opposite_phase_deterministic,all_noisy_deterministic"
SMOKE_METHODS="$BASE_METHODS,$COUNTERFACTUAL_METHODS"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SMOKE_PROMPT_INDEX="${V188_SMOKE_PROMPT_INDEX:-5}"
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
    python "$ROOT/scripts/prepare_v188_robustness_matrix.py" prepare \
        --v187-decision "$V187_DECISION" \
        --v187-input-manifest "$V187_INPUT" \
        --v187-published "$V187_PUBLISHED" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" "$CHECKPOINT" "$MANIFEST"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v188_robustness_matrix.py" verify \
        --manifest "$MANIFEST"
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v187_unseen128_confirmation.py \
            tests/test_v188_robustness_matrix.py)
    fi
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
kind, name, key = sys.argv[2:]
if kind == "method":
    value = payload["method_templates"][name][key]
else:
    rows = {row["key"]: row for row in payload["scopes"]}
    value = rows[name][key]
if isinstance(value, (list, dict)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
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

shard_size() {
    local prompt_count="$1" rank="$2" stride="$3"
    if [[ "$rank" -ge "$prompt_count" ]]; then
        echo 0
    else
        echo $(( (prompt_count - 1 - rank) / stride + 1 ))
    fi
}

run_shard() {
    local scope_key="$1" scope_root="$2" method="$3" prompt_file="$4"
    local prompt_count="$5" frames="$6" seed="$7" rank="$8" stride="$9"
    local gpu="${10}"
    local raw_dir="$scope_root/raw/$method"
    local shard_name="shard$(printf '%03d' "$rank")"
    local log="$scope_root/logs/$method/$shard_name.log"
    local marker="$scope_root/status/$method/$shard_name.done"
    local trace="$scope_root/traces/$method/$shard_name.schedule.jsonl"
    local videos
    videos="$(shard_size "$prompt_count" "$rank" "$stride")"

    [[ "$videos" -gt 0 ]] || return
    if [[ "$FORCE" == "1" ]]; then
        clear_shard "$raw_dir" "$prompt_count" "$rank" "$stride"
        rm -f "$marker" "$trace"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v188-skip] scope=$scope_key method=$method rank=$rank"
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
            started="$(date +%s)"
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$prompt_file" --output_folder "$raw_dir" \
                --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" \
                --prompt_offset "$rank"
            elapsed=$(( $(date +%s) - started ))
            echo "[v188-runtime] scope=$scope_key method=$method shard=$rank videos=$videos elapsed_seconds=$elapsed frames=$frames seed=$seed"
        ) >"$log" 2>&1
    else
        local schedule operator history_policy head_map
        schedule="$(manifest_value method "$method" schedule)"
        operator="$(manifest_value method "$method" operator)"
        history_policy="$(manifest_value method "$method" history_policy)"
        head_map="$(manifest_value method "$method" head_map)"
        (
            cd "$PF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
            export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
            export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
            export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS=0,10,20,29
            export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS=0,6,10
            export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
            export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
            started="$(date +%s)"
            python inference.py \
                --config_path "$PF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$prompt_file" --output_folder "$raw_dir" \
                --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" \
                --prompt_offset "$rank" \
                --pyramidkv_head_config_path "$head_map" \
                --pyramidkv_history_polarity \
                --pyramidkv_history_support_policy "$history_policy" \
                --pyramidkv_history_suppress_policy "$history_policy" \
                --pyramidkv_cache_compatibility_denoise_schedule "$schedule" \
                --pyramidkv_cache_compatibility_denoise_coverage_policy "$operator"
            elapsed=$(( $(date +%s) - started ))
            echo "[v188-runtime] scope=$scope_key method=$method shard=$rank videos=$videos elapsed_seconds=$elapsed frames=$frames seed=$seed"
        ) >"$log" 2>&1
    fi
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    IFS=',' read -r -a methods <<<"$SMOKE_METHODS"
    [[ "$GPUS_PER_NODE" -ge "${#methods[@]}" ]] || {
        echo "[error] smoke needs six GPUs"
        exit 2
    }
    local prompt_file="$INPUT_ROOT/prompts/$MECHANISM_SCOPE.txt"
    local scope_root="$OUT_BASE/smoke"
    local -a pids=()
    local slot failed=0 pid
    for slot in "${!methods[@]}"; do
        run_shard smoke "$scope_root" "${methods[$slot]}" "$prompt_file" 32 120 10000 \
            "$SMOKE_PROMPT_INDEX" 32 "${GPUS[$slot]}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] v188 smoke failed"; exit 1; }
}

generate_scope() {
    local scope_key="$1" methods_csv="$2"
    preflight
    [[ "$NUM_NODES" -eq 4 && "$GPUS_PER_NODE" -eq 8 && "$WORLD_SHARDS" -eq 32 ]] || {
        echo "[error] v188 generation is frozen to 4 nodes x 8 GPUs"
        exit 2
    }
    local prompt_count frames seed prompt_file
    prompt_count="$(manifest_value scope "$scope_key" prompt_count)"
    frames="$(manifest_value scope "$scope_key" num_output_frames)"
    seed="$(manifest_value scope "$scope_key" seed)"
    prompt_file="$(manifest_value scope "$scope_key" prompt_file)"
    local scope_root="$OUT_BASE/$scope_key"
    IFS=',' read -r -a methods <<<"$methods_csv"
    local count="${#methods[@]}"
    local rotation=$((NODE_RANK % count))
    local offset method slot rank pid failed
    for ((offset=0; offset<count; offset++)); do
        method="${methods[$(((rotation + offset) % count))]}"
        echo "[v188-method] scope=$scope_key method=$method node=$NODE_RANK order_slot=$offset"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            run_shard "$scope_key" "$scope_root" "$method" "$prompt_file" \
                "$prompt_count" "$frames" "$seed" "$rank" "$WORLD_SHARDS" \
                "${GPUS[$slot]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v188 scope=$scope_key method=$method failed on node=$NODE_RANK"
            exit 1
        }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v188_robustness_matrix.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --scope "$scope" --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
}

status() {
    python - "$OUT_BASE" "$MANIFEST" "$SMOKE_PROMPT_INDEX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
smoke = int(sys.argv[3])
scopes = [
    ("smoke", 1, list(manifest["method_templates"])),
    *[
        (row["key"], int(row["prompt_count"]), row["generated_methods"])
        for row in manifest["scopes"]
    ],
]
for scope, count, methods in scopes:
    print(f"[{scope}]")
    for method in methods:
        raw = root / scope / "raw" / method
        observed = {
            int(path.name.split("-", 1)[0])
            for path in raw.glob("*-0_ema.mp4")
            if path.name.split("-", 1)[0].isdigit()
        }
        expected = {smoke} if scope == "smoke" else set(range(count))
        logs = list((root / scope / "logs" / method).glob("*.log"))
        failures = sum(
            "Traceback (most recent call last)" in path.read_text(
                encoding="utf-8", errors="replace"
            )
            for path in logs
        )
        print(
            f"{method}: videos={len(observed)}/{len(expected)} "
            f"missing={sorted(expected - observed)} logs={len(logs)} failures={failures}"
        )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_BASE/v188_robustness_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$target" \
        --exclude='*/raw' --exclude='*/published' \
        --exclude='*/vbench_comparison/published' \
        inputs smoke replica64_seed20000 long60_seed10000_32 \
        mechanism32_seed10000
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    generate-replica) generate_scope "$REPLICA_SCOPE" "$BASE_METHODS" ;;
    generate-long) generate_scope "$LONG_SCOPE" "$BASE_METHODS" ;;
    generate-mechanism) generate_scope "$MECHANISM_SCOPE" "$COUNTERFACTUAL_METHODS" ;;
    status) status ;;
    audit-smoke) audit_scope smoke ;;
    audit-replica) audit_scope "$REPLICA_SCOPE" ;;
    audit-long) audit_scope "$LONG_SCOPE" ;;
    audit-mechanism) audit_scope "$MECHANISM_SCOPE" ;;
    package) package ;;
esac
