#!/usr/bin/env bash
# Hard-gated MovieGen-128 paper confirmation at 30 and 60 seconds.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate30|generate60|status|audit-smoke|audit30|audit60|package) ;;
    *)
        echo "usage: bash scripts/run_v208_paper_confirmation_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate30 generate60 status audit-smoke audit30 audit60 package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V208_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
V207_ROOT="${V207_OUT_ROOT:-$ROOT/runs/v207_context_budget_phase_recovery}"
V207_INPUTS="${V207_INPUT_MANIFEST:-$V207_ROOT/inputs/manifest.json}"
V207_REPORT="${V207_REPORT:-$V207_ROOT/screen32/analysis/v207_context_budget_phase.json}"
V207_PARITY_REPORT="${V207_PARITY_REPORT:-$V207_ROOT/parity/report/parity_report.json}"
V207_PARITY_INPUTS="${V207_PARITY_INPUTS:-$V207_ROOT/parity/inputs/manifest.json}"
OUT_BASE="${V208_OUT_ROOT:-$ROOT/runs/v208_paper_confirmation}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/prompts/moviegen_qwen_full128.txt"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-20800}"
SMOKE_PROMPT_INDEX="${V208_SMOKE_PROMPT_INDEX:-3}"
FORCE="${FORCE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
GPUS_PER_NODE="${#GPUS[@]}"
WORLD_SHARDS=$((NUM_NODES * GPUS_PER_NODE))
[[ "$NODE_RANK" -ge 0 && "$NODE_RANK" -lt "$NUM_NODES" ]] || {
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"; exit 2;
}
[[ "$SEED" -eq 20800 ]] || { echo "[error] v208 seed is frozen to 20800"; exit 2; }
[[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt 128 ]] || {
    echo "[error] V208_SMOKE_PROMPT_INDEX must be within [0,127]"; exit 2;
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$PF:$SF:${PYTHONPATH:-}"
}

scrub_experiment_env() {
    local key
    while IFS='=' read -r key _; do
        case "$key" in
            LIFECACHE_*|HEAD_ROLE_*|STRUCTURED_MEMORY_*|COMMIT_FORCING_*|\
            SCENE_TRANSITION_*|CACHE_COMPAT_*|PYRAMIDKV_*|SF_PARITY_*) unset "$key" ;;
        esac
    done < <(env)
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v208_paper_confirmation.py" prepare \
        --source-prompts "$SOURCE_PROMPTS" \
        --v207-input-manifest "$V207_INPUTS" --v207-report "$V207_REPORT" \
        --parity-report "$V207_PARITY_REPORT" \
        --output-root "$INPUT_ROOT"
    bind_runtime
}

bind_runtime() {
    python "$ROOT/scripts/bind_v208_runtime.py" \
        --manifest "$MANIFEST" --parity-input-manifest "$V207_PARITY_INPUTS" \
        --repo-root "$ROOT" --checkpoint "$CHECKPOINT" \
        --sf-config "$SF_CONFIG" --pf-config "$PF_CONFIG" \
        --output "$INPUT_ROOT/runtime_contract.json"
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload["methods"][sys.argv[2]][sys.argv[3]]
print("" if value is None else value)
PY
}

requested_methods() {
    if [[ -z "${METHODS:-}" ]]; then
        printf '%s\n' sf_native recent21 ours
    else
        tr ',' '\n' <<<"$METHODS"
    fi
}

scope_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = next(item for item in payload["scopes"] if item["key"] == sys.argv[2])
print(row[sys.argv[3]])
PY
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" "$CHECKPOINT" "$MANIFEST" "$PROMPTS" "$V207_PARITY_REPORT"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v208_paper_confirmation.py" verify --manifest "$MANIFEST"
    [[ -s "$INPUT_ROOT/runtime_contract.json" ]] || {
        echo "[error] runtime contract missing; run prepare on node 0"; exit 2;
    }
    bind_runtime
    mapfile -t requested < <(requested_methods)
    declare -A seen=()
    local method
    for method in "${requested[@]}"; do
        [[ "$method" == "sf_native" || "$method" == "recent21" || "$method" == "ours" ]] || {
            echo "[error] unsupported v208 method: $method"; exit 2;
        }
        [[ -z "${seen[$method]:-}" ]] || { echo "[error] duplicate method: $method"; exit 2; }
        seen[$method]=1
    done
    local -a runtime_paths=(
        third_party/Self-Forcing/inference.py
        third_party/Pyramid-Forcing/inference.py
        third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
        third_party/Pyramid-Forcing/pyramidkv/denoise_schedule.py
        third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py
        scripts/prepare_v208_paper_confirmation.py
        scripts/run_v208_paper_confirmation_32gpu.sh
        scripts/audit_v208_paper_confirmation.py
    )
    git -C "$ROOT" diff --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v208 runtime has unstaged changes"; exit 2;
    }
    git -C "$ROOT" diff --cached --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v208 runtime has staged changes"; exit 2;
    }
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v207_context_budget_phase_screen.py \
            tests/test_v208_paper_confirmation.py)
    fi
    echo "[v208-preflight] PASS parent=$(manifest_value ours parent_method) methods=${#requested[@]}"
}

configure_cache_runtime() {
    export PYRAMIDKV_DENSE_COMPAT_ATTENTION=1
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
    export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS="$(seq -s, 0 29)"
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS="$(seq -s, 0 11)"
}

shard_complete() {
    local raw_dir="$1" prompt_count="$2" rank="$3" stride="$4"
    local index
    for ((index=rank; index<prompt_count; index+=stride)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] || return 1
    done
}

run_shard() {
    local scope_root="$1" method="$2" prompt_count="$3" frames="$4" rank="$5" stride="$6" gpu="$7"
    local raw_dir="$scope_root/raw/$method"
    local shard_name="shard$(printf '%02d' "$rank")"
    local log="$scope_root/logs/$method/$shard_name.log"
    local marker="$scope_root/status/$method/$shard_name.done"
    local trace="$scope_root/traces/$method/$shard_name.schedule.jsonl"
    local started_at
    [[ "$rank" -lt "$prompt_count" ]] || return
    if [[ "$FORCE" != "1" && -s "$marker" ]] && shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v208-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    # A partial shard is regenerated together with its trace and log, at most
    # four prompts per shard on 32 GPUs. This keeps audit and timing coherent.
    if [[ -d "$raw_dir" ]]; then
        local index
        for ((index=rank; index<prompt_count; index+=stride)); do
            rm -f "$raw_dir/${index}-0_ema.mp4"
        done
    fi
    rm -f "$marker" "$trace"
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")" "$(dirname "$trace")"
    started_at="$(date +%s)"
    if [[ "$method" == "sf_native" ]]; then
        (
            cd "$SF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$frames" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" --prompt_offset "$rank"
        ) >"$log" 2>&1
    else
        local operator schedule head_map history_policy archive_capacity
        operator="$(manifest_value "$method" operator)"
        schedule="$(manifest_value "$method" schedule)"
        head_map="$(manifest_value "$method" head_map)"
        history_policy="$(manifest_value "$method" history_policy)"
        archive_capacity="$(manifest_value "$method" archive_capacity)"
        (
            cd "$PF"
            scrub_experiment_env
            configure_cache_runtime
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
            python inference.py \
                --config_path "$PF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$frames" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" --prompt_offset "$rank" \
                --model_local_attn_size 21 --pyramidkv_head_config_path "$head_map" \
                --pyramidkv_history_polarity \
                --pyramidkv_history_support_policy "$history_policy" \
                --pyramidkv_history_suppress_policy "$history_policy" \
                --pyramidkv_cache_compatibility_denoise_schedule "$schedule" \
                --pyramidkv_cache_compatibility_denoise_coverage_policy "$operator" \
                --pyramidkv_cache_compatibility_read_budget_frames 21 \
                --pyramidkv_semantic_retrieval_archive_capacity "$archive_capacity"
        ) >"$log" 2>&1
    fi
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    local finished_at videos=0 index
    finished_at="$(date +%s)"
    for ((index=rank; index<prompt_count; index+=stride)); do
        [[ -s "$raw_dir/${index}-0_ema.mp4" ]] && videos=$((videos + 1))
    done
    printf '[V208Runtime] method=%s elapsed_seconds=%s videos=%s gpu=%s rank=%s stride=%s\n' \
        "$method" "$((finished_at-started_at))" "$videos" "$gpu" "$rank" "$stride" >>"$log"
    printf '[V208Provenance] runtime_sha256=%s input_sha256=%s git=%s\n' \
        "$(sha256sum "$INPUT_ROOT/runtime_contract.json" | cut -d' ' -f1)" \
        "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" "$(git -C "$ROOT" rev-parse HEAD)" >>"$log"
    printf 'ok\n' >"$marker"
}

run_scope() {
    local scope="$1" prompt_count="$2" frames="$3" smoke_mode="$4"
    local scope_root="$OUT_BASE/$scope"
    mapfile -t methods < <(requested_methods)
    local method_count="${#methods[@]}"
    local shift=$((NODE_RANK % method_count))
    local -a ordered=()
    local index
    for ((index=0; index<method_count; index++)); do
        ordered+=("${methods[$(((index+shift)%method_count))]}")
    done
    if [[ "$smoke_mode" == "1" ]]; then
        local -a pids=()
        local slot pid failed=0
        [[ "$GPUS_PER_NODE" -ge "$method_count" ]] || { echo "[error] smoke needs three GPUs"; exit 2; }
        for slot in "${!ordered[@]}"; do
            run_shard "$scope_root" "${ordered[$slot]}" 128 "$frames" \
                "$SMOKE_PROMPT_INDEX" 128 "${GPUS[$slot]}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v208 smoke failed"; exit 1; }
        return
    fi
    local method slot rank pid failed
    for method in "${ordered[@]}"; do
        echo "[v208-method] scope=$scope method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            if [[ "$rank" -lt "$prompt_count" ]]; then
                run_shard "$scope_root" "$method" "$prompt_count" "$frames" "$rank" \
                    "$WORLD_SHARDS" "${GPUS[$slot]}" &
                pids+=("$!")
            fi
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || { echo "[error] v208 $scope/$method failed"; exit 1; }
    done
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v208_paper_confirmation.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --scope "$scope" --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
    if [[ "$scope" != "smoke" ]]; then
        python "$ROOT/scripts/analyze_v208_efficiency.py" \
            --manifest "$MANIFEST" --run-root "$OUT_BASE/$scope" --scope "$scope" \
            --output "$OUT_BASE/$scope/analysis/v208_efficiency.json"
    fi
}

status() {
    python - "$OUT_BASE" "$MANIFEST" "$SMOKE_PROMPT_INDEX" <<'PY'
import json, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
smoke = int(sys.argv[3])
for scope, expected in (("smoke", {smoke}), ("main30", set(range(128))), ("long60", set(range(128)))):
    print(f"[{scope}]")
    for method in manifest["method_order"]:
        observed = {int(p.name.split("-", 1)[0]) for p in (root/scope/"raw"/method).glob("*-0_ema.mp4") if p.name.split("-", 1)[0].isdigit()}
        print(f"{method}: videos={len(observed)}/{len(expected)} missing={sorted(expected-observed)}")
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_BASE/v208_paper_confirmation_artifacts.tar.gz"
    local -a items=(inputs)
    local scope folder
    for scope in smoke main30 long60; do
        for folder in contracts audits logs analysis; do
            [[ ! -d "$OUT_BASE/$scope/$folder" ]] || items+=("$scope/$folder")
        done
        if [[ "${INCLUDE_RAW_TRACES:-0}" == "1" && -d "$OUT_BASE/$scope/traces" ]]; then
            items+=("$scope/traces")
        fi
    done
    tar -C "$OUT_BASE" -czf "$target" "${items[@]}"
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) preflight; run_scope smoke 128 120 1 ;;
    generate30) preflight; run_scope main30 128 "$(scope_value main30 num_output_frames)" 0 ;;
    generate60) preflight; run_scope long60 128 "$(scope_value long60 num_output_frames)" 0 ;;
    status) status ;;
    audit-smoke) audit_scope smoke ;;
    audit30) audit_scope main30 ;;
    audit60) audit_scope long60 ;;
    package) package ;;
esac
