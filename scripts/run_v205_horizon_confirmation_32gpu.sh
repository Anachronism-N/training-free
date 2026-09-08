#!/usr/bin/env bash
# Profiling-disjoint confirmation for v201-selected horizon candidates.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate128|status|audit-smoke|audit-confirm|package) ;;
    *)
        echo "usage: bash scripts/run_v205_horizon_confirmation_32gpu.sh ACTION"
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
V201_ROOT="${V201_OUT_ROOT:-$ROOT/runs/v201_head_phase_horizon_sf_screen}"
V201_INPUT="${V201_INPUT_MANIFEST:-$V201_ROOT/inputs/manifest.json}"
V201_RUN="${V201_RUN_ROOT:-$V201_ROOT/screen32}"
V201_PUBLISHED="${V201_PUBLISHED:-$V201_RUN/published_manifest.json}"
V201_COMPARISON="${V201_COMPARISON_MANIFEST:-$V201_RUN/vbench_comparison/comparison_manifest.json}"
V201_DECISION="${V201_DECISION:-$V201_RUN/analysis/v201_head_phase_horizon.json}"
FRESH_PROMPT_MANIFEST="${V205_FRESH_PROMPT_MANIFEST:-$ROOT/runs/v180_rccp_fresh128/inputs/manifest.json}"
OUT_BASE="${V205_OUT_ROOT:-$ROOT/runs/v205_horizon_confirmation}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/prompts/moviegen_profile_disjoint_0128_0255.txt"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-20500}"
SMOKE_PROMPT_INDEX="${V205_SMOKE_PROMPT_INDEX:-7}"
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
[[ "$FRAMES" -eq 120 && "$SEED" -eq 20500 ]] || {
    echo "[error] v205 is frozen at 120 latent frames and seed 20500"; exit 2;
}
[[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt 128 ]] || {
    echo "[error] V205_SMOKE_PROMPT_INDEX must be within [0,127]"; exit 2;
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
            SCENE_TRANSITION_*|CACHE_COMPAT_*|PYRAMIDKV_*) unset "$key" ;;
        esac
    done < <(env)
}

verify_sampling_parity() {
    python - "$SF_CONFIG" "$PF_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    sf = yaml.safe_load(handle)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    pf = yaml.safe_load(handle)

def read(payload, dotted):
    value = payload
    for key in dotted.split("."):
        value = value[key]
    return value

shared = (
    "denoising_step_list",
    "warp_denoising_step",
    "ts_schedule",
    "num_train_timestep",
    "timestep_shift",
    "guidance_scale",
    "negative_prompt",
    "image_or_video_shape",
    "num_frame_per_block",
    "model_kwargs.timestep_shift",
)
mismatches = {
    key: (read(sf, key), read(pf, key))
    for key in shared
    if read(sf, key) != read(pf, key)
}
if mismatches:
    raise SystemExit(f"SF/PF sampling config mismatch: {mismatches}")
if read(sf, "model_kwargs.local_attn_size") != 21:
    raise SystemExit("canonical SF must retain local_attn_size=21")
if read(pf, "model_kwargs.local_attn_size") != -1:
    raise SystemExit("cache runtime must expose full history with local_attn_size=-1")
if not bool(pf.get("use_pyramidkv")) or not bool(pf.get("use_adaptive_pyramidkv")):
    raise SystemExit("cache runtime requires adaptive PyramidKV plumbing")
if not bool(pf.get("sink_grid_decoupling")):
    raise SystemExit("cache runtime requires sink-grid decoupling")
print("[v205-config] PASS shared_sampling=true sf_window=21 cache_history=full")
PY
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v205_horizon_confirmation.py" prepare \
        --v201-decision "$V201_DECISION" \
        --v201-input-manifest "$V201_INPUT" \
        --v201-published "$V201_PUBLISHED" \
        --v201-comparison-manifest "$V201_COMPARISON" \
        --fresh-prompt-manifest "$FRESH_PROMPT_MANIFEST" \
        --output-root "$INPUT_ROOT"
}

manifest_methods() {
    python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
for method in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["method_order"]:
    print(method)
PY
}

requested_methods() {
    if [[ -z "${METHODS:-}" ]]; then
        manifest_methods
    else
        tr ',' '\n' <<<"$METHODS"
    fi
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" "$CHECKPOINT" \
        "$MANIFEST" "$PROMPTS"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    verify_sampling_parity
    python "$ROOT/scripts/prepare_v205_horizon_confirmation.py" verify \
        --manifest "$MANIFEST"
    mapfile -t available < <(manifest_methods)
    mapfile -t requested < <(requested_methods)
    [[ "${#requested[@]}" -gt 0 ]] || { echo "[error] no v205 methods"; exit 2; }
    declare -A seen=()
    local method
    for method in "${requested[@]}"; do
        [[ " ${available[*]} " == *" $method "* ]] || {
            echo "[error] unsupported v205 method: $method"; exit 2;
        }
        [[ -z "${seen[$method]:-}" ]] || {
            echo "[error] duplicate v205 method: $method"; exit 2;
        }
        seen[$method]=1
    done
    local -a runtime_paths=(
        third_party/Self-Forcing/inference.py
        third_party/Self-Forcing/configs/self_forcing_dmd.yaml
        third_party/Pyramid-Forcing/inference.py
        third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml
        third_party/Pyramid-Forcing/pyramidkv
        scripts/prepare_v205_horizon_confirmation.py
        scripts/audit_v205_horizon_confirmation.py
    )
    git -C "$ROOT" diff --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v205 runtime has unstaged changes"; exit 2;
    }
    git -C "$ROOT" diff --cached --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v205 runtime has staged changes"; exit 2;
    }
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v200_head_phase_horizon.py \
            tests/test_v201_head_phase_horizon.py \
            tests/test_v205_horizon_confirmation.py)
    fi
    echo "[v205-preflight] PASS methods=${#requested[@]} prompts=128 gate=v201"
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["methods"][sys.argv[2]][sys.argv[3]])
PY
}

configure_cache_runtime() {
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
    return 0
}

run_shard() {
    local scope_root="$1" method="$2" prompt_count="$3" rank="$4"
    local stride="$5" gpu="$6" trace_enabled="$7"
    local raw_dir="$scope_root/raw/$method"
    local shard_name="shard$(printf '%02d' "$rank")"
    local log="$scope_root/logs/$method/$shard_name.log"
    local marker="$scope_root/status/$method/$shard_name.done"
    local trace="$scope_root/traces/$method/$shard_name.schedule.jsonl"
    [[ "$rank" -lt "$prompt_count" ]] || return
    if [[ "$FORCE" == "1" ]]; then
        local index
        for ((index=rank; index<prompt_count; index+=stride)); do
            rm -f "$raw_dir/${index}-0_ema.mp4"
        done
        rm -f "$marker" "$trace"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
       shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"; then
        echo "[v205-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")"
    if [[ "$method" == "sf_native" ]]; then
        (
            cd "$SF"
            scrub_experiment_env
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
            export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0
            export COMMIT_FORCING_ENABLE=0 HEAD_ROLE_ENABLE=0
            export HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPTS" --output_folder "$raw_dir" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt --skip_existing \
                --end_idx "$prompt_count" --prompt_stride "$stride" \
                --prompt_offset "$rank"
        ) >"$log" 2>&1
    else
        local operator history_policy horizon_map head_map
        operator="$(manifest_value "$method" operator)"
        history_policy="$(manifest_value "$method" history_policy)"
        horizon_map="$(manifest_value "$method" horizon_map)"
        head_map="$(manifest_value "$method" head_bank_map)"
        (
            cd "$PF"
            scrub_experiment_env
            configure_cache_runtime
            export CUDA_VISIBLE_DEVICES="$gpu"
            if [[ "$trace_enabled" == "1" ]]; then
                mkdir -p "$(dirname "$trace")"
                export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
            else
                unset PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH
            fi
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
                --pyramidkv_cache_compatibility_denoise_schedule head_phase_horizon \
                --pyramidkv_cache_compatibility_denoise_coverage_policy "$operator" \
                --pyramidkv_cache_compatibility_horizon_map "$horizon_map"
        ) >"$log" 2>&1
    fi
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

run_methods() {
    local scope_root="$1" smoke_mode="$2"
    mapfile -t methods < <(requested_methods)
    local method_count="${#methods[@]}"
    local shift=$((NODE_RANK % method_count))
    local -a ordered=()
    local index
    for ((index=0; index<method_count; index++)); do
        ordered+=("${methods[$(((index + shift) % method_count))]}")
    done
    local method slot rank pid failed trace
    if [[ "$smoke_mode" == "1" ]]; then
        local cursor=0
        while [[ "$cursor" -lt "$method_count" ]]; do
            local -a pids=()
            for slot in "${!GPUS[@]}"; do
                index=$((cursor + slot))
                [[ "$index" -lt "$method_count" ]] || break
                method="${ordered[$index]}"
                trace=0
                [[ "$method" != "sf_native" ]] && trace=1
                run_shard "$scope_root" "$method" 128 "$SMOKE_PROMPT_INDEX" 128 \
                    "${GPUS[$slot]}" "$trace" &
                pids+=("$!")
            done
            failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            [[ "$failed" -eq 0 ]] || { echo "[error] v205 smoke failed"; exit 1; }
            cursor=$((cursor + GPUS_PER_NODE))
        done
        return
    fi
    for method in "${ordered[@]}"; do
        echo "[v205-method] method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            trace=0
            [[ "$method" != "sf_native" && "$rank" -eq 0 ]] && trace=1
            run_shard "$scope_root" "$method" 128 "$rank" "$WORLD_SHARDS" \
                "${GPUS[$slot]}" "$trace" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v205 method=$method failed on node=$NODE_RANK"; exit 1;
        }
    done
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    run_methods "$OUT_BASE/smoke" 1
}

generate128() {
    preflight
    [[ "$NUM_NODES" -eq 4 && "$GPUS_PER_NODE" -eq 8 && "$WORLD_SHARDS" -eq 32 ]] || {
        echo "[error] v205 generate128 is frozen to 4 nodes x 8 GPUs"; exit 2;
    }
    run_methods "$OUT_BASE/confirm128" 0
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v205_horizon_confirmation.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --scope "$scope" --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
}

status() {
    python - "$OUT_BASE" "$MANIFEST" "$SMOKE_PROMPT_INDEX" <<'PY'
import json, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
smoke = int(sys.argv[3])
methods = json.loads(manifest.read_text(encoding="utf-8"))["method_order"]
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
    local target="$OUT_BASE/v205_horizon_confirmation_diagnostics.tar.gz"
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
