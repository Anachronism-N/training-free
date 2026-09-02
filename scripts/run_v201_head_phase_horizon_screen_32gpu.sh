#!/usr/bin/env bash
# Classifier-holdout causal screen for frozen Head x Phase x AR-horizon maps.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|smoke|generate32|status|audit-smoke|audit-screen|package) ;;
    *)
        echo "usage: bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh ACTION"
        echo "actions: prepare preflight smoke generate32 status audit-smoke audit-screen package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
V189_ROOT="${V189_OUT_ROOT:-$ROOT/runs/v189_structured_head_phase_profile}"
V189_MANIFEST="${V189_MANIFEST:-$V189_ROOT/inputs/manifest.json}"
V200_ROOT="${V200_OUT_ROOT:-$ROOT/runs/v200_head_phase_horizon_audit}"
V200_ANALYSIS="${V200_ANALYSIS:-$V200_ROOT/analysis/analysis.json}"
OUT_BASE="${V201_OUT_ROOT:-$ROOT/runs/v201_head_phase_horizon_causal_screen}"
INPUT_ROOT="$OUT_BASE/inputs"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPTS="$INPUT_ROOT/prompts/moviegen_qwen_holdout32.txt"

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-20100}"
SMOKE_PROMPT_INDEX="${V201_SMOKE_PROMPT_INDEX:-3}"
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
[[ "$FRAMES" -eq 120 && "$SEED" -eq 20100 ]] || {
    echo "[error] v201 is frozen at 120 latent frames and seed 20100"; exit 2;
}
[[ "$SMOKE_PROMPT_INDEX" -ge 0 && "$SMOKE_PROMPT_INDEX" -lt 32 ]] || {
    echo "[error] V201_SMOKE_PROMPT_INDEX must be within [0,31]"; exit 2;
}

activate_env() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
}

prepare() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] prepare requires node 0"; exit 2; }
    activate_env
    python "$ROOT/scripts/prepare_v201_head_phase_horizon_screen.py" prepare \
        --v189-manifest "$V189_MANIFEST" --v200-analysis "$V200_ANALYSIS" \
        --output-root "$INPUT_ROOT"
}

manifest_methods() {
    python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for method in payload["method_order"]:
    print(method)
PY
}

requested_methods() {
    if [[ -z "${METHODS:-}" ]]; then
        manifest_methods
        return
    fi
    tr ',' '\n' <<<"$METHODS"
}

preflight() {
    activate_env
    for path in "$PF" "$CONFIG" "$CHECKPOINT" "$MANIFEST" "$PROMPTS"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    python "$ROOT/scripts/prepare_v201_head_phase_horizon_screen.py" verify \
        --manifest "$MANIFEST"
    mapfile -t available < <(manifest_methods)
    mapfile -t requested < <(requested_methods)
    [[ "${#requested[@]}" -gt 0 ]] || { echo "[error] no v201 methods"; exit 2; }
    declare -A seen=()
    local method
    for method in "${requested[@]}"; do
        [[ " ${available[*]} " == *" $method "* ]] || {
            echo "[error] unsupported v201 method: $method"; exit 2;
        }
        [[ -z "${seen[$method]:-}" ]] || {
            echo "[error] duplicate v201 method: $method"; exit 2;
        }
        seen[$method]=1
    done
    local -a runtime_paths=(
        third_party/Pyramid-Forcing/inference.py
        third_party/Pyramid-Forcing/pyramidkv/denoise_schedule.py
        third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
        scripts/analyze_v200_head_phase_horizon.py
        scripts/prepare_v201_head_phase_horizon_screen.py
    )
    git -C "$ROOT" diff --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v201 runtime has unstaged changes"; exit 2;
    }
    git -C "$ROOT" diff --cached --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked v201 runtime has staged changes"; exit 2;
    }
    if [[ "$NODE_RANK" -eq 0 && "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v200_head_phase_horizon.py \
            tests/test_v201_head_phase_horizon.py)
    fi
    echo "[v201-preflight] PASS methods=${#requested[@]} prompts=32 gate=v200"
}

manifest_value() {
    python - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
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
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS="$(seq -s, 0 29)"
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS="$(seq -s, 0 11)"
    unset CACHE_COMPAT_PROFILE CACHE_COMPAT_PROFILE_CONTRACT
    unset PYRAMIDKV_CACHE_COMPAT_DENOISE_SCHEDULE
    unset PYRAMIDKV_CACHE_COMPAT_HEAD_PHASE_MAP
    unset PYRAMIDKV_CACHE_COMPAT_HORIZON_MAP
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
    local trace="$scope_root/traces/$method/$shard_name.schedule.jsonl"
    local operator history_policy horizon_map head_map
    operator="$(manifest_value "$method" operator)"
    history_policy="$(manifest_value "$method" history_policy)"
    horizon_map="$(manifest_value "$method" horizon_map)"
    head_map="$(manifest_value "$method" head_bank_map)"

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
        echo "[v201-skip] scope=$(basename "$scope_root") method=$method rank=$rank"
        return
    fi
    mkdir -p "$raw_dir" "$(dirname "$log")" "$(dirname "$marker")" \
        "$(dirname "$trace")"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$trace"
        python inference.py \
            --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
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
    shard_complete "$raw_dir" "$prompt_count" "$rank" "$stride"
    printf 'ok\n' >"$marker"
}

run_methods() {
    local scope_root="$1" prompt_count="$2" smoke_mode="$3"
    mapfile -t methods < <(requested_methods)
    local method_count="${#methods[@]}"
    local shift=$((NODE_RANK % method_count))
    local -a ordered=()
    local index
    for ((index=0; index<method_count; index++)); do
        ordered+=("${methods[$(((index + shift) % method_count))]}")
    done
    local method slot rank pid failed start
    if [[ "$smoke_mode" == "1" ]]; then
        start="$SMOKE_PROMPT_INDEX"
        local cursor=0
        while [[ "$cursor" -lt "$method_count" ]]; do
            local -a pids=()
            for slot in "${!GPUS[@]}"; do
                index=$((cursor + slot))
                [[ "$index" -lt "$method_count" ]] || break
                method="${ordered[$index]}"
                run_shard "$scope_root" "$method" 32 "$start" 32 \
                    "${GPUS[$slot]}" &
                pids+=("$!")
            done
            failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            [[ "$failed" -eq 0 ]] || {
                echo "[error] v201 smoke wave failed"; exit 1;
            }
            cursor=$((cursor + GPUS_PER_NODE))
        done
        return
    fi
    for method in "${ordered[@]}"; do
        echo "[v201-method] method=$method node=$NODE_RANK"
        local -a pids=()
        for slot in "${!GPUS[@]}"; do
            rank=$((NODE_RANK * GPUS_PER_NODE + slot))
            if [[ "$rank" -lt "$prompt_count" ]]; then
                run_shard "$scope_root" "$method" "$prompt_count" "$rank" \
                    "$WORLD_SHARDS" "${GPUS[$slot]}" &
                pids+=("$!")
            fi
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" -eq 0 ]] || {
            echo "[error] v201 method=$method failed on node=$NODE_RANK"; exit 1;
        }
    done
}

smoke() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] smoke requires node 0"; exit 2; }
    preflight
    configure_runtime
    run_methods "$OUT_BASE/smoke" 32 1
}

generate32() {
    preflight
    configure_runtime
    run_methods "$OUT_BASE/screen32" 32 0
}

audit_scope() {
    local scope="$1"
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] audit requires node 0"; exit 2; }
    preflight
    python "$ROOT/scripts/audit_v201_head_phase_horizon_screen.py" \
        --run-root "$OUT_BASE/$scope" --input-manifest "$MANIFEST" \
        --scope "$scope" --smoke-prompt-index "$SMOKE_PROMPT_INDEX"
}

status() {
    python - "$OUT_BASE" "$MANIFEST" "$SMOKE_PROMPT_INDEX" <<'PY'
import json, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
smoke_index = int(sys.argv[3])
methods = json.loads(manifest.read_text(encoding="utf-8"))["method_order"]
for scope, expected in (("smoke", {smoke_index}), ("screen32", set(range(32)))):
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
            f"missing={sorted(expected-observed)} logs={len(logs)} "
            f"failures={failures}"
        )
PY
}

package() {
    [[ "$NODE_RANK" -eq 0 ]] || { echo "[error] package requires node 0"; exit 2; }
    local target="$OUT_BASE/v201_head_phase_horizon_diagnostics.tar.gz"
    tar -C "$OUT_BASE" -czf "$target" \
        inputs smoke/contracts smoke/audits smoke/published_manifest.json \
        smoke/logs smoke/traces screen32/contracts screen32/audits \
        screen32/published_manifest.json screen32/metrics screen32/analysis \
        screen32/logs screen32/traces
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    smoke) smoke ;;
    generate32) generate32 ;;
    status) status ;;
    audit-smoke) audit_scope smoke ;;
    audit-screen) audit_scope screen32 ;;
    package) package ;;
esac
