#!/usr/bin/env bash
# Short-trajectory numerical parity gate for canonical SF and candidate cache runtime.
set -euo pipefail

ACTION="${1:-}"
case "$ACTION" in
    prepare|preflight|run|status|analyze|package) ;;
    *)
        echo "usage: bash scripts/run_v207_sf_parity.sh ACTION"
        echo "actions: prepare preflight run status analyze package"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_PLAIN_CONFIG="${PF_PLAIN_CONFIG:-$PF/configs/self-forcing.yaml}"
PF_ADAPTIVE_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}}"
SOURCE_PROMPTS="${V207_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
OUT_BASE="${V207_OUT_ROOT:-$ROOT/runs/v207_context_budget_phase_recovery}"
PARITY_ROOT="$OUT_BASE/parity"
INPUT_ROOT="$PARITY_ROOT/inputs"
TRACE_ROOT="$PARITY_ROOT/traces"
REPORT_ROOT="$PARITY_ROOT/report"
MANIFEST="$INPUT_ROOT/manifest.json"
PROMPT="$INPUT_ROOT/prompt.txt"
HEAD_MAP="$INPUT_ROOT/all_profile_banks.csv"

GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 3 ]] || { echo "[error] parity run requires at least 3 GPUs"; exit 2; }
FRAMES=9
SEED=20703
FORCE="${FORCE:-0}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"

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
    activate_env
    python "$ROOT/scripts/prepare_v207_sf_parity.py" prepare \
        --repo-root "$ROOT" --source-prompts "$SOURCE_PROMPTS" \
        --checkpoint "$CHECKPOINT" --sf-config "$SF_CONFIG" \
        --pf-plain-config "$PF_PLAIN_CONFIG" \
        --pf-adaptive-config "$PF_ADAPTIVE_CONFIG" \
        --output-root "$INPUT_ROOT"
}

preflight() {
    activate_env
    for path in "$SF" "$PF" "$CHECKPOINT" "$SOURCE_PROMPTS" "$MANIFEST" \
        "$PROMPT" "$HEAD_MAP" "$SF_CONFIG" "$PF_PLAIN_CONFIG" \
        "$PF_ADAPTIVE_CONFIG"; do
        [[ -e "$path" ]] || { echo "[error] missing $path; run prepare"; exit 2; }
    done
    local -a runtime_paths=(
        src/lifecycle_kv/parity_trace.py
        third_party/Self-Forcing/pipeline/causal_inference.py
        third_party/Self-Forcing/wan/modules/causal_model.py
        third_party/Pyramid-Forcing/pipeline/causal_inference.py
        third_party/Pyramid-Forcing/wan/modules/causal_model.py
        third_party/Pyramid-Forcing/wan/modules/attention/core.py
        third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
        scripts/prepare_v207_sf_parity.py
        scripts/compare_v207_sf_parity.py
        scripts/run_v207_sf_parity.sh
    )
    git -C "$ROOT" diff --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked parity runtime has unstaged changes"; exit 2;
    }
    git -C "$ROOT" diff --cached --quiet -- "${runtime_paths[@]}" || {
        echo "[error] tracked parity runtime has staged changes"; exit 2;
    }
    python "$ROOT/scripts/prepare_v207_sf_parity.py" verify \
        --repo-root "$ROOT" --manifest "$MANIFEST"
    if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
        (cd "$ROOT" && python -m pytest -q \
            tests/test_v173_cache_compatibility.py \
            tests/test_v207_sf_parity.py)
    fi
    echo "[v207-parity-preflight] PASS frames=$FRAMES blocks=3 runs=4"
}

configure_trace() {
    local run_kind="$1" trace_dir="$2"
    export SF_PARITY_TRACE_DIR="$trace_dir"
    export SF_PARITY_RUN_KIND="$run_kind"
    export SF_PARITY_CONTRACT_SHA256
    SF_PARITY_CONTRACT_SHA256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
    export SF_PARITY_TRACE_LAYERS=all
    export SF_PARITY_SAMPLE_VALUES=4096
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONHASHSEED=20703
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
}

configure_adaptive() {
    export PYRAMIDKV_DENSE_COMPAT_ATTENTION=1
    export PYRAMIDKV_ROPE_REFERENCE=1
    export PYRAMIDKV_CPP_STRATEGY=0 PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_DISABLE_M6_FASTPATH=1 PYRAMIDKV_PATH_AB=0
    export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
    export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_LAYERS="$(seq -s, 0 29)"
    export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_HEADS="$(seq -s, 0 11)"
}

run_one() {
    local run_kind="$1" gpu="$2"
    local run_root="$PARITY_ROOT/runs/$run_kind"
    local trace_dir="$TRACE_ROOT/$run_kind"
    local log="$PARITY_ROOT/logs/$run_kind.log"
    local marker="$PARITY_ROOT/status/$run_kind.done"
    if [[ "$FORCE" == "1" ]]; then
        rm -rf "$run_root" "$trace_dir"
        rm -f "$log" "$marker"
        [[ "$run_kind" != "adaptive_recent21" ]] || rm -f "$PARITY_ROOT/cache_schedule.jsonl"
    fi
    if [[ "$FORCE" != "1" && -s "$marker" && -s "$trace_dir/events.jsonl" ]]; then
        echo "[v207-parity-skip] $run_kind"
        return
    fi
    mkdir -p "$run_root" "$trace_dir" "$(dirname "$log")" "$(dirname "$marker")"
    (
        scrub_experiment_env
        configure_trace "$run_kind" "$trace_dir"
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [[ "$run_kind" == sf_native_* ]]; then
            cd "$SF"
            python inference.py \
                --config_path "$SF_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPT" --output_folder "$run_root" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt \
                --start_idx 0 --end_idx 1
        elif [[ "$run_kind" == "pf_plain_sf21" ]]; then
            cd "$PF"
            python inference.py \
                --config_path "$PF_PLAIN_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPT" --output_folder "$run_root" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt \
                --start_idx 0 --end_idx 1 --model_local_attn_size 21
        elif [[ "$run_kind" == "adaptive_recent21" ]]; then
            cd "$PF"
            configure_adaptive
            export PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH="$PARITY_ROOT/cache_schedule.jsonl"
            python inference.py \
                --config_path "$PF_ADAPTIVE_CONFIG" --checkpoint_path "$CHECKPOINT" \
                --data_path "$PROMPT" --output_folder "$run_root" \
                --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
                --use_ema --save_with_index --reseed_per_prompt \
                --start_idx 0 --end_idx 1 --model_local_attn_size 21 \
                --pyramidkv_head_config_path "$HEAD_MAP" \
                --pyramidkv_history_polarity \
                --pyramidkv_history_support_policy retrieval \
                --pyramidkv_history_suppress_policy retrieval \
                --pyramidkv_cache_compatibility_denoise_schedule recent \
                --pyramidkv_cache_compatibility_denoise_coverage_policy retrieval \
                --pyramidkv_cache_compatibility_read_budget_frames 21 \
                --pyramidkv_semantic_retrieval_archive_capacity 12
        else
            echo "[error] unknown parity run $run_kind"
            exit 2
        fi
    ) >"$log" 2>&1
    [[ -s "$trace_dir/events.jsonl" ]] || { echo "[error] empty trace: $run_kind"; exit 1; }
    [[ -s "$run_root/0-0_ema.mp4" ]] || { echo "[error] missing video: $run_kind"; exit 1; }
    printf 'ok\n' >"$marker"
}

run_all() {
    preflight
    if [[ "${SF_PARITY_SINGLE_GPU:-1}" == "1" ]]; then
        # Numerical parity must not confound runtime changes with cross-GPU
        # kernel scheduling. Run every trajectory sequentially on the exact
        # same device; the jobs are intentionally short (9 latent frames).
        local gpu="${GPUS[0]}"
        run_one sf_native_a "$gpu"
        run_one sf_native_b "$gpu"
        run_one pf_plain_sf21 "$gpu"
        run_one adaptive_recent21 "$gpu"
        return
    fi
    local -a pids=()
    run_one sf_native_a "${GPUS[0]}" & pids+=("$!")
    run_one pf_plain_sf21 "${GPUS[1]}" & pids+=("$!")
    run_one adaptive_recent21 "${GPUS[2]}" & pids+=("$!")
    local pid failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "[error] first parity wave failed"; exit 1; }
    run_one sf_native_b "${GPUS[0]}"
}

status() {
    local run_kind
    for run_kind in sf_native_a sf_native_b pf_plain_sf21 adaptive_recent21; do
        local trace="$TRACE_ROOT/$run_kind/events.jsonl"
        local marker="$PARITY_ROOT/status/$run_kind.done"
        local events=0
        [[ -f "$trace" ]] && events="$(wc -l <"$trace")"
        printf '%s marker=%s events=%s\n' "$run_kind" "$([[ -s "$marker" ]] && echo yes || echo no)" "$events"
    done
}

analyze() {
    preflight
    local run_kind
    for run_kind in sf_native_a sf_native_b pf_plain_sf21 adaptive_recent21; do
        [[ -s "$PARITY_ROOT/status/$run_kind.done" ]] || {
            echo "[error] incomplete run: $run_kind"; exit 2;
        }
    done
    python "$ROOT/scripts/compare_v207_sf_parity.py" \
        --trace-root "$TRACE_ROOT" --output-dir "$REPORT_ROOT"
}

package() {
    [[ -s "$REPORT_ROOT/parity_report.json" ]] || { echo "[error] run analyze first"; exit 2; }
    local target="$PARITY_ROOT/v207_sf_runtime_parity.tar.gz"
    tar -C "$PARITY_ROOT" -czf "$target" \
        inputs report logs status cache_schedule.jsonl traces/*/trace_meta.json \
        traces/*/events.jsonl
    echo "$target"
}

case "$ACTION" in
    prepare) prepare ;;
    preflight) preflight ;;
    run) run_all ;;
    status) status ;;
    analyze) analyze ;;
    package) package ;;
esac
