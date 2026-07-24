#!/usr/bin/env bash
# Matched-seed v78 confirmation and weak-priority factorization on 16 H20s.
# Usage: bash scripts/run_v90_priority_factorization_16gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
PROMPTS="${PROMPTS:-$ROOT/prompts/v86_single_long_complex_16.txt}"
FRAMES="${FRAMES:-120}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v90_priority_factorization_screen}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROOT/runs/v86_role_transition_screen}"
CONTROL_DIR="${CONTROL_DIR:-$ROOT/runs/v82_probecache_control_labels}"
LEARNED_LABEL="${LEARNED_LABEL:-$CONTROL_DIR/learned.csv}"
INVERSE_LABEL="${INVERSE_LABEL:-$CONTROL_DIR/inverse.csv}"
RANDOM_LABEL="${RANDOM_LABEL:-$CONTROL_DIR/random_2026.csv}"
DERIVED_LABEL_DIR="${DERIVED_LABEL_DIR:-$OUT_ROOT/labels}"
PF_BINARY_LABEL="${PF_BINARY_LABEL:-$DERIVED_LABEL_DIR/pf_binary.csv}"
WAVE_ONLY_LABEL="${WAVE_ONLY_LABEL:-$DERIVED_LABEL_DIR/wave_only.csv}"
VEIL_ONLY_LABEL="${VEIL_ONLY_LABEL:-$DERIVED_LABEL_DIR/veil_only.csv}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v90 requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PF_LABELS" "$PROMPTS" \
    "$LEARNED_LABEL" "$INVERSE_LABEL" "$RANDOM_LABEL"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 16 ]] || {
    echo "[error] v90 expects 16 prompts, found $PROMPT_COUNT"
    exit 2
}
for method in pf v78 pf_binary_balanced learned_balanced; do
    path="$BASELINE_ROOT/$method"
    count=0
    if [[ -d "$path" ]]; then
        count="$(find "$path" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    fi
    [[ "$count" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] baseline $method has $count/$PROMPT_COUNT videos at $path"
        exit 2
    }
done

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0

mkdir -p \
    "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/configs" \
    "$OUT_ROOT/diagnostics" "$DERIVED_LABEL_DIR"
python "$ROOT/scripts/build_pf_transition_controls.py" \
    --pf-csv "$PF_LABELS" \
    --output-dir "$DERIVED_LABEL_DIR"

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v90_priority_factorization\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'BASELINE_ROOT=%s\n' "$BASELINE_ROOT"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$OUT_ROOT/run_manifest.env"

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

write_config() {
    local name="$1" method="$2" seed="$3" role_csv="$4"
    local p_scale="$5" r_scale="$6" p_age="$7" r_age="$8"
    local bias="$9"
    shift 9
    local layer_start="$1" layer_end="$2"
    local role_sha256=""
    if [[ -n "$role_csv" && -s "$role_csv" ]]; then
        role_sha256="$(sha256sum "$role_csv" | awk '{print $1}')"
    fi
    {
        printf 'name=%s\n' "$name"
        printf 'method=%s\n' "$method"
        printf 'seed=%s\n' "$seed"
        printf 'prompt_file=%s\n' "$PROMPTS"
        printf 'expected_videos=%s\n' "$PROMPT_COUNT"
        printf 'frames=%s\n' "$FRAMES"
        printf 'role_csv=%s\n' "$role_csv"
        printf 'role_sha256=%s\n' "$role_sha256"
        printf 'persistent_novelty_scale=%s\n' "$p_scale"
        printf 'reactive_novelty_scale=%s\n' "$r_scale"
        printf 'persistent_max_age=%s\n' "$p_age"
        printf 'reactive_max_age=%s\n' "$r_age"
        printf 'reactive_utility_bias=%s\n' "$bias"
        printf 'role_layer_start=%s\n' "$layer_start"
        printf 'role_layer_end=%s\n' "$layer_end"
    } >"$OUT_ROOT/configs/$name.env"
}

require_clean_or_skip() {
    local name="$1" output="$2" trace="${3:-}"
    local existing
    existing="$(video_count "$output")"
    if [[ "$FORCE" != "1" && "$existing" -eq "$PROMPT_COUNT" ]]; then
        if [[ -z "$trace" || -s "$trace" ]]; then
            echo "[skip] $name"
            return 10
        fi
    fi
    if [[ "$existing" -ne 0 ]]; then
        echo "[error] $name has $existing existing videos; use a clean OUT_ROOT"
        return 2
    fi
    return 0
}

run_pf() {
    local name="$1" gpu="$2" seed="$3"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    require_clean_or_skip "$name" "$output"
    local ready=$?
    [[ "$ready" -eq 10 ]] && return 0
    [[ "$ready" -eq 0 ]] || return "$ready"
    write_config "$name" pf "$seed" "" 1 1 6 6 0 0 -1
    mkdir -p "$output"
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] $name produced $(video_count "$output")/$PROMPT_COUNT videos"
        return 1
    }
}

# name gpu seed role_csv persistent_scale reactive_scale persistent_age
# reactive_age reactive_bias layer_start layer_end
run_transition() {
    local name="$1" gpu="$2" seed="$3" role_csv="$4"
    local p_scale="$5" r_scale="$6" p_age="$7" r_age="$8"
    local bias="$9"
    shift 9
    local layer_start="$1" layer_end="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.transition.jsonl"
    local method="v78_uniform"
    local role_args=()
    if [[ -n "$role_csv" ]]; then
        method="v90_weak_priority"
        role_args=(
            --pyramidkv_cache_transition_role_conditioning
            --pyramidkv_cache_transition_role_config_path "$role_csv"
            --pyramidkv_cache_transition_persistent_label 1
            --pyramidkv_cache_transition_reactive_labels=-1
            --pyramidkv_cache_transition_persistent_min_novelty_scale "$p_scale"
            --pyramidkv_cache_transition_reactive_min_novelty_scale "$r_scale"
            --pyramidkv_cache_transition_persistent_max_age_blocks "$p_age"
            --pyramidkv_cache_transition_reactive_max_age_blocks "$r_age"
            --pyramidkv_cache_transition_reactive_utility_bias "$bias"
            --pyramidkv_cache_transition_role_layer_start "$layer_start"
            --pyramidkv_cache_transition_role_layer_end "$layer_end"
        )
    fi
    require_clean_or_skip "$name" "$output" "$trace"
    local ready=$?
    [[ "$ready" -eq 10 ]] && return 0
    [[ "$ready" -eq 0 ]] || return "$ready"
    write_config \
        "$name" "$method" "$seed" "$role_csv" "$p_scale" "$r_scale" \
        "$p_age" "$r_age" "$bias" "$layer_start" "$layer_end"
    mkdir -p "$output"
    rm -f "$trace"
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_cache_transition \
            --pyramidkv_cache_transition_mode full \
            --pyramidkv_cache_transition_min_reliability .55 \
            --pyramidkv_cache_transition_min_novelty .01 \
            --pyramidkv_cache_transition_max_commit_fraction .75 \
            --pyramidkv_cache_transition_stagger_period 1 \
            --pyramidkv_cache_transition_max_age_blocks 6 \
            --pyramidkv_cache_transition_branches both \
            --pyramidkv_cache_transition_denoise_weight 2 \
            --pyramidkv_cache_transition_trace_path "$trace" \
            --pyramidkv_cache_transition_debug \
            "${role_args[@]}"
    ) >"$log" 2>&1
    [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] $name produced $(video_count "$output")/$PROMPT_COUNT videos"
        return 1
    }
    [[ -s "$trace" ]] || {
        echo "[error] missing transition trace $trace"
        return 1
    }
}

PIDS=()
STATUS=0
launch() {
    "$@" &
    PIDS+=("$!")
}

# Six matched-seed cells test the paper core. Ten seed-0 cells factorize the
# weak-priority hypothesis without changing PF's read topology.
launch run_pf pf_s1 "${GPUS[0]}" 1
launch run_transition v78_s1 "${GPUS[1]}" 1 "" 1 1 6 6 0 0 -1
launch run_pf pf_s2 "${GPUS[2]}" 2
launch run_transition v78_s2 "${GPUS[3]}" 2 "" 1 1 6 6 0 0 -1
launch run_pf pf_s3 "${GPUS[4]}" 3
launch run_transition v78_s3 "${GPUS[5]}" 3 "" 1 1 6 6 0 0 -1
launch run_transition pf_priority_b005 "${GPUS[6]}" 0 "$PF_BINARY_LABEL" 1 1 6 6 .05 0 -1
launch run_transition pf_priority_b010 "${GPUS[7]}" 0 "$PF_BINARY_LABEL" 1 1 6 6 .10 0 -1
launch run_transition learned_priority_b005 "${GPUS[8]}" 0 "$LEARNED_LABEL" 1 1 6 6 .05 0 -1
launch run_transition inverse_priority_b005 "${GPUS[9]}" 0 "$INVERSE_LABEL" 1 1 6 6 .05 0 -1
launch run_transition random_priority_b005 "${GPUS[10]}" 0 "$RANDOM_LABEL" 1 1 6 6 .05 0 -1
launch run_transition pf_age_only "${GPUS[11]}" 0 "$PF_BINARY_LABEL" 1 1 8 4 0 0 -1
launch run_transition pf_novelty_only "${GPUS[12]}" 0 "$PF_BINARY_LABEL" 1.5 .5 6 6 0 0 -1
launch run_transition wave_priority_b005 "${GPUS[13]}" 0 "$WAVE_ONLY_LABEL" 1 1 6 6 .05 0 -1
launch run_transition veil_priority_b005 "${GPUS[14]}" 0 "$VEIL_ONLY_LABEL" 1 1 6 6 .05 0 -1
launch run_transition pf_priority_late "${GPUS[15]}" 0 "$PF_BINARY_LABEL" 1 1 6 6 .05 15 30

echo "[v90] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

shopt -s nullglob
traces=("$OUT_ROOT"/traces/*.transition.jsonl)
if [[ "${#traces[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${traces[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
fi

echo "[v90] completed status=$STATUS"
exit "$STATUS"
