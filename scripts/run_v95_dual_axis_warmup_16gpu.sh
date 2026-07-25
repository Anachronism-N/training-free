#!/usr/bin/env bash
# MovieGenBench-32 causal screen for prompt-guided warmup and update priority.
# Usage: bash scripts/run_v95_dual_axis_warmup_16gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
PRIMARY_REPORT="${PRIMARY_REPORT:-$ROOT/runs/v81_probecache_profile/labels/probecache_profile_report.json}"
REPLICA_REPORT="${REPLICA_REPORT:-$ROOT/runs/v82_probecache_profile_replica/labels/probecache_profile_report.json}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v95_dual_axis_warmup32}"
LABEL_DIR="${LABEL_DIR:-$OUT_ROOT/labels}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"
PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-1}"

METHODS=(
    pf v78 prompt_priority_b005 prompt_priority_b010
    random_priority_b005 inverse_priority_b005 remote_priority_b005
    pfbinary_priority_b005 prompt_middle_w2 prompt_middle_w4
    prompt_history_w2 prompt_history_w4 prompt_history_w4_r6
    random_history_w4_r6 inverse_history_w4_r6 dual_axis_full
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v95 requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PF_LABELS" "$PROMPTS" \
    "$PRIMARY_REPORT" "$REPLICA_REPORT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 32 ]] || {
    echo "[error] v95 expects 32 prompts, found $PROMPT_COUNT"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p \
    "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/configs" \
    "$OUT_ROOT/status" "$OUT_ROOT/diagnostics" "$LABEL_DIR"

if [[ "$PRELOAD_PYRAMIDKV" == "1" ]]; then
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
        python -c "from pyramidkv import _ops; _ops._ensure_loaded(); print('[PyramidKVPreload] ok', flush=True)"
    ) >"$OUT_ROOT/logs/pyramidkv_preload.log" 2>&1 || exit 2
fi

python "$ROOT/scripts/build_prompt_contrastive_head_maps.py" \
    --profile-report "$PRIMARY_REPORT" \
    --replica-report "$REPLICA_REPORT" \
    --pf-csv "$PF_LABELS" \
    --output-dir "$LABEL_DIR" \
    --random-seed 2026 || exit 2

PF_BINARY="$LABEL_DIR/pf_binary.csv"
PROMPT_MAP="$LABEL_DIR/prompt_pfcount.csv"
RANDOM_MAP="$LABEL_DIR/prompt_random_pfcount.csv"
INVERSE_MAP="$LABEL_DIR/prompt_inverse_pfcount.csv"
REMOTE_MAP="$LABEL_DIR/remote_pfcount.csv"
for path in "$PF_BINARY" "$PROMPT_MAP" "$RANDOM_MAP" "$INVERSE_MAP" "$REMOTE_MAP"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
PF_SHA256="$(sha256sum "$PF_LABELS" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v95_dual_axis_warmup32\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'PF_LABELS=%s\n' "$PF_LABELS"
    printf 'PF_SHA256=%s\n' "$PF_SHA256"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$OUT_ROOT/run_manifest.env"

write_config() {
    local name="$1" transition="$2" role_csv="$3" role_bias="$4"
    local warm_mode="$5" warm_blocks="$6" warm_span="$7"
    local role_sha=""
    [[ -z "$role_csv" ]] || role_sha="$(sha256sum "$role_csv" | awk '{print $1}')"
    {
        printf 'name=%s\n' "$name"
        printf 'read_csv=%s\n' "$PF_LABELS"
        printf 'read_sha256=%s\n' "$PF_SHA256"
        printf 'transition=%s\n' "$transition"
        printf 'role_csv=%s\n' "$role_csv"
        printf 'role_sha256=%s\n' "$role_sha"
        printf 'role_bias=%s\n' "$role_bias"
        printf 'warm_mode=%s\n' "$warm_mode"
        printf 'warm_blocks=%s\n' "$warm_blocks"
        printf 'warm_release_span=%s\n' "$warm_span"
        printf 'reseed_per_prompt=1\n'
    } >"$OUT_ROOT/configs/$name.env"
}

run_cell() {
    local name="$1" gpu="$2" transition="$3" role_csv="$4" role_bias="$5"
    local warm_mode="$6" warm_blocks="$7" warm_span="$8"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local marker="$OUT_ROOT/status/$name.done"
    local transition_trace="$OUT_ROOT/traces/$name.transition.jsonl"
    local warm_trace="$OUT_ROOT/traces/$name.warmup.jsonl"
    local transition_args=() role_args=() warm_args=()

    mkdir -p "$output"
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        python "$ROOT/scripts/audit_indexed_videos.py" \
            --video-dir "$output" --start-idx 0 --end-idx 32 \
            >/dev/null 2>&1; then
        echo "[skip] $name"
        return 0
    fi
    rm -f "$marker" "$transition_trace" "$warm_trace"
    write_config \
        "$name" "$transition" "$role_csv" "$role_bias" \
        "$warm_mode" "$warm_blocks" "$warm_span"

    if [[ "$transition" == "1" ]]; then
        transition_args=(
            --pyramidkv_cache_transition
            --pyramidkv_cache_transition_mode full
            --pyramidkv_cache_transition_min_reliability .55
            --pyramidkv_cache_transition_min_novelty .01
            --pyramidkv_cache_transition_max_commit_fraction .75
            --pyramidkv_cache_transition_stagger_period 1
            --pyramidkv_cache_transition_max_age_blocks 6
            --pyramidkv_cache_transition_branches both
            --pyramidkv_cache_transition_denoise_weight 2
            --pyramidkv_cache_transition_trace_path "$transition_trace"
            --pyramidkv_cache_transition_debug
        )
    fi
    if [[ -n "$role_csv" ]]; then
        role_args=(--pyramidkv_cache_transition_role_config_path "$role_csv")
        if [[ "$transition" == "1" && "$role_bias" != "none" ]]; then
            role_args+=(
                --pyramidkv_cache_transition_role_conditioning
                --pyramidkv_cache_transition_persistent_label 1
                --pyramidkv_cache_transition_reactive_labels=-1
                --pyramidkv_cache_transition_persistent_min_novelty_scale 1
                --pyramidkv_cache_transition_reactive_min_novelty_scale 1
                --pyramidkv_cache_transition_persistent_max_age_blocks 6
                --pyramidkv_cache_transition_reactive_max_age_blocks 6
                --pyramidkv_cache_transition_reactive_utility_bias "$role_bias"
                --pyramidkv_cache_transition_role_layer_start 0
                --pyramidkv_cache_transition_role_layer_end -1
            )
        fi
    fi
    if [[ "$warm_mode" != "none" ]]; then
        warm_args=(
            --pyramidkv_prompt_warmup
            --pyramidkv_prompt_warmup_blocks "$warm_blocks"
            --pyramidkv_prompt_warmup_release_span "$warm_span"
            --pyramidkv_prompt_warmup_mode "$warm_mode"
            --pyramidkv_prompt_warmup_shield_labels=-1
            --pyramidkv_prompt_warmup_layer_start 0
            --pyramidkv_prompt_warmup_layer_end -1
            --pyramidkv_prompt_warmup_trace_path "$warm_trace"
            --pyramidkv_prompt_warmup_debug
        )
    fi

    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" \
            --num_samples 1 --use_ema --save_with_index \
            --start_idx 0 --end_idx 32 --reseed_per_prompt \
            --pyramidkv_head_config_path "$PF_LABELS" \
            "${transition_args[@]}" "${role_args[@]}" "${warm_args[@]}"
    ) >"$log" 2>&1

    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx 0 --end-idx 32 \
        --output-json "$OUT_ROOT/diagnostics/$name.audit.json" \
        >"$OUT_ROOT/diagnostics/$name.audit.log" 2>&1 || return 1
    [[ "$transition" != "1" || -s "$transition_trace" ]] || return 1
    [[ "$warm_mode" == "none" || -s "$warm_trace" ]] || return 1
    grep -q '\[PyramidKVHeadMap\]' "$log" || return 1
    if [[ "$warm_mode" != "none" ]]; then
        grep -q '\[PromptWarmupShield\]' "$log" || return 1
    fi
    printf 'ok\n' >"$marker"
}

PIDS=()
STATUS=0
launch() {
    run_cell "$@" &
    PIDS+=("$!")
}

launch pf                       "${GPUS[0]}"  0 ""            none none    0 0
launch v78                      "${GPUS[1]}"  1 ""            none none    0 0
launch prompt_priority_b005     "${GPUS[2]}"  1 "$PROMPT_MAP" .05  none    0 0
launch prompt_priority_b010     "${GPUS[3]}"  1 "$PROMPT_MAP" .10  none    0 0
launch random_priority_b005     "${GPUS[4]}"  1 "$RANDOM_MAP" .05  none    0 0
launch inverse_priority_b005    "${GPUS[5]}"  1 "$INVERSE_MAP" .05 none    0 0
launch remote_priority_b005     "${GPUS[6]}"  1 "$REMOTE_MAP" .05  none    0 0
launch pfbinary_priority_b005   "${GPUS[7]}"  1 "$PF_BINARY"  .05  none    0 0
launch prompt_middle_w2         "${GPUS[8]}"  0 "$PROMPT_MAP" none middle  2 0
launch prompt_middle_w4         "${GPUS[9]}"  0 "$PROMPT_MAP" none middle  4 0
launch prompt_history_w2        "${GPUS[10]}" 0 "$PROMPT_MAP" none history 2 0
launch prompt_history_w4        "${GPUS[11]}" 0 "$PROMPT_MAP" none history 4 0
launch prompt_history_w4_r6     "${GPUS[12]}" 0 "$PROMPT_MAP" none history 4 6
launch random_history_w4_r6     "${GPUS[13]}" 0 "$RANDOM_MAP" none history 4 6
launch inverse_history_w4_r6    "${GPUS[14]}" 0 "$INVERSE_MAP" none history 4 6
launch dual_axis_full           "${GPUS[15]}" 1 "$PROMPT_MAP" .05  history 4 6

echo "[v95] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

mapfile -t TRANSITION_TRACES < <(
    find "$OUT_ROOT/traces" -maxdepth 1 -type f -name '*.transition.jsonl' | sort
)
mapfile -t WARMUP_TRACES < <(
    find "$OUT_ROOT/traces" -maxdepth 1 -type f -name '*.warmup.jsonl' | sort
)
if [[ "${#TRANSITION_TRACES[@]}" -ne 8 ]]; then
    echo "[error] expected 8 transition traces, found ${#TRANSITION_TRACES[@]}"
    STATUS=1
else
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${TRANSITION_TRACES[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
fi
if [[ "${#WARMUP_TRACES[@]}" -ne 8 ]]; then
    echo "[error] expected 8 warmup traces, found ${#WARMUP_TRACES[@]}"
    STATUS=1
else
    python "$ROOT/scripts/summarize_prompt_warmup_trace.py" \
        "${WARMUP_TRACES[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/prompt_warmup_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/prompt_warmup_summary.md" \
        >"$OUT_ROOT/diagnostics/prompt_warmup_summary.log" 2>&1 || STATUS=1
fi

echo "[v95] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
