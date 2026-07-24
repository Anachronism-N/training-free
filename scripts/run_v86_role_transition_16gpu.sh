#!/usr/bin/env bash
# Functional-role-conditioned cache-transition experiments on one 16-H20 node.
# Usage: bash scripts/run_v86_role_transition_16gpu.sh smoke|screen|confirm|ultralong|switch
set -uo pipefail

MODE="${1:-screen}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
CONTROL_DIR="${CONTROL_DIR:-$ROOT/runs/v82_probecache_control_labels}"
REPLICA_PROFILE_ROOT="${REPLICA_PROFILE_ROOT:-$ROOT/runs/v82_probecache_profile_replica}"
LEARNED_LABEL="${LEARNED_LABEL:-$CONTROL_DIR/learned.csv}"
REPLICA_LABEL="${REPLICA_LABEL:-$REPLICA_PROFILE_ROOT/labels/probecache_binary_labels.csv}"
PF_BINARY_LABEL="${PF_BINARY_LABEL:-$CONTROL_DIR/pf_binary.csv}"
INVERSE_LABEL="${INVERSE_LABEL:-$CONTROL_DIR/inverse.csv}"
RANDOM_LABEL="${RANDOM_LABEL:-$CONTROL_DIR/random_2026.csv}"
SCREEN_PROMPTS="${SCREEN_PROMPTS:-$ROOT/prompts/probecache_v82_diagnostic_complex_3.txt}"
CONFIRM_PROMPTS="${CONFIRM_PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
ULTRALONG_PROMPTS="${ULTRALONG_PROMPTS:-$ROOT/prompts/probecache_v82_ultralong_complex_6.txt}"
SWITCH_PROMPTS="${SWITCH_PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-$ROOT/prompts/smoke_identity_motion.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v86_role_transition_${MODE}}"
CONSENSUS_LABEL="${CONSENSUS_LABEL:-$OUT_ROOT/labels/consensus.csv}"
CONSENSUS_REPORT="${CONSENSUS_REPORT:-$OUT_ROOT/labels/consensus.json}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

case "$MODE" in
    smoke)
        PROMPTS="$SMOKE_PROMPTS"
        FRAMES="${FRAMES:-12}"
        REQUIRED_GPUS=4
        ;;
    screen)
        PROMPTS="$SCREEN_PROMPTS"
        FRAMES="${FRAMES:-120}"
        REQUIRED_GPUS=16
        ;;
    confirm)
        PROMPTS="$CONFIRM_PROMPTS"
        FRAMES="${FRAMES:-120}"
        REQUIRED_GPUS=16
        ;;
    ultralong)
        PROMPTS="$ULTRALONG_PROMPTS"
        FRAMES="${FRAMES:-240}"
        REQUIRED_GPUS=8
        ;;
    switch)
        PROMPTS="$SWITCH_PROMPTS"
        FRAMES="${FRAMES:-120}"
        REQUIRED_GPUS=8
        ;;
    *)
        echo "[error] mode must be smoke, screen, confirm, ultralong, or switch"
        exit 2
        ;;
esac

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge "$REQUIRED_GPUS" ]] || {
    echo "[error] $MODE requires at least $REQUIRED_GPUS GPU ids"
    exit 2
}
for path in "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PROMPTS"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
if [[ "$MODE" == "smoke" ]]; then
    [[ -s "$LEARNED_LABEL" ]] || {
        echo "[error] missing role labels $LEARNED_LABEL"
        exit 2
    }
else
    for path in \
        "$LEARNED_LABEL" "$REPLICA_LABEL" "$PF_BINARY_LABEL" \
        "$INVERSE_LABEL" "$RANDOM_LABEL"; do
        [[ -s "$path" ]] || { echo "[error] missing role labels $path"; exit 2; }
    done
fi

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
if [[ "$MODE" != "smoke" ]]; then
    python "$ROOT/scripts/build_transition_role_consensus.py" \
        --primary "$LEARNED_LABEL" \
        --replica "$REPLICA_LABEL" \
        --output-csv "$CONSENSUS_LABEL" \
        --output-json "$CONSENSUS_REPORT"
fi

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -gt 0 ]] || { echo "[error] no prompts in $PROMPTS"; exit 2; }
case "$MODE" in
    screen|switch)
        [[ "$PROMPT_COUNT" -eq 3 ]] || {
            echo "[error] $MODE expects 3 prompts, found $PROMPT_COUNT"
            exit 2
        }
        ;;
    confirm)
        [[ "$PROMPT_COUNT" -eq 12 ]] || {
            echo "[error] confirm expects 12 prompts, found $PROMPT_COUNT"
            exit 2
        }
        ;;
    ultralong)
        [[ "$PROMPT_COUNT" -eq 6 ]] || {
            echo "[error] ultralong expects 6 prompts, found $PROMPT_COUNT"
            exit 2
        }
        ;;
esac
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/configs" "$OUT_ROOT/diagnostics"
{
    printf 'MODE=%s\n' "$MODE"
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
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
        printf 'prompt_file=%s\n' "$PROMPTS"
        printf 'expected_videos=%s\n' "$PROMPT_COUNT"
        printf 'frames=%s\n' "$FRAMES"
        printf 'seed=%s\n' "$seed"
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

run_pf() {
    local name="$1" gpu="$2" seed="$3"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    write_config "$name" pf "$seed" "" 1 1 6 6 0 0 -1
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return 0
    fi
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
    [[ "$(video_count "$output")" -ge "$PROMPT_COUNT" ]] || {
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
        method="v86_role_transition"
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
    write_config \
        "$name" "$method" "$seed" "$role_csv" "$p_scale" "$r_scale" \
        "$p_age" "$r_age" "$bias" "$layer_start" "$layer_end"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" && -s "$trace" ]]; then
        echo "[skip] $name"
        return 0
    fi
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
    [[ "$(video_count "$output")" -ge "$PROMPT_COUNT" ]] || {
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

launch_smoke() {
    launch run_pf pf_s0 "${GPUS[0]}" 0
    launch run_transition v78_s0 "${GPUS[1]}" 0 "" 1 1 6 6 0 0 -1
    launch run_transition learned_neutral_s0 "${GPUS[2]}" 0 "$LEARNED_LABEL" 1 1 6 6 0 0 -1
    launch run_transition learned_balanced_s0 "${GPUS[3]}" 0 "$LEARNED_LABEL" 1.5 .5 8 4 .1 0 -1
}

launch_screen() {
    launch run_pf pf "${GPUS[0]}" 0
    launch run_transition v78 "${GPUS[1]}" 0 "" 1 1 6 6 0 0 -1
    launch run_transition learned_neutral "${GPUS[2]}" 0 "$LEARNED_LABEL" 1 1 6 6 0 0 -1
    launch run_transition learned_balanced "${GPUS[3]}" 0 "$LEARNED_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition replica_balanced "${GPUS[4]}" 0 "$REPLICA_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition pf_binary_balanced "${GPUS[5]}" 0 "$PF_BINARY_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition inverse_balanced "${GPUS[6]}" 0 "$INVERSE_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition random_balanced "${GPUS[7]}" 0 "$RANDOM_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition learned_conservative "${GPUS[8]}" 0 "$LEARNED_LABEL" 2 .75 8 5 .05 0 -1
    launch run_transition learned_open "${GPUS[9]}" 0 "$LEARNED_LABEL" 1.25 .25 8 3 .15 0 -1
    launch run_transition learned_no_bias "${GPUS[10]}" 0 "$LEARNED_LABEL" 1.5 .5 8 4 0 0 -1
    launch run_transition learned_age_only "${GPUS[11]}" 0 "$LEARNED_LABEL" 1 1 8 4 0 0 -1
    launch run_transition consensus_balanced "${GPUS[12]}" 0 "$CONSENSUS_LABEL" 1.5 .5 8 4 .1 0 -1
    launch run_transition learned_early "${GPUS[13]}" 0 "$LEARNED_LABEL" 1.5 .5 8 4 .1 0 15
    launch run_transition learned_late "${GPUS[14]}" 0 "$LEARNED_LABEL" 1.5 .5 8 4 .1 15 30
    launch run_transition pf_binary_conservative "${GPUS[15]}" 0 "$PF_BINARY_LABEL" 2 .75 8 5 .05 0 -1
}

launch_confirm() {
    for seed in 0 1 2 3; do
        launch run_pf "pf_s$seed" "${GPUS[$seed]}" "$seed"
        launch run_transition "v78_s$seed" "${GPUS[$((seed + 4))]}" "$seed" "" 1 1 6 6 0 0 -1
        launch run_transition "learned_s$seed" "${GPUS[$((seed + 8))]}" "$seed" "$LEARNED_LABEL" 1.5 .5 8 4 .1 0 -1
        launch run_transition "pf_binary_s$seed" "${GPUS[$((seed + 12))]}" "$seed" "$PF_BINARY_LABEL" 1.5 .5 8 4 .1 0 -1
    done
}

launch_long_matrix() {
    for seed in 0 1; do
        local offset="$((seed * 4))"
        launch run_pf "pf_s$seed" "${GPUS[$offset]}" "$seed"
        launch run_transition "v78_s$seed" "${GPUS[$((offset + 1))]}" "$seed" "" 1 1 6 6 0 0 -1
        launch run_transition "learned_s$seed" "${GPUS[$((offset + 2))]}" "$seed" "$LEARNED_LABEL" 1.5 .5 8 4 .1 0 -1
        launch run_transition "pf_binary_s$seed" "${GPUS[$((offset + 3))]}" "$seed" "$PF_BINARY_LABEL" 1.5 .5 8 4 .1 0 -1
    done
}

echo "[v86] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
case "$MODE" in
    smoke) launch_smoke ;;
    screen) launch_screen ;;
    confirm) launch_confirm ;;
    ultralong|switch) launch_long_matrix ;;
esac
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

echo "[v86] completed status=$STATUS"
exit "$STATUS"
