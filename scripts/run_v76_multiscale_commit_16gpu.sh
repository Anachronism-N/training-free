#!/usr/bin/env bash
# Commit Forcing v2 multiscale-bank screen for a 16-GPU server.
# Usage: bash scripts/run_v76_multiscale_commit_16gpu.sh smoke|screen|baselines
set -uo pipefail

MODE="${1:-screen}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
ECHO="${ECHO_REPO:-$ROOT/third_party/Echo-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
DEFAULT_PROMPTS="$ROOT/prompts/lifecache_v3_single_long_complex_12.txt"
[[ "$MODE" != "smoke" ]] || DEFAULT_PROMPTS="$ROOT/prompts/smoke_identity_motion.txt"
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
FRAMES="${FRAMES:-120}"
[[ "$MODE" != "smoke" ]] || FRAMES="${SMOKE_FRAMES:-12}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v76_multiscale_commit_${MODE}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
case "$MODE" in
    smoke) required_gpus=4 ;;
    baselines) required_gpus=3 ;;
    screen) required_gpus=16 ;;
    *)
        echo "[error] mode must be smoke, screen, or baselines"
        exit 2
        ;;
esac
[[ "${#GPUS[@]}" -ge "$required_gpus" ]] || {
    echo "[error] $MODE requires at least $required_gpus visible GPU ids"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

for path in "$SF" "$SF_CONFIG" "$SF_CHECKPOINT" "$PROMPTS"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -gt 0 ]] || { echo "[error] no prompts in $PROMPTS"; exit 2; }
if [[ "$MODE" == "screen" && "$PROMPT_COUNT" -ne 12 ]]; then
    echo "[error] screen requires exactly 12 prompts, found $PROMPT_COUNT"
    exit 2
fi

PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/diagnostics"
{
    printf 'MODE=%s\n' "$MODE"
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
} >"$OUT_ROOT/run_manifest.env"

PIDS=()
BATCH_STATUS=0

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

gpu_snapshot() {
    local gpu="$1"
    echo "[gpu-before] requested_device=$gpu"
    nvidia-smi -i "$gpu" \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader 2>&1 || true
}

clear_interventions() {
    export LIFECACHE_ENABLE=0
    export STRUCTURED_MEMORY_ENABLE=0
    export STRUCTURED_MEMORY_TRACE_ENABLED=0
    export HEAD_ROLE_ENABLE=0
    export HEAD_ROLE_POOL_ENABLE=0
    export SCENE_TRANSITION_RESET=0
    export SF_FULL_ATTN_MAX_FRAMES=""
    export COMMIT_FORCING_ENABLE=0
}

run_native() {
    local name="$1" gpu="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$SF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        clear_interventions
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_pf() {
    local name="$1" gpu="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    for path in "$PF" "$PF_CONFIG" "$PF_CHECKPOINT"; do
        [[ -e "$path" ]] || { echo "[error] missing PF input $path"; return 2; }
    done
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        clear_interventions
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_echo() {
    local name="$1" gpu="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    for path in "$ECHO" "$ECHO_CONFIG" "$ECHO_CHECKPOINT"; do
        [[ -e "$path" ]] || { echo "[error] missing Echo input $path"; return 2; }
    done
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$ECHO" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        clear_interventions
        export ECHO_VERBOSE=1
        python inference.py \
            --config_path "$ECHO_CONFIG" --checkpoint_path "$ECHO_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

# Args after gpu:
# timesteps bank cap origin_cap origin_use recent_use summary_cap summary_use
# merge motion_gate motion_high renoise trigger trigger_rel admission
run_commit() {
    local name="$1" gpu="$2" timesteps="$3" bank="$4" capacity="$5"
    local origin_capacity="$6" origin_use="$7" recent_use="$8"
    local summary_capacity="$9"
    shift 9
    local summary_use="$1" merge="$2" motion_gate="$3" motion_high="$4"
    local renoise="$5" trigger="$6" trigger_rel="$7" admission="$8"
    local start_frame=12
    [[ "$MODE" != "smoke" ]] || start_frame=3
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" && -s "$trace" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    rm -f "$trace"
    (
        gpu_snapshot "$gpu"
        cd "$SF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        clear_interventions
        export COMMIT_FORCING_ENABLE=1
        export COMMIT_FORCING_TIMESTEPS="$timesteps"
        export COMMIT_FORCING_START_FRAME="$start_frame"
        export COMMIT_FORCING_TRIGGER_MODE="$trigger"
        export COMMIT_FORCING_TRIGGER_RELIABILITY="$trigger_rel"
        export COMMIT_FORCING_REFERENCE_MODE=hybrid
        export COMMIT_FORCING_REFERENCE_CAPACITY="$capacity"
        export COMMIT_FORCING_ORIGIN_CAPACITY="$origin_capacity"
        export COMMIT_FORCING_ORIGIN_USE="$origin_use"
        export COMMIT_FORCING_TRUSTED_USE="$recent_use"
        export COMMIT_FORCING_TRUSTED_MIN_GAP=3
        export COMMIT_FORCING_ADMISSION_RELIABILITY="$admission"
        export COMMIT_FORCING_RELIABILITY_EMA_DECAY=.90
        export COMMIT_FORCING_BANK_MODE="$bank"
        export COMMIT_FORCING_SUMMARY_CAPACITY="$summary_capacity"
        export COMMIT_FORCING_SUMMARY_USE="$summary_use"
        export COMMIT_FORCING_SUMMARY_MERGE_MODE="$merge"
        export COMMIT_FORCING_SUMMARY_RELIABILITY_POWER=2
        export COMMIT_FORCING_MERGE_MOTION_TOLERANCE=.75
        export COMMIT_FORCING_MOTION_GATE="$motion_gate"
        export COMMIT_FORCING_MOTION_HIGH_RATIO="$motion_high"
        export COMMIT_FORCING_MOTION_EMA_DECAY=.90
        export COMMIT_FORCING_RENOISE_MODE="$renoise"
        export COMMIT_FORCING_SEED=91021
        export COMMIT_FORCING_TRACE_PATH="$trace"
        export COMMIT_FORCING_DEBUG=1
        env | grep '^COMMIT_FORCING_' | sort >"$output/run_config.env"
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

launch_baselines() {
    run_native sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
}

launch_smoke() {
    run_native sf_native_smoke "${GPUS[0]}" & PIDS+=("$!")
    run_commit v74_fresh_smoke "${GPUS[1]}" \
        500,250 fifo 4 1 1 1 0 0 adaptive 0 1.35 fresh always .45 .30 &
    PIDS+=("$!")
    run_commit fifo_trajectory_smoke "${GPUS[2]}" \
        500,250 fifo 4 1 1 1 0 0 adaptive 0 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit multiscale_full_smoke "${GPUS[3]}" \
        500,250 multiscale 6 1 1 1 2 1 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
}

launch_screen() {
    run_native sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
    run_commit v74_hybrid_fresh "${GPUS[3]}" \
        500,250 fifo 4 1 1 1 0 0 adaptive 0 1.35 fresh always .45 .30 &
    PIDS+=("$!")
    run_commit v74_origin2_fresh "${GPUS[4]}" \
        500,250 fifo 4 2 2 1 0 0 adaptive 0 1.35 fresh always .45 .30 &
    PIDS+=("$!")
    run_commit fifo_hybrid_trajectory "${GPUS[5]}" \
        500,250 fifo 4 1 1 1 0 0 adaptive 0 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit fifo_origin2_trajectory "${GPUS[6]}" \
        500,250 fifo 4 2 2 1 0 0 adaptive 0 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_fresh_nomotion "${GPUS[7]}" \
        500,250 multiscale 6 1 1 1 2 1 adaptive 0 1.35 fresh always .45 .30 &
    PIDS+=("$!")
    run_commit ms_trajectory_nomotion "${GPUS[8]}" \
        500,250 multiscale 6 1 1 1 2 1 adaptive 0 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_full_motion "${GPUS[9]}" \
        500,250 multiscale 6 1 1 1 2 1 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_origin2_motion "${GPUS[10]}" \
        500,250 multiscale 6 2 2 1 2 1 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_summary2_motion "${GPUS[11]}" \
        500,250 multiscale 7 1 1 1 3 2 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_representative_motion "${GPUS[12]}" \
        500,250 multiscale 6 1 1 1 2 1 representative 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_mean_motion "${GPUS[13]}" \
        500,250 multiscale 6 1 1 1 2 1 mean 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_no_summary_read "${GPUS[14]}" \
        500,250 multiscale 6 1 1 1 2 0 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
    run_commit ms_t250_motion "${GPUS[15]}" \
        250 multiscale 6 1 1 1 2 1 adaptive 1 1.35 trajectory always .45 .30 &
    PIDS+=("$!")
}

echo "[v76] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT" \
    "frames=$FRAMES out=$OUT_ROOT"
"launch_$MODE"
for pid in "${PIDS[@]}"; do
    wait "$pid" || BATCH_STATUS=1
done

shopt -s nullglob
traces=("$OUT_ROOT"/traces/*.jsonl)
if [[ "${#traces[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_commit_forcing_trace.py" \
        "${traces[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/commit_trace_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/commit_trace_summary.md" \
        >"$OUT_ROOT/diagnostics/commit_trace_summary.log" 2>&1 || BATCH_STATUS=1
fi

echo "[v76] completed status=$BATCH_STATUS"
exit "$BATCH_STATUS"
