#!/usr/bin/env bash
# Commit Forcing screen and confirmation matrix for a 16-GPU server.
# Usage: bash scripts/run_v74_commit_forcing_16gpu.sh screen|confirm|smoke
set -uo pipefail

MODE="${1:-screen}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
DEFAULT_PROMPTS="$ROOT/prompts/lifecache_v3_single_long_complex_12.txt"
[[ "$MODE" != "smoke" ]] || DEFAULT_PROMPTS="$ROOT/prompts/smoke_identity_motion.txt"
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
FRAMES="${FRAMES:-120}"
[[ "$MODE" != "smoke" ]] || FRAMES="${SMOKE_FRAMES:-12}"
SEED="${SEED:-0}"
DEFAULT_OUT_ROOT="$ROOT/runs/v74_commit_${MODE}_12p_30s"
[[ "$MODE" != "smoke" ]] || DEFAULT_OUT_ROOT="$ROOT/runs/v74_commit_smoke"
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
required_gpus=16
[[ "$MODE" == "smoke" ]] && required_gpus=3
[[ "${#GPUS[@]}" -ge "$required_gpus" ]] || {
    echo "[error] $MODE requires at least $required_gpus visible GPU ids"
    exit 2
}
[[ "$MODE" =~ ^(screen|confirm|smoke)$ ]] || {
    echo "[error] mode must be screen, confirm, or smoke"
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
if [[ "$MODE" != "smoke" && "$PROMPT_COUNT" -ne 12 ]]; then
    echo "[error] $MODE requires exactly 12 prompts, found $PROMPT_COUNT"
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
    local name="$1" gpu="$2" seed="$3"
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
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_commit() {
    local name="$1" gpu="$2" seed="$3" timesteps="$4" reference_mode="$5"
    local capacity="$6" origin_capacity="$7" origin_use="$8" trusted_use="$9"
    shift 9
    local start_frame="$1" trigger_mode="$2" trigger_reliability="$3"
    local admission_reliability="$4"
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
        export COMMIT_FORCING_TRIGGER_MODE="$trigger_mode"
        export COMMIT_FORCING_TRIGGER_RELIABILITY="$trigger_reliability"
        export COMMIT_FORCING_REFERENCE_MODE="$reference_mode"
        export COMMIT_FORCING_REFERENCE_CAPACITY="$capacity"
        export COMMIT_FORCING_ORIGIN_CAPACITY="$origin_capacity"
        export COMMIT_FORCING_ORIGIN_USE="$origin_use"
        export COMMIT_FORCING_TRUSTED_USE="$trusted_use"
        export COMMIT_FORCING_TRUSTED_MIN_GAP=3
        export COMMIT_FORCING_ADMISSION_RELIABILITY="$admission_reliability"
        export COMMIT_FORCING_RELIABILITY_EMA_DECAY=0.90
        export COMMIT_FORCING_SEED=91021
        export COMMIT_FORCING_TRACE_PATH="$trace"
        export COMMIT_FORCING_DEBUG=1
        env | grep '^COMMIT_FORCING_' | sort >"$output/run_config.env"
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_pf() {
    local name="$1" gpu="$2" seed="$3"
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
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

launch_screen() {
    run_native sf_native "${GPUS[0]}" "$SEED" & PIDS+=("$!")
    run_commit origin_t500 "${GPUS[1]}" "$SEED" \
        500 origin 1 1 1 0 12 always .45 .30 & PIDS+=("$!")
    run_commit origin_t250 "${GPUS[2]}" "$SEED" \
        250 origin 1 1 1 0 12 always .45 .30 & PIDS+=("$!")
    run_commit origin_t500_250 "${GPUS[3]}" "$SEED" \
        500,250 origin 1 1 1 0 12 always .45 .30 & PIDS+=("$!")
    run_commit origin_t750_500_250 "${GPUS[4]}" "$SEED" \
        750,500,250 origin 1 1 1 0 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_t500 "${GPUS[5]}" "$SEED" \
        500 hybrid 4 1 1 1 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_t250 "${GPUS[6]}" "$SEED" \
        250 hybrid 4 1 1 1 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_t500_250 "${GPUS[7]}" "$SEED" \
        500,250 hybrid 4 1 1 1 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_t750_500_250 "${GPUS[8]}" "$SEED" \
        750,500,250 hybrid 4 1 1 1 12 always .45 .30 & PIDS+=("$!")
    run_commit trusted_t500_250 "${GPUS[9]}" "$SEED" \
        500,250 trusted 4 0 0 1 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_origin2 "${GPUS[10]}" "$SEED" \
        500,250 hybrid 4 2 2 1 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_trusted2 "${GPUS[11]}" "$SEED" \
        500,250 hybrid 4 1 1 2 12 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_start21 "${GPUS[12]}" "$SEED" \
        500,250 hybrid 4 1 1 1 21 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_admit045 "${GPUS[13]}" "$SEED" \
        500,250 hybrid 4 1 1 1 12 always .45 .45 & PIDS+=("$!")
    run_commit hybrid_admit015 "${GPUS[14]}" "$SEED" \
        500,250 hybrid 4 1 1 1 12 always .45 .15 & PIDS+=("$!")
    run_commit hybrid_unreliable045 "${GPUS[15]}" "$SEED" \
        500,250 hybrid 4 1 1 1 12 unreliable .45 .30 & PIDS+=("$!")
}

launch_confirm() {
    local index=0 seed
    for seed in 0 1 2 3; do
        run_native "sf_native_s${seed}" "${GPUS[$index]}" "$seed" &
        PIDS+=("$!"); index=$((index + 1))
        run_pf "pf_s${seed}" "${GPUS[$index]}" "$seed" &
        PIDS+=("$!"); index=$((index + 1))
        run_commit "origin_t500_250_s${seed}" "${GPUS[$index]}" "$seed" \
            500,250 origin 1 1 1 0 12 always .45 .30 & PIDS+=("$!"); index=$((index + 1))
        run_commit "hybrid_t500_250_s${seed}" "${GPUS[$index]}" "$seed" \
            500,250 hybrid 4 1 1 1 12 always .45 .30 & PIDS+=("$!"); index=$((index + 1))
    done
}

launch_smoke() {
    run_native sf_native_smoke "${GPUS[0]}" "$SEED" & PIDS+=("$!")
    run_commit origin_smoke "${GPUS[1]}" "$SEED" \
        500,250 origin 1 1 1 0 3 always .45 .30 & PIDS+=("$!")
    run_commit hybrid_smoke "${GPUS[2]}" "$SEED" \
        500,250 hybrid 4 1 1 1 3 always .45 .30 & PIDS+=("$!")
}

echo "[v74] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT" \
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

echo "[v74] completed status=$BATCH_STATUS"
exit "$BATCH_STATUS"
