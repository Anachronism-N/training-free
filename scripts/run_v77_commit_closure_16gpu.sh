#!/usr/bin/env bash
# Closure experiments requested by docs/77 for the best v74 FIFO/fresh branch.
# Usage: bash scripts/run_v77_commit_closure_16gpu.sh smoke|screen|baselines
set -uo pipefail

MODE="${1:-screen}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
ECHO="${ECHO_REPO:-$ROOT/third_party/Echo-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
DEFAULT_PROMPTS="$ROOT/prompts/lifecache_v3_single_long_complex_12.txt"
[[ "$MODE" != "smoke" ]] || DEFAULT_PROMPTS="$ROOT/prompts/smoke_identity_motion.txt"
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
FRAMES="${FRAMES:-120}"
[[ "$MODE" != "smoke" ]] || FRAMES="${SMOKE_FRAMES:-12}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v77_commit_closure_${MODE}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
case "$MODE" in
    smoke) required_gpus=4 ;;
    baselines) required_gpus=3 ;;
    screen) required_gpus=16 ;;
    *) echo "[error] mode must be smoke, screen, or baselines"; exit 2 ;;
esac
[[ "${#GPUS[@]}" -ge "$required_gpus" ]] || {
    echo "[error] $MODE requires at least $required_gpus GPU ids"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

for path in "$SF" "$PF" "$ECHO" "$SF_CONFIG" "$PF_CONFIG" "$ECHO_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$ECHO_CHECKPOINT" "$PROMPTS"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
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

run_baseline() {
    local kind="$1" name="$2" gpu="$3"
    local repo config checkpoint
    case "$kind" in
        sf) repo="$SF"; config="$SF_CONFIG"; checkpoint="$SF_CHECKPOINT" ;;
        pf) repo="$PF"; config="$PF_CONFIG"; checkpoint="$PF_CHECKPOINT" ;;
        echo) repo="$ECHO"; config="$ECHO_CONFIG"; checkpoint="$ECHO_CHECKPOINT" ;;
        *) echo "[error] unknown baseline $kind"; return 2 ;;
    esac
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$repo" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        clear_interventions
        [[ "$kind" != "echo" ]] || export ECHO_VERBOSE=1
        python inference.py \
            --config_path "$config" --checkpoint_path "$checkpoint" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

# name gpu timesteps trigger trigger_rel block_interval strength timestep_strengths ramp
run_commit() {
    local name="$1" gpu="$2" timesteps="$3" trigger="$4" trigger_rel="$5"
    local interval="$6" strength="$7" timestep_strengths="$8" ramp="$9"
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
        export COMMIT_FORCING_START_FRAME=12
        [[ "$MODE" != "smoke" ]] || export COMMIT_FORCING_START_FRAME=3
        export COMMIT_FORCING_TRIGGER_MODE="$trigger"
        export COMMIT_FORCING_TRIGGER_RELIABILITY="$trigger_rel"
        export COMMIT_FORCING_REFERENCE_MODE=hybrid
        export COMMIT_FORCING_REFERENCE_CAPACITY=4
        export COMMIT_FORCING_ORIGIN_CAPACITY=1
        export COMMIT_FORCING_ORIGIN_USE=1
        export COMMIT_FORCING_TRUSTED_USE=1
        export COMMIT_FORCING_TRUSTED_MIN_GAP=3
        export COMMIT_FORCING_ADMISSION_RELIABILITY=.30
        export COMMIT_FORCING_RELIABILITY_EMA_DECAY=.90
        export COMMIT_FORCING_BANK_MODE=fifo
        export COMMIT_FORCING_SUMMARY_CAPACITY=0
        export COMMIT_FORCING_SUMMARY_USE=0
        export COMMIT_FORCING_MOTION_GATE=0
        export COMMIT_FORCING_RENOISE_MODE=fresh
        export COMMIT_FORCING_BLOCK_INTERVAL="$interval"
        export COMMIT_FORCING_CORRECTION_STRENGTH="$strength"
        export COMMIT_FORCING_TIMESTEP_STRENGTHS="$timestep_strengths"
        export COMMIT_FORCING_RAMP_BLOCKS="$ramp"
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
    run_baseline sf sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_baseline pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_baseline echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
}

launch_smoke() {
    run_commit v74_default "${GPUS[0]}" 500,250 always .45 1 1 "" 0 & PIDS+=("$!")
    run_commit every2 "${GPUS[1]}" 500,250 always .45 2 1 "" 0 & PIDS+=("$!")
    run_commit timestep_smooth "${GPUS[2]}" 500,250 always .45 1 1 "500:1,250:.5" 0 & PIDS+=("$!")
    run_commit ramp4 "${GPUS[3]}" 500,250 always .45 1 1 "" 4 & PIDS+=("$!")
}

launch_screen() {
    run_baseline sf sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_baseline pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_baseline echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
    run_commit v74_default "${GPUS[3]}" 500,250 always .45 1 1 "" 0 & PIDS+=("$!")
    run_commit trigger_r035 "${GPUS[4]}" 500,250 unreliable .35 1 1 "" 0 & PIDS+=("$!")
    run_commit trigger_r045 "${GPUS[5]}" 500,250 unreliable .45 1 1 "" 0 & PIDS+=("$!")
    run_commit trigger_r055 "${GPUS[6]}" 500,250 unreliable .55 1 1 "" 0 & PIDS+=("$!")
    run_commit every2 "${GPUS[7]}" 500,250 always .45 2 1 "" 0 & PIDS+=("$!")
    run_commit every3 "${GPUS[8]}" 500,250 always .45 3 1 "" 0 & PIDS+=("$!")
    run_commit strength075 "${GPUS[9]}" 500,250 always .45 1 .75 "" 0 & PIDS+=("$!")
    run_commit strength050 "${GPUS[10]}" 500,250 always .45 1 .50 "" 0 & PIDS+=("$!")
    run_commit t500_only "${GPUS[11]}" 500 always .45 1 1 "" 0 & PIDS+=("$!")
    run_commit t500_100_t250_075 "${GPUS[12]}" 500,250 always .45 1 1 "500:1,250:.75" 0 & PIDS+=("$!")
    run_commit t500_100_t250_050 "${GPUS[13]}" 500,250 always .45 1 1 "500:1,250:.5" 0 & PIDS+=("$!")
    run_commit ramp4 "${GPUS[14]}" 500,250 always .45 1 1 "" 4 & PIDS+=("$!")
    run_commit every2_strength075 "${GPUS[15]}" 500,250 always .45 2 .75 "" 0 & PIDS+=("$!")
}

echo "[v77] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
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

echo "[v77] completed status=$BATCH_STATUS"
exit "$BATCH_STATUS"
