#!/usr/bin/env bash
# PF cache-transition screen for a 16-GPU server.
# Usage: bash scripts/run_v78_cache_transition_16gpu.sh smoke|screen|baselines
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
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v78_cache_transition_${MODE}}"
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
    echo "[error] $MODE requires at least $required_gpus GPU ids"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

for path in "$SF" "$PF" "$ECHO" "$SF_CONFIG" "$PF_CONFIG" "$ECHO_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$ECHO_CHECKPOINT" "$PROMPTS"; do
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

prepare_pf_environment() {
    # Audit and controlled cells must use the same Python strategy path.
    export PYRAMIDKV_USE_CPP_STRATEGY=0
    export PYRAMIDKV_USE_CPP_PACK=0
    export PYRAMIDKV_USE_MEGA_CACHE=0
}

run_sf() {
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
        export COMMIT_FORCING_ENABLE=0
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_echo() {
    local name="$1" gpu="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$ECHO" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ECHO_VERBOSE=1
        python inference.py \
            --config_path "$ECHO_CONFIG" --checkpoint_path "$ECHO_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_pf() {
    local name="$1" gpu="$2"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        gpu_snapshot "$gpu"
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        prepare_pf_environment
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

# name gpu mode min_rel min_novelty fraction period max_age branches denoise_weight
run_transition() {
    local name="$1" gpu="$2" transition_mode="$3" min_rel="$4"
    local min_novelty="$5" fraction="$6" period="$7" max_age="$8"
    local branches="$9"
    shift 9
    local denoise_weight="$1"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" && -s "$trace" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    rm -f "$trace"
    {
        printf 'mode=%s\n' "$transition_mode"
        printf 'min_reliability=%s\n' "$min_rel"
        printf 'min_novelty=%s\n' "$min_novelty"
        printf 'max_commit_fraction=%s\n' "$fraction"
        printf 'stagger_period=%s\n' "$period"
        printf 'max_age_blocks=%s\n' "$max_age"
        printf 'branches=%s\n' "$branches"
        printf 'denoise_weight=%s\n' "$denoise_weight"
    } >"$output/run_config.env"
    (
        gpu_snapshot "$gpu"
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        prepare_pf_environment
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_cache_transition \
            --pyramidkv_cache_transition_mode "$transition_mode" \
            --pyramidkv_cache_transition_min_reliability "$min_rel" \
            --pyramidkv_cache_transition_min_novelty "$min_novelty" \
            --pyramidkv_cache_transition_max_commit_fraction "$fraction" \
            --pyramidkv_cache_transition_stagger_period "$period" \
            --pyramidkv_cache_transition_max_age_blocks "$max_age" \
            --pyramidkv_cache_transition_branches "$branches" \
            --pyramidkv_cache_transition_denoise_weight "$denoise_weight" \
            --pyramidkv_cache_transition_trace_path "$trace" \
            --pyramidkv_cache_transition_debug
    ) >"$log" 2>&1
}

launch_baselines() {
    run_sf sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
}

launch_smoke() {
    run_pf pf_official "${GPUS[0]}" & PIDS+=("$!")
    run_transition pf_transition_audit "${GPUS[1]}" audit .55 0 1 1 6 both 2 &
    PIDS+=("$!")
    run_transition pf_transition_gate "${GPUS[2]}" gate .55 .01 1 1 6 both 2 &
    PIDS+=("$!")
    run_transition pf_transition_full "${GPUS[3]}" full .55 .01 .5 2 6 both 2 &
    PIDS+=("$!")
}

launch_screen() {
    run_sf sf_native "${GPUS[0]}" & PIDS+=("$!")
    run_pf pf_official "${GPUS[1]}" & PIDS+=("$!")
    run_echo echo_pc "${GPUS[2]}" & PIDS+=("$!")
    run_transition pf_audit "${GPUS[3]}" audit .55 0 1 1 6 both 2 & PIDS+=("$!")
    run_transition gate_r045 "${GPUS[4]}" gate .45 0 1 1 6 both 2 & PIDS+=("$!")
    run_transition gate_r055 "${GPUS[5]}" gate .55 0 1 1 6 both 2 & PIDS+=("$!")
    run_transition gate_r065 "${GPUS[6]}" gate .65 0 1 1 6 both 2 & PIDS+=("$!")
    run_transition gate_r055_n001 "${GPUS[7]}" gate .55 .01 1 1 6 both 2 & PIDS+=("$!")
    run_transition stagger_half "${GPUS[8]}" stagger 0 0 .5 2 6 both 0 & PIDS+=("$!")
    run_transition stagger_third "${GPUS[9]}" stagger 0 0 .34 3 6 both 0 & PIDS+=("$!")
    run_transition full_r045 "${GPUS[10]}" full .45 .01 .5 2 6 both 2 & PIDS+=("$!")
    run_transition full_r055 "${GPUS[11]}" full .55 .01 .5 2 6 both 2 & PIDS+=("$!")
    run_transition full_r065 "${GPUS[12]}" full .65 .01 .5 2 6 both 2 & PIDS+=("$!")
    run_transition full_age4 "${GPUS[13]}" full .55 .01 .5 2 4 both 2 & PIDS+=("$!")
    run_transition full_cond "${GPUS[14]}" full .55 .01 .5 2 6 cond 2 & PIDS+=("$!")
    run_transition full_budget075_p1 "${GPUS[15]}" full .55 .01 .75 1 6 both 2 & PIDS+=("$!")
}

echo "[v78] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
"launch_$MODE"
for pid in "${PIDS[@]}"; do
    wait "$pid" || BATCH_STATUS=1
done

shopt -s nullglob
traces=("$OUT_ROOT"/traces/*.jsonl)
if [[ "${#traces[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${traces[@]}" --strict \
        --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
        --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
        >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || BATCH_STATUS=1
fi

echo "[v78] completed status=$BATCH_STATUS"
exit "$BATCH_STATUS"
