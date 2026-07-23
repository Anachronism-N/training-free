#!/usr/bin/env bash
# ProbeCache screen: 16 cells in parallel for single-prompt or scene-switch tasks.
# Usage: bash scripts/run_v81_probecache_16gpu.sh single|switch|smoke
set -euo pipefail

TASK="${1:-single}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
ECHO="${ECHO_REPO:-$ROOT/third_party/Echo-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
HEAD_CSV="${HEAD_CSV:-$ROOT/runs/v81_probecache_profile/labels/probecache_binary_labels.csv}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

case "$TASK" in
    single) PROMPTS="${PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}" ;;
    switch) PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}" ;;
    smoke)
        PROMPTS="${PROMPTS:-$ROOT/prompts/smoke_identity_motion.txt}"
        FRAMES="${SMOKE_FRAMES:-12}"
        ;;
    *) echo "[error] task must be single, switch, or smoke"; exit 2 ;;
esac
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v81_probecache_${TASK}}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 16 ]] || { echo "[error] 16 GPU ids required"; exit 2; }
for path in "$SF" "$PF" "$ECHO" "$SF_CONFIG" "$PF_CONFIG" "$ECHO_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$ECHO_CHECKPOINT" "$HEAD_CSV" "$PROMPTS"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -gt 0 ]] || { echo "[error] empty prompt file"; exit 2; }

mkdir -p "$OUT_ROOT"/{logs,traces,configs}
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0

video_count() {
    local path="$1"
    [[ -d "$path" ]] || { printf '0'; return; }
    find "$path" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

run_sf() {
    local name="$1" gpu="$2" output="$OUT_ROOT/$1" log="$OUT_ROOT/logs/$1.log"
    [[ "$FORCE" == "1" || "$(video_count "$output")" -lt "$PROMPT_COUNT" ]] || return
    mkdir -p "$output"
    (
        cd "$SF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_pf() {
    local name="$1" gpu="$2" output="$OUT_ROOT/$1" log="$OUT_ROOT/logs/$1.log"
    [[ "$FORCE" == "1" || "$(video_count "$output")" -lt "$PROMPT_COUNT" ]] || return
    mkdir -p "$output"
    (
        source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-longlive}"
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        python inference.py --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_echo() {
    local name="$1" gpu="$2" output="$OUT_ROOT/$1" log="$OUT_ROOT/logs/$1.log"
    [[ "$FORCE" == "1" || "$(video_count "$output")" -lt "$PROMPT_COUNT" ]] || return
    mkdir -p "$output"
    (
        cd "$ECHO" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ECHO_VERBOSE=1
        python inference.py --config_path "$ECHO_CONFIG" --checkpoint_path "$ECHO_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

# name gpu mode archive topk prompt_weight trust extra...
run_ours() {
    local name="$1" gpu="$2" mode="$3" archive="$4" topk="$5"
    local prompt_weight="$6" trust="$7"
    shift 7
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.probecache.jsonl"
    local transition_trace="$OUT_ROOT/traces/$name.transition.jsonl"
    [[ "$FORCE" == "1" || "$(video_count "$output")" -lt "$PROMPT_COUNT" || ! -s "$trace" ]] || return
    mkdir -p "$output"
    rm -f "$trace" "$transition_trace"
    local transition_args=()
    if [[ "$trust" == "1" ]]; then
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
        )
    fi
    {
        printf 'task=%s\nmode=%s\narchive=%s\ntopk=%s\nprompt_weight=%s\ntrust=%s\n' \
            "$TASK" "$mode" "$archive" "$topk" "$prompt_weight" "$trust"
    } >"$OUT_ROOT/configs/$name.env"
    (
        source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-longlive}"
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        python inference.py --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_head_config_path "$HEAD_CSV" \
            --pyramidkv_probecache --pyramidkv_probecache_mode "$mode" \
            --pyramidkv_probecache_archive_max_frames "$archive" \
            --pyramidkv_probecache_persistent_top_k "$topk" \
            --pyramidkv_probecache_reactive_top_k "$topk" \
            --pyramidkv_probecache_prompt_weight "$prompt_weight" \
            --pyramidkv_probecache_trace_path "$trace" \
            --pyramidkv_probecache_debug \
            "${transition_args[@]}" "$@"
    ) >"$log" 2>&1
}

pids=()
run_sf sf_native "${GPUS[0]}" & pids+=("$!")
run_pf pf_official "${GPUS[1]}" & pids+=("$!")
run_echo echo_pc "${GPUS[2]}" & pids+=("$!")
run_ours ours_audit "${GPUS[3]}" audit 24 4 .15 1 & pids+=("$!")
run_ours ours_persistent "${GPUS[4]}" persistent 24 4 .15 1 & pids+=("$!")
run_ours ours_reactive "${GPUS[5]}" reactive 24 4 .15 1 & pids+=("$!")
run_ours ours_full "${GPUS[6]}" full 24 4 .15 1 & pids+=("$!")
run_ours ours_no_trust "${GPUS[7]}" full 24 4 .15 0 & pids+=("$!")
run_ours ours_archive12 "${GPUS[8]}" full 12 4 .15 1 & pids+=("$!")
run_ours ours_archive36 "${GPUS[9]}" full 36 4 .15 1 & pids+=("$!")
run_ours ours_topk2 "${GPUS[10]}" full 24 2 .15 1 & pids+=("$!")
run_ours ours_topk6 "${GPUS[11]}" full 24 6 .15 1 & pids+=("$!")
run_ours ours_prompt0 "${GPUS[12]}" full 24 4 0 1 & pids+=("$!")
run_ours ours_prompt30 "${GPUS[13]}" full 24 4 .30 1 & pids+=("$!")
run_ours ours_open_gate "${GPUS[14]}" full 24 4 .15 1 \
    --pyramidkv_probecache_min_similarity -1 \
    --pyramidkv_probecache_min_margin 0 \
    --pyramidkv_probecache_max_entropy 1 & pids+=("$!")
run_ours ours_conservative "${GPUS[15]}" full 24 4 .15 1 \
    --pyramidkv_probecache_min_similarity .20 \
    --pyramidkv_probecache_min_margin .05 \
    --pyramidkv_probecache_max_entropy .80 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
    echo "[error] one or more cells failed; inspect $OUT_ROOT/logs"
    exit 1
fi
python "$ROOT/scripts/summarize_probecache_trace.py" \
    "$OUT_ROOT"/traces/*.probecache.jsonl \
    --strict \
    --output-json "$OUT_ROOT/probecache_trace_summary.json" \
    --output-md "$OUT_ROOT/probecache_trace_summary.md"
echo "[v81] completed task=$TASK prompts=$PROMPT_COUNT frames=$FRAMES out=$OUT_ROOT"
