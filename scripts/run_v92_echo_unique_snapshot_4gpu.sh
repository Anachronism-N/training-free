#!/usr/bin/env bash
# Echo-Forcing coherent uniqueness snapshot factorization.
# Usage: bash scripts/run_v92_echo_unique_snapshot_4gpu.sh
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
ECHO="${ECHO_REPO:-$ROOT/third_party/Echo-Forcing}"
CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/paper_scene_switch_echo_3.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v92_echo_unique_snapshot}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
FORCE="${FORCE:-0}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 4 ]] || {
    echo "[error] Echo uniqueness screen requires exactly 4 GPU ids"
    exit 2
}
for path in "$ECHO" "$CONFIG" "$CHECKPOINT" "$PROMPTS"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 3 ]] || {
    echo "[error] expected 3 scene-switch prompts, found $PROMPT_COUNT"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$ECHO:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/configs"

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

run_cell() {
    local name="$1" gpu="$2" mode="$3" uniqueness_weight="$4"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local existing
    existing="$(video_count "$output")"
    if [[ "$FORCE" != "1" && "$existing" -eq "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return 0
    fi
    if [[ "$existing" -ne 0 ]]; then
        echo "[error] $name has $existing existing videos; use a clean OUT_ROOT"
        return 2
    fi
    mkdir -p "$output"
    {
        printf 'name=%s\n' "$name"
        printf 'mode=%s\n' "$mode"
        printf 'uniqueness_weight=%s\n' "$uniqueness_weight"
        printf 'endpoint_bonus=0.05\n'
        printf 'prompt_file=%s\n' "$PROMPTS"
        printf 'expected_videos=%s\n' "$PROMPT_COUNT"
    } >"$OUT_ROOT/configs/$name.env"
    (
        cd "$ECHO" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export ECHO_VERBOSE=1
        export ECHO_COMPRESS_MODE="$mode"
        export ECHO_COMPRESS_UNIQUENESS_WEIGHT="$uniqueness_weight"
        export ECHO_COMPRESS_ENDPOINT_BONUS=.05
        python inference.py \
            --config_path "$CONFIG" \
            --checkpoint_path "$CHECKPOINT" \
            --output_folder "$output" \
            --data_path "$PROMPTS" \
            --seed 0 --num_samples 1 --use_ema --save_with_index
    ) >"$log" 2>&1
    [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]] || {
        echo "[error] $name produced $(video_count "$output")/$PROMPT_COUNT videos"
        return 1
    }
    if [[ "$mode" == "coherent_unique" ]] && ! grep -q '\[EchoUnique\]' "$log"; then
        echo "[error] $name produced no EchoUnique diagnostics"
        return 1
    fi
}

PIDS=()
STATUS=0
run_cell echo_score_weighted "${GPUS[0]}" score_weighted .25 & PIDS+=("$!")
run_cell echo_token_select "${GPUS[1]}" token_select .25 & PIDS+=("$!")
run_cell echo_coherent_u015 "${GPUS[2]}" coherent_unique .15 & PIDS+=("$!")
run_cell echo_coherent_u030 "${GPUS[3]}" coherent_unique .30 & PIDS+=("$!")
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

echo "[v92-echo] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
