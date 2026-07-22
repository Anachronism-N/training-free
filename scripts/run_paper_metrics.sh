#!/usr/bin/env bash
# Run post-review metrics for a canonical paper matrix.
set -uo pipefail

TASK="${1:-}"
GPU="${2:-0}"
if [[ "$TASK" != "single" && "$TASK" != "scene" ]]; then
    echo "usage: HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh single|scene [gpu]"
    exit 2
fi
if [[ "${HUMAN_REVIEW_DONE:-0}" != "1" ]]; then
    echo "[blocked] freeze the blind scorecard, then rerun with HUMAN_REVIEW_DONE=1"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SEED="${SEED:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"

if [[ "$TASK" == "single" ]]; then
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/paper_single_30s_s${SEED}}"
    PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_single_long_complex_3.txt}"
    METHODS=(sf_native sf_pyramid_forcing sf_echo_forcing ours_all_heads ours_role)
    BASELINE="sf_native"
else
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/paper_scene_30s_s${SEED}}"
    PROMPTS="${PROMPTS:-$ROOT/prompts/paper_scene_switch_sf_3.txt}"
    METHODS=(sf_segmented_reset sf_echo_forcing ours_all_heads ours_role)
    BASELINE="sf_segmented_reset"
fi
METRICS_ROOT="${METRICS_ROOT:-$RUN_ROOT/metrics}"

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

[[ -f "$RUN_ROOT/blind_review/scorecard.csv" ]] || {
    echo "[error] missing blind scorecard: $RUN_ROOT/blind_review/scorecard.csv"
    exit 2
}
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts: $PROMPTS"; exit 2; }
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    directory="$RUN_ROOT/$method"
    [[ -d "$directory" ]] || { echo "[error] missing video directory: $directory"; exit 2; }
    count="$(find "$directory" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    [[ "$count" -ge 3 ]] || { echo "[error] $directory has only $count videos"; exit 2; }
    VIDEO_DIRS+=("$directory")
done
mkdir -p "$METRICS_ROOT"

status=0
echo "[metrics] task=$TASK gpu=$GPU run_root=$RUN_ROOT"
python "$ROOT/scripts/evaluate_comprehensive.py" \
    --video_dirs "${VIDEO_DIRS[@]}" \
    --prompts "$PROMPTS" \
    --output "$METRICS_ROOT/comprehensive.json" \
    --gpu 0 --sample_frames 64 --batch_size 8 \
    >"$METRICS_ROOT/comprehensive.log" 2>&1 || status=1

if [[ "$TASK" == "scene" ]]; then
    python "$ROOT/scripts/evaluate_hrem_v2.py" \
        --run-root "$RUN_ROOT" --prompt-count 3 \
        --methods "${METHODS[@]}" --baseline "$BASELINE" \
        --output "$METRICS_ROOT/aba_return.json" \
        >"$METRICS_ROOT/aba_return.log" 2>&1 || status=1
fi

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" ]] || { echo "[error] missing VBench evaluator: $EVAL"; exit 2; }
    [[ -f "$INFO" ]] || { echo "[error] missing VBench info: $INFO"; exit 2; }
    DIMS=(
        subject_consistency
        background_consistency
        aesthetic_quality
        imaging_quality
        motion_smoothness
        dynamic_degree
    )
    for method in "${METHODS[@]}"; do
        output="$METRICS_ROOT/vbench_long/$method"
        mkdir -p "$output"
        echo "[vbench] method=$method"
        (
            cd "$VBENCH_ROOT"
            python "$EVAL" \
                --videos_path "$RUN_ROOT/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO"
        ) >"$output/run.log" 2>&1 || status=1
    done
fi

echo "[done] metrics=$METRICS_ROOT status=$status"
echo "[feedback] return blind scorecard, metrics JSON, traces/*_diagnosis.json, and logs/*.log"
exit "$status"
