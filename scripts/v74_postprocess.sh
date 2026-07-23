#!/usr/bin/env bash
# Post-review metrics for a v74 Commit Forcing run.
set -uo pipefail

[[ "${HUMAN_REVIEW_DONE:-0}" == "1" ]] || {
    echo "[blocked] freeze blind_review/scorecard.csv, then set HUMAN_REVIEW_DONE=1"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v74_commit_screen_12p_30s}"
PROMPTS="${PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
GPU="${GPU:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
METRICS_ROOT="$RUN_ROOT/metrics"

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

[[ -f "$PROMPTS" ]] || { echo "[error] missing $PROMPTS"; exit 2; }
[[ -f "$RUN_ROOT/blind_review/scorecard.csv" ]] || {
    echo "[error] missing $RUN_ROOT/blind_review/scorecard.csv"
    exit 2
}

if [[ -n "${METHODS_CSV:-}" ]]; then
    IFS=',' read -r -a METHODS <<<"$METHODS_CSV"
else
    mapfile -t METHODS < <(
        find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d \
            ! -name logs ! -name traces ! -name diagnostics \
            ! -name metrics ! -name blind_review \
            -printf '%f\n' | sort
    )
fi
[[ "${#METHODS[@]}" -gt 0 ]] || { echo "[error] no method directories"; exit 2; }

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    directory="$RUN_ROOT/$method"
    count="$(find "$directory" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    [[ "$count" -ge "$PROMPT_COUNT" ]] || {
        echo "[error] $method has $count/$PROMPT_COUNT videos"
        exit 2
    }
    VIDEO_DIRS+=("$directory")
done
mkdir -p "$METRICS_ROOT"

status=0
shopt -s nullglob
traces=("$RUN_ROOT"/traces/*.jsonl)
if [[ "${#traces[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_commit_forcing_trace.py" \
        "${traces[@]}" --strict \
        --output-json "$METRICS_ROOT/commit_trace_summary.json" \
        --output-md "$METRICS_ROOT/commit_trace_summary.md" \
        >"$METRICS_ROOT/commit_trace_summary.log" 2>&1 || status=1
fi

python "$ROOT/scripts/evaluate_comprehensive.py" \
    --video_dirs "${VIDEO_DIRS[@]}" \
    --prompts "$PROMPTS" \
    --output "$METRICS_ROOT/comprehensive.json" \
    --gpu 0 --sample_frames 64 --batch_size 8 \
    >"$METRICS_ROOT/comprehensive.log" 2>&1 || status=1

python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "$RUN_ROOT" --output "$METRICS_ROOT/temporal_jump.csv" \
    >"$METRICS_ROOT/temporal_jump.log" 2>&1 || status=1

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$INFO" ]] || {
        echo "[error] VBench-Long is missing under $VBENCH_ROOT"
        exit 2
    }
    DIMS=(
        subject_consistency background_consistency aesthetic_quality
        imaging_quality motion_smoothness dynamic_degree
    )
    for method in "${METHODS[@]}"; do
        output="$METRICS_ROOT/vbench_long/$method"
        mkdir -p "$output"
        (
            cd "$VBENCH_ROOT" || exit 2
            python "$EVAL" \
                --videos_path "$RUN_ROOT/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO"
        ) >"$output/run.log" 2>&1 || status=1
    done
fi

echo "[v74-post] methods=${#METHODS[@]} metrics=$METRICS_ROOT status=$status"
exit "$status"
