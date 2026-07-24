#!/usr/bin/env bash
# Quality metrics for v86 after the human-review sheet has been frozen.
# Usage: HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v86_role_transition.sh screen|confirm|ultralong|switch
set -euo pipefail

TASK="${1:-}"
case "$TASK" in
    screen|confirm|ultralong|switch) ;;
    *) echo "usage: $0 screen|confirm|ultralong|switch"; exit 2 ;;
esac
[[ "${HUMAN_REVIEW_DONE:-0}" == "1" ]] || {
    echo "[blocked] freeze human review before setting HUMAN_REVIEW_DONE=1"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v86_role_transition_${TASK}}"
GPU="${GPU:-0}"
RUN_VBENCH="${RUN_VBENCH:-0}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"

case "$TASK" in
    screen)
        PROMPTS="${PROMPTS:-$ROOT/prompts/probecache_v82_diagnostic_complex_3.txt}"
        METHODS=(
            pf v78 learned_neutral learned_balanced replica_balanced
            pf_binary_balanced inverse_balanced random_balanced
            learned_conservative learned_open learned_no_bias
            learned_age_only consensus_balanced learned_early learned_late
            pf_binary_conservative
        )
        ;;
    confirm)
        PROMPTS="${PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
        METHODS=()
        for seed in 0 1 2 3; do
            METHODS+=("pf_s$seed" "v78_s$seed" "learned_s$seed" "pf_binary_s$seed")
        done
        ;;
    ultralong)
        PROMPTS="${PROMPTS:-$ROOT/prompts/probecache_v82_ultralong_complex_6.txt}"
        METHODS=(
            pf_s0 v78_s0 learned_s0 pf_binary_s0
            pf_s1 v78_s1 learned_s1 pf_binary_s1
        )
        ;;
    switch)
        PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
        METHODS=(
            pf_s0 v78_s0 learned_s0 pf_binary_s0
            pf_s1 v78_s1 learned_s1 pf_binary_s1
        )
        ;;
esac

EXPECTED="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
METRICS="$RUN_ROOT/metrics"
mkdir -p "$METRICS"
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    video_dir="$RUN_ROOT/$method"
    log="$RUN_ROOT/logs/$method.log"
    [[ -d "$video_dir" ]] || { echo "[error] missing $video_dir"; exit 2; }
    count="$(find "$video_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    [[ "$count" -ge "$EXPECTED" ]] || {
        echo "[error] $method has $count/$EXPECTED videos"
        exit 2
    }
    [[ -s "$log" ]] || { echo "[error] missing log $log"; exit 2; }
    if grep -Eqi 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|KeyError:' "$log"; then
        echo "[error] failure signature in $log"
        exit 2
    fi
    VIDEO_DIRS+=("$video_dir")
done

mapfile -t TRACES < <(
    find "$RUN_ROOT/traces" -maxdepth 1 -type f \
        -name '*.transition.jsonl' | sort
)
[[ "${#TRACES[@]}" -gt 0 ]] || {
    echo "[error] no cache-transition traces under $RUN_ROOT/traces"
    exit 2
}
python "$ROOT/scripts/summarize_cache_transition_trace.py" \
    "${TRACES[@]}" --strict \
    --output-json "$METRICS/cache_transition_summary.json" \
    --output-md "$METRICS/cache_transition_summary.md" \
    >"$METRICS/cache_transition_summary.log" 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
python "$ROOT/scripts/evaluate_comprehensive.py" \
    --video_dirs "${VIDEO_DIRS[@]}" \
    --prompts "$PROMPTS" \
    --output "$METRICS/comprehensive.json" \
    --gpu 0 --sample_frames 64 --batch_size 8 \
    >"$METRICS/comprehensive.log" 2>&1

python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "${VIDEO_DIRS[@]}" --output "$METRICS/temporal_jump.csv" \
    >"$METRICS/temporal_jump.log" 2>&1

if [[ "$TASK" == "switch" ]]; then
    python "$ROOT/scripts/evaluate_hrem_v2.py" \
        --run-root "$RUN_ROOT" --prompt-count "$EXPECTED" \
        --methods "${METHODS[@]}" --baseline pf_s0 \
        --output "$METRICS/aba_return.json" \
        >"$METRICS/aba_return.log" 2>&1
fi

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$INFO" ]] || {
        echo "[error] VBench-Long missing under $VBENCH_ROOT"
        exit 2
    }
    DIMS=(
        subject_consistency background_consistency aesthetic_quality
        imaging_quality motion_smoothness dynamic_degree
    )
    for method in "${METHODS[@]}"; do
        output="$METRICS/vbench_long/$method"
        mkdir -p "$output"
        (
            cd "$VBENCH_ROOT"
            python "$EVAL" \
                --videos_path "$RUN_ROOT/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO"
        ) >"$output/run.log" 2>&1
    done
fi

echo "[v86] task=$TASK metrics=$METRICS"
