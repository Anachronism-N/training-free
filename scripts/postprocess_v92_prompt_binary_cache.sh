#!/usr/bin/env bash
# Review-first metrics for the v92 prompt-binary cache screen.
# Usage: HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v92_prompt_binary_cache.sh
set -euo pipefail

[[ "${HUMAN_REVIEW_DONE:-0}" == "1" ]] || {
    echo "[blocked] freeze human review before setting HUMAN_REVIEW_DONE=1"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v92_prompt_binary_cache_screen}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROOT/runs/v86_role_transition_screen}"
PROMPTS="${PROMPTS:-$ROOT/prompts/v86_single_long_complex_16.txt}"
GPU="${GPU:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
VBENCH_GPU_LIST="${VBENCH_GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"

METHODS=(
    pf_binary_read pf_binary_read_v78
    prompt_pfcount_read prompt_pfcount_read_v78
    prompt_kmeans_read prompt_kmeans_read_v78
    prompt_replica_read_v78 prompt_consensus_read_v78
    prompt_inverse_read_v78 prompt_random_read_v78
    remote_read_v78 role_score_read_v78
    pf_read_prompt_priority prompt_read_prompt_priority
    prompt_read_v78_coverage pf_binary_read_v78_coverage
)
BASELINE_METHODS=(pf v78)
EXPECTED="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$EXPECTED" -eq 16 ]] || {
    echo "[error] expected 16 prompts, found $EXPECTED"
    exit 2
}

validate_method() {
    local root="$1" method="$2"
    local video_dir="$root/$method" log="$root/logs/$method.log"
    [[ -d "$video_dir" ]] || { echo "[error] missing $video_dir"; return 2; }
    local count
    count="$(find "$video_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    [[ "$count" -eq "$EXPECTED" ]] || {
        echo "[error] $method has $count/$EXPECTED videos"
        return 2
    }
    [[ -s "$log" ]] || { echo "[error] missing log $log"; return 2; }
    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|KeyError:' \
        "$log"; then
        echo "[error] failure signature in $log"
        return 2
    fi
}

VIDEO_DIRS=()
for method in "${BASELINE_METHODS[@]}"; do
    validate_method "$BASELINE_ROOT" "$method"
    VIDEO_DIRS+=("$BASELINE_ROOT/$method")
done
for method in "${METHODS[@]}"; do
    validate_method "$RUN_ROOT" "$method"
    VIDEO_DIRS+=("$RUN_ROOT/$method")
done

METRICS="$RUN_ROOT/metrics"
mkdir -p "$METRICS"
{
    printf 'BASELINE_ROOT=%s\n' "$BASELINE_ROOT"
    printf 'BASELINE_METHODS=%s\n' "${BASELINE_METHODS[*]}"
    printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'VBENCH_DIMS=%s\n' "$VBENCH_DIMS"
} >"$METRICS/comparison_sources.env"

mapfile -t TRACES < <(
    find "$RUN_ROOT/traces" -maxdepth 1 -type f \
        -name '*.transition.jsonl' | sort
)
[[ "${#TRACES[@]}" -eq 13 ]] || {
    echo "[error] expected 13 transition traces, found ${#TRACES[@]}"
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
    --gpu 0 --sample_frames 64 --batch_size 8 --skip_m3 \
    >"$METRICS/comprehensive.log" 2>&1

python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "${VIDEO_DIRS[@]}" --output "$METRICS/temporal_jump.csv" \
    >"$METRICS/temporal_jump.log" 2>&1

python "$ROOT/scripts/analyze_v92_metrics.py" \
    --comprehensive "$METRICS/comprehensive.json" \
    --temporal-jump "$METRICS/temporal_jump.csv" \
    --trace-summary "$METRICS/cache_transition_summary.json" \
    --label-manifest "$RUN_ROOT/labels/prompt_contrastive_manifest.json" \
    --output-json "$METRICS/v92_analysis.json" \
    --output-md "$METRICS/v92_analysis.md" \
    >"$METRICS/v92_analysis.log" 2>&1

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$INFO" ]] || {
        echo "[error] VBench-Long missing under $VBENCH_ROOT"
        exit 2
    }
    read -r -a DIMS <<<"$VBENCH_DIMS"
    IFS=',' read -r -a VBENCH_GPUS <<<"$VBENCH_GPU_LIST"
    [[ "${#VBENCH_GPUS[@]}" -eq "${#METHODS[@]}" ]] || {
        echo "[error] VBench needs exactly ${#METHODS[@]} GPU ids"
        exit 2
    }
    VBENCH_PIDS=()
    VBENCH_STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        gpu="${VBENCH_GPUS[$index]}"
        output="$METRICS/vbench_long/$method"
        mkdir -p "$output"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            cd "$VBENCH_ROOT"
            python "$EVAL" \
                --videos_path "$RUN_ROOT/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO"
        ) >"$output/run.log" 2>&1 &
        VBENCH_PIDS+=("$!")
    done
    for pid in "${VBENCH_PIDS[@]}"; do
        wait "$pid" || VBENCH_STATUS=1
    done
    [[ "$VBENCH_STATUS" -eq 0 ]] || {
        echo "[error] at least one VBench-Long job failed"
        exit 1
    }
fi

echo "[v92] metrics=$METRICS"
