#!/usr/bin/env bash
# Blind-review preparation and post-review metrics for ProbeCache.
# Usage:
#   bash scripts/postprocess_v81_probecache.sh prepare single
#   HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v81_probecache.sh metrics single
set -euo pipefail

PHASE="${1:-}"
TASK="${2:-single}"
[[ "$PHASE" == "prepare" || "$PHASE" == "metrics" ]] || {
    echo "usage: $0 prepare|metrics single|switch"
    exit 2
}
[[ "$TASK" == "single" || "$TASK" == "switch" ]] || {
    echo "usage: $0 prepare|metrics single|switch"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v81_probecache_${TASK}}"
GPU="${GPU:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
if [[ "$TASK" == "single" ]]; then
    PROMPTS="${PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
else
    PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
fi
DEFAULT_METHODS="sf_native,pf_official,echo_pc,ours_persistent,ours_reactive,ours_full,ours_no_trust,ours_open_gate,ours_conservative"
IFS=',' read -r -a METHODS <<<"${METHODS_CSV:-$DEFAULT_METHODS}"
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -gt 0 ]] || { echo "[error] empty prompts"; exit 2; }

for method in "${METHODS[@]}"; do
    count="$(find "$RUN_ROOT/$method" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)"
    [[ "$count" -ge "$PROMPT_COUNT" ]] || {
        echo "[error] $method has $count/$PROMPT_COUNT videos"
        exit 2
    }
done

if [[ "$PHASE" == "prepare" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$RUN_ROOT" \
        --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" \
        --output "$RUN_ROOT/blind_review" \
        --prompt-count "$PROMPT_COUNT" \
        --seed 20260723 --force
    echo "[review] freeze $RUN_ROOT/blind_review/scorecard.csv before metrics"
    exit 0
fi

[[ "${HUMAN_REVIEW_DONE:-0}" == "1" ]] || {
    echo "[blocked] set HUMAN_REVIEW_DONE=1 only after freezing blind scores"
    exit 2
}
[[ -f "$RUN_ROOT/blind_review/scorecard.csv" ]] || {
    echo "[error] missing blind scorecard"
    exit 2
}

METRICS="$RUN_ROOT/metrics"
mkdir -p "$METRICS"
export CUDA_VISIBLE_DEVICES="$GPU"
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do VIDEO_DIRS+=("$RUN_ROOT/$method"); done

python "$ROOT/scripts/summarize_probecache_trace.py" \
    "$RUN_ROOT"/traces/*.probecache.jsonl --strict \
    --output-json "$METRICS/probecache_trace_summary.json" \
    --output-md "$METRICS/probecache_trace_summary.md" \
    >"$METRICS/probecache_trace_summary.log" 2>&1

python "$ROOT/scripts/evaluate_comprehensive.py" \
    --video_dirs "${VIDEO_DIRS[@]}" \
    --prompts "$PROMPTS" \
    --output "$METRICS/comprehensive.json" \
    --gpu 0 --sample_frames 64 --batch_size 8 \
    >"$METRICS/comprehensive.log" 2>&1

python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "$RUN_ROOT" --output "$METRICS/temporal_jump.csv" \
    >"$METRICS/temporal_jump.log" 2>&1

if [[ "$TASK" == "switch" ]]; then
    python "$ROOT/scripts/evaluate_hrem_v2.py" \
        --run-root "$RUN_ROOT" --prompt-count "$PROMPT_COUNT" \
        --methods "${METHODS[@]}" --baseline sf_native \
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
echo "[v81] metrics complete: $METRICS"
