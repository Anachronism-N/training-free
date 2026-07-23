#!/usr/bin/env bash
# Post-review quality metrics for v82. Never run before freezing human scores.
# Usage: HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v82_probecache.sh TASK
# TASK: labels, confirm, ultralong, switch
set -euo pipefail

TASK="${1:-}"
case "$TASK" in
    labels|confirm|ultralong|switch) ;;
    *) echo "usage: $0 labels|confirm|ultralong|switch"; exit 2 ;;
esac
[[ "${HUMAN_REVIEW_DONE:-0}" == "1" ]] || {
    echo "[blocked] freeze human review before setting HUMAN_REVIEW_DONE=1"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
GPU="${GPU:-0}"
RUN_VBENCH="${RUN_VBENCH:-0}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"

case "$TASK" in
    labels)
        RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v82_probecache_labels}"
        PROMPTS="${PROMPTS:-$ROOT/prompts/probecache_v82_diagnostic_complex_3.txt}"
        REPLICA="profile_replica"
        [[ -d "$RUN_ROOT/$REPLICA" ]] || REPLICA="random_2028_fallback"
        METHODS=(
            pf v78 learned "$REPLICA" pf_binary inverse
            random_2026 random_2027 remote_only prompt_only
            layer_early layer_middle layer_late layer_first_half
            layer_second_half learned_audit
        )
        ;;
    confirm)
        RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v82_probecache_confirm}"
        PROMPTS="${PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
        METHODS=(
            pf_s1 pf_s2 pf_s3 v78_s1 v78_s2 v78_s3
            learned_s1 learned_s2 learned_s3
            pf_binary_s1 pf_binary_s2 pf_binary_s3
            learned_open_s1 learned_open_s2
            learned_conservative_s1 learned_conservative_s2
        )
        ;;
    ultralong)
        RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v82_probecache_ultralong}"
        PROMPTS="${PROMPTS:-$ROOT/prompts/probecache_v82_ultralong_complex_6.txt}"
        REPLICA="replica"
        [[ -d "$RUN_ROOT/${REPLICA}_s0" ]] || REPLICA="random_2028"
        METHODS=(
            sf_s0 sf_s1 pf_s0 pf_s1 v78_s0 v78_s1
            learned_s0 learned_s1 pf_binary_s0 pf_binary_s1
            "${REPLICA}_s0" "${REPLICA}_s1"
            learned_open_s0 learned_open_s1
            learned_conservative_s0 learned_conservative_s1
        )
        ;;
    switch)
        RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v82_probecache_switch}"
        PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
        METHODS=(
            sf_native pf_official echo_pc ours_audit ours_persistent
            ours_reactive ours_full ours_no_trust ours_archive12
            ours_archive36 ours_topk2 ours_topk6 ours_prompt0
            ours_prompt30 ours_open_gate ours_conservative
        )
        ;;
esac

python "$ROOT/scripts/audit_probecache_experiment_runs.py" \
    --run-root "$RUN_ROOT" --strict

METRICS="$RUN_ROOT/metrics"
mkdir -p "$METRICS"
export CUDA_VISIBLE_DEVICES="$GPU"
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    [[ -d "$RUN_ROOT/$method" ]] || {
        echo "[error] missing method directory: $RUN_ROOT/$method"
        exit 2
    }
    VIDEO_DIRS+=("$RUN_ROOT/$method")
done

mapfile -t TRACES < <(
    find "$RUN_ROOT/traces" -maxdepth 1 -type f \
        -name '*.probecache.jsonl' | sort
)
if [[ "${#TRACES[@]}" -gt 0 ]]; then
    python "$ROOT/scripts/summarize_probecache_trace.py" \
        "${TRACES[@]}" --strict \
        --output-json "$METRICS/probecache_trace_summary.json" \
        --output-md "$METRICS/probecache_trace_summary.md" \
        >"$METRICS/probecache_trace_summary.log" 2>&1
fi

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
        --run-root "$RUN_ROOT" --prompt-count 3 \
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
echo "[v82] task=$TASK metrics=$METRICS"
