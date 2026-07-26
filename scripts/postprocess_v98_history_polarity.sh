#!/usr/bin/env bash
# Audit v98 generation, run VBench-Long and diagnostics, and freeze review.
set -uo pipefail

MODE="${1:-screen32}"
[[ "$MODE" == "screen32" || "$MODE" == "main128" ]] || {
    echo "usage: $0 screen32|main128"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FORCE_METRICS="${FORCE_METRICS:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_COMPREHENSIVE="${RUN_COMPREHENSIVE:-1}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-4}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree}"

if [[ "$MODE" == "screen32" ]]; then
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v98_history_polarity_screen32}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
else
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v98_history_polarity_main128}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
fi

METHODS=(
    sf_native
    pf_native
    pf_explicit_parity
    pf_aw_hybrid_merge
    history_polarity_hybrid_merge
    history_polarity_stride_merge
    history_polarity_hybrid_merge_v78
    positive_rate_half_hybrid_merge
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 8 ]] || {
    echo "[error] v98 postprocess requires exactly eight local GPU ids"
    exit 2
}
for path in "$CONDA_SH" "$PROMPTS" "$RUN_ROOT/maps/history_polarity_manifest.json"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
for node in 0 1 2 3; do
    [[ -s "$RUN_ROOT/status/node${node}.done" ]] || {
        echo "[error] node $node generation is incomplete"
        exit 2
    }
done

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

METRICS="$RUN_ROOT/metrics"
mkdir -p \
    "$METRICS/logs" "$METRICS/comprehensive_parts" \
    "$METRICS/vbench_long" "$METRICS/status"

STATUS=0
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    VIDEO_DIRS+=("$RUN_ROOT/$method")
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$RUN_ROOT/$method" --start-idx 0 --end-idx "$EXPECTED" \
        --output-json "$METRICS/$method.video_audit.json" \
        >"$METRICS/logs/$method.video_audit.log" 2>&1 || STATUS=1
    for shard in 0 1 2 3; do
        log="$RUN_ROOT/logs/$method.shard$shard.log"
        config="$RUN_ROOT/configs/$method.shard$shard.env"
        marker="$RUN_ROOT/status/$method.shard$shard.done"
        for path in "$log" "$config" "$marker"; do
            [[ -s "$path" ]] || {
                echo "[error] missing generation artifact $path"
                STATUS=1
            }
        done
        if [[ -s "$log" ]] && grep -Eqi \
            'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|PyramidKVPolicyTraceError' \
            "$log"; then
            echo "[error] failure signature in $log"
            STATUS=1
        fi
    done
done
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"

python "$ROOT/scripts/audit_v98_policy_traces.py" \
    --run-root "$RUN_ROOT" \
    --expected-layers "${PYRAMIDKV_POLICY_TRACE_LAYERS:-0,7,15,23,29}" \
    --output-json "$METRICS/policy_trace_audit.json" \
    --output-md "$METRICS/policy_trace_audit.md" \
    --strict >"$METRICS/logs/policy_trace_audit.log" 2>&1 || exit 1

mapfile -t TRANSITION_TRACES < <(
    find "$RUN_ROOT/traces" -maxdepth 1 -type f \
        -name 'history_polarity_hybrid_merge_v78.shard*.transition.jsonl' |
        sort
)
[[ "${#TRANSITION_TRACES[@]}" -eq 4 ]] || {
    echo "[error] expected four v78 transition traces, found ${#TRANSITION_TRACES[@]}"
    exit 2
}
python "$ROOT/scripts/summarize_cache_transition_trace.py" \
    "${TRANSITION_TRACES[@]}" --strict \
    --output-json "$METRICS/cache_transition_summary.json" \
    --output-md "$METRICS/cache_transition_summary.md" \
    >"$METRICS/logs/cache_transition_summary.log" 2>&1 || exit 1

BLIND_REVIEW="$RUN_ROOT/blind_review"
if [[ ! -d "$BLIND_REVIEW" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$RUN_ROOT" --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" --prompt-count "$EXPECTED" \
        --seed 20260727 --output "$BLIND_REVIEW" || exit 1
fi

{
    printf 'MODE=%s\n' "$MODE"
    printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'EXPECTED=%s\n' "$EXPECTED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'SAMPLE_FRAMES=%s\n' "$SAMPLE_FRAMES"
    printf 'TEMPORAL_FRAME_STEP=%s\n' "$TEMPORAL_FRAME_STEP"
    printf 'VBENCH_DIMS=%s\n' "$VBENCH_DIMS"
    printf 'BLIND_REVIEW=%s\n' "$BLIND_REVIEW"
} >"$METRICS/metric_manifest.env"

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$INFO" ]] || {
        echo "[error] VBench-Long missing under $VBENCH_ROOT"
        exit 2
    }
    read -r -a DIMS <<<"$VBENCH_DIMS"
    PIDS=()
    STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        output="$METRICS/vbench_long/$method"
        marker="$METRICS/status/vbench.$method.done"
        mkdir -p "$output"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output/results.json" ]]; then
            continue
        fi
        rm -f "$marker"
        (
            export CUDA_VISIBLE_DEVICES="${GPUS[$index]}"
            cd "$VBENCH_ROOT" || exit 2
            python "$EVAL" \
                --videos_path "$RUN_ROOT/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO" &&
                test -s "$output/results.json" &&
                printf 'ok\n' >"$marker"
        ) >"$output/run.log" 2>&1 &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || STATUS=1
    done
    [[ "$STATUS" -eq 0 ]] || exit 1
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS/vbench_long" \
        --methods "${METHODS[@]}" --dimensions "${DIMS[@]}" \
        --output-json "$METRICS/vbench_long_summary.json" \
        --output-csv "$METRICS/vbench_long_summary.csv" \
        --output-md "$METRICS/vbench_long_summary.md" \
        >"$METRICS/logs/collect_vbench.log" 2>&1 || exit 1
fi

if [[ "$RUN_COMPREHENSIVE" == "1" ]]; then
    PIDS=()
    STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        output="$METRICS/comprehensive_parts/$method.json"
        marker="$METRICS/status/comprehensive.$method.done"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output" ]]; then
            continue
        fi
        rm -f "$marker"
        (
            export CUDA_VISIBLE_DEVICES="${GPUS[$index]}"
            python "$ROOT/scripts/evaluate_comprehensive.py" \
                --video_dirs "$RUN_ROOT/$method" \
                --prompts "$PROMPTS" --output "$output" \
                --gpu 0 --sample_frames "$SAMPLE_FRAMES" \
                --batch_size 8 --skip_m3 &&
                printf 'ok\n' >"$marker"
        ) >"$METRICS/logs/comprehensive.$method.log" 2>&1 &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || STATUS=1
    done
    [[ "$STATUS" -eq 0 ]] || exit 1
    PARTS=()
    for method in "${METHODS[@]}"; do
        PARTS+=("$METRICS/comprehensive_parts/$method.json")
    done
    python "$ROOT/scripts/merge_comprehensive_results.py" \
        "${PARTS[@]}" --output "$METRICS/comprehensive.json" \
        --expected-methods "${METHODS[@]}" --expected-videos "$EXPECTED" \
        >"$METRICS/logs/merge_comprehensive.log" 2>&1 || exit 1
fi

if [[ "$RUN_TEMPORAL" == "1" ]]; then
    python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${VIDEO_DIRS[@]}" --frame-step "$TEMPORAL_FRAME_STEP" \
        --output "$METRICS/temporal_jump.csv" \
        >"$METRICS/logs/temporal_jump.log" 2>&1 || exit 1
fi

for path in \
    "$METRICS/comprehensive.json" \
    "$METRICS/vbench_long_summary.json" \
    "$METRICS/temporal_jump.csv"; do
    [[ -s "$path" ]] || {
        echo "[error] required metric output missing: $path"
        exit 2
    }
done
python "$ROOT/scripts/analyze_v98_history_polarity.py" \
    --comprehensive "$METRICS/comprehensive.json" \
    --vbench "$METRICS/vbench_long_summary.json" \
    --temporal-jump "$METRICS/temporal_jump.csv" \
    --map-manifest "$RUN_ROOT/maps/history_polarity_manifest.json" \
    --policy-audit "$METRICS/policy_trace_audit.json" \
    --output-json "$METRICS/v98_analysis.json" \
    --output-md "$METRICS/v98_analysis.md" \
    >"$METRICS/logs/v98_analysis.log" 2>&1 || exit 1

echo "[v98-postprocess] mode=$MODE metrics=$METRICS"
echo "[review] freeze $BLIND_REVIEW/scorecard.csv before opening metrics or key_private.json"
