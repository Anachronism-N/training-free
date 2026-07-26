#!/usr/bin/env bash
# VBench-Long, comprehensive metrics, temporal diagnostics, and blind review.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v96_binary_cache32}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/runs/v96_qk_head_profile}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE_METRICS="${FORCE_METRICS:-0}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-4}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree}"

METHODS=(
    pf
    pf_binary_cyclic pf_binary_merge pf_binary_recent
    cfg_cyclic cfg_merge semantic_cyclic semantic_merge
    consensus_cyclic consensus_merge consensus_recent
    consensus_merge_v78 consensus_cyclic_v78
    random_merge inverse_merge pf_binary_merge_v78
)
EXPECTED=32

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v96 postprocess requires exactly 16 GPU ids"
    exit 2
}
source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"

METRICS="$RUN_ROOT/metrics"
mkdir -p \
    "$METRICS/logs" "$METRICS/comprehensive_parts" \
    "$METRICS/vbench_long" "$METRICS/status"
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    VIDEO_DIRS+=("$RUN_ROOT/$method")
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$RUN_ROOT/$method" --start-idx 0 --end-idx "$EXPECTED" \
        --output-json "$METRICS/$method.video_audit.json" \
        >"$METRICS/logs/$method.video_audit.log" 2>&1 || exit 1
    log="$RUN_ROOT/logs/$method.log"
    [[ -s "$log" ]] || { echo "[error] missing $log"; exit 2; }
    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|KeyError:' \
        "$log"; then
        echo "[error] failure signature in $log"
        exit 1
    fi
done

BLIND_REVIEW="$RUN_ROOT/blind_review"
if [[ ! -d "$BLIND_REVIEW" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$RUN_ROOT" --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" --prompt-count "$EXPECTED" \
        --seed 20260726 --output "$BLIND_REVIEW" || exit 1
fi

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

python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
    "${VIDEO_DIRS[@]}" --frame-step "$TEMPORAL_FRAME_STEP" \
    --output "$METRICS/temporal_jump.csv" \
    >"$METRICS/logs/temporal_jump.log" 2>&1 || exit 1

python "$ROOT/scripts/analyze_v96_binary_cache.py" \
    --comprehensive "$METRICS/comprehensive.json" \
    --temporal-jump "$METRICS/temporal_jump.csv" \
    --vbench "$METRICS/vbench_long_summary.json" \
    --head-profile "$PROFILE_ROOT/labels/qk_head_threshold_report.json" \
    --transition-traces "$RUN_ROOT/diagnostics/cache_transition_summary.json" \
    --output-json "$METRICS/v96_analysis.json" \
    --output-md "$METRICS/v96_analysis.md" \
    >"$METRICS/logs/v96_analysis.log" 2>&1 || exit 1

echo "[v96-postprocess] metrics=$METRICS"
echo "[review] freeze $BLIND_REVIEW/scorecard.csv before opening metrics or ${BLIND_REVIEW}_private/key_private.json"
