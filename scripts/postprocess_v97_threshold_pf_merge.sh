#!/usr/bin/env bash
# Validate traces, run metrics, and produce the v97 decision report.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v97_threshold_pf_merge32}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/runs/v97_qk_head_scores}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FORCE_METRICS="${FORCE_METRICS:-0}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-4}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree}"

METHODS=(
    prompt_tau_0p0_merge
    prompt_tau_0p5_merge
    prompt_tau_1p0_merge
    prompt_tau_1p5_merge
    prompt_tau_2p0_merge
    prompt_tau_1p0_cyclic
    prompt_tau_1p0_recent
    prompt_tau_1p0_random_merge
    prompt_tau_1p0_reversed_merge
    sign_rpos_0p5_stride_merge
    pf_ar_stride_merge
    pf_aw_stride_merge
    pf_native
    pf_anchor_extended_recent
    pf_wave_extended_recent
    pf_veil_extended_recent
)
EXPECTED=32

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v97 postprocess requires exactly 16 GPU ids"
    exit 2
}
for path in \
    "$CONDA_SH" "$PROMPTS" \
    "$PROFILE_ROOT/maps/head_map_classification_report.json"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

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
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|KeyError:|PyramidKVPolicyTraceError' \
        "$log"; then
        echo "[error] failure signature in $log"
        exit 1
    fi
done

python "$ROOT/scripts/summarize_v97_policy_traces.py" \
    --trace-dir "$RUN_ROOT/traces" \
    --config-dir "$RUN_ROOT/configs" \
    --methods "${METHODS[@]}" \
    --expected-layers "${PYRAMIDKV_POLICY_TRACE_LAYERS:-0,7,15,23,29}" \
    --output-json "$METRICS/policy_trace_audit.json" \
    --output-md "$METRICS/policy_trace_audit.md" \
    --strict >"$METRICS/logs/policy_trace_audit.log" 2>&1 || exit 1

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

python "$ROOT/scripts/analyze_v97_threshold_pf_merge.py" \
    --comprehensive "$METRICS/comprehensive.json" \
    --temporal-jump "$METRICS/temporal_jump.csv" \
    --vbench "$METRICS/vbench_long_summary.json" \
    --classification "$PROFILE_ROOT/maps/head_map_classification_report.json" \
    --policy-traces "$METRICS/policy_trace_audit.json" \
    --output-json "$METRICS/v97_analysis.json" \
    --output-md "$METRICS/v97_analysis.md" \
    >"$METRICS/logs/v97_analysis.log" 2>&1 || exit 1

echo "[v97-postprocess] metrics=$METRICS"
echo "[review] freeze $BLIND_REVIEW/scorecard.csv before opening metrics or key_private.json"
