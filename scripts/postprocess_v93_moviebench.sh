#!/usr/bin/env bash
# Parallel post-processing for v93 main or head32.
# Usage: bash scripts/postprocess_v93_moviebench.sh main|head32
set -uo pipefail

MODE="${1:-}"
[[ "$MODE" == "main" || "$MODE" == "head32" ]] || {
    echo "usage: $0 main|head32"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
RUN_COMPREHENSIVE="${RUN_COMPREHENSIVE:-1}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
FORCE_METRICS="${FORCE_METRICS:-0}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-4}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality dynamic_degree}"

if [[ "$MODE" == "main" ]]; then
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v93_moviebench128_main}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
    TRACE_EXPECTED=10
    METHODS=(
        sf_native pf echo_pc v78
        pf_binary_read_v78 prompt_pfcount_read_v78
        prompt_kmeans_read_v78 veil_priority_b005
    )
else
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v93_moviebench32_head}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
    TRACE_EXPECTED=12
    METHODS=(
        pf pf_binary_read prompt_pfcount_read prompt_kmeans_read
        v78 pf_binary_read_v78 prompt_pfcount_read_v78 prompt_kmeans_read_v78
        prompt_replica_read_v78 prompt_consensus_read_v78
        prompt_inverse_read_v78 prompt_random_read_v78
        remote_read_v78 role_score_read_v78
        pf_read_prompt_priority prompt_read_prompt_priority
    )
fi

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v93 postprocess requires exactly 16 GPU ids"
    exit 2
}
[[ -f "$PROMPTS" ]] || { echo "[error] missing prompts $PROMPTS"; exit 2; }
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED" ]] || {
    echo "[error] expected $EXPECTED prompts, found $PROMPT_COUNT"
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

STATUS=0
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    video_dir="$RUN_ROOT/$method"
    VIDEO_DIRS+=("$video_dir")
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$video_dir" --start-idx 0 --end-idx "$EXPECTED" \
        --output-json "$METRICS/$method.video_audit.json" \
        >"$METRICS/logs/$method.video_audit.log" 2>&1 || STATUS=1
    if [[ "$MODE" == "main" ]]; then
        logs=("$RUN_ROOT/logs/$method.shard0.log" "$RUN_ROOT/logs/$method.shard1.log")
    else
        logs=("$RUN_ROOT/logs/$method.log")
    fi
    for log in "${logs[@]}"; do
        [[ -s "$log" ]] || {
            echo "[error] missing generation log $log"
            STATUS=1
            continue
        }
        if grep -Eqi \
            'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|KeyError:' \
            "$log"; then
            echo "[error] failure signature in $log"
            STATUS=1
        fi
    done
done
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"

mapfile -t TRACES < <(
    find "$RUN_ROOT/traces" -maxdepth 1 -type f \
        -name '*.transition.jsonl' | sort
)
[[ "${#TRACES[@]}" -eq "$TRACE_EXPECTED" ]] || {
    echo "[error] expected $TRACE_EXPECTED transition traces, found ${#TRACES[@]}"
    exit 2
}
python "$ROOT/scripts/summarize_cache_transition_trace.py" \
    "${TRACES[@]}" --strict \
    --output-json "$METRICS/cache_transition_summary.json" \
    --output-md "$METRICS/cache_transition_summary.md" \
    >"$METRICS/logs/cache_transition_summary.log" 2>&1 || exit 1

BLIND_REVIEW="$RUN_ROOT/blind_review"
if [[ ! -d "$BLIND_REVIEW" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$RUN_ROOT" --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" --prompt-count "$EXPECTED" \
        --seed 20260725 --output "$BLIND_REVIEW" || exit 1
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
    VBENCH_PIDS=()
    VBENCH_STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        gpu="${GPUS[$index]}"
        output="$METRICS/vbench_long/$method"
        marker="$METRICS/status/vbench.$method.done"
        mkdir -p "$output"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output/results.json" ]]; then
            echo "[skip] VBench $method"
            continue
        fi
        rm -f "$marker"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
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
        VBENCH_PIDS+=("$!")
    done
    for pid in "${VBENCH_PIDS[@]}"; do
        wait "$pid" || VBENCH_STATUS=1
    done
    [[ "$VBENCH_STATUS" -eq 0 ]] || {
        echo "[error] at least one VBench-Long job failed"
        exit 1
    }
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS/vbench_long" \
        --methods "${METHODS[@]}" --dimensions "${DIMS[@]}" \
        --output-json "$METRICS/vbench_long_summary.json" \
        --output-csv "$METRICS/vbench_long_summary.csv" \
        --output-md "$METRICS/vbench_long_summary.md" \
        >"$METRICS/logs/collect_vbench.log" 2>&1 || exit 1
fi

if [[ "$RUN_COMPREHENSIVE" == "1" ]]; then
    COMP_PIDS=()
    COMP_STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        gpu="${GPUS[$index]}"
        output="$METRICS/comprehensive_parts/$method.json"
        log="$METRICS/logs/comprehensive.$method.log"
        marker="$METRICS/status/comprehensive.$method.done"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output" ]]; then
            echo "[skip] comprehensive $method"
            continue
        fi
        rm -f "$marker"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            python "$ROOT/scripts/evaluate_comprehensive.py" \
                --video_dirs "$RUN_ROOT/$method" \
                --prompts "$PROMPTS" --output "$output" \
                --gpu 0 --sample_frames "$SAMPLE_FRAMES" \
                --batch_size 8 --skip_m3 &&
                printf 'ok\n' >"$marker"
        ) >"$log" 2>&1 &
        COMP_PIDS+=("$!")
    done
    for pid in "${COMP_PIDS[@]}"; do
        wait "$pid" || COMP_STATUS=1
    done
    [[ "$COMP_STATUS" -eq 0 ]] || {
        echo "[error] at least one comprehensive metric job failed"
        exit 1
    }
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
    "$METRICS/temporal_jump.csv" \
    "$METRICS/vbench_long_summary.json"; do
    [[ -s "$path" ]] || {
        echo "[error] required metric output missing: $path"
        exit 2
    }
done

python "$ROOT/scripts/analyze_v93_moviebench.py" \
    --mode "$MODE" \
    --comprehensive "$METRICS/comprehensive.json" \
    --temporal-jump "$METRICS/temporal_jump.csv" \
    --vbench "$METRICS/vbench_long_summary.json" \
    --trace-summary "$METRICS/cache_transition_summary.json" \
    --label-manifest "$RUN_ROOT/labels/prompt_contrastive_manifest.json" \
    --output-json "$METRICS/v93_analysis.json" \
    --output-md "$METRICS/v93_analysis.md" \
    >"$METRICS/logs/v93_analysis.log" 2>&1 || exit 1

echo "[v93-postprocess] mode=$MODE metrics=$METRICS"
echo "[review] freeze $BLIND_REVIEW/scorecard.csv before opening ${BLIND_REVIEW}_private/key_private.json or metrics"
