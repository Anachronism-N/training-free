#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
FULL_JSON="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUT_ROOT="$ROOT/runs/vbench_long/v43_archive_30s"
DIMS=(
    subject_consistency
    background_consistency
    aesthetic_quality
    imaging_quality
    motion_smoothness
    dynamic_degree
)

run_method() {
    local gpu="$1"
    local name="$2"
    local videos="$3"
    local output="$OUT_ROOT/$name"
    mkdir -p "$output"
    (
        cd "$VBENCH_ROOT"
        CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL" \
            --videos_path "$videos" \
            --dimension "${DIMS[@]}" \
            --mode long_custom_input \
            --dev_flag \
            --num_of_samples_per_prompt 1 \
            --output_path "$output" \
            --full_json_dir "$FULL_JSON"
    ) > "$output/run.log" 2>&1
}

run_method 4 pf "$ROOT/runs/REVIEW_v43_archive_30s/pf" &
pid_pf=$!
run_method 5 conf "$ROOT/runs/REVIEW_v43_archive_30s/conf" &
pid_conf=$!
run_method 6 query "$ROOT/runs/REVIEW_v43_archive_30s/query" &
pid_query=$!
run_method 7 lag5 "$ROOT/runs/REVIEW_v43_archive_30s/lag5" &
pid_lag=$!

status=0
for pid in "$pid_pf" "$pid_conf" "$pid_query" "$pid_lag"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
