#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
FULL_JSON="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUT_ROOT="$ROOT/runs/vbench_long/v44_headroute_30s"
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

run_method 4 stable1 "$ROOT/runs/REVIEW_v44_headroute_30s/stable1" &
pid_stable1=$!
run_method 5 stable12 "$ROOT/runs/REVIEW_v44_headroute_30s/stable12" &
pid_stable12=$!
run_method 6 stable1g30 "$ROOT/runs/REVIEW_v44_headroute_30s/stable1g30" &
pid_stable1g30=$!

status=0
for pid in "$pid_stable1" "$pid_stable12" "$pid_stable1g30"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
