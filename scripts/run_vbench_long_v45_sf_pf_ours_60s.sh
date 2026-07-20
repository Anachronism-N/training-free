#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
FULL_JSON="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUT_ROOT="$ROOT/runs/vbench_long/v45_sf_pf_ours_60s"
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

run_method 4 sf_native "$ROOT/runs/REVIEW_v45_sf_pf_ours_60s/sf_native" &
pid_sf_native=$!
run_method 5 sf_pf "$ROOT/runs/REVIEW_v45_sf_pf_ours_60s/sf_pf" &
pid_sf_pf=$!
run_method 6 ours "$ROOT/runs/REVIEW_v45_sf_pf_ours_60s/ours" &
pid_ours=$!

status=0
for pid in "$pid_sf_native" "$pid_sf_pf" "$pid_ours"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
