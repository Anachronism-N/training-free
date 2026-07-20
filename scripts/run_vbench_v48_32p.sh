#!/usr/bin/env bash
set -euo pipefail

# VBench-Long evaluation for 32-prompt ablation
# Run after run_v48_32p_ablation.sh completes

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
FULL_JSON="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUT_ROOT="$ROOT/runs/vbench_long/v48_32p_ablation"
DIMS=(
    subject_consistency
    background_consistency
    aesthetic_quality
    imaging_quality
    motion_smoothness
    dynamic_degree
)

run_eval() {
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

# Map ablation variants to video directories
declare -A VARIANTS
VARIANTS[pf]="$ROOT/runs/v35_pf_value_refresh/20260720_v48_32p_pf/pf_refresh_pf_32p"
VARIANTS[static]="$ROOT/runs/v35_pf_value_refresh/20260720_v48_32p_static/pf_refresh_static_32p"
VARIANTS[adaptive]="$ROOT/runs/v35_pf_value_refresh/20260720_v48_32p_adaptive/pf_refresh_adaptive_32p"
VARIANTS[full]="$ROOT/runs/v35_pf_value_refresh/20260720_v48_32p_full/pf_refresh_full_32p"

gpu=0
pids=()
for name in pf static adaptive full; do
    videos="${VARIANTS[$name]}"
    if [[ ! -d "$videos" ]]; then
        echo "SKIP $name: $videos not found"
        continue
    fi
    echo "Evaluating $name on GPU $gpu..."
    run_eval "$gpu" "$name" "$videos" &
    pids+=($!)
    gpu=$((gpu + 1))
    if [[ $gpu -ge 4 ]]; then gpu=0; fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo "=== VBench Results ==="
for name in pf static adaptive full; do
    result="$OUT_ROOT/$name/results.json"
    if [[ -f "$result" ]]; then
        echo -e "\n$name:"
        python3 -c "
import json
with open('$result') as f:
    data = json.load(f)
for dim in ['subject_consistency','background_consistency','aesthetic_quality','imaging_quality','motion_smoothness','dynamic_degree']:
    if dim in data:
        print(f'  {dim}: {data[dim]:.5f}')
" 2>/dev/null || echo "  (parse error)"
    fi
done

exit "$status"
