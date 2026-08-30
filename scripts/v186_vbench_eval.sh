#!/usr/bin/env bash
# v186 vbench eval: one method's 9 dimensions across available GPUs
set -euo pipefail
METHOD="$1"

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
COMPARISON_MANIFEST="$ROOT/runs/v186_long60_comparison/vbench_comparison/comparison_manifest.json"
VIDEOS_PATH="$ROOT/runs/v186_long60_comparison/vbench_comparison/published/$METHOD"
FULL_JSON_DIR="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUTPUT_BASE="$ROOT/runs/v186_long60_comparison/metrics/vbench_long_parts/$METHOD"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"

DIMS=(subject_consistency background_consistency temporal_flickering motion_smoothness overall_consistency dynamic_degree aesthetic_quality imaging_quality temporal_style)

# Assign dims to GPUs 0-7; 9th dim waits for first free GPU
pids=()
gpu_for_dim() {
    # dims 0-7 -> GPU 0-7, dim 8 (temporal_style) -> GPU 0 (after first finishes)
    echo $(( $1 % 8 ))
}

for i in "${!DIMS[@]}"; do
    dim="${DIMS[$i]}"
    gpu=$(( i % 8 ))
    output_dir="$OUTPUT_BASE/$dim"
    mkdir -p "$output_dir"
    if [[ -s "$output_dir/v129_${dim}_eval_results.json" ]]; then
        echo "[skip] $METHOD/$dim already done"
        continue
    fi
    echo "[v186] $METHOD/$dim on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
        --vbench-root "$VBENCH_ROOT" \
        --comparison-manifest "$COMPARISON_MANIFEST" \
        --videos_path "$VIDEOS_PATH" \
        --dimension "$dim" \
        --output_path "$output_dir" \
        --full_json_dir "$FULL_JSON_DIR" \
        --num_of_samples_per_prompt 1 \
        --dev_flag \
        --local-models \
        --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub" \
        > "$OUTPUT_BASE/${dim}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || echo "[warn] some dims failed for $METHOD"
echo "[v186] all dims dispatched for $METHOD"
