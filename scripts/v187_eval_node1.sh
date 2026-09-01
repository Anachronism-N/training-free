#!/bin/bash
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
OUTPUT_BASE="$ROOT/runs/v187_hybrid_retrieval/metrics/vbench_long_parts/pf_hybrid_retrieval"
DIMS=(dynamic_degree aesthetic_quality imaging_quality temporal_style)
for i in "${!DIMS[@]}"; do
    dim="${DIMS[$i]}"
    output_dir="$OUTPUT_BASE/$dim"
    mkdir -p "$output_dir"
    if [[ -s "$output_dir/v129_${dim}_eval_results.json" ]]; then
        echo "[skip] $dim already done"; continue
    fi
    echo "[v187-node1] $dim on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
        --vbench-root "$VBENCH_ROOT" \
        --comparison-manifest "$ROOT/runs/v187_hybrid_retrieval/vbench_comparison/comparison_manifest.json" \
        --videos_path "$ROOT/runs/v187_hybrid_retrieval/vbench_comparison/published/pf_hybrid_retrieval" \
        --dimension "$dim" \
        --output_path "$output_dir" \
        --full_json_dir "$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json" \
        --num_of_samples_per_prompt 1 --dev_flag \
        --local-models --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub" \
        > "$OUTPUT_BASE/${dim}_node1.log" 2>&1 &
done
wait
echo "node1 dims done"
