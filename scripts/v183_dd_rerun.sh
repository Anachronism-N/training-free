#!/bin/bash
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
MANIFEST="$ROOT/runs/v180_rccp_fresh128/recovery_v183/vbench_comparison/comparison_manifest.json"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
METHODS=(sf_native rccp_matched all_recent all_coverage)
for i in "${!METHODS[@]}"; do
    m="${METHODS[$i]}"
    out="$ROOT/runs/v180_rccp_fresh128/recovery_v183/metrics/vbench_long_parts/$m/dynamic_degree"
    mkdir -p "$out"
    rm -f "$out/v129_dynamic_degree_eval_results.json"
    echo "[v183-dd] $m on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
        --vbench-root "$VBENCH_ROOT" \
        --comparison-manifest "$MANIFEST" \
        --videos_path "$ROOT/runs/v180_rccp_fresh128/recovery_v183/vbench_comparison/published/$m" \
        --dimension dynamic_degree \
        --output_path "$out" \
        --full_json_dir "$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json" \
        --num_of_samples_per_prompt 1 --dev_flag \
        --local-models --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub" \
        > "$out/rerun.log" 2>&1 &
done
wait
echo "all done"
