#!/bin/bash
# Rerun dynamic_degree with continuous flow-magnitude metric
# for the core 30s comparison methods (v183/v184/v185/v187)
set -euo pipefail

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"

# method_name|manifest|videos_path|output_path
TARGETS=(
  "v183_all_recent|$ROOT/runs/v180_rccp_fresh128/recovery_v183/vbench_comparison/comparison_manifest.json|$ROOT/runs/v180_rccp_fresh128/recovery_v183/vbench_comparison/published/all_recent|$ROOT/runs/v180_rccp_fresh128/recovery_v183/metrics/vbench_long_parts/all_recent/dynamic_degree"
  "v184_retrieval|$ROOT/runs/v184_retrieval_128/vbench_comparison/comparison_manifest.json|$ROOT/runs/v184_retrieval_128/vbench_comparison/published/all_coverage_retrieval|$ROOT/runs/v184_retrieval_128/metrics/vbench_long_parts/all_coverage_retrieval/dynamic_degree"
  "v185_pf_native|$ROOT/runs/v185_pf_baseline/vbench_comparison/comparison_manifest.json|$ROOT/runs/v185_pf_baseline/vbench_comparison/published/pf_native|$ROOT/runs/v185_pf_baseline/metrics/vbench_long_parts/pf_native/dynamic_degree"
  "v187_hybrid|$ROOT/runs/v187_hybrid_retrieval/vbench_comparison/comparison_manifest.json|$ROOT/runs/v187_hybrid_retrieval/vbench_comparison/published/pf_hybrid_retrieval|$ROOT/runs/v187_hybrid_retrieval/metrics/vbench_long_parts/pf_hybrid_retrieval/dynamic_degree"
)

for i in "${!TARGETS[@]}"; do
    IFS='|' read -r name manifest videos out <<<"${TARGETS[$i]}"
    mkdir -p "$out/cont"
    echo "[cont-dd] $name on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
        --vbench-root "$VBENCH_ROOT" \
        --comparison-manifest "$manifest" \
        --videos_path "$videos" \
        --dimension dynamic_degree \
        --output_path "$out/cont" \
        --full_json_dir "$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json" \
        --num_of_samples_per_prompt 1 --dev_flag \
        --local-models --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub" \
        > "$out/cont/run.log" 2>&1 &
done
wait
echo "=== all continuous dd runs done ==="
for t in "${TARGETS[@]}"; do
    IFS='|' read -r name _ _ out <<<"$t"
    f="$out/cont/continuous_flow_magnitude.json"
    if [ -f "$f" ]; then
        echo "$name: $(cat $f)"
    else
        echo "$name: MISSING continuous_flow_magnitude.json"
    fi
done
