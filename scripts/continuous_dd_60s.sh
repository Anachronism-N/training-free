#!/bin/bash
# Continuous flow-magnitude metric for 60s methods (v181 + v186)
set -euo pipefail
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"

V181="$ROOT/runs/v181_rccp_long_stress/scopes/long60_seed0"
V186="$ROOT/runs/v186_long60_comparison"

TARGETS=(
  "v181_sf_native|0|$V181/vbench_comparison/comparison_manifest.json|$V181/vbench_comparison/published/sf_native|$V181/metrics/vbench_long_parts/sf_native/dynamic_degree/cont"
  "v181_rccp_matched|1|$V181/vbench_comparison/comparison_manifest.json|$V181/vbench_comparison/published/rccp_matched|$V181/metrics/vbench_long_parts/rccp_matched/dynamic_degree/cont"
  "v181_all_recent|2|$V181/vbench_comparison/comparison_manifest.json|$V181/vbench_comparison/published/all_recent|$V181/metrics/vbench_long_parts/all_recent/dynamic_degree/cont"
  "v186_pf_native|3|$V186/vbench_comparison/comparison_manifest.json|$V186/vbench_comparison/published/pf_native|$V186/metrics/vbench_long_parts/pf_native/dynamic_degree/cont"
  "v186_retrieval|4|$V186/vbench_comparison/comparison_manifest.json|$V186/vbench_comparison/published/all_coverage_retrieval|$V186/metrics/vbench_long_parts/all_coverage_retrieval/dynamic_degree/cont"
)

for t in "${TARGETS[@]}"; do
    IFS='|' read -r name gpu manifest videos out <<<"$t"
    mkdir -p "$out"
    echo "[cont-dd-60s] $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
        --vbench-root "$VBENCH_ROOT" \
        --comparison-manifest "$manifest" \
        --videos_path "$videos" \
        --dimension dynamic_degree \
        --output_path "$out" \
        --full_json_dir "$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json" \
        --num_of_samples_per_prompt 1 --dev_flag \
        --local-models --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub" \
        > "$out/run.log" 2>&1 &
done
wait
echo "=== all 60s continuous dd runs done ==="
for t in "${TARGETS[@]}"; do
    IFS='|' read -r name _ _ _ out <<<"$t"
    f="$out/continuous_flow_magnitude.json"
    [ -f "$f" ] && echo "$name: $(cat $f)" || echo "$name: MISSING"
done
