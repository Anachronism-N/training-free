#!/usr/bin/env bash
set -euo pipefail
DIMENSION="$1"
GPU="$2"

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
VBENCH_ROOT="$ROOT/../research_sprint/bench_baselines/VBench"
COMPARISON_MANIFEST="$ROOT/runs/v184_retrieval_128/vbench_comparison/comparison_manifest.json"
VIDEOS_PATH="$ROOT/runs/v184_retrieval_128/vbench_comparison/published/all_coverage_retrieval"
FULL_JSON_DIR="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OUTPUT_BASE="$ROOT/runs/v184_retrieval_128/metrics/vbench_long_parts/all_coverage_retrieval"

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

output_dir="$OUTPUT_BASE/$DIMENSION"
mkdir -p "$output_dir"
echo "=== Evaluating $DIMENSION on GPU $GPU ==="
python "$ROOT/scripts/eval_vbench_long_prompt_aware.py" \
    --vbench-root "$VBENCH_ROOT" \
    --comparison-manifest "$COMPARISON_MANIFEST" \
    --videos_path "$VIDEOS_PATH" \
    --dimension "$DIMENSION" \
    --output_path "$output_dir" \
    --full_json_dir "$FULL_JSON_DIR" \
    --num_of_samples_per_prompt 1 \
    --dev_flag \
    --local-models \
    --torch-hub-dir "$ROOT/runs/_model_cache/torch_hub"
echo "=== Done: $DIMENSION ==="
