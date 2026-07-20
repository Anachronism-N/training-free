#!/usr/bin/env bash
set -euo pipefail

# Run head classification diagnostic with REAL inference data
# Uses GPU 4 (free during 32-prompt ablation on GPU 0-3)
# Runs 1 prompt, 120 frames (30s), with HEAD_DIAGNOSTIC=1 env var

ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF_CONFIG="$ROOT/third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
PROMPT_FILE="$ROOT/prompts/review_3_prompts.txt"
OUTPUT_DIR="$ROOT/runs/head_diagnostic/videos"

mkdir -p "$OUTPUT_DIR"

cd "$ROOT/third_party/Pyramid-Forcing"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/Pyramid-Forcing:${PYTHONPATH:-}"

# Run inference with diagnostic enabled
# gate=0.0 so memory doesn't affect output, but archive is built for measurement
CUDA_VISIBLE_DEVICES=4 HEAD_DIAGNOSTIC=1 python inference.py \
    --config_path "$PF_CONFIG" \
    --output_folder "$OUTPUT_DIR" \
    --checkpoint_path "$ROOT/third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt" \
    --data_path "$PROMPT_FILE" \
    --num_output_frames 120 \
    --seed 0 \
    --num_samples 1 \
    --use_ema \
    --save_with_index \
    --pyramidkv_structured_memory \
    --pyramidkv_structured_memory_storage_mode archive \
    --pyramidkv_structured_memory_archive_max_frames 64 \
    --pyramidkv_structured_memory_top_k_frames 3 \
    --pyramidkv_structured_memory_readout_gate 0.0 \
    --pyramidkv_structured_memory_layer_start 15 \
    --pyramidkv_structured_memory_layer_end 21 \
    2>&1 | tee "$ROOT/logs/head_diagnostic.log"

echo "Diagnostic inference complete."
echo "Check runs/head_diagnostic/diagnostic_report.json for results."
