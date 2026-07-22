#!/bin/bash
# Quick SF test with longlive conda env
set -uo pipefail
source /apdcephfs_gy2/share_303214315/cedricnie/activate_conda_gy2.sh

# Fix: remove torch-base lib path, add system lib paths
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|:/opt/conda/envs/torch-base/lib||g')
export LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/src:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing/scripts" \
python inference.py \
    --config_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing/configs/self_forcing_dmd.yaml \
    --output_folder /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs/hrem_smoke/test_longlive \
    --checkpoint_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt \
    --data_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/prompts/sf_simple_test.txt \
    --num_output_frames 30 \
    --seed 0 \
    --num_samples 1 \
    --use_ema \
    --save_with_index \
    2>&1 | tail -20
