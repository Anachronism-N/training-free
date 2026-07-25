#!/usr/bin/env bash
set -x
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Pyramid-Forcing || exit 1
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate longlive || exit 1
export PYTHONPATH=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/src:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Pyramid-Forcing:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0

LOG=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs/v90_priority_factorization_screen/logs/pf_age_only_restart.log

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/pyramid-forcing.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/prompts/v86_single_long_complex_16.txt \
    --output_folder /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs/v90_priority_factorization_screen/pf_age_only \
    --num_output_frames 120 --seed 0 --num_samples 1 --use_ema --save_with_index \
    --start_idx 11 --end_idx 16 \
    --pyramidkv_cache_transition \
    --pyramidkv_cache_transition_mode full \
    --pyramidkv_cache_transition_min_reliability .55 \
    --pyramidkv_cache_transition_min_novelty .01 \
    --pyramidkv_cache_transition_max_commit_fraction .75 \
    --pyramidkv_cache_transition_stagger_period 1 \
    --pyramidkv_cache_transition_max_age_blocks 6 \
    --pyramidkv_cache_transition_branches both \
    --pyramidkv_cache_transition_denoise_weight 2 \
    --pyramidkv_cache_transition_trace_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs/v90_priority_factorization_screen/traces/pf_age_only.transition.jsonl \
    --pyramidkv_cache_transition_debug \
    --pyramidkv_cache_transition_role_conditioning \
    --pyramidkv_cache_transition_role_config_path /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/runs/v90_priority_factorization_screen/labels/pf_binary.csv \
    --pyramidkv_cache_transition_persistent_label 1 \
    --pyramidkv_cache_transition_reactive_labels=-1 \
    --pyramidkv_cache_transition_persistent_min_novelty_scale 1 \
    --pyramidkv_cache_transition_reactive_min_novelty_scale 1 \
    --pyramidkv_cache_transition_persistent_max_age_blocks 8 \
    --pyramidkv_cache_transition_reactive_max_age_blocks 4 \
    --pyramidkv_cache_transition_reactive_utility_bias 0 \
    --pyramidkv_cache_transition_role_layer_start 0 \
    --pyramidkv_cache_transition_role_layer_end -1 \
    >"$LOG" 2>&1
