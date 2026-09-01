#!/bin/bash
ROOT="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
PF="$ROOT/third_party/Pyramid-Forcing"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/src:$ROOT:$PF:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LIFECACHE_ENABLE=0 STRUCTURED_MEMORY_ENABLE=0 COMMIT_FORCING_ENABLE=0
export HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0 SCENE_TRANSITION_RESET=0
cd "$PF"
MISSING=(0 16 32 48 64 80 96 112)
for i in "${!MISSING[@]}"; do
    p="${MISSING[$i]}"
    CUDA_VISIBLE_DEVICES=$i python inference.py \
        --config_path "$PF/configs/pyramid-forcing.yaml" \
        --checkpoint_path /apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt \
        --data_path "$ROOT/prompts/moviegen_128_full.txt" \
        --output_folder "$ROOT/runs/v187_hybrid_retrieval/raw" \
        --num_output_frames 120 --seed 0 --num_samples 1 \
        --use_ema --save_with_index --reseed_per_prompt --skip_existing \
        --end_idx 128 --prompt_stride 128 --prompt_offset $p \
        --pyramidkv_pf_hybrid_retrieval \
        > "$ROOT/runs/v187_hybrid_retrieval/logs/backfill_$p.log" 2>&1 &
done
wait
echo "backfill done"
