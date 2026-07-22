#!/bin/bash
# HREM paper experiment: multi-prompt A-B-A on Self-Forcing
# 5 soft-transition prompts, native vs HREM, seed 0
set -uo pipefail

REPO="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free"
# Activate conda directly (bypasses activate_conda_gy2.sh path issue)
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export LD_LIBRARY_PATH="/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/lib:${LD_LIBRARY_PATH}"

SF="$REPO/third_party/Self-Forcing"
CKPT="$SF/checkpoints/self_forcing_dmd.pt"
mkdir -p "$REPO/runs/hrem_paper"

PFILE="$REPO/prompts/aba_multi_soft.txt"

run() {
    local name="$1" idx="$2" use_hr="$3"
    local out="$REPO/runs/hrem_paper/${name}_p${idx}"
    local log="$REPO/runs/hrem_paper/${name}_p${idx}.log"
    mkdir -p "$out"
    echo "[$name p$idx] running..."
    (
        cd "$SF"
        export PYTHONPATH="$REPO/src:$SF/scripts"
        if [ "$use_hr" = "1" ]; then
            export HEAD_ROLE_ENABLE=1
            export HEAD_ROLE_SPLIT_MODE=fixed
        fi
        CUDA_VISIBLE_DEVICES=1 python inference.py \
            --config_path "$SF/configs/self_forcing_dmd.yaml" \
            --output_folder "$out" \
            --checkpoint_path "$CKPT" \
            --data_path "$PFILE" \
            --num_output_frames 120 --seed 0 --num_samples 1 \
            --use_ema --save_with_index --start_idx "$idx" --end_idx $((idx+1))
    ) > "$log" 2>&1
    local rc=$?
    echo "[$name p$idx] done rc=$rc => $out"
}

for idx in 0 1 2 3 4; do
    run "native" $idx 0
    run "hrem" $idx 1
done
echo "ALL DONE"
