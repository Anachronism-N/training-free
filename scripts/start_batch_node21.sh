#!/bin/bash
export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd $REPO_ROOT
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT/third_party/Pyramid-Forcing:$REPO_ROOT/scripts
OUT_ROOT=$REPO_ROOT/runs/v125_moviebench128_main/ours6_9434cf7084d6
PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
python scripts/batch_inference_runner.py --rank 0 --num-nodes 4 --gpu 0 \
  --out-root $OUT_ROOT --prompts $PROMPTS \
  --methods sf_native,pf_native,ours_landmark_motion1,ours_landmark_retrieval1_age24 \
  > $OUT_ROOT/logs/batch_r0_gpu0.log 2>&1 &
python scripts/batch_inference_runner.py --rank 0 --num-nodes 4 --gpu 1 \
  --out-root $OUT_ROOT --prompts $PROMPTS \
  --methods ours_landmark_retrieval_motion,ours_prototype_motion1,ours_prototype_retrieval1_age24,ours_prototype_retrieval_motion \
  > $OUT_ROOT/logs/batch_r0_gpu1.log 2>&1 &
wait
echo "ALL BATCH DONE rank=0"
