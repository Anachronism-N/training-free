#!/bin/bash
# Kill old batch runner for retrieval_motion and restart with placeholder fix
for pid in $(pgrep -f '[i]nference.py.*retrieval_motion'); do kill -9 $pid 2>/dev/null; done
sleep 2
for pid in $(pgrep -f '[b]atch_inference.*retrieval_motion'); do kill -9 $pid 2>/dev/null; done
sleep 2
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export PYTHONPATH=$PWD/src:$PWD/third_party/Pyramid-Forcing:$PWD/scripts
O=$PWD/runs/v125_moviebench128_main/ours6_9434cf7084d6
P=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
CUDA_VISIBLE_DEVICES=0 nohup python scripts/batch_inference_runner.py --rank 0 --num-nodes 4 --gpu 0 --out-root $O --prompts $P --methods ours_prototype_retrieval_motion >$O/logs/batch_last_final.log 2>&1 &
echo "started PID=$!"
sleep 15
tail -5 $O/logs/batch_last_final.log 2>/dev/null
