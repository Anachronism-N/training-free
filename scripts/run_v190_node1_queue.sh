#!/bin/bash
# node1 queue: after retrieval_all_coverage, run membership_shift
R=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
# wait for current method to finish
while pgrep -f "inference.py" > /dev/null 2>&1; do sleep 60; done
for m in retrieval_membership_shift retrieval_dense_phase; do
    n=$(ls $R/runs/v190_head_phase_causal_screen/screen32/raw/$m/*.mp4 2>/dev/null | wc -l)
    [ "$n" -ge 32 ] && { echo "[skip] $m $n/32"; continue; }
    echo "[queue] starting $m $(date)"
    bash $R/scripts/run_v190_method.sh $m
    echo "[queue] $m done $(date)"
done
echo "NODE1 QUEUE DONE"
