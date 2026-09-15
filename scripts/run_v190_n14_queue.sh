#!/bin/bash
# .14 queue: after retrieval_compatible, run phase_shift
R=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
while pgrep -f "inference.py" > /dev/null 2>&1; do sleep 60; done
for m in retrieval_phase_shift; do
    n=$(ls $R/runs/v190_head_phase_causal_screen/screen32/raw/$m/*.mp4 2>/dev/null | wc -l)
    [ "$n" -ge 32 ] && { echo "[skip] $m $n/32"; continue; }
    echo "[queue] starting $m $(date)"
    bash $R/scripts/run_v190_method.sh $m
    echo "[queue] $m done $(date)"
done
echo "N14 QUEUE DONE"
