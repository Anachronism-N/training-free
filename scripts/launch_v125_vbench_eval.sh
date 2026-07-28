#!/bin/bash
# VBench-Long evaluation launcher for all 4 nodes
# Run this script after split completes
REPO=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
CONDA="source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh && conda activate longlive"
ENV="export REPO_ROOT=$REPO && export PYTHONPATH=$REPO/src:$REPO/third_party/Pyramid-Forcing:$REPO/scripts && export NUM_NODES=4"

# Kill all occupiers first
for ip in 29.232.240.221 29.127.50.121 29.232.228.21; do
    ssh -o ConnectTimeout=5 $ip "for pid in \$(pgrep -f '[g]pu_occupier'); do kill -9 \$pid 2>/dev/null; done; rm -f /tmp/gpu_occupier.pid" 2>/dev/null
    echo "$ip: occupier killed"
done
for pid in $(pgrep -f '[g]pu_occupier'); do kill -9 $pid 2>/dev/null; done
rm -f /tmp/gpu_occupier.pid
echo "local: occupier killed"

# Run preflight
echo "=== VBench preflight ==="
$CONDA && cd $REPO && $ENV && NODE_RANK=0 GPU_LIST=0 bash scripts/run_v125_vbench_long.sh preflight 2>&1 | tail -5

# Launch eval on all 4 nodes
echo "=== Launching VBench eval on 4 nodes ==="
# node42 (rank 0)
setsid bash -c "$CONDA && cd $REPO && $ENV && NODE_RANK=0 GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v125_vbench_long.sh eval" </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_node0.log 2>&1 &
echo "node42 eval PID=$!"

# node221 (rank 1)
ssh -o ConnectTimeout=10 29.232.240.221 "setsid bash -c '$CONDA && cd $REPO && $ENV && NODE_RANK=1 GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v125_vbench_long.sh eval' </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_node1.log 2>&1 &" 2>/dev/null
echo "node221 eval launched"

# node121 (rank 2)
ssh -o ConnectTimeout=10 29.127.50.121 "setsid bash -c '$CONDA && cd $REPO && $ENV && NODE_RANK=2 GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v125_vbench_long.sh eval' </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_node2.log 2>&1 &" 2>/dev/null
echo "node121 eval launched"

# node21 (rank 3)
ssh -o ConnectTimeout=10 29.232.228.21 "setsid bash -c '$CONDA && cd $REPO && $ENV && NODE_RANK=3 GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v125_vbench_long.sh eval' </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_node3.log 2>&1 &" 2>/dev/null
echo "node21 eval launched"

echo "=== All 4 nodes eval launched ==="
