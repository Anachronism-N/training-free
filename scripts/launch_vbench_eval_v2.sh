#!/bin/bash
# Launch VBench-Long eval on all 4 nodes with correct NODE_RANK
REPO=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
CACHE=$REPO/runs/v125_moviebench128_main/vbench_cache
CONDA_SH=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh

# Kill any remaining occupiers
for ip in 29.232.240.221 29.127.50.121 29.232.228.21; do
    ssh -o ConnectTimeout=5 $ip "for pid in \$(pgrep -f '[g]pu_occupier'); do kill -9 \$pid 2>/dev/null; done; rm -f /tmp/gpu_occupier.pid" 2>/dev/null
done
for pid in $(pgrep -f '[g]pu_occupier'); do kill -9 $pid 2>/dev/null; done
rm -f /tmp/gpu_occupier.pid
echo "occupiers killed"

# Launch eval on node42 (rank 0)
source $CONDA_SH && conda activate longlive && cd $REPO
export REPO_ROOT=$REPO
export PYTHONPATH=$REPO/src:$REPO/third_party/Pyramid-Forcing:$REPO/scripts
export NUM_NODES=4 NODE_RANK=0 GPU_LIST=0,1,2,3,4,5,6,7
export VBENCH_CACHE_DIR=$CACHE
setsid bash scripts/run_v125_vbench_long.sh eval </dev/null >runs/v125_moviebench128_main/comparison_quality8/eval_r0.log 2>&1 &
echo "node42 rank0 PID=$!"

# Launch eval on node221 (rank 1)
ssh -o ConnectTimeout=10 29.232.240.221 "source $CONDA_SH && conda activate longlive && cd $REPO && export REPO_ROOT=$REPO && export PYTHONPATH=$REPO/src:$REPO/third_party/Pyramid-Forcing:$REPO/scripts && export NUM_NODES=4 NODE_RANK=1 GPU_LIST=0,1,2,3,4,5,6,7 && export VBENCH_CACHE_DIR=$CACHE && setsid bash scripts/run_v125_vbench_long.sh eval </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_r1.log 2>&1 & echo launched" 2>&1
echo "node221 rank1 launched"

# Launch eval on node121 (rank 2)
ssh -o ConnectTimeout=10 29.127.50.121 "source $CONDA_SH && conda activate longlive && cd $REPO && export REPO_ROOT=$REPO && export PYTHONPATH=$REPO/src:$REPO/third_party/Pyramid-Forcing:$REPO/scripts && export NUM_NODES=4 NODE_RANK=2 GPU_LIST=0,1,2,3,4,5,6,7 && export VBENCH_CACHE_DIR=$CACHE && setsid bash scripts/run_v125_vbench_long.sh eval </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_r2.log 2>&1 & echo launched" 2>&1
echo "node121 rank2 launched"

# Launch eval on node21 (rank 3)
ssh -o ConnectTimeout=10 29.232.228.21 "source $CONDA_SH && conda activate longlive && cd $REPO && export REPO_ROOT=$REPO && export PYTHONPATH=$REPO/src:$REPO/third_party/Pyramid-Forcing:$REPO/scripts && export NUM_NODES=4 NODE_RANK=3 GPU_LIST=0,1,2,3,4,5,6,7 && export VBENCH_CACHE_DIR=$CACHE && setsid bash scripts/run_v125_vbench_long.sh eval </dev/null >$REPO/runs/v125_moviebench128_main/comparison_quality8/eval_r3.log 2>&1 & echo launched" 2>&1
echo "node21 rank3 launched"

echo "=== All 4 nodes eval launched ==="
