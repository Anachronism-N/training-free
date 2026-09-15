#!/bin/bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/src:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Pyramid-Forcing:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing:${PYTHONPATH:-}"
NODE_RANK=$1 NUM_NODES=2 GPU_LIST=0,1,2,3,4,5,6,7 RUN_UNIT_TESTS=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh profile128
