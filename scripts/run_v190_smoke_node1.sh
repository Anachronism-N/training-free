#!/bin/bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/src:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Pyramid-Forcing:/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/third_party/Self-Forcing:${PYTHONPATH:-}"
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3 RUN_UNIT_TESTS=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh smoke
