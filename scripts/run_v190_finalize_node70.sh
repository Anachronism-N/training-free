#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
RAW="$ROOT/runs/v190_head_phase_causal_screen/screen32/raw/retrieval_head_only"
while true; do
    count=$(ls "$RAW"/*.mp4 2>/dev/null | wc -l)
    echo "[v190-finalizer] head_only=$count/32 $(date)"
    [[ "$count" -ge 32 ]] && break
    sleep 120
done
while pgrep -f "[r]un_v201_head_phase_horizon_screen_32gpu.sh|[p]ython inference.py.*v201_head_phase_horizon" >/dev/null 2>&1; do
    echo "[v190-finalizer] waiting for v201 GPU work on this node $(date)"
    sleep 120
done
NODE_RANK=0 NUM_NODES=1 RUN_UNIT_TESTS=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-screen
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh prepare
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh split
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v190_vbench_long.sh eval
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh decision
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh package
echo "[v190-finalizer] COMPLETE $(date)"
