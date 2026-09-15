#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
H=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
restore() { pgrep -f "^bash $H" >/dev/null || nohup bash "$H" > /tmp/gpu_auto_holder.log 2>&1 & }
trap restore EXIT
pids=$(pgrep -f "^bash $H" || true); [ -z "$pids" ] || kill $pids
pids=$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z "$pids" ] || kill $pids
sleep 5
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=4 RUN_UNIT_TESTS=0 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh prepare
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 RUN_UNIT_TESTS=0 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh smoke
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=1 RUN_UNIT_TESTS=0 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh audit-smoke
