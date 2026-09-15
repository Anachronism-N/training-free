#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
V209_OUTPUT_ROOT="${V209_OUT_ROOT:-$ROOT/runs/v209_sf_protocol_budget}"
restore_holder() {
    cd "$ROOT"
    rm -f /tmp/gpu_occupier.pid
    pgrep -f '[g]pu_occupier.py' >/dev/null || nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 &
}
trap restore_holder EXIT
pkill -f '[g]pu_occupier.py' 2>/dev/null || true
sleep 5
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
V209_OUT_ROOT="$V209_OUTPUT_ROOT" SHARED_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh prepare
V209_OUT_ROOT="$V209_OUTPUT_ROOT" SHARED_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-native
V209_OUT_ROOT="$V209_OUTPUT_ROOT" SHARED_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
cat "$V209_OUTPUT_ROOT/canary_production/analysis/decision.md"
V209_OUT_ROOT="$V209_OUTPUT_ROOT" SHARED_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-runtime
V209_OUT_ROOT="$V209_OUTPUT_ROOT" SHARED_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
cat "$V209_OUTPUT_ROOT/canary_production/analysis/decision.md"
