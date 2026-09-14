#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
READY="$ROOT/runs/v201_head_phase_horizon_sf_screen/status/smoke_ready"
while [[ ! -f "$READY" ]]; do sleep 60; done
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
restore_holder() {
    if [[ -x "$AUTO_HOLDER" ]] && ! pgrep -f "^bash $AUTO_HOLDER" >/dev/null 2>&1; then
        nohup bash "$AUTO_HOLDER" > /tmp/gpu_auto_holder.log 2>&1 &
    fi
}
trap restore_holder EXIT
holder_pids=$(pgrep -f "^bash $AUTO_HOLDER" || true)
[[ -z "$holder_pids" ]] || kill $holder_pids 2>/dev/null || true
dummy_pids=$(pgrep -f "^python3 .*/dummy_train.py" || true)
[[ -z "$dummy_pids" ]] || kill $dummy_pids 2>/dev/null || true
sleep 5
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
METHODS=retrieval_all_coverage,retrieval_static_top10,retrieval_horizon_top10,retrieval_horizon_shift_top10 \
    PF_CHECKPOINT=/tmp/self_forcing_dmd.pt NODE_RANK=0 NUM_NODES=1 \
    GPU_LIST=4,5,6,7 RUN_UNIT_TESTS=0 \
    bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh generate32
echo "[v201-node231] assigned methods complete $(date)"
