#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
HEAD_PID="${1:-3594864}"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
restore_holder() {
    if [[ -x "$AUTO_HOLDER" ]] && ! pgrep -f "^bash $AUTO_HOLDER" >/dev/null 2>&1; then
        nohup bash "$AUTO_HOLDER" > /tmp/gpu_auto_holder.log 2>&1 &
    fi
}
trap restore_holder EXIT
while kill -0 "$HEAD_PID" 2>/dev/null; do sleep 60; done
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
# Regenerate the corrupted local-control videos and overwrite stale OOM logs.
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt METHODS=all_recent FORCE=1 NODE_RANK=0 \
    NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 RUN_UNIT_TESTS=0 \
    bash scripts/run_v190_head_phase_causal_screen_32gpu.sh generate32
NODE_RANK=0 NUM_NODES=1 RUN_UNIT_TESTS=0 \
    bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-screen
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh prepare
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh split
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
    bash scripts/run_v190_vbench_long.sh eval
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_vbench_long.sh decision
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh package
echo "[v190-repair-finalize] COMPLETE $(date)"
