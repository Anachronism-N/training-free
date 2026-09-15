#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
restore_holder() {
    cd "$ROOT"
    rm -f /tmp/gpu_occupier.pid
    if ! pgrep -f '[g]pu_occupier.py' >/dev/null 2>&1; then
        nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 &
    fi
}
trap restore_holder EXIT
pkill -f '[g]pu_occupier.py' 2>/dev/null || true
sleep 5
cd "$ROOT"
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:$ROOT/third_party/Pyramid-Forcing:$ROOT/third_party/Self-Forcing:${PYTHONPATH:-}"
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=1 \
    bash scripts/run_v207_sf_parity.sh prepare
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=1 GPU_LIST=0,1,2 \
    bash scripts/run_v207_sf_parity.sh run
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=0 \
    bash scripts/run_v207_sf_parity.sh analyze
PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=0 \
    bash scripts/run_v207_sf_parity.sh package
cat "$ROOT/runs/v207_context_budget_phase_recovery/parity/report/parity_report.md"
