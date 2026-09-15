#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
OUT=$ROOT/runs/v207_context_budget_phase_recovery_replica_node14
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
V207_OUT_ROOT="$OUT" PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=0 bash scripts/run_v207_sf_parity.sh prepare
V207_OUT_ROOT="$OUT" PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=0 GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh run
V207_OUT_ROOT="$OUT" PF_CHECKPOINT=/tmp/self_forcing_dmd.pt RUN_UNIT_TESTS=0 bash scripts/run_v207_sf_parity.sh analyze
cat "$OUT/parity/report/parity_report.md"
