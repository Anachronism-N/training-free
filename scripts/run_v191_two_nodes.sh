#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODE0=28.7.187.25
NODE1=28.7.192.14
DECISION="$ROOT/runs/v190_head_phase_causal_screen/screen32/analysis/v190_head_phase_causal_screen.json"
V198="$ROOT/runs/v198_audited_long60/analysis/v198_long60_operator.json"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v191_remote_stage.sh' '$2' '${3:-0}' '${4:-1}' '${5:-0,1,2,3,4,5,6,7}'"
}
restore() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$NODE0" \
        "cd '$ROOT' && if ! pgrep -f '[g]pu_occupier.py' >/dev/null; then nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 & fi" || true
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$NODE1" \
        "if ! pgrep -f '^bash $AUTO_HOLDER' >/dev/null; then nohup bash '$AUTO_HOLDER' > /tmp/gpu_auto_holder.log 2>&1 & fi" || true
}
trap restore EXIT
while [[ ! -s "$DECISION" || ! -s "$V198" ]]; do sleep 120; done
# Stop only known holder programs; never touch unknown GPU workloads.
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$NODE0" \
    "pkill -f '[g]pu_occupier.py' 2>/dev/null || true; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$NODE1" \
    "pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true
sleep 10
remote "$NODE0" prepare 0 2
remote "$NODE0" smoke 0 1 0,1,2
remote "$NODE0" audit-smoke 0 1
remote "$NODE0" generate128 0 2 & p0=$!
remote "$NODE1" generate128 1 2 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" audit-confirm 0 2
remote "$NODE0" vbench-prepare 0 2
remote "$NODE0" vbench-split 0 2 & p0=$!
remote "$NODE1" vbench-split 1 2 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" vbench-eval 0 2 & p0=$!
remote "$NODE1" vbench-eval 1 2 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" vbench-collect 0 2
remote "$NODE0" vbench-decision 0 2
remote "$NODE0" package 0 2
echo "[v191-controller] COMPLETE $(date)"
