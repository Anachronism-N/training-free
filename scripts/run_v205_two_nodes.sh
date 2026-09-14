#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODE0=28.7.192.121
NODE1=28.7.192.231
DECISION="$ROOT/runs/v201_head_phase_horizon_sf_screen/screen32/analysis/v201_head_phase_horizon.json"
MOTION="$ROOT/runs/v201_head_phase_horizon_sf_screen/screen32/analysis/v203_v201_continuous_motion.json"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v205_remote_stage.sh' '$2' '${3:-0}' '${4:-1}' '${5:-0,1,2,3,4,5,6,7}'"
}
restore() {
    for ip in "$NODE0" "$NODE1"; do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
            "if ! pgrep -f '^bash $AUTO_HOLDER' >/dev/null; then nohup bash '$AUTO_HOLDER' > /tmp/gpu_auto_holder.log 2>&1 & fi" || true
    done
}
trap restore EXIT
while [[ ! -s "$DECISION" || ! -s "$MOTION" ]]; do sleep 120; done
for ip in "$NODE0" "$NODE1"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true
done
sleep 10
remote "$NODE0" prepare 0 2
remote "$NODE0" smoke 0 1 0,1,2
remote "$NODE0" audit-smoke 0 1
remote "$NODE0" generate128 0 2 & p0=$!
remote "$NODE1" generate128 1 2 4,5,6,7 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" audit-confirm 0 2
remote "$NODE0" vbench-prepare 0 2
remote "$NODE0" vbench-split 0 2 & p0=$!
remote "$NODE1" vbench-split 1 2 4,5,6,7 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" vbench-eval 0 2 & p0=$!
remote "$NODE1" vbench-eval 1 2 4,5,6,7 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" vbench-collect 0 2
remote "$NODE0" vbench-decision 0 2
remote "$NODE0" vbench-motion-compute 0 2 & p0=$!
remote "$NODE1" vbench-motion-compute 1 2 4,5,6,7 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" vbench-motion-collect 0 2
remote "$NODE0" vbench-motion-analyze 0 2
remote "$NODE0" vbench-package 0 2
remote "$NODE0" package 0 1
echo "[v205-controller] COMPLETE $(date)"
