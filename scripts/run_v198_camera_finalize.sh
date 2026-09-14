#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODE0=28.7.187.25
NODE1=28.7.192.14
REPORT="$ROOT/runs/v198_audited_long60/analysis/v198_long60_operator.json"
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v198_remote_stage.sh' '$2' '${3:-0}' '${4:-1}'"
}
while [[ ! -s "$REPORT" ]]; do sleep 60; done
remote "$NODE0" camera-compute 0 2 & p0=$!
remote "$NODE1" camera-compute 1 2 & p1=$!
wait "$p0"; wait "$p1"
remote "$NODE0" camera-collect 0 2
remote "$NODE0" decision 0 2
remote "$NODE0" package 0 2
echo "[v198-camera-finalize] COMPLETE $(date)"
