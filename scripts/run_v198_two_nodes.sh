#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODE0=28.7.187.25
NODE1=28.7.192.14
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v198_remote_stage.sh' '$2' '${3:-0}' '${4:-1}'"
}

echo "[v198-controller] audit-all $(date)"
remote "$NODE0" audit-all 0 1
echo "[v198-controller] prepare $(date)"
remote "$NODE0" prepare 0 2

echo "[v198-controller] split $(date)"
remote "$NODE0" split 0 2 & p0=$!
remote "$NODE1" split 1 2 & p1=$!
wait "$p0"; wait "$p1"

echo "[v198-controller] eval $(date)"
remote "$NODE0" eval 0 2 & p0=$!
remote "$NODE1" eval 1 2 & p1=$!
wait "$p0"; wait "$p1"

echo "[v198-controller] collect $(date)"
remote "$NODE0" collect 0 2
echo "[v198-controller] core complete $(date)"
