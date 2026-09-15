#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODES=(28.7.187.25 28.7.192.14 28.7.192.121)
OUT="$ROOT/runs/v207_horizon_seed_replication"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v207_remote_stage.sh' '$2' '${3:-0}' '${4:-3}' '${5:-0,1,2,3,4,5,6,7}'"
}
restore() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"${NODES[0]}" \
        "cd '$ROOT'; rm -f /tmp/gpu_occupier.pid; if ! pgrep -f '[g]pu_occupier.py' >/dev/null; then nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 & fi" || true
    for ip in "${NODES[@]:1}"; do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
            "if ! pgrep -f '^bash $AUTO_HOLDER' >/dev/null; then nohup bash '$AUTO_HOLDER' > /tmp/gpu_auto_holder.log 2>&1 & fi" || true
    done
}
trap restore EXIT
for ip in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "pkill -f '[g]pu_occupier.py' 2>/dev/null || true; pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true
done
# Prefer node-local model files when available; CEPH model reads can stall many
# simultaneous workers during initialization.
for ip in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "if [[ -d /tmp/Wan2.1-T2V-1.3B ]]; then mountpoint -q '$ROOT/third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B' || mount --bind /tmp/Wan2.1-T2V-1.3B '$ROOT/third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B'; fi" || true
done
sleep 10
remote "${NODES[0]}" smoke 0 1
remote "${NODES[0]}" audit-smoke 0 1
pids=(); for rank in 0 1 2; do remote "${NODES[$rank]}" generate32 "$rank" 3 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" audit-screen 0 1
remote "${NODES[0]}" vbench-prepare 0 3
pids=(); for rank in 0 1 2; do remote "${NODES[$rank]}" vbench-split "$rank" 3 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
pids=(); for rank in 0 1 2; do remote "${NODES[$rank]}" vbench-eval "$rank" 3 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-collect 0 3
remote "${NODES[0]}" vbench-decision 0 3
remote "${NODES[0]}" vbench-motion-compute 0 3 & p0=$!
remote "${NODES[1]}" vbench-motion-compute 1 3 & p1=$!
remote "${NODES[2]}" vbench-motion-compute 2 3 & p2=$!
wait "$p0"; wait "$p1"; wait "$p2"
remote "${NODES[0]}" vbench-motion-collect 0 3
remote "${NODES[0]}" vbench-motion-analyze 0 3
remote "${NODES[0]}" vbench-package 0 3
remote "${NODES[0]}" package 0 1
echo "[v207-controller] COMPLETE $(date)"
