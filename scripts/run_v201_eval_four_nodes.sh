#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODES=(28.7.187.25 28.7.192.14 28.7.192.121 28.7.192.231)
GPUS=(0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7 4,5,6,7)
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v201_remote_stage.sh' '$2' '$3' '4' '$4'"
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
sleep 10
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-split "$rank" "${GPUS[$rank]}" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-eval "$rank" "${GPUS[$rank]}" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-collect 0 "${GPUS[0]}"
remote "${NODES[0]}" vbench-decision 0 "${GPUS[0]}"
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-motion-compute "$rank" "${GPUS[$rank]}" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-motion-collect 0 "${GPUS[0]}"
remote "${NODES[0]}" vbench-motion-analyze 0 "${GPUS[0]}"
remote "${NODES[0]}" vbench-package 0 "${GPUS[0]}"
remote "${NODES[0]}" package 0 "${GPUS[0]}"
echo "[v201-eval-four] COMPLETE $(date)"
