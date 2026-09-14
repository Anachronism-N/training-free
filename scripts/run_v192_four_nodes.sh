#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODES=(28.7.187.25 28.7.192.14 28.7.192.121 28.7.187.70)
V191="$ROOT/runs/v191_head_phase_confirmation/confirm128/analysis/v191_head_phase_confirmation.json"
V205="$ROOT/runs/v205_horizon_confirmation/confirm128/analysis/v205_continuous_motion.json"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v192_remote_stage.sh' '$2' '${3:-0}' '${4:-4}' '${5:-seed2026_30s_128}'"
}
restore() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"${NODES[0]}" \
        "cd '$ROOT' && if ! pgrep -f '[g]pu_occupier.py' >/dev/null; then nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 & fi" || true
    for ip in "${NODES[@]:1}"; do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
            "if ! pgrep -f '^bash $AUTO_HOLDER' >/dev/null; then nohup bash '$AUTO_HOLDER' > /tmp/gpu_auto_holder.log 2>&1 & fi" || true
    done
}
trap restore EXIT
while [[ ! -s "$V191" || ! -s "$V205" ]]; do sleep 180; done
# Four fully free eight-GPU nodes are required by the frozen v192 contract.
for ip in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids; pkill -f '[g]pu_occupier.py' 2>/dev/null || true" || true
done
sleep 10
remote "${NODES[0]}" prepare 0 4
remote "${NODES[0]}" smoke 0 1 seed2026_30s_128
pids=()
for rank in 0 1 2 3; do remote "${NODES[$rank]}" generate-all "$rank" 4 & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" audit-all 0 4
for scope in seed2026_30s_128 long60_seed10000_32; do
    remote "${NODES[0]}" vbench-prepare 0 4 "$scope"
    pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-split "$rank" 4 "$scope" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
    pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-eval "$rank" 4 "$scope" & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
    remote "${NODES[0]}" vbench-collect 0 4 "$scope"
    remote "${NODES[0]}" vbench-decision 0 4 "$scope"
done
remote "${NODES[0]}" package 0 4
echo "[v192-controller] COMPLETE $(date)"
