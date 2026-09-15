#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODES=(28.7.187.25 28.7.192.14 28.7.192.121 28.7.187.70)
PARITY="$ROOT/runs/v207_context_budget_phase_recovery/parity/report/parity_report.json"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v207_official_remote.sh' '$2' '${3:-0}' '${4:-4}'"
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
while [[ ! -s "$PARITY" ]]; do sleep 60; done
python3 - "$PARITY" <<'PY'
import json, sys
report=json.load(open(sys.argv[1]))
if report.get('parity_pass') is not True:
    raise SystemExit('v207 parity gate failed: '+str(report.get('decision')))
print('[v207-controller] parity gate passed')
PY
for ip in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "pkill -f '[g]pu_occupier.py' 2>/dev/null || true; pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true
done
sleep 10
remote "${NODES[0]}" prepare 0 4
remote "${NODES[0]}" smoke 0 1
remote "${NODES[0]}" audit-smoke 0 1
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" generate32 "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" audit-screen 0 4
remote "${NODES[0]}" efficiency 0 4
remote "${NODES[0]}" vbench-prepare 0 4
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-split "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-eval "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-collect 0 4
remote "${NODES[0]}" vbench-decision 0 4
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-motion-compute "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-motion-collect 0 4
remote "${NODES[0]}" vbench-package 0 4
remote "${NODES[0]}" package 0 4
echo "[v207-controller] COMPLETE $(date)"
