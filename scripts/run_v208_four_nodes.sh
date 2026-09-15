#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
NODES=(28.7.187.25 28.7.192.14 28.7.192.121 28.7.187.70)
V207="$ROOT/runs/v207_context_budget_phase_recovery/screen32/analysis/v207_context_budget_phase.json"
PARITY="$ROOT/runs/v207_context_budget_phase_recovery/parity/report/parity_report.json"
AUTO_HOLDER=/apdcephfs_cq12/share_1297902/wtaoozhong/gpu_holder_tool/gpu_auto_holder.sh
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$1" \
        "bash '$ROOT/scripts/run_v208_remote_stage.sh' '$2' '${3:-0}' '${4:-4}' '${5:-main30}'"
}
restore() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"${NODES[0]}" \
        "cd '$ROOT'; rm -f /tmp/gpu_occupier.pid; pgrep -f '[g]pu_occupier.py' >/dev/null || nohup /opt/conda/envs/torch-base/bin/python scripts/gpu_occupier.py > /tmp/gpu_occupier.log 2>&1 &" || true
    for ip in "${NODES[@]:1}"; do ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
        "pgrep -f '^bash $AUTO_HOLDER' >/dev/null || nohup bash '$AUTO_HOLDER' > /tmp/gpu_auto_holder.log 2>&1 &" || true; done
}
trap restore EXIT
while [[ ! -s "$V207" || ! -s "$PARITY" ]]; do sleep 120; done
python3 - "$PARITY" "$V207" <<'PY'
import json,sys
parity=json.load(open(sys.argv[1])); report=json.load(open(sys.argv[2]))
if parity.get('parity_pass') is not True: raise SystemExit('v207 parity did not pass')
if len(report.get('selected_for_fresh128') or []) != 1: raise SystemExit('v207 selected no unique candidate')
PY
for ip in "${NODES[@]}"; do ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$ip" \
    "pkill -f '[g]pu_occupier.py' 2>/dev/null || true; pids=\$(pgrep -f '^bash $AUTO_HOLDER' || true); [ -z \"\$pids\" ] || kill \$pids; pids=\$(pgrep -f '^python3 .*/dummy_train.py' || true); [ -z \"\$pids\" ] || kill \$pids" || true; done
sleep 10
remote "${NODES[0]}" prepare 0 4
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" generate30 "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" audit30 0 4
remote "${NODES[0]}" vbench-prepare 0 4 main30
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-split "$rank" 4 main30 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-eval "$rank" 4 main30 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-collect 0 4 main30
remote "${NODES[0]}" vbench-decision 0 4 main30
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-motion-compute "$rank" 4 main30 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-motion-collect 0 4 main30
remote "${NODES[0]}" vbench-motion-analyze 0 4 main30
remote "${NODES[0]}" vbench-package 0 4 main30
remote "${NODES[0]}" package 0 4
# Long60 is run after main30 regardless of sign, per frozen reporting rule.
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" generate60 "$rank" 4 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" audit60 0 4
remote "${NODES[0]}" vbench-prepare 0 4 long60
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-split "$rank" 4 long60 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
pids=(); for rank in 0 1 2 3; do remote "${NODES[$rank]}" vbench-eval "$rank" 4 long60 & pids+=("$!"); done; for pid in "${pids[@]}"; do wait "$pid"; done
remote "${NODES[0]}" vbench-collect 0 4 long60
remote "${NODES[0]}" vbench-decision 0 4 long60
remote "${NODES[0]}" vbench-package 0 4 long60
remote "${NODES[0]}" package 0 4
echo "[v208-controller] COMPLETE $(date)"
