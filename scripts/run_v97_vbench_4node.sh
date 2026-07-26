#!/usr/bin/env bash
# v97 vbench-long: 4-node × 4-method parallel launcher.
# Each node runs 4 methods on GPUs 0-3. All 16 methods run in one wave.
# Resume-aware: skips methods that already produced results.json.
set -uo pipefail

ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
RUN_ROOT="$ROOT/runs/v97_threshold_pf_merge32"
METRICS="$RUN_ROOT/metrics"
RUNNER="$ROOT/runs/v97_eval_smoke.sh"
LOGDIR="$RUN_ROOT/metrics/logs"
mkdir -p "$LOGDIR" "$METRICS/status" "$METRICS/vbench_long"

# Node assignments: "ip:gpu:method" — 4 methods per node, GPUs 0-3
ASSIGNMENTS=(
    # node 42 (local)
    "29.232.228.42:0:pf_native"
    "29.232.228.42:1:prompt_tau_0p0_merge"
    "29.232.228.42:2:prompt_tau_0p5_merge"
    "29.232.228.42:3:prompt_tau_1p0_merge"
    # node 221
    "29.232.240.221:0:prompt_tau_1p5_merge"
    "29.232.240.221:1:prompt_tau_2p0_merge"
    "29.232.240.221:2:prompt_tau_1p0_cyclic"
    "29.232.240.221:3:prompt_tau_1p0_recent"
    # node 121
    "29.127.50.121:0:prompt_tau_1p0_random_merge"
    "29.127.50.121:1:prompt_tau_1p0_reversed_merge"
    "29.127.50.121:2:sign_rpos_0p5_stride_merge"
    "29.127.50.121:3:pf_ar_stride_merge"
    # node 21
    "29.232.228.21:0:pf_aw_stride_merge"
    "29.232.228.21:1:pf_anchor_extended_recent"
    "29.232.228.21:2:pf_wave_extended_recent"
    "29.232.228.21:3:pf_veil_extended_recent"
)

LOCAL_IP="29.232.228.42"

echo "[v97-vbench-multinode] start $(date)"
echo "[v97-vbench-multinode] ${#ASSIGNMENTS[@]} methods to launch (4 dims, no dynamic_degree)"

PIDS=()
for a in "${ASSIGNMENTS[@]}"; do
    IFS=':' read -r ip gpu method <<<"$a"
    marker="$METRICS/status/vbench.$method.done"
    out="$METRICS/vbench_long/$method/results.json"
    if [[ -s "$marker" && -s "$out" ]]; then
        echo "[vbench] SKIP $method (already done)"
        continue
    fi
    log="$LOGDIR/vbench.$method.log"
    if [[ "$ip" == "$LOCAL_IP" ]]; then
        nohup bash "$RUNNER" "$method" vbench "$gpu" >"$log" 2>&1 &
    else
        nohup ssh -o StrictHostKeyChecking=no "$ip" \
            "bash $RUNNER $method vbench $gpu" >"$log" 2>&1 &
    fi
    echo "[vbench] LAUNCH $method on $ip GPU$gpu (pid=$!)"
    PIDS+=("$!")
done

echo "[v97-vbench-multinode] waiting for ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

echo "[v97-vbench-multinode] all launched jobs done $(date)"
echo "=== STATUS ==="
for m in prompt_tau_0p0_merge prompt_tau_0p5_merge prompt_tau_1p0_merge prompt_tau_1p5_merge \
         prompt_tau_2p0_merge prompt_tau_1p0_cyclic prompt_tau_1p0_recent prompt_tau_1p0_random_merge \
         prompt_tau_1p0_reversed_merge sign_rpos_0p5_stride_merge pf_ar_stride_merge pf_aw_stride_merge \
         pf_native pf_anchor_extended_recent pf_wave_extended_recent pf_veil_extended_recent; do
    marker="$METRICS/status/vbench.$m.done"
    out="$METRICS/vbench_long/$m/results.json"
    if [[ -s "$marker" && -s "$out" ]]; then
        echo "  [OK] $m"
    else
        echo "  [PENDING/FAIL] $m"
    fi
done
