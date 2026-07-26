#!/usr/bin/env bash
# Relaunch remote vbench + comp on 3 nodes with proxy env vars.
# vbench on GPUs 0-3, comp on GPUs 4-7. Runs both simultaneously.
set -uo pipefail

ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
RUNNER="$ROOT/runs/v97_eval_smoke.sh"
LOGDIR="$ROOT/runs/v97_threshold_pf_merge32/metrics/logs"
mkdir -p "$LOGDIR"

# vbench assignments (all 12 remote methods, 4 per node)
VBENCH_ASSIGNMENTS=(
    "29.232.240.221:0:prompt_tau_1p5_merge"
    "29.232.240.221:1:prompt_tau_2p0_merge"
    "29.232.240.221:2:prompt_tau_1p0_cyclic"
    "29.232.240.221:3:prompt_tau_1p0_recent"
    "29.127.50.121:0:prompt_tau_1p0_random_merge"
    "29.127.50.121:1:prompt_tau_1p0_reversed_merge"
    "29.127.50.121:2:sign_rpos_0p5_stride_merge"
    "29.127.50.121:3:pf_ar_stride_merge"
    "29.232.228.21:0:pf_aw_stride_merge"
    "29.232.228.21:1:pf_anchor_extended_recent"
    "29.232.228.21:2:pf_wave_extended_recent"
    "29.232.228.21:3:pf_veil_extended_recent"
)

# comp assignments (11 remaining: pf_native + tau_0p0/0p5/1p0/1p5 already done)
COMP_ASSIGNMENTS=(
    "29.232.240.221:4:prompt_tau_2p0_merge"
    "29.232.240.221:5:prompt_tau_1p0_cyclic"
    "29.232.240.221:6:prompt_tau_1p0_recent"
    "29.232.240.221:7:prompt_tau_1p0_random_merge"
    "29.127.50.121:4:prompt_tau_1p0_reversed_merge"
    "29.127.50.121:5:sign_rpos_0p5_stride_merge"
    "29.127.50.121:6:pf_ar_stride_merge"
    "29.127.50.121:7:pf_aw_stride_merge"
    "29.232.228.21:4:pf_anchor_extended_recent"
    "29.232.228.21:5:pf_wave_extended_recent"
    "29.232.228.21:6:pf_veil_extended_recent"
)

echo "[remote-relaunch] start $(date)"
PIDS=()

# Launch vbench
for a in "${VBENCH_ASSIGNMENTS[@]}"; do
    IFS=':' read -r ip gpu method <<<"$a"
    log="$LOGDIR/vbench.$method.log"
    nohup ssh -o StrictHostKeyChecking=no "$ip" \
        "bash $RUNNER $method vbench $gpu" >"$log" 2>&1 &
    echo "[vbench] $method on $ip GPU$gpu"
    PIDS+=("$!")
done

# Launch comp
for a in "${COMP_ASSIGNMENTS[@]}"; do
    IFS=':' read -r ip gpu method <<<"$a"
    log="$LOGDIR/comp.$method.log"
    nohup ssh -o StrictHostKeyChecking=no "$ip" \
        "bash $RUNNER $method comp $gpu" >"$log" 2>&1 &
    echo "[comp] $method on $ip GPU$gpu"
    PIDS+=("$!")
done

echo "[remote-relaunch] ${#PIDS[@]} jobs launched, waiting..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done
echo "[remote-relaunch] done $(date)"
