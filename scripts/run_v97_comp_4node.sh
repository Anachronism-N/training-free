#!/usr/bin/env bash
# v97 comprehensive (DINO): 4-node parallel launcher using idle GPUs 4-7.
# Runs alongside vbench-long (which uses GPUs 0-3). 15 methods (pf_native
# already done by comp smoke). All run in one wave.
set -uo pipefail

ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
RUN_ROOT="$ROOT/runs/v97_threshold_pf_merge32"
METRICS="$RUN_ROOT/metrics"
RUNNER="$ROOT/runs/v97_eval_smoke.sh"
LOGDIR="$RUN_ROOT/metrics/logs"
mkdir -p "$LOGDIR" "$METRICS/status" "$METRICS/comprehensive_parts"

# Node assignments: "ip:gpu:method" — uses GPUs 4-7 (idle while vbench uses 0-3)
ASSIGNMENTS=(
    # node 42 — GPU1 was comp smoke (done), GPUs 5,6,7 idle
    "29.232.228.42:1:prompt_tau_0p0_merge"
    "29.232.228.42:5:prompt_tau_0p5_merge"
    "29.232.228.42:6:prompt_tau_1p0_merge"
    "29.232.228.42:7:prompt_tau_1p5_merge"
    # node 221 — GPUs 4-7
    "29.232.240.221:4:prompt_tau_2p0_merge"
    "29.232.240.221:5:prompt_tau_1p0_cyclic"
    "29.232.240.221:6:prompt_tau_1p0_recent"
    "29.232.240.221:7:prompt_tau_1p0_random_merge"
    # node 121 — GPUs 4-7
    "29.127.50.121:4:prompt_tau_1p0_reversed_merge"
    "29.127.50.121:5:sign_rpos_0p5_stride_merge"
    "29.127.50.121:6:pf_ar_stride_merge"
    "29.127.50.121:7:pf_aw_stride_merge"
    # node 21 — GPUs 4-7
    "29.232.228.21:4:pf_anchor_extended_recent"
    "29.232.228.21:5:pf_wave_extended_recent"
    "29.232.228.21:6:pf_veil_extended_recent"
)

LOCAL_IP="29.232.228.42"

echo "[v97-comp-multinode] start $(date)"
echo "[v97-comp-multinode] ${#ASSIGNMENTS[@]} methods to launch (pf_native already done)"

PIDS=()
for a in "${ASSIGNMENTS[@]}"; do
    IFS=':' read -r ip gpu method <<<"$a"
    marker="$METRICS/status/comprehensive.$method.done"
    out="$METRICS/comprehensive_parts/$method.json"
    if [[ -s "$marker" && -s "$out" ]]; then
        echo "[comp] SKIP $method (already done)"
        continue
    fi
    log="$LOGDIR/comp.$method.log"
    if [[ "$ip" == "$LOCAL_IP" ]]; then
        nohup bash "$RUNNER" "$method" comp "$gpu" >"$log" 2>&1 &
    else
        nohup ssh -o StrictHostKeyChecking=no "$ip" \
            "bash $RUNNER $method comp $gpu" >"$log" 2>&1 &
    fi
    echo "[comp] LAUNCH $method on $ip GPU$gpu (pid=$!)"
    PIDS+=("$!")
done

echo "[v97-comp-multinode] waiting for ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

echo "[v97-comp-multinode] all launched jobs done $(date)"
echo "=== STATUS ==="
for m in prompt_tau_0p0_merge prompt_tau_0p5_merge prompt_tau_1p0_merge prompt_tau_1p5_merge \
         prompt_tau_2p0_merge prompt_tau_1p0_cyclic prompt_tau_1p0_recent prompt_tau_1p0_random_merge \
         prompt_tau_1p0_reversed_merge sign_rpos_0p5_stride_merge pf_ar_stride_merge pf_aw_stride_merge \
         pf_native pf_anchor_extended_recent pf_wave_extended_recent pf_veil_extended_recent; do
    marker="$METRICS/status/comprehensive.$m.done"
    out="$METRICS/comprehensive_parts/$m.json"
    if [[ -s "$marker" && -s "$out" ]]; then
        echo "  [OK] $m"
    else
        echo "  [PENDING/FAIL] $m"
    fi
done
