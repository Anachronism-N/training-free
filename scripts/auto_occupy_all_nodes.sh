#!/usr/bin/env bash
# Auto GPU occupy: check all 4 nodes, occupy idle GPUs.
# Run this every 45 minutes via cron.
set -uo pipefail

ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
OCCUPIER="$ROOT/scripts/gpu_occupier.py"
CONDA_SH=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=longlive
NODES=("29.232.228.42" "29.232.240.221" "29.127.50.121" "29.232.228.21")
LOCAL_IP="29.232.228.42"
LOG="$ROOT/logs/gpu_occupy.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] === auto-occupy check ==="

for ip in "${NODES[@]}"; do
  if [[ "$ip" == "$LOCAL_IP" ]]; then
    # Local node
    source "$CONDA_SH" 2>/dev/null && conda activate "$CONDA_ENV" 2>/dev/null
    # Check if already occupying
    if [[ -f /tmp/gpu_occupier.pid ]]; then
      echo "[$ip] already occupying (PID file exists)"
      continue
    fi
    # Check if idle
    if python3 "$OCCUPIER" --check-idle 2>/dev/null; then
      echo "[$ip] GPUs idle, starting occupier"
      nohup python3 "$OCCUPIER" >>"$LOG" 2>&1 &
      echo "$!" > /tmp/gpu_occupier_launcher.pid
    else
      echo "[$ip] GPUs busy (other processes running), skipping"
    fi
  else
    # Remote node
    # Check if already occupying
    if timeout 8 ssh -o StrictHostKeyChecking=no "$ip" "test -f /tmp/gpu_occupier.pid" 2>/dev/null; then
      echo "[$ip] already occupying (PID file exists)"
      continue
    fi
    # Check if idle
    if timeout 15 ssh -o StrictHostKeyChecking=no "$ip" "
      source $CONDA_SH 2>/dev/null && conda activate $CONDA_ENV 2>/dev/null
      python3 $OCCUPIER --check-idle
    " 2>/dev/null; then
      echo "[$ip] GPUs idle, starting occupier"
      timeout 10 ssh -o StrictHostKeyChecking=no "$ip" "
        source $CONDA_SH 2>/dev/null && conda activate $CONDA_ENV 2>/dev/null
        nohup python3 $OCCUPIER >>/tmp/gpu_occupy.log 2>&1 &
      " 2>/dev/null
      echo "[$ip] occupier launched"
    else
      echo "[$ip] GPUs busy or unreachable, skipping"
    fi
  fi
done

echo "[$(date)] === check complete ==="
