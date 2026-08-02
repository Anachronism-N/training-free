#!/usr/bin/env bash
# Background GPU monitor: every 30 min, check all 4 nodes.
# If a node has no v129/v120 process running AND GPUs are idle, launch gpu_occupier.
# If v129 is running, do nothing (it's using GPUs).
# Logs to /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free/logs/gpu_monitor.log
set -uo pipefail

REPO=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
LOGFILE=$REPO/logs/gpu_monitor.log
CONDA_ENV=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3
NODES=("29.127.81.4:0" "29.127.80.81:1" "29.191.208.222:2" "29.232.242.104:3")
INTERVAL=1800  # 30 minutes

mkdir -p $REPO/logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOGFILE
}

check_and_occupy() {
  local entry="$1"
  local ip="${entry%%:*}"
  local rank="${entry##*:}"

  # Check if v129/v120 process is running on this node
  local v129_count
  if [ "$ip" = "29.127.81.4" ]; then
    v129_count=$(ps aux | grep -E "run_v129|run_v120_moviebench32|inference.py" | grep -v grep | wc -l)
  else
    v129_count=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $ip "ps aux | grep -E 'run_v129|run_v120_moviebench32|inference.py' | grep -v grep | wc -l" 2>/dev/null || echo "0")
  fi

  # Check GPU utilization (max of all GPUs)
  local gpu_util
  if [ "$ip" = "29.127.81.4" ]; then
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -rn | head -1 | tr -d ' ')
  else
    gpu_util=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $ip "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -rn | head -1 | tr -d ' '" 2>/dev/null || echo "0")
  fi

  # Check if occupier already running
  local occupier_running
  if [ "$ip" = "29.127.81.4" ]; then
    occupier_running=$(ps aux | grep "gpu_occupier" | grep -v grep | wc -l)
  else
    occupier_running=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $ip "ps aux | grep 'gpu_occupier' | grep -v grep | wc -l" 2>/dev/null || echo "0")
  fi

  log "  node$rank ($ip): v129_procs=$v129_count max_gpu_util=${gpu_util}% occupier_running=$occupier_running"

  # Decision: if no v129 process AND GPU util < 30% AND no occupier running -> launch occupier
  if [ "$v129_count" -le 0 ] && [ "$gpu_util" -lt 30 ] && [ "$occupier_running" -le 0 ]; then
    log "  -> IDLE: launching gpu_occupier on $ip (node$rank)"
    if [ "$ip" = "29.127.81.4" ]; then
      rm -f /tmp/gpu_occupier.pid
      nohup bash -c "source $CONDA_ENV/etc/profile.d/conda.sh && conda activate longlive && exec python3 $REPO/scripts/gpu_occupier.py" >/tmp/gpu_occupier.log 2>&1 & disown
      log "  -> node0 occupier launched PID=$!"
    else
      ssh -o ConnectTimeout=10 -o BatchMode=yes $ip "rm -f /tmp/gpu_occupier.pid; setsid bash -c 'source $CONDA_ENV/etc/profile.d/conda.sh && conda activate longlive && python3 $REPO/scripts/gpu_occupier.py' </dev/null >/tmp/gpu_occupier.log 2>&1 &" 2>/dev/null
      log "  -> $ip occupier launched"
    fi
  elif [ "$v129_count" -gt 0 ]; then
    log "  -> BUSY: v129 running, skipping occupy"
  elif [ "$occupier_running" -gt 0 ]; then
    log "  -> OK: occupier already running"
  else
    log "  -> WAIT: GPU util ${gpu_util}% >= 30%, monitoring"
  fi
}

log "=== GPU monitor started (interval=${INTERVAL}s) ==="

while true; do
  log "--- check cycle ---"
  for entry in "${NODES[@]}"; do
    check_and_occupy "$entry"
  done
  log "--- sleeping ${INTERVAL}s ---"
  sleep $INTERVAL
done
