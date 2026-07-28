#!/bin/bash
# Auto-monitor and restart crashed batch runners on all nodes
# Runs every 5 minutes, checks for crashed/stalled processes
LOG=/tmp/v125_auto_monitor.log
REPO=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

echo "[monitor] started at $(date)" >> "$LOG"

while true; do
  done_count=$(find "$REPO/runs/v125_moviebench128_main" -name "*.done.json" 2>/dev/null | wc -l)
  msg="[$(date '+%H:%M:%S')] done=${done_count}/1024"

  # Check node21 (rank 0)
  n21_batch=$(ssh -o ConnectTimeout=5 29.232.228.21 "pgrep -c -f 'batch_inference' 2>/dev/null || echo 0" 2>/dev/null)
  n21_active=$(ssh -o ConnectTimeout=5 29.232.228.21 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | grep -v '0 %' | wc -l" 2>/dev/null)
  msg="$msg n21(b=$n21_batch/a=$n21_active)"
  if [ "${n21_batch:-0}" -eq 0 ] 2>/dev/null; then
    echo "[monitor] node21 batch runner DEAD, restarting..." >> "$LOG"
    ssh -o ConnectTimeout=10 29.232.228.21 "setsid bash $REPO/scripts/start_batch_node21.sh </dev/null >/dev/null 2>&1 &" 2>/dev/null
  fi

  # Check node121 (rank 2)
  n121_batch=$(ssh -o ConnectTimeout=5 29.127.50.121 "pgrep -c -f 'batch_inference' 2>/dev/null || echo 0" 2>/dev/null)
  n121_active=$(ssh -o ConnectTimeout=5 29.127.50.121 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | grep -v '0 %' | wc -l" 2>/dev/null)
  msg="$msg n121(b=$n121_batch/a=$n121_active)"
  if [ "${n121_batch:-0}" -eq 0 ] 2>/dev/null; then
    echo "[monitor] node121 batch runner DEAD, restarting..." >> "$LOG"
    ssh -o ConnectTimeout=10 29.127.50.121 "setsid bash /tmp/start_batch.sh </dev/null >/dev/null 2>&1 &" 2>/dev/null
  fi

  # Check node221 (rank 1)
  n221_batch=$(ssh -o ConnectTimeout=5 29.232.240.221 "pgrep -c -f 'batch_inference' 2>/dev/null || echo 0" 2>/dev/null)
  n221_active=$(ssh -o ConnectTimeout=5 29.232.240.221 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | grep -v '0 %' | wc -l" 2>/dev/null)
  msg="$msg n221(b=$n221_batch/a=$n221_active)"
  if [ "${n221_batch:-0}" -eq 0 ] 2>/dev/null; then
    echo "[monitor] node221 batch runner DEAD, restarting..." >> "$LOG"
    ssh -o ConnectTimeout=10 29.232.240.221 "setsid bash /tmp/start_batch_1gpu.sh </dev/null >/dev/null 2>&1 &" 2>/dev/null
  fi

  # Check node42 (should have occupier)
  n42_occ=$(pgrep -c -f 'gpu_occupier' 2>/dev/null || echo 0)
  msg="$msg n42(occ=$n42_occ)"
  if [ "${n42_occ:-0}" -eq 0 ] 2>/dev/null; then
    echo "[monitor] node42 occupier DEAD, restarting..." >> "$LOG"
    setsid python "$REPO/scripts/gpu_occupier.py" </dev/null >>/tmp/gpu_occupier.log 2>&1 &
  fi

  echo "$msg" >> "$LOG"

  # Check if all done
  if [ "$done_count" -ge 1024 ] 2>/dev/null; then
    echo "[monitor] ALL 1024 DONE! Starting occupiers on all nodes..." >> "$LOG"
    for ip in 29.232.240.221 29.127.50.121 29.232.228.21; do
      ssh -o ConnectTimeout=10 $ip "source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh && conda activate longlive && cd $REPO && setsid python scripts/gpu_occupier.py </dev/null >>/tmp/gpu_occupier.log 2>&1 &" 2>/dev/null
    done
    echo "[monitor] All occupiers started. Exiting." >> "$LOG"
    break
  fi

  sleep 300
done
