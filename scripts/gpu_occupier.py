#!/usr/bin/env python3
"""GPU occupier — allocates matrices and runs matmul to hold GPUs.

Usage:
  python3 gpu_occupier.py                # occupy all GPUs on this node
  python3 gpu_occupier.py --status       # show GPU status
  python3 gpu_occupier.py --stop         # kill all occupier processes

Based on the GPU占卡.md documentation: allocates 8192x8192 float16 matrices
(~384MB/card) and runs torch.matmul at ~80% duty cycle.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import torch

DUTY_CYCLE = 1.0          # 100% utilization (no sleep) to keep avg > 30%
MATMUL_SIZE = 8192        # 8192x8192 float16 ~ 128MB per matrix, 3 matrices ~ 384MB
SLEEP_DURATION = 0.0      # no sleep at 100% duty cycle
PID_FILE = "/tmp/gpu_occupier.pid"


def is_gpu_idle(threshold_mb=1000):
    """Check if ALL GPUs on this node are idle."""
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_used = torch.cuda.memory_allocated(i)  # not reliable for other procs
        # Use nvidia-smi for actual usage
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "--id", str(i)],
                capture_output=True, text=True, timeout=5
            )
            mem_used_mb = int(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired):
            continue
        if mem_used_mb >= threshold_mb:
            return False
    # Also check for compute processes
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            return False
    except subprocess.TimeoutExpired:
        pass
    return True


def occupy_gpus():
    """Occupy all GPUs with matmul loop."""
    if os.path.exists(PID_FILE):
        print(f"[occupier] already running (PID file exists: {PID_FILE})")
        return

    n = torch.cuda.device_count()
    if n == 0:
        print("[occupier] no CUDA GPUs found")
        return

    print(f"[occupier] occupying {n} GPUs with {MATMUL_SIZE}x{MATMUL_SIZE} matmul")
    matrices = []
    for i in range(n):
        torch.cuda.set_device(i)
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float16, device=f"cuda:{i}")
        b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float16, device=f"cuda:{i}")
        matrices.append((a, b))

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    def cleanup(signum=None, frame=None):
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("[occupier] releasing GPUs and exiting")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    on_time = DUTY_CYCLE * 0.1   # matmul duration
    off_time = (1 - DUTY_CYCLE) * 0.1  # sleep duration (0 at 100%)

    print(f"[occupier] duty cycle: 100% (no sleep)")
    while True:
        for i, (a, b) in enumerate(matrices):
            torch.cuda.set_device(i)
            torch.matmul(a, b)
        if off_time > 0:
            time.sleep(off_time)


def status():
    """Show GPU status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        print(f"[status] GPU usage:\n{result.stdout}")
    except subprocess.TimeoutExpired:
        print("[status] nvidia-smi timeout")
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = f.read().strip()
        print(f"[status] occupier PID: {pid}")
    else:
        print("[status] no occupier running")


def stop():
    """Kill occupier process."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[stop] sent SIGTERM to PID {pid}")
            time.sleep(2)
            os.remove(PID_FILE)
        except ProcessLookupError:
            print(f"[stop] PID {pid} not found, removing stale PID file")
            os.remove(PID_FILE)
    else:
        print("[stop] no occupier PID file found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU occupier")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--check-idle", action="store_true",
                        help="Check if GPUs are idle, exit 0 if idle, 1 if busy")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.stop:
        stop()
    elif args.check_idle:
        idle = is_gpu_idle()
        print(f"idle={idle}")
        sys.exit(0 if idle else 1)
    else:
        occupy_gpus()
