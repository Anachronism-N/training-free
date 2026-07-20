#!/usr/bin/env python3
"""Create three-way comparison videos for v4.6: SF / PF / Ours side by side."""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free")
SF_DIR = ROOT / "runs/v35_pf_value_refresh/20260720_v46_sf_native_60s/pf_refresh_sf_native_60s"
PF_DIR = ROOT / "runs/v35_pf_value_refresh/20260720_v46_pf_60s/pf_refresh_pf_60s"
OURS_DIR = ROOT / "runs/v35_pf_value_refresh/20260720_v46_ours_60s/pf_refresh_ours_60s"
OUT_DIR = ROOT / "runs/REVIEW_v46_optimized_60s/comparisons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for i in range(3):
    sf_video = SF_DIR / f"{i}-0_ema.mp4"
    pf_video = PF_DIR / f"{i}-0_ema.mp4"
    ours_video = OURS_DIR / f"{i}-0_ema.mp4"
    if not all(p.exists() for p in [sf_video, pf_video, ours_video]):
        print(f"Skipping {i}: missing files")
        continue

    output = OUT_DIR / f"{i}_threeway.mp4"
    # Use ffmpeg filter_complex to stack horizontally
    cmd = [
        "ffmpeg", "-y",
        "-i", str(sf_video),
        "-i", str(pf_video),
        "-i", str(ours_video),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[v0];"
        "[1:v]setpts=PTS-STARTPTS[v1];"
        "[2:v]setpts=PTS-STARTPTS[v2];"
        "[v0][v1][v2]hstack=inputs=3[v]",
        "-map", "[v]",
        "-c:v", "mpeg4", "-q:v", "2",
        "-movflags", "+faststart",
        str(output),
    ]
    print(f"Creating {output}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
    else:
        print(f"  OK: {output}")

print("Done")
