#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path, width: int, stride: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            height = max(1, round(frame.shape[0] * width / frame.shape[1]))
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
        index += 1
    capture.release()
    return frames


def evaluate_video(path: Path, width: int, stride: int) -> dict[str, float | int | str]:
    frames = read_frames(path, width=width, stride=stride)
    if len(frames) < 2:
        raise ValueError(f"Need at least two sampled frames: {path}")

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    luma = np.asarray([gray.mean() / 255.0 for gray in grays], dtype=np.float64)
    sharpness = np.asarray(
        [cv2.Laplacian(gray, cv2.CV_64F).var() for gray in grays], dtype=np.float64
    )
    flow_means = []
    dynamic_ratios = []
    for previous, current in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.linalg.norm(flow, axis=2)
        flow_means.append(float(np.mean(magnitude)))
        dynamic_ratios.append(float(np.mean(magnitude > 1.0)))

    quarter = max(1, len(luma) // 4)
    return {
        "video": path.name,
        "sampled_frames": len(frames),
        "flow_mean": float(np.mean(flow_means)),
        "flow_median": float(np.median(flow_means)),
        "dynamic_pixel_ratio": float(np.mean(dynamic_ratios)),
        "luma_first_quarter": float(np.mean(luma[:quarter])),
        "luma_last_quarter": float(np.mean(luma[-quarter:])),
        "luma_relative_change": float(
            np.mean(luma[-quarter:]) / max(np.mean(luma[:quarter]), 1e-8) - 1.0
        ),
        "sharpness_mean": float(np.mean(sharpness)),
        "sharpness_last_quarter": float(np.mean(sharpness[-quarter:])),
    }


def aggregate(items: list[dict[str, float | int | str]]) -> dict[str, float]:
    keys = (
        "flow_mean",
        "flow_median",
        "dynamic_pixel_ratio",
        "luma_relative_change",
        "sharpness_mean",
        "sharpness_last_quarter",
    )
    return {key: float(np.mean([float(item[key]) for item in items])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    result = {}
    for video_dir in args.video_dirs:
        items = [
            evaluate_video(path, width=args.width, stride=args.stride)
            for path in sorted(video_dir.glob("*.mp4"))
        ]
        result[video_dir.name] = {"videos": items, "aggregate": aggregate(items)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
