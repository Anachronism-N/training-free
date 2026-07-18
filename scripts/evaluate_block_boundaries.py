#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_gray_frames(path: Path, width: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height = max(1, round(frame.shape[0] * width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    capture.release()
    return frames


def summarize(values: np.ndarray, boundary_mask: np.ndarray) -> dict[str, float]:
    boundary = values[boundary_mask]
    regular = values[~boundary_mask]
    boundary_mean = float(boundary.mean()) if boundary.size else 0.0
    regular_mean = float(regular.mean()) if regular.size else 0.0
    return {
        "boundary_mean": boundary_mean,
        "regular_mean": regular_mean,
        "boundary_ratio": boundary_mean / max(regular_mean, 1e-8),
        "boundary_p95": float(np.percentile(boundary, 95)) if boundary.size else 0.0,
        "regular_p95": float(np.percentile(regular, 95)) if regular.size else 0.0,
    }


def evaluate_video(path: Path, width: int, period: int, first_boundary: int) -> dict:
    frames = load_gray_frames(path, width)
    if len(frames) < 2:
        raise ValueError(f"Video has fewer than two frames: {path}")

    low_frequency = [cv2.GaussianBlur(frame, (0, 0), 5.0) for frame in frames]
    edges = [cv2.Laplacian(frame, cv2.CV_32F, ksize=3) for frame in frames]
    low_diffs = np.asarray(
        [np.mean(np.abs(right - left)) for left, right in zip(low_frequency, low_frequency[1:])]
    )
    edge_diffs = np.asarray(
        [np.mean(np.abs(right - left)) for left, right in zip(edges, edges[1:])]
    )

    destination_indices = np.arange(1, len(frames))
    boundary_mask = (
        (destination_indices >= first_boundary)
        & ((destination_indices - first_boundary) % period == 0)
    )
    return {
        "frames": len(frames),
        "boundary_transitions": int(boundary_mask.sum()),
        "low_frequency": summarize(low_diffs, boundary_mask),
        "edge": summarize(edge_diffs, boundary_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--period", type=int, default=12)
    parser.add_argument("--first-boundary", type=int, default=9)
    args = parser.parse_args()

    results = {}
    for video_dir in args.video_dirs:
        videos = sorted(video_dir.glob("*.mp4"))
        per_video = {
            video.name: evaluate_video(
                video, args.width, args.period, args.first_boundary
            )
            for video in videos
        }
        if per_video:
            results[video_dir.name] = {
                "videos": per_video,
                "aggregate": {
                    key: float(
                        np.mean(
                            [metrics[key]["boundary_ratio"] for metrics in per_video.values()]
                        )
                    )
                    for key in ("low_frequency", "edge")
                },
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
