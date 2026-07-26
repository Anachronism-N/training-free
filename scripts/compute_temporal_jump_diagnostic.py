#!/usr/bin/env python3
"""Compute lightweight temporal discontinuity diagnostics for generated videos.

The metric is intended for paired debugging of the same prompt and seed. It is
not a replacement for VBench, RAFT-based motion metrics, or human review.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - depends on the server runtime
    cv2 = None
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--expected-videos", type=int)
    return parser.parse_args()


INDEXED_VIDEO_PATTERN = re.compile(r"^(\d+)-(\d+)_[^.]+\.mp4$")


def _indexed_video_paths(
    inputs: list[Path],
    *,
    expected_videos: int | None = None,
    sample_idx: int = 0,
) -> list[tuple[str, int, Path]]:
    indexed: dict[tuple[str, int], Path] = {}
    directory_methods: dict[str, Path] = {}
    malformed: list[str] = []
    unexpected_samples: list[str] = []
    for value in inputs:
        if value.is_dir():
            method = value.name
            resolved_root = value.resolve()
            prior_root = directory_methods.get(method)
            if prior_root is not None and prior_root != resolved_root:
                raise ValueError(
                    f"input directories have duplicate method name {method!r}: "
                    f"{prior_root}, {resolved_root}"
                )
            directory_methods[method] = resolved_root
            candidates = sorted(value.rglob("*.mp4"))
        elif value.suffix.lower() == ".mp4" and value.is_file():
            method = value.parent.name
            candidates = [value]
        else:
            raise FileNotFoundError(f"missing video input: {value}")
        for path in candidates:
            match = INDEXED_VIDEO_PATTERN.fullmatch(path.name)
            if match is None:
                malformed.append(str(path))
                continue
            prompt_index = int(match.group(1))
            current_sample = int(match.group(2))
            if current_sample != sample_idx:
                unexpected_samples.append(str(path))
                continue
            key = (method, prompt_index)
            resolved = path.resolve()
            if key in indexed:
                raise ValueError(
                    f"duplicate sample-{sample_idx} video for {key}: "
                    f"{indexed[key]}, {resolved}"
                )
            indexed[key] = resolved

    if malformed or unexpected_samples:
        raise ValueError(
            "non-canonical temporal inputs: "
            f"malformed={malformed[:10]} "
            f"unexpected_sample_indices={unexpected_samples[:10]}"
        )
    if not indexed:
        return []

    methods = sorted({method for method, _ in indexed})
    if expected_videos is not None:
        if expected_videos <= 0:
            raise ValueError("expected-videos must be positive")
        # Accept either a per-method count (directory-oriented callers) or
        # the total Cartesian row count (callers that expand individual
        # files before invoking this script).
        per_method_count = expected_videos
        if expected_videos == len(indexed):
            if expected_videos % len(methods):
                raise ValueError(
                    "total expected-videos is not divisible by method count"
                )
            per_method_count = expected_videos // len(methods)
        expected_indices = set(range(per_method_count))
    else:
        maximum = max(prompt_index for _, prompt_index in indexed)
        expected_indices = set(range(maximum + 1))
    coverage_failures = {}
    for method in methods:
        actual = {
            prompt_index
            for row_method, prompt_index in indexed
            if row_method == method
        }
        if actual != expected_indices:
            coverage_failures[method] = {
                "missing": sorted(expected_indices - actual),
                "extra": sorted(actual - expected_indices),
            }
    if coverage_failures:
        raise ValueError(
            f"temporal prompt coverage mismatch: {coverage_failures}"
        )
    return [
        (method, prompt_index, indexed[(method, prompt_index)])
        for method in methods
        for prompt_index in sorted(expected_indices)
    ]


def _video_paths(inputs: list[Path]) -> list[Path]:
    """Compatibility wrapper returning strictly indexed paths."""
    return [
        path
        for _, _, path in _indexed_video_paths(inputs)
    ]


def _resize_gray(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _summary(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(array.max()),
    }


def _robust_outlier_fraction(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 4:
        return 0.0
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    threshold = median + 3.0 * max(mad, 1e-8)
    return float(np.mean(array > threshold))


def analyze_video(path: Path, *, max_width: int, frame_step: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    decoded = 0
    retained = 0
    previous = None
    appearance_delta: list[float] = []
    flow_speed: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            keep = decoded % frame_step == 0
            decoded += 1
            if not keep:
                continue
            gray = _resize_gray(frame, max_width)
            retained += 1
            if previous is not None:
                appearance_delta.append(
                    float(np.mean(np.abs(gray.astype(np.float32) - previous)) / 255.0)
                )
                flow = cv2.calcOpticalFlowFarneback(
                    previous,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                magnitude = np.linalg.norm(flow, axis=-1)
                flow_speed.append(float(np.median(magnitude)))
            previous = gray
    finally:
        capture.release()
    if retained < 3:
        raise RuntimeError(f"video has fewer than three retained frames: {path}")

    flow_acceleration = np.abs(np.diff(np.asarray(flow_speed, dtype=np.float64))).tolist()
    appearance = _summary(appearance_delta, "appearance_delta")
    speed = _summary(flow_speed, "flow_speed")
    acceleration = _summary(flow_acceleration, "flow_accel")
    eps = 1e-6
    appearance_jump_ratio = appearance["appearance_delta_p95"] / max(
        appearance["appearance_delta_median"], eps
    )
    flow_jump_ratio = acceleration["flow_accel_p95"] / max(
        speed["flow_speed_median"], eps
    )
    return {
        "video": str(path),
        "decoded_frames": decoded,
        "retained_frames": retained,
        "fps": fps,
        "duration_seconds": decoded / fps if fps > 0 else 0.0,
        **appearance,
        **speed,
        **acceleration,
        "appearance_jump_ratio": appearance_jump_ratio,
        "flow_jump_ratio": flow_jump_ratio,
        "appearance_outlier_fraction": _robust_outlier_fraction(appearance_delta),
        "flow_accel_outlier_fraction": _robust_outlier_fraction(flow_acceleration),
        "temporal_jump": 0.5 * (appearance_jump_ratio + flow_jump_ratio),
    }


def main() -> None:
    args = parse_args()
    if cv2 is None or np is None:
        raise SystemExit(
            "compute_temporal_jump_diagnostic.py requires "
            "opencv-python and numpy"
        )
    if args.max_width <= 0 or args.frame_step <= 0:
        raise ValueError("max-width and frame-step must be positive")
    videos = _indexed_video_paths(
        args.inputs,
        expected_videos=args.expected_videos,
    )
    if not videos:
        raise ValueError("no MP4 files found")
    rows = []
    for index, (method, prompt_index, path) in enumerate(videos, start=1):
        row = analyze_video(
            path,
            max_width=args.max_width,
            frame_step=args.frame_step,
        )
        row = {
            "method": method,
            "prompt_index": prompt_index,
            "sample_index": 0,
            **row,
        }
        rows.append(row)
        print(
            f"[temporal-jump] {index}/{len(videos)} "
            f"score={row['temporal_jump']:.5f} video={path}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[temporal-jump] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
