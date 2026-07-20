#!/usr/bin/env python3
"""Evaluate explicit A-B-A scene-return experiments.

Reports DINO centroid similarities for full frames and background crops:
A1-A2 return similarity, B-A2 leakage, and their margin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_comprehensive import extract_dino_features  # noqa: E402


def decode(path: Path) -> np.ndarray:
    container = av.open(str(path))
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return np.stack(frames)


def sample_segment(frames: np.ndarray, start: float, end: float, count: int = 8) -> np.ndarray:
    total = len(frames)
    lo = int(total * start)
    hi = max(lo + 1, int(total * end))
    indices = np.linspace(lo, hi - 1, count, dtype=int)
    return frames[indices]


def features(frames: np.ndarray, background: bool = False) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 255.0
    if background:
        # Suppress central subject: use outer border quadrants for scene/layout.
        _, _, h, w = tensor.shape
        top = tensor[:, :, : h // 3]
        bottom = tensor[:, :, 2 * h // 3 :]
        tensor = torch.cat([top, bottom], dim=2)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False).half()
    return F.normalize(extract_dino_features(tensor, batch_size=8).float(), dim=-1)


def centroid_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = F.normalize(a.mean(dim=0), dim=0)
    cb = F.normalize(b.mean(dim=0), dim=0)
    return float(torch.dot(ca, cb).item())


def evaluate_video(path: Path) -> dict:
    frames = decode(path)
    # Avoid segment boundaries: A1 [8%,27%], B [40%,60%], A2 [76%,96%].
    a1 = sample_segment(frames, 0.08, 0.27)
    b = sample_segment(frames, 0.40, 0.60)
    a2 = sample_segment(frames, 0.76, 0.96)
    result = {}
    for name, bg in (("full", False), ("background", True)):
        fa1, fb, fa2 = features(a1, bg), features(b, bg), features(a2, bg)
        return_sim = centroid_similarity(fa1, fa2)
        leakage = centroid_similarity(fb, fa2)
        result[name] = {
            "a1_a2": return_sim,
            "b_a2": leakage,
            "return_margin": return_sim - leakage,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/aba_v59_metrics.json")
    args = parser.parse_args()
    base = args.root / "runs/v35_pf_value_refresh"
    methods = {
        "pf": base / "20260720_v59_pf/pf_refresh_pf",
        "correct": base / "20260720_v59_correct/pf_refresh_correct",
        "wrong_b": base / "20260720_v59_wrong_b/pf_refresh_wrong_b",
        "shuffled_v": base / "20260720_v59_shuffled_v/pf_refresh_shuffled_v",
        "abstain": base / "20260720_v59_abstain/pf_refresh_abstain",
    }
    output = {"per_video": {}, "aggregate": {}}
    for method, directory in methods.items():
        values = []
        for index in range(3):
            path = directory / f"{index}-0_ema.mp4"
            metric = evaluate_video(path)
            output["per_video"][f"{method}/{index}"] = metric
            values.append(metric)
        aggregate = {}
        for view in ("full", "background"):
            aggregate[view] = {
                key: float(np.mean([value[view][key] for value in values]))
                for key in ("a1_a2", "b_a2", "return_margin")
            }
        output["aggregate"][method] = aggregate
        print(method, aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
