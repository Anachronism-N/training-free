#!/usr/bin/env python3
"""Check whether v188 60-second runs reproduce their v187 first-30s prefix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from prepare_v188_robustness_matrix import BASE_METHODS, scope_config


SAMPLE_COUNT = 16
PREFIX_FRAMES = 477
MEAN_MAE_THRESHOLD = 1.5
MIN_PSNR_THRESHOLD = 35.0


def sampled_pair(source: Path, extended: Path) -> dict:
    import cv2

    source_capture = cv2.VideoCapture(str(source))
    extended_capture = cv2.VideoCapture(str(extended))
    if not source_capture.isOpened() or not extended_capture.isOpened():
        raise ValueError(f"cannot open prefix pair: {source} / {extended}")
    indices = np.linspace(0, PREFIX_FRAMES - 1, SAMPLE_COUNT, dtype=np.int64)
    maes = []
    psnrs = []
    try:
        for index in indices:
            frames = []
            for capture in (source_capture, extended_capture):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"failed to decode frame {index}: {source} / {extended}")
                frames.append(frame.astype(np.float32))
            if frames[0].shape != frames[1].shape:
                raise ValueError(f"prefix frame shape mismatch at {index}")
            difference = frames[0] - frames[1]
            mae = float(np.mean(np.abs(difference)))
            mse = float(np.mean(difference * difference))
            psnr = 120.0 if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            maes.append(mae)
            psnrs.append(psnr)
    finally:
        source_capture.release()
        extended_capture.release()
    mean_mae = float(np.mean(maes))
    min_psnr = float(min(psnrs))
    return {
        "sampled_frame_indices": indices.tolist(),
        "mean_absolute_pixel_error": mean_mae,
        "max_sample_mae": float(max(maes)),
        "minimum_sample_psnr": min_psnr,
        "exact_sample_fraction": float(np.mean(np.asarray(maes) == 0.0)),
        "prefix_equivalent": bool(
            mean_mae <= MEAN_MAE_THRESHOLD and min_psnr >= MIN_PSNR_THRESHOLD
        ),
    }


def audit(input_manifest: Path, run_base: Path) -> dict:
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v188_post_confirmation_robustness_matrix":
        raise ValueError("invalid v188 input manifest")
    scope = scope_config(manifest, "long60_seed10000_32")
    run_root = run_base / scope["key"]
    published_path = run_root / "published_manifest.json"
    if not published_path.is_file():
        raise ValueError("audit the v188 long60 generation before prefix comparison")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("scope") != scope["key"]
        or tuple(rows) != BASE_METHODS
    ):
        raise ValueError("invalid v188 long60 published evidence")

    source_rows = manifest["v187_provenance"]["source_methods"]
    results = {}
    for method in BASE_METHODS:
        local_dir = Path(rows[method]["video_dir"])
        source_dir = Path(source_rows[method]["video_dir"])
        method_rows = []
        for item in scope["prompt_items"]:
            local_index = int(item["index"])
            v187_index = int(item["v187_index"])
            source = source_dir / f"{v187_index:06d}.mp4"
            extended = local_dir / f"{local_index:06d}.mp4"
            if not source.is_file() or not extended.is_file():
                raise ValueError(f"missing prefix comparison video: {source} / {extended}")
            method_rows.append(
                {
                    "prompt_index": local_index,
                    "v187_index": v187_index,
                    "source_index": int(item["source_index"]),
                    "source_video_size": source.stat().st_size,
                    "extended_video_size": extended.stat().st_size,
                    **sampled_pair(source, extended),
                }
            )
        results[method] = {
            "pair_count": len(method_rows),
            "equivalent_pair_fraction": float(
                np.mean([row["prefix_equivalent"] for row in method_rows])
            ),
            "mean_pair_mae": float(
                np.mean([row["mean_absolute_pixel_error"] for row in method_rows])
            ),
            "minimum_pair_psnr": float(
                min(row["minimum_sample_psnr"] for row in method_rows)
            ),
            "pairs": method_rows,
        }
    supported = all(
        row["equivalent_pair_fraction"] == 1.0 for row in results.values()
    )
    return {
        "version": 1,
        "experiment": "v188_long60_prefix_reproducibility",
        "scope": scope["key"],
        "sample_count_per_video": SAMPLE_COUNT,
        "prefix_frames": PREFIX_FRAMES,
        "thresholds": {
            "mean_mae_le": MEAN_MAE_THRESHOLD,
            "minimum_psnr_ge": MIN_PSNR_THRESHOLD,
        },
        "methods": results,
        "prefix_reproducibility_supported": supported,
        "decision_role": "diagnostic_not_method_selection_gate",
        "interpretation": (
            "Failure indicates duration-dependent or nondeterministic prefix drift; "
            "late-half effects remain measurable but cannot be treated as a pure extension."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input_manifest, args.run_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "[v188-prefix] "
        f"supported={str(report['prefix_reproducibility_supported']).lower()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
