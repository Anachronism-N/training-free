#!/usr/bin/env python3
"""Compute deterministic camera-compensated motion diagnostics for video grids.

The script is intentionally a diagnostic rather than a paper metric.  Dense
Farneback flow is decomposed into a robust global affine field and a residual
field.  This prevents a camera pan from being counted as evidence that the
generated subject kept moving.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:  # pragma: no cover - OpenCV is supplied by the server environment.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


VERSION = 1
ACTIVE_RESIDUAL_NDPS = 0.01
LOW_RESIDUAL_NDPS = 0.0025
CSV_FIELDS = (
    "method",
    "prompt_index",
    "sample_index",
    "video",
    "decoded_frames",
    "retained_frames",
    "flow_transition_count",
    "fps",
    "duration_seconds",
    "frame_step",
    "sample_interval_seconds",
    "analysis_width",
    "analysis_height",
    "raw_motion_ndps_median",
    "global_motion_ndps_median",
    "residual_motion_ndps_median",
    "residual_motion_p90_ndps_median",
    "residual_transition_active_fraction",
    "residual_active_area_fraction_mean",
    "late_residual_motion_ratio",
    "longest_low_residual_run_fraction",
    "residual_accel_outlier_fraction",
    "camera_motion_fraction_median",
    "camera_fit_valid_fraction",
    "camera_fit_inlier_fraction_median",
    "camera_fit_error_nd_median",
    "residual_energy_concentration_mean",
    "residual_direction_entropy_mean",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return sha256(path)


def load_video_grid(manifest_path: Path) -> tuple[dict, list[tuple[str, int, Path]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_count = int(manifest.get("prompt_count", -1))
    methods = manifest.get("methods") or ()
    method_names = tuple(str(row.get("key", "")) for row in methods)
    if (
        prompt_count <= 0
        or not method_names
        or len(set(method_names)) != len(method_names)
    ):
        raise ValueError("comparison manifest has no complete method grid")
    grid = []
    for row, method in zip(methods, method_names):
        directory = Path(str(row.get("video_dir", "")))
        if not directory.is_dir():
            raise FileNotFoundError(f"missing comparison video directory: {directory}")
        expected = {f"{index:06d}-0.mp4" for index in range(prompt_count)}
        observed = {path.name for path in directory.glob("*.mp4")}
        if observed != expected:
            raise ValueError(
                f"camera-motion grid mismatch for {method}: "
                f"missing={sorted(expected - observed)[:12]} "
                f"extra={sorted(observed - expected)[:12]}"
            )
        grid.extend(
            (method, index, (directory / f"{index:06d}-0.mp4").resolve())
            for index in range(prompt_count)
        )
    return manifest, grid


def _normalized_design(
    height: int, width: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x_norm = 2.0 * xx.astype(np.float64) / max(width - 1, 1) - 1.0
    y_norm = 2.0 * yy.astype(np.float64) / max(height - 1, 1) - 1.0
    design = np.column_stack((x_norm.ravel(), y_norm.ravel(), np.ones(xx.size)))
    return design, yy.ravel(), xx.ravel()


def robust_global_affine(
    flow: np.ndarray,
    *,
    sample_stride: int = 8,
    iterations: int = 6,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Fit a deterministic robust affine displacement field.

    Flow is normalized by the image diagonal before fitting.  Iteratively
    reweighted least squares starts from the median translation, so a compact
    moving foreground does not pull the global camera estimate toward itself.
    """

    if flow.ndim != 3 or flow.shape[2] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow must be a finite HxWx2 array")
    height, width = flow.shape[:2]
    if height < 2 or width < 2 or sample_stride <= 0:
        raise ValueError("flow geometry and sample stride must be positive")
    diagonal = math.hypot(height, width)
    design, yy, xx = _normalized_design(height, width, sample_stride)
    response = flow[yy, xx].astype(np.float64) / diagonal
    if response.shape[0] < 6:
        translation = np.median(response, axis=0)
        beta = np.vstack((np.zeros((2, 2)), translation))
        valid = False
    else:
        beta = np.vstack((np.zeros((2, 2)), np.median(response, axis=0)))
        valid = True
        for _ in range(iterations):
            residual = np.linalg.norm(response - design @ beta, axis=1)
            median = float(np.median(residual))
            cutoff = max(2.5 * median, 1e-6)
            weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_response = response * np.sqrt(weights)[:, None]
            fitted, _, rank, _ = np.linalg.lstsq(
                weighted_design, weighted_response, rcond=None
            )
            if rank < 3 or not np.isfinite(fitted).all():
                valid = False
                break
            beta = fitted
    full_design, full_y, full_x = _normalized_design(height, width, 1)
    global_normalized = (full_design @ beta).reshape(height, width, 2)
    sampled_residual = np.linalg.norm(response - design @ beta, axis=1)
    median_residual = float(np.median(sampled_residual))
    mad = float(np.median(np.abs(sampled_residual - median_residual)))
    inlier_threshold = max(median_residual + 3.0 * 1.4826 * mad, 5e-4)
    inlier_fraction = float(np.mean(sampled_residual <= inlier_threshold))
    if inlier_fraction < 0.25:
        valid = False
    # Keep these variables explicit: they document that the full field follows
    # the same row-major geometry as the sampled fit.
    assert full_y.size == full_x.size == height * width
    return global_normalized, {
        "valid": bool(valid),
        "inlier_fraction": inlier_fraction,
        "median_fit_error_nd": median_residual,
    }


def _direction_entropy(vectors: np.ndarray, magnitude: np.ndarray) -> float:
    threshold = max(ACTIVE_RESIDUAL_NDPS, float(np.percentile(magnitude, 75)))
    active = magnitude >= threshold
    if not np.any(active):
        return 0.0
    angles = np.arctan2(vectors[..., 1][active], vectors[..., 0][active])
    weights = magnitude[active]
    bins = np.floor((angles + math.pi) * 8.0 / (2.0 * math.pi)).astype(int) % 8
    histogram = np.bincount(bins, weights=weights, minlength=8).astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1e-12)
    nonzero = probabilities > 0
    return float(
        -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / math.log(8.0)
    )


def transition_diagnostics(
    flow: np.ndarray, *, delta_seconds: float
) -> dict[str, float]:
    if delta_seconds <= 0:
        raise ValueError("delta_seconds must be positive")
    height, width = flow.shape[:2]
    diagonal = math.hypot(height, width)
    normalized = flow.astype(np.float64) / diagonal
    global_field, fit = robust_global_affine(flow)
    residual = normalized - global_field
    raw_speed = np.linalg.norm(normalized, axis=-1) / delta_seconds
    global_speed = np.linalg.norm(global_field, axis=-1) / delta_seconds
    residual_speed = np.linalg.norm(residual, axis=-1) / delta_seconds
    energy = np.square(residual_speed).ravel()
    top_count = max(1, math.ceil(0.10 * energy.size))
    top_energy = float(np.partition(energy, energy.size - top_count)[-top_count:].sum())
    total_energy = float(energy.sum())
    raw_median = float(np.median(raw_speed))
    global_median = float(np.median(global_speed))
    residual_median = float(np.median(residual_speed))
    return {
        "raw_median": raw_median,
        "global_median": global_median,
        "residual_median": residual_median,
        "residual_p90": float(np.percentile(residual_speed, 90)),
        "residual_active_area": float(np.mean(residual_speed >= ACTIVE_RESIDUAL_NDPS)),
        "camera_fraction": global_median / max(global_median + residual_median, 1e-12),
        "fit_valid": float(bool(fit["valid"])),
        "fit_inlier_fraction": float(fit["inlier_fraction"]),
        "fit_error_nd": float(fit["median_fit_error_nd"]),
        "residual_energy_concentration": top_energy / max(total_energy, 1e-12),
        "residual_direction_entropy": _direction_entropy(
            residual / delta_seconds, residual_speed
        ),
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.tolist():
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _robust_outlier_fraction(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 3.0 * max(1.4826 * mad, 1e-10)
    return float(np.mean(values > threshold))


def aggregate_transitions(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one flow transition is required")

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    residual_p90 = values("residual_p90")
    late_count = max(1, math.ceil(residual_p90.size * 0.25))
    earlier = residual_p90[:-late_count]
    if not earlier.size:
        earlier = residual_p90
    late = residual_p90[-late_count:]
    low = residual_p90 < LOW_RESIDUAL_NDPS
    acceleration = np.abs(np.diff(residual_p90))
    return {
        "raw_motion_ndps_median": float(np.median(values("raw_median"))),
        "global_motion_ndps_median": float(np.median(values("global_median"))),
        "residual_motion_ndps_median": float(np.median(values("residual_median"))),
        "residual_motion_p90_ndps_median": float(np.median(residual_p90)),
        "residual_transition_active_fraction": float(
            np.mean(residual_p90 >= ACTIVE_RESIDUAL_NDPS)
        ),
        "residual_active_area_fraction_mean": float(
            np.mean(values("residual_active_area"))
        ),
        "late_residual_motion_ratio": float(np.median(late))
        / max(float(np.median(earlier)), 1e-12),
        "longest_low_residual_run_fraction": float(_longest_true_run(low))
        / residual_p90.size,
        "residual_accel_outlier_fraction": _robust_outlier_fraction(acceleration),
        "camera_motion_fraction_median": float(np.median(values("camera_fraction"))),
        "camera_fit_valid_fraction": float(np.mean(values("fit_valid"))),
        "camera_fit_inlier_fraction_median": float(
            np.median(values("fit_inlier_fraction"))
        ),
        "camera_fit_error_nd_median": float(np.median(values("fit_error_nd"))),
        "residual_energy_concentration_mean": float(
            np.mean(values("residual_energy_concentration"))
        ),
        "residual_direction_entropy_mean": float(
            np.mean(values("residual_direction_entropy"))
        ),
    }


def _resize_gray(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def analyze_video(path: Path, *, max_width: int, frame_step: int) -> dict[str, object]:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("compute_v193_camera_motion.py requires opencv-python")
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS: {path}")
    decoded = retained = 0
    previous = None
    analysis_shape = None
    transitions = []
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
            analysis_shape = gray.shape
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                transitions.append(
                    transition_diagnostics(flow, delta_seconds=frame_step / fps)
                )
            previous = gray
    finally:
        capture.release()
    if retained < 3 or analysis_shape is None:
        raise RuntimeError(f"video has fewer than three retained frames: {path}")
    return {
        "video": str(path.resolve()),
        "decoded_frames": decoded,
        "retained_frames": retained,
        "flow_transition_count": len(transitions),
        "fps": fps,
        "duration_seconds": decoded / fps,
        "frame_step": frame_step,
        "sample_interval_seconds": frame_step / fps,
        "analysis_width": int(analysis_shape[1]),
        "analysis_height": int(analysis_shape[0]),
        **aggregate_transitions(transitions),
    }


def _runtime_contract(
    manifest_path: Path,
    *,
    max_width: int,
    frame_step: int,
    num_shards: int,
) -> dict:
    return {
        "version": VERSION,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__)),
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "max_width": max_width,
        "frame_step": frame_step,
        "active_residual_ndps": ACTIVE_RESIDUAL_NDPS,
        "low_residual_ndps": LOW_RESIDUAL_NDPS,
        "num_shards": num_shards,
        "numpy_version": np.__version__,
        "opencv_version": getattr(cv2, "__version__", "unavailable"),
        "deterministic_global_model": "IRLS full-affine over diagonal-normalized flow",
    }


def compute_part(
    manifest_path: Path,
    output: Path,
    contract_path: Path,
    *,
    max_width: int,
    frame_step: int,
    workers: int,
    shard_index: int,
    num_shards: int,
) -> dict:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("compute_v193_camera_motion.py requires opencv-python")
    if (
        max_width <= 0
        or frame_step <= 0
        or workers <= 0
        or num_shards <= 0
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError("invalid v193 compute geometry or shard configuration")
    manifest, grid = load_video_grid(manifest_path)
    selected = [
        item for index, item in enumerate(grid) if index % num_shards == shard_index
    ]
    if not selected:
        raise ValueError(f"v193 shard {shard_index} has no videos")

    def run_one(item: tuple[str, int, Path]) -> dict:
        method, prompt_index, path = item
        return {
            "method": method,
            "prompt_index": prompt_index,
            "sample_index": 0,
            **analyze_video(path, max_width=max_width, frame_step=frame_step),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, item): item for item in selected}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[v193-motion] shard={shard_index}/{num_shards} "
                f"{completed}/{len(selected)} method={row['method']} "
                f"prompt={row['prompt_index']} residual_p90="
                f"{row['residual_motion_p90_ndps_median']:.6g}"
            )
    rows.sort(key=lambda row: (str(row["method"]), int(row["prompt_index"])))
    csv_digest = _write_csv(output, rows)
    contract = {
        **_runtime_contract(
            manifest_path,
            max_width=max_width,
            frame_step=frame_step,
            num_shards=num_shards,
        ),
        "kind": "part",
        "shard_index": shard_index,
        "row_count": len(rows),
        "methods": [str(row["key"]) for row in manifest["methods"]],
        "prompt_count": int(manifest["prompt_count"]),
        "output_csv": str(output.resolve()),
        "output_csv_sha256": csv_digest,
    }
    _write_json(contract_path, contract)
    return contract


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"unexpected v193 CSV schema: {path}")
        return list(reader)


def merge_parts(
    manifest_path: Path,
    parts_dir: Path,
    output: Path,
    contract_path: Path,
    *,
    expected_shards: int,
) -> dict:
    if expected_shards <= 0:
        raise ValueError("expected_shards must be positive")
    manifest, grid = load_video_grid(manifest_path)
    expected_methods = [str(row["key"]) for row in manifest["methods"]]
    expected_keys = {(method, prompt) for method, prompt, _ in grid}
    base_contract = None
    rows_by_key = {}
    part_sources = []
    for shard in range(expected_shards):
        csv_path = parts_dir / f"part_{shard:02d}_of_{expected_shards:02d}.csv"
        metadata_path = csv_path.with_suffix(".contract.json")
        if not csv_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing v193 shard artifacts: {csv_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {
            key: metadata[key]
            for key in (
                "version",
                "implementation_sha256",
                "comparison_manifest_sha256",
                "max_width",
                "frame_step",
                "active_residual_ndps",
                "low_residual_ndps",
                "num_shards",
                "numpy_version",
                "opencv_version",
                "deterministic_global_model",
            )
        }
        if base_contract is None:
            base_contract = comparable
        if (
            comparable != base_contract
            or metadata.get("implementation_sha256") != sha256(Path(__file__))
            or metadata.get("comparison_manifest_sha256") != sha256(manifest_path)
            or metadata.get("kind") != "part"
            or int(metadata.get("shard_index", -1)) != shard
            or int(metadata.get("num_shards", -1)) != expected_shards
            or metadata.get("methods") != expected_methods
            or int(metadata.get("prompt_count", -1)) != int(manifest["prompt_count"])
            or Path(str(metadata.get("output_csv", ""))).resolve() != csv_path.resolve()
            or metadata.get("output_csv_sha256") != sha256(csv_path)
        ):
            raise ValueError(f"mixed or drifted v193 part: {csv_path}")
        part_rows = _read_rows(csv_path)
        if len(part_rows) != int(metadata.get("row_count", -1)):
            raise ValueError(f"v193 part row count drifted: {csv_path}")
        for row in part_rows:
            key = (str(row["method"]), int(row["prompt_index"]))
            if key in rows_by_key:
                raise ValueError(f"duplicate v193 row: {key}")
            expected_path = next(
                path for method, prompt, path in grid if (method, prompt) == key
            )
            actual_path = Path(row["video"])
            if not actual_path.is_file() or not actual_path.samefile(expected_path):
                raise ValueError(f"v193 row is not bound to comparison video: {key}")
            rows_by_key[key] = row
        part_sources.append(
            {
                "csv": str(csv_path.resolve()),
                "csv_sha256": sha256(csv_path),
                "contract": str(metadata_path.resolve()),
                "contract_sha256": sha256(metadata_path),
            }
        )
    if set(rows_by_key) != expected_keys:
        raise ValueError(
            "v193 merged grid mismatch: "
            f"missing={sorted(expected_keys - set(rows_by_key))[:12]} "
            f"extra={sorted(set(rows_by_key) - expected_keys)[:12]}"
        )
    ordered_rows = [rows_by_key[(method, prompt)] for method, prompt, _ in grid]
    csv_digest = _write_csv(output, ordered_rows)
    contract = {
        **(base_contract or {}),
        "kind": "merged",
        "methods": expected_methods,
        "prompt_count": int(manifest["prompt_count"]),
        "row_count": len(ordered_rows),
        "comparison_manifest": str(manifest_path.resolve()),
        "output_csv": str(output.resolve()),
        "output_csv_sha256": csv_digest,
        "part_sources": part_sources,
    }
    _write_json(contract_path, contract)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    compute = subparsers.add_parser("compute")
    compute.add_argument("--comparison-manifest", type=Path, required=True)
    compute.add_argument("--output", type=Path, required=True)
    compute.add_argument("--contract", type=Path, required=True)
    compute.add_argument("--max-width", type=int, default=256)
    compute.add_argument("--frame-step", type=int, default=8)
    compute.add_argument("--workers", type=int, default=8)
    compute.add_argument("--shard-index", type=int, default=0)
    compute.add_argument("--num-shards", type=int, default=1)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--comparison-manifest", type=Path, required=True)
    merge.add_argument("--parts-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "compute":
        report = compute_part(
            args.comparison_manifest,
            args.output,
            args.contract,
            max_width=args.max_width,
            frame_step=args.frame_step,
            workers=args.workers,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    else:
        report = merge_parts(
            args.comparison_manifest,
            args.parts_dir,
            args.output,
            args.contract,
            expected_shards=args.expected_shards,
        )
    print(
        f"[v193-motion] PASS action={args.action} rows={report['row_count']} "
        f"contract={args.contract}"
    )


if __name__ == "__main__":
    main()
