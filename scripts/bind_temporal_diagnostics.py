#!/usr/bin/env python3
"""Bind a temporal-diagnostic CSV to its exact comparison manifest and videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_contract(comparison_manifest: Path, temporal_csv: Path) -> dict:
    manifest = load_json(comparison_manifest)
    methods = tuple(str(row["key"]) for row in manifest.get("methods") or ())
    prompt_count = int(manifest.get("prompt_count", -1))
    if not methods or prompt_count <= 0:
        raise ValueError("comparison manifest has no complete method grid")
    video_dirs = {
        str(row["key"]): Path(str(row["video_dir"]))
        for row in manifest["methods"]
    }
    expected = {
        (method, prompt) for method in methods for prompt in range(prompt_count)
    }
    observed = {}
    with temporal_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row.get("method", ""))
            prompt = int(row.get("prompt_index", -1))
            sample = int(row.get("sample_index", -1))
            key = (method, prompt)
            if key in observed:
                raise ValueError(f"duplicate temporal diagnostic row: {key}")
            if method not in video_dirs or sample != 0 or not 0 <= prompt < prompt_count:
                raise ValueError(f"invalid temporal diagnostic row: {key}, sample={sample}")
            actual = Path(str(row.get("video", "")))
            expected_video = video_dirs[method] / f"{prompt:06d}-0.mp4"
            if not actual.is_file() or not expected_video.is_file():
                raise ValueError(f"missing temporal input video for {key}")
            if not actual.samefile(expected_video):
                raise ValueError(
                    f"temporal input is not the comparison video for {key}: {actual}"
                )
            observed[key] = str(actual.resolve())
    if set(observed) != expected:
        raise ValueError(
            "temporal diagnostic grid mismatch: "
            f"missing={sorted(expected-set(observed))[:12]} "
            f"extra={sorted(set(observed)-expected)[:12]}"
        )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "comparison_manifest": str(comparison_manifest.resolve()),
        "comparison_manifest_sha256": sha256(comparison_manifest),
        "temporal_csv": str(temporal_csv.resolve()),
        "temporal_csv_sha256": sha256(temporal_csv),
        "methods": list(methods),
        "prompt_count": prompt_count,
        "row_count": len(observed),
        "video_grid_verified_by_samefile": True,
    }


def write_contract(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def verify_contract(
    path: Path,
    comparison_manifest: Path,
    temporal_csv: Path,
) -> dict:
    expected = build_contract(comparison_manifest, temporal_csv)
    observed = load_json(path)
    if observed != expected:
        raise ValueError("temporal diagnostic provenance contract drifted")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind", "verify"))
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "bind":
        payload = build_contract(args.comparison_manifest, args.temporal_csv)
        digest = write_contract(args.output, payload)
    else:
        payload = verify_contract(
            args.output, args.comparison_manifest, args.temporal_csv
        )
        digest = sha256(args.output)
    print(
        "[temporal-contract] PASS "
        f"action={args.action} rows={payload['row_count']} sha256={digest}"
    )


if __name__ == "__main__":
    main()
