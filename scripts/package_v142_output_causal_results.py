#!/usr/bin/env python3
"""Package bounded v142 analysis artifacts for repository review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ANALYSIS_FILES = (
    "analysis_report.json",
    "analysis_summary.md",
    "natural_head_summary.csv",
    "natural_context_head_summary.csv",
    "aba_head_summary.csv",
    "natural_profile_audit.csv",
    "aba_profile_audit.csv",
)
INPUT_FILES = ("suite_metadata.json",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(analysis_dir: Path, input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ANALYSIS_FILES:
        source = analysis_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / name
        shutil.copy2(source, target)
        copied.append(target)
    for name in INPUT_FILES:
        source = input_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / name
        shutil.copy2(source, target)
        copied.append(target)
    inventory = {
        "version": 1,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(copied)
        ],
        "excluded": [
            "raw .pt profiles",
            "generated videos",
            "server logs",
            "prompt text and manifests containing full benchmark prompts",
        ],
    }
    (output_dir / "bundle_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package(args.analysis_dir, args.input_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
