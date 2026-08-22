#!/usr/bin/env python3
"""Materialize prompt-correct VBench-Long inputs for v190 screen32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "overall_consistency",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_style",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v190 VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def prepare(run_root: Path, comparison_root: Path) -> dict:
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v190 generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v190_head_phase_causal_generation"
        or published.get("scope") != "screen32"
        or contract.get("scope") != "screen32"
        or int(contract.get("prompt_count", -1)) != 32
        or contract.get("prompt_indices") != list(range(32))
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("invalid v190 generation artifacts")
    rows = {str(row["key"]): row for row in published["methods"]}
    methods = [str(value) for value in contract["methods"]]
    if set(rows) != set(methods) or methods[0] != "all_recent":
        raise ValueError("v190 method membership drift")
    control_aliases = {
        str(key): str(value)
        for key, value in (contract.get("control_aliases") or {}).items()
    }
    if any(
        alias in methods or target not in methods
        for alias, target in control_aliases.items()
    ):
        raise ValueError("v190 control alias drift")
    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    prompt_items = contract.get("prompt_items") or ()
    if (
        len(prompts) != 32
        or len(prompt_items) != 32
        or sha256(prompt_path) != contract["prompt_file_sha256"]
        or prompts != [str(row["text"]) for row in prompt_items]
    ):
        raise ValueError("v190 prompt provenance drift")
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    for method in methods:
        row = rows[method]
        if row.get("ok") is not True:
            raise ValueError(f"v190 method did not pass audit: {method}")
        source_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(32)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"v190 incomplete video set: {method}")
        target_dir = comparison_root / "published" / method
        for index in range(32):
            mode = link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            links[mode] += 1
        comparison_methods.append(
            {
                "key": method,
                "role": row["role"],
                "operator": row["operator"],
                "phase_map_id": row["phase_map_id"],
                "coverage_count_by_call": row["coverage_count_by_call"],
                "coverage_cell_count": row["coverage_cell_count"],
                "coverage_exposure_fraction": row[
                    "coverage_exposure_fraction"
                ],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )
    payload = {
        "version": 1,
        "experiment": "v190_head_phase_causal_vbench_screen32",
        "development_only": True,
        "prompt_count": 32,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": contract["seed"],
        "methods": comparison_methods,
        "control_aliases": control_aliases,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": sha256(contract_path),
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise RuntimeError("frozen v190 VBench manifest differs")
    manifest_path.write_bytes(encoded)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "methods": len(methods),
        "videos": len(methods) * 32,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v190-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
