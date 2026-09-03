#!/usr/bin/env python3
"""Materialize prompt-correct VBench-Long inputs for the v201 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v201_head_phase_horizon_screen import (
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
    sha256,
)

EXPERIMENT = "v201_head_phase_horizon_causal_vbench_screen32"
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


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v201 VBench input: {target}")
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
        raise ValueError("v201 generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v201_head_phase_horizon_causal_generation"
        or published.get("scope") != "screen32"
        or contract.get("scope") != "screen32"
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or contract.get("primary_baseline") != "sf_native"
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
        or int(contract.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(contract.get("seed", -1)) != SEED
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("invalid v201 generation artifacts")
    rows = {str(row["key"]): row for row in published["methods"]}
    methods = [str(value) for value in contract["methods"]]
    if (
        not methods
        or methods[0] != "sf_native"
        or set(rows) != set(methods)
        or any(rows[key].get("ok") is not True for key in methods)
    ):
        raise ValueError("v201 method membership or audit drift")
    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    prompt_items = contract.get("prompt_items") or ()
    if (
        len(prompts) != PROMPT_COUNT
        or len(prompt_items) != PROMPT_COUNT
        or sha256(prompt_path) != contract["prompt_file_sha256"]
        or prompts != [str(row["text"]) for row in prompt_items]
    ):
        raise ValueError("v201 prompt provenance drift")
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    for method in methods:
        row = rows[method]
        source_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"v201 incomplete video set: {method}")
        target_dir = comparison_root / "published" / method
        expected_targets = {f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)}
        unexpected = {path.name for path in target_dir.glob("*.mp4")} - expected_targets
        if unexpected:
            raise RuntimeError(
                f"refusing stale v201 VBench videos for {method}: {sorted(unexpected)}"
            )
        for index in range(PROMPT_COUNT):
            mode = link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            links[mode] += 1
        if {path.name for path in target_dir.glob("*.mp4")} != expected_targets:
            raise RuntimeError(f"incomplete v201 VBench target set: {method}")
        comparison_methods.append(
            {
                "key": method,
                "role": row["role"],
                "runtime": row["runtime"],
                "operator": row.get("operator"),
                "routing_map_id": row.get("routing_map_id"),
                "map_classification": row.get("map_classification"),
                "coverage_count_by_position": row.get("coverage_count_by_position"),
                "coverage_exposure_count": row.get("coverage_exposure_count"),
                "coverage_exposure_fraction": row.get("coverage_exposure_fraction"),
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": SEED,
        "operators": contract["operators"],
        "primary_baseline": "sf_native",
        "operator_contracts": contract["operator_contracts"],
        "methods": comparison_methods,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": sha256(contract_path),
        },
        "claim_boundary": (
            "Aggregate means are descriptive. The v201 decision requires "
            "paired prompt-level full and half-window comparisons, exact "
            "equal-exposure controls, and automatic temporal safety."
        ),
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise RuntimeError("frozen v201 VBench manifest differs")
    manifest_path.write_bytes(encoded)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "methods": len(methods),
        "videos": len(methods) * PROMPT_COUNT,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v201-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
