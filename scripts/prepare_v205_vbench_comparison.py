#!/usr/bin/env python3
"""Materialize audited v205 videos as prompt-correct VBench-Long inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v201_head_phase_horizon_screen import sha256
from prepare_v205_horizon_confirmation import (
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
    verify,
)


EXPERIMENT = "v205_profile_disjoint_head_phase_horizon_vbench"
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
            raise RuntimeError(f"refusing mixed v205 VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen v205 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(run_root: Path, comparison_root: Path, input_manifest: Path) -> dict:
    frozen = verify(input_manifest)
    methods = tuple(str(value) for value in frozen["method_order"])
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v205 confirm128 generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v205_profile_disjoint_head_phase_horizon_generation"
        or published.get("scope") != "confirm128"
        or published.get("confirmatory") is not True
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != "confirm128"
        or contract.get("confirmatory") is not True
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
        or int(contract.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(contract.get("seed", -1)) != SEED
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != methods
        or tuple(rows) != methods
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError("invalid or mixed v205 generation artifacts")

    prompt_path = Path(frozen["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    prompt_items = frozen.get("prompt_items") or ()
    if (
        len(prompts) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in prompt_items]
        or sha256(prompt_path) != frozen["prompt_file_sha256"]
    ):
        raise ValueError("v205 prompt contract drifted")

    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    input_video_hashes = {}
    for method in methods:
        source_row = rows[method]
        source_dir = Path(source_row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete v205 canonical video set")
        audit_path = Path(source_row["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_row["audit_sha256"]:
            raise ValueError(f"{method}: source audit drifted")
        target_dir = comparison_root / "published" / method
        expected_targets = {f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)}
        unexpected = {path.name for path in target_dir.glob("*.mp4")} - expected_targets
        if unexpected:
            raise RuntimeError(
                f"refusing stale v205 VBench videos for {method}: {sorted(unexpected)}"
            )
        method_hashes = {}
        for index in range(PROMPT_COUNT):
            source_video = source_dir / f"{index:06d}.mp4"
            method_hashes[source_video.name] = sha256(source_video)
            mode = link_or_validate(
                source_video, target_dir / f"{index:06d}-0.mp4"
            )
            links[mode] += 1
        if {path.name for path in target_dir.glob("*.mp4")} != expected_targets:
            raise RuntimeError(f"incomplete v205 VBench target set: {method}")
        input_video_hashes[method] = method_hashes
        config = frozen["methods"][method]
        comparison_methods.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "routing_map_id": config.get("routing_map_id"),
                "v201_evidence_tier": config.get("v201_evidence_tier"),
                "v201_mechanism_supported": config.get(
                    "v201_mechanism_supported"
                ),
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": source_row["audit_sha256"],
            }
        )

    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "confirmatory": True,
        "prompt_suite": "moviegen_profile_disjoint_source_indices_128_255",
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": SEED,
        "primary_baseline": "sf_native",
        "selected_v201_candidates": frozen["selected_v201_candidates"],
        "v201_provenance": frozen["v201_provenance"],
        "methods": comparison_methods,
        "input_video_sha256": input_video_hashes,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": sha256(input_manifest),
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "generation_contract": str(contract_path.resolve()),
            "generation_contract_sha256": sha256(contract_path),
        },
        "claim_boundary": frozen["claim_boundary"],
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(methods),
        "videos": len(methods) * PROMPT_COUNT,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.input_manifest)
    print(
        "[v205-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
