#!/usr/bin/env python3
"""Materialize one audited v208 scope for prompt-correct VBench-Long."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v207_context_budget_phase_screen import sha256
from prepare_v207_vbench_comparison import DIMENSIONS
from prepare_v208_paper_confirmation import (
    DECODED_CONTRACT,
    EXPERIMENT as INPUT_EXPERIMENT,
    METHOD_ORDER,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
    selection_splits,
)


def experiment(scope: str) -> str:
    return f"v208_{scope}_vbench_long"


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v208 VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def prepare(run_root: Path, comparison_root: Path, scope: str) -> dict:
    if scope not in NUM_OUTPUT_FRAMES:
        raise ValueError(f"unsupported v208 scope: {scope}")
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v208 generation scope must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v208_paper_confirmation_generation"
        or published.get("scope") != scope
        or contract.get("scope") != scope
        or contract.get("paper_stage") is not True
        or contract.get("methods") != list(METHOD_ORDER)
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or int(contract.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES[scope]
        or contract.get("decoded_video_contract") != DECODED_CONTRACT[scope]
        or int(contract.get("seed", -1)) != SEED
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("invalid v208 generation artifacts")
    rows = {str(row["key"]): row for row in published["methods"]}
    selection_splits(contract["prompt_items"])
    if set(rows) != set(METHOD_ORDER) or any(not rows[key]["ok"] for key in METHOD_ORDER):
        raise ValueError("v208 method membership or audit drift")
    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompts) != PROMPT_COUNT
        or sha256(prompt_path) != contract["prompt_file_sha256"]
        or prompts != [str(row["text"]) for row in contract["prompt_items"]]
    ):
        raise ValueError("v208 prompt provenance drift")
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    methods = []
    for method in METHOD_ORDER:
        source_dir = Path(rows[method]["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"v208 incomplete video set: {scope}/{method}")
        target_dir = comparison_root / "published" / method
        targets = {f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)}
        stale = {path.name for path in target_dir.glob("*.mp4")} - targets
        if stale:
            raise RuntimeError(f"refusing stale v208 VBench videos: {method}/{sorted(stale)}")
        for index in range(PROMPT_COUNT):
            mode = link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            links[mode] += 1
        methods.append(
            {
                "key": method,
                "role": rows[method]["role"],
                "parent_method": rows[method].get("parent_method"),
                "operator": rows[method].get("operator"),
                "schedule": rows[method].get("schedule"),
                "read_frame_equivalents": rows[method]["read_frame_equivalents"],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )
    payload = {
        "version": 1,
        "experiment": experiment(scope),
        "source_experiment": INPUT_EXPERIMENT,
        "scope": scope,
        "paper_stage": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": contract["prompt_items"],
        "num_output_frames": NUM_OUTPUT_FRAMES[scope],
        "decoded_video_contract": DECODED_CONTRACT[scope],
        "seed": SEED,
        "frozen_parent_method": contract["frozen_parent_method"],
        "primary_baseline": "sf_native",
        "equal_budget_control": "recent21",
        "methods": methods,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": sha256(contract_path),
        },
        "claim_boundary": (
            "Primary tests use the 96 prompts outside v207 selection. Historical "
            "prompt freshness is unverified. Full128 is a descriptive benchmark "
            "table. A pass requires positive support and full/late non-inferiority "
            "against SF and Recent21, plus temporal safety. Dynamic Degree is "
            "reported but excluded from the confirmatory decision."
        ),
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise RuntimeError("frozen v208 VBench manifest differs")
    manifest_path.write_bytes(encoded)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "scope": scope,
        "methods": len(methods),
        "videos": len(methods) * PROMPT_COUNT,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(NUM_OUTPUT_FRAMES), required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.scope)
    print(
        "[v208-vbench-prepare] "
        f"scope={report['scope']} methods={report['methods']} "
        f"videos={report['videos']} links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
