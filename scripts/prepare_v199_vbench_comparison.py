#!/usr/bin/env python3
"""Materialize the audited v199 videos for prompt-correct VBench-Long."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v199_retrieval_storage_attribution import (
    EXPERIMENT,
    METHODS,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    sha256,
    verify,
)

VBENCH_EXPERIMENT = "v199_retrieval_storage_attribution_vbench"


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v199 VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def prepare(run_root: Path, comparison_root: Path) -> dict:
    input_path = run_root / "inputs" / "manifest.json"
    published_path = run_root / "published_manifest.json"
    inputs = verify(input_path)
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if (
        published.get("experiment") != EXPERIMENT
        or published.get("ok") is not True
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("input_manifest_sha256") != sha256(input_path)
        or tuple(row.get("key") for row in published.get("methods") or ()) != METHODS
    ):
        raise ValueError("v199 generation has not passed the frozen audit")
    published_rows = {row["key"]: row for row in published["methods"]}
    input_rows = {row["key"]: row for row in inputs["methods"]}
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    methods = []
    for method in METHODS:
        source_dir = Path(published_rows[method]["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"incomplete v199 published grid: {method}")
        target_dir = comparison_root / "published" / method
        for index in range(PROMPT_COUNT):
            mode = link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            links[mode] += 1
        methods.append(
            {
                **input_rows[method],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "generation_audit": published_rows[method]["path"],
                "generation_audit_sha256": published_rows[method]["sha256"],
            }
        )
    payload = {
        "version": 1,
        "experiment": VBENCH_EXPERIMENT,
        "source_experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_file": inputs["prompt_file"],
        "prompt_file_sha256": inputs["prompt_file_sha256"],
        "prompt_items": inputs["prompt_items"],
        "source_indices": inputs["source_indices"],
        "seed": inputs["seed"],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "clips_per_video": NUM_OUTPUT_FRAMES // 8,
        "decoded_video_contract": inputs["decoded_video_contract"],
        "methods": methods,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "input_manifest": str(input_path.resolve()),
            "input_manifest_sha256": sha256(input_path),
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
        },
        "selection_policy": (
            "Choose the smallest archive capacity that passes full/late "
            "noninferiority and temporal safety unless a larger archive has "
            "paired CI-supported benefit."
        ),
        "manual_review_required": False,
        "claim_boundary": inputs["claim_boundary"],
    }
    path = comparison_root / "comparison_manifest.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError("frozen v199 VBench comparison differs")
    path.write_bytes(encoded)
    return {
        "manifest": str(path.resolve()),
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
    report = prepare(args.run_root.resolve(), args.comparison_root.resolve())
    print(
        "[v199-vbench-prepare] PASS "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
