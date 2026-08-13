#!/usr/bin/env python3
"""Materialize prompt-correct VBench-Long inputs for v178."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_v174_vbench_comparison as base


def prepare(run_root: Path, comparison_root: Path) -> dict:
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v178 generation must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v178_rccp_holdout_generation"
        or contract.get("experiment") != "v178_rccp_holdout_generation"
        or published.get("profile_contract") != "v177"
        or contract.get("profile_contract") != "v177"
        or published.get("generation_prompts_used_for_membership") is not False
        or contract.get("generation_prompts_used_for_membership") is not False
        or published.get("experiment_contract_sha256") != base.sha256(contract_path)
    ):
        raise ValueError("invalid, leaked, or mixed v178 generation artifacts")
    methods = tuple(row["key"] for row in published["methods"])
    if methods != tuple(contract["methods"]):
        raise ValueError("v178 method order differs from generation contract")
    if methods != (
        "matched",
        "all_recent",
        "hard_negative_0",
        "hard_negative_1",
        "hard_negative_2",
        "hard_negative_3",
    ):
        raise ValueError("v178 method set drift")
    prompt_count = int(contract["prompt_count"])
    prompts_path = Path(contract["prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    source_prompt_ids = [int(value) for value in contract["source_prompt_ids"]]
    if (
        prompt_count != 32
        or len(prompts) != prompt_count
        or base.sha256(prompts_path) != contract["prompt_file_sha256"]
    ):
        raise ValueError("v178 holdout prompt hash or cardinality drift")
    if len(source_prompt_ids) != prompt_count or len(set(source_prompt_ids)) != prompt_count:
        raise ValueError("v178 source prompt mapping drift")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    for row in published["methods"]:
        method = str(row["key"])
        source_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical video set")
        target_dir = comparison_root / "published" / method
        for index in range(prompt_count):
            mode = base.link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            link_counts[mode] += 1
        comparison_methods.append(
            {
                "key": method,
                "role": row["role"],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )
    payload = {
        "version": 1,
        "experiment": "v178_rccp_holdout_vbench",
        "profile_contract": "v177",
        "prompt_suite": "moviegen_qwen_rccp_untouched_generation32",
        "prompt_count": prompt_count,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": [
            {
                "index": index,
                "source_index": source_prompt_ids[index],
                "text": prompts[index],
            }
            for index in range(prompt_count)
        ],
        "generation_prompts_used_for_membership": False,
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": 0,
        "methods": comparison_methods,
        "vbench_long_dimensions": list(base.DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": base.sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": base.sha256(contract_path),
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = base.write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(methods),
        "videos": len(methods) * prompt_count,
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v178-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']} manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
