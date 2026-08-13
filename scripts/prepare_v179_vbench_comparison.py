#!/usr/bin/env python3
"""Materialize only the two new v179 cells for incremental VBench-Long."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_v174_vbench_comparison as base
from prepare_v179_head_attribution import GENERATED_METHODS, METHODS, verify


def prepare(run_root: Path, comparison_root: Path) -> dict:
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v179 generation must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_path = Path(contract.get("input_manifest", ""))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v179_rccp_head_attribution_generation"
        or contract.get("experiment") != "v179_rccp_head_attribution_generation"
        or published.get("profile_contract") != "v177"
        or contract.get("profile_contract") != "v177"
        or published.get("generation_prompts_used_for_membership") is not False
        or contract.get("generation_prompts_used_for_membership") is not False
        or published.get("experiment_contract_sha256") != base.sha256(contract_path)
        or not input_path.is_file()
        or contract.get("input_manifest_sha256") != base.sha256(input_path)
    ):
        raise ValueError("invalid, leaked, or mixed v179 generation artifacts")
    inputs = verify(input_path)
    if (
        tuple(contract.get("methods") or ()) != METHODS
        or tuple(contract.get("generated_methods") or ()) != GENERATED_METHODS
        or tuple(row["key"] for row in published.get("methods") or ()) != METHODS
    ):
        raise ValueError("v179 generated method set drift")

    prompts_path = Path(contract["prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    source_prompt_ids = [int(value) for value in contract["source_prompt_ids"]]
    if (
        len(prompts) != 32
        or len(source_prompt_ids) != 32
        or len(set(source_prompt_ids)) != 32
        or base.sha256(prompts_path) != contract["prompt_file_sha256"]
        or contract["prompt_file_sha256"] != inputs["prompt_file_sha256"]
    ):
        raise ValueError("v179 prompt hash, cardinality, or source mapping drift")

    published_rows = {row["key"]: row for row in published["methods"]}
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    methods = []
    for method in GENERATED_METHODS:
        row = published_rows.get(method) or {}
        source_dir = Path(str(row.get("video_dir", "")))
        expected = {f"{index:06d}.mp4" for index in range(32)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical video set")
        target_dir = comparison_root / "published" / method
        for index in range(32):
            mode = base.link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / f"{index:06d}-0.mp4",
            )
            link_counts[mode] += 1
        methods.append(
            {
                "key": method,
                "role": row["role"],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )

    payload = {
        "version": 1,
        "experiment": "v179_rccp_head_attribution_vbench_incremental",
        "profile_contract": "v177",
        "prompt_suite": "moviegen_qwen_rccp_untouched_generation32",
        "prompt_count": 32,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": [
            {
                "index": index,
                "source_index": source_prompt_ids[index],
                "text": prompts[index],
            }
            for index in range(32)
        ],
        "generation_prompts_used_for_membership": False,
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": 0,
        "methods": methods,
        "factorial_design": contract["factorial_design"],
        "profile_top1_head": contract["profile_top1_head"],
        "reused_metric_cells": {
            method: {
                "v178_paired_result": inputs["v178_paired_result"],
                "v178_paired_result_sha256": inputs["v178_paired_result_sha256"],
            }
            for method in ("all_recent", "matched")
        },
        "required_metric_runtime_fingerprint": inputs[
            "v178_metric_runtime_fingerprint"
        ],
        "vbench_long_dimensions": list(base.DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": base.sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": base.sha256(contract_path),
            "input_manifest": str(input_path.resolve()),
            "input_manifest_sha256": base.sha256(input_path),
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = base.write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(methods),
        "new_videos": len(methods) * 32,
        "reused_metric_cells": 2,
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v179-vbench-prepare] "
        f"new_methods={report['methods']} new_videos={report['new_videos']} "
        f"reused_cells={report['reused_metric_cells']} "
        f"manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
