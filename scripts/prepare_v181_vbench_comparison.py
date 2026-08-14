#!/usr/bin/env python3
"""Materialize prompt-correct VBench-Long inputs for one v181 scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_v174_vbench_comparison as base
from prepare_v181_rccp_long_stress import METHODS, verify

EXPERIMENT = "v181_rccp_long_stress_vbench"


def prepare(run_root: Path, comparison_root: Path, scope_key: str) -> dict:
    scope_root = run_root / "scopes" / scope_key
    published_path = scope_root / "published_manifest.json"
    contract_path = scope_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError(f"v181 {scope_key} must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_manifest = verify(Path(contract["input_manifest"]))
    scope = next(
        (row for row in input_manifest["scopes"] if row["key"] == scope_key),
        None,
    )
    if scope is None:
        raise ValueError(f"v181 input manifest has no scope {scope_key}")
    if (
        published.get("ok") is not True
        or published.get("complete") is not True
        or published.get("experiment") != "v181_rccp_long_stress_generation"
        or contract.get("experiment") != "v181_rccp_long_stress_generation"
        or published.get("scope") != scope_key
        or contract.get("scope") != scope_key
        or published.get("profile_contract") != "v177"
        or contract.get("profile_contract") != "v177"
        or published.get("evaluation_prompts_used_for_membership") is not False
        or contract.get("evaluation_prompts_used_for_membership") is not False
        or published.get("experiment_contract_sha256") != base.sha256(contract_path)
        or contract.get("input_manifest_sha256")
        != base.sha256(Path(contract["input_manifest"]))
    ):
        raise ValueError(f"invalid, leaked, or mixed v181 scope {scope_key}")
    methods = tuple(row["key"] for row in published.get("methods") or ())
    if methods != METHODS or tuple(contract.get("methods") or ()) != METHODS:
        raise ValueError(f"v181 {scope_key} method set or order drift")

    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    source_indices = [int(value) for value in contract["source_prompt_indices"]]
    prompt_count = int(scope["prompt_count"])
    if (
        int(contract.get("prompt_count", -1)) != prompt_count
        or len(prompts) != prompt_count
        or source_indices != scope["prompt_source_indices"]
        or base.sha256(prompt_path) != contract["prompt_file_sha256"]
        or contract["prompt_file_sha256"] != scope["prompt_file_sha256"]
        or int(contract.get("num_output_frames", -1)) != int(scope["num_output_frames"])
        or int(contract.get("seed", -1)) != int(scope["seed"])
        or contract.get("decoded_video_contract") != scope["decoded_video_contract"]
    ):
        raise ValueError(f"v181 {scope_key} prompt, seed, or duration drift")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    for row in published["methods"]:
        method = str(row["key"])
        source_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{scope_key}/{method}: incomplete canonical videos")
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
        "experiment": EXPERIMENT,
        "profile_contract": "v177",
        "scope": scope_key,
        "scope_priority": scope["priority"],
        "prompt_suite": f"moviegen_source_indices_{source_indices[0]:04d}_{source_indices[-1]:04d}",
        "prompt_count": prompt_count,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": [
            {
                "index": index,
                "source_index": source_indices[index],
                "text": prompts[index],
            }
            for index in range(prompt_count)
        ],
        "evaluation_prompts_used_for_membership": False,
        "num_output_frames": int(scope["num_output_frames"]),
        "decoded_video_contract": scope["decoded_video_contract"],
        "seed": int(scope["seed"]),
        "reseed_per_prompt": True,
        "pf_required": False,
        "methods": comparison_methods,
        "vbench_long_dimensions": list(base.DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": base.sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": base.sha256(contract_path),
            "input_manifest": contract["input_manifest"],
            "input_manifest_sha256": contract["input_manifest_sha256"],
            "v178_paired_result": contract["v178_paired_result"],
            "v178_paired_result_sha256": contract["v178_paired_result_sha256"],
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = base.write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "scope": scope_key,
        "methods": len(methods),
        "videos": len(methods) * prompt_count,
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.scope)
    print(
        "[v181-vbench-prepare] "
        f"scope={report['scope']} methods={report['methods']} "
        f"videos={report['videos']} links={report['link_counts']} "
        f"manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
