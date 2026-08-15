#!/usr/bin/env python3
"""Materialize prompt-correct VBench inputs from the recovered v180 grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_v174_vbench_comparison as base
from audit_v183_v180_recovery import METHODS, PROMPT_COUNT


EXPERIMENT = "v183_v180_recovery_vbench"


def prepare(recovery_root: Path, comparison_root: Path) -> dict:
    published_path = recovery_root / "published_manifest.json"
    contract_path = recovery_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("run the full v183 recovery audit before VBench preparation")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("complete") is not True
        or published.get("experiment") != "v183_v180_recovery_generation"
        or contract.get("experiment") != "v183_v180_recovery_generation"
        or published.get("experiment_contract_sha256") != base.sha256(contract_path)
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(contract.get("methods") or ()) != METHODS
        or contract.get("evaluation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid or mixed v183 recovery artifacts")
    formal = bool(contract.get("formal_rccp_membership_claim_allowed"))
    if formal != bool(published.get("formal_rccp_membership_claim_allowed")):
        raise ValueError("v183 recovery claim scope drift")

    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    source_indices = [int(value) for value in contract["source_prompt_indices"]]
    if (
        len(prompts) != PROMPT_COUNT
        or source_indices != list(range(128, 256))
        or base.sha256(prompt_path) != contract["prompt_file_sha256"]
    ):
        raise ValueError("v183 prompt mapping or hash drift")

    observed_methods = tuple(row.get("key") for row in published.get("methods") or ())
    if observed_methods != METHODS:
        raise ValueError("v183 published method order drift")
    comparison_methods = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    expected_names = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    for row in published["methods"]:
        method = str(row["key"])
        source_dir = Path(row["video_dir"])
        if {path.name for path in source_dir.glob("*.mp4")} != expected_names:
            raise ValueError(f"{method}: incomplete recovered video set")
        target_dir = comparison_root / "published" / method
        for index in range(PROMPT_COUNT):
            link_counts[
                base.link_or_validate(
                    source_dir / f"{index:06d}.mp4",
                    target_dir / f"{index:06d}-0.mp4",
                )
            ] += 1
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
        "source_experiment": contract["source_experiment"],
        "evidence_scope": contract["evidence_scope"],
        "formal_rccp_membership_claim_allowed": formal,
        "claim_boundary": contract["claim_boundary"],
        "prompt_suite": "moviegen_source_indices_0128_0255",
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": [
            {
                "index": index,
                "source_index": source_indices[index],
                "text": prompts[index],
            }
            for index in range(PROMPT_COUNT)
        ],
        "evaluation_prompts_used_for_membership": False,
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
            "source_input_manifest": contract["source_input_manifest"],
            "source_input_manifest_sha256": contract["source_input_manifest_sha256"],
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = base.write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(METHODS),
        "videos": len(METHODS) * PROMPT_COUNT,
        "link_counts": link_counts,
        "evidence_scope": payload["evidence_scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.recovery_root, args.comparison_root)
    print(
        "[v183-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"scope={report['evidence_scope']} links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
