#!/usr/bin/env python3
"""Materialize formal or provisional incremental v179 VBench inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_v174_vbench_comparison as base
from prepare_v179_head_attribution import (
    GENERATED_METHODS,
    METHODS,
    _validate_v178_gate,
    verify,
)


def _validate_provisional_v178(path: Path, prompt_count: int) -> dict:
    if not path.is_file():
        raise ValueError("provisional v178 paired result is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "v178_rccp_holdout_vbench_provisional"
        or payload.get("profile_contract") != "v177"
        or payload.get("provisional") is not True
        or payload.get("membership_decision_allowed") is not False
        or payload.get("membership_hypothesis_gate") is not None
        or payload.get("decision") != "provisional_only_no_membership_decision"
        or int(payload.get("prompt_count", -1)) != prompt_count
        or not isinstance(payload.get("metric_runtime_fingerprint"), dict)
    ):
        raise ValueError("invalid provisional v178 paired result")
    provenance = payload.get("input_provenance") or {}
    comparison_path = Path(str(provenance.get("comparison_manifest", "")))
    summary_path = Path(str(provenance.get("metric_summary", "")))
    for artifact, digest, label in (
        (comparison_path, provenance.get("comparison_manifest_sha256"), "comparison"),
        (summary_path, provenance.get("metric_summary_sha256"), "summary"),
    ):
        if not artifact.is_file() or base.sha256(artifact) != digest:
            raise ValueError(f"provisional v178 {label} provenance drift")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (
        comparison.get("experiment") != "v178_rccp_holdout_vbench_provisional"
        or comparison.get("provisional") is not True
        or int(comparison.get("prompt_count", -1)) != prompt_count
    ):
        raise ValueError("invalid provisional v178 comparison contract")
    payload["_comparison"] = comparison
    return payload


def prepare(
    run_root: Path,
    comparison_root: Path,
    *,
    provisional_count: int | None = None,
    v178_paired_path: Path | None = None,
) -> dict:
    provisional = provisional_count is not None
    if provisional:
        prompt_count = int(provisional_count)
        if not 1 <= prompt_count < 32:
            raise ValueError("provisional-count must be in [1, 31]")
        scope_root = run_root / f"provisional_{prompt_count:02d}"
        generation_experiment = (
            "v179_rccp_head_attribution_generation_provisional"
        )
        comparison_experiment = (
            "v179_rccp_head_attribution_vbench_provisional"
        )
    else:
        prompt_count = 32
        scope_root = run_root
        generation_experiment = "v179_rccp_head_attribution_generation"
        comparison_experiment = (
            "v179_rccp_head_attribution_vbench_incremental"
        )

    published_path = scope_root / "published_manifest.json"
    contract_path = scope_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v179 generation must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    input_path = Path(contract.get("input_manifest", ""))
    if (
        published.get("ok") is not True
        or bool(published.get("complete")) != (not provisional)
        or bool(published.get("provisional")) != provisional
        or bool(published.get("attribution_decision_allowed")) != (not provisional)
        or published.get("experiment") != generation_experiment
        or contract.get("experiment") != generation_experiment
        or published.get("profile_contract") != "v177"
        or contract.get("profile_contract") != "v177"
        or published.get("generation_prompts_used_for_membership") is not False
        or contract.get("generation_prompts_used_for_membership") is not False
        or bool(contract.get("provisional")) != provisional
        or bool(contract.get("attribution_decision_allowed")) != (not provisional)
        or published.get("experiment_contract_sha256") != base.sha256(contract_path)
        or not input_path.is_file()
        or contract.get("input_manifest_sha256") != base.sha256(input_path)
    ):
        raise ValueError("invalid, leaked, or mixed v179 generation artifacts")
    inputs = verify(input_path)

    if int(contract.get("prompt_count", -1)) != prompt_count:
        raise ValueError("v179 generation prompt count drift")
    prompts_path = Path(contract["prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    source_prompt_ids = [int(value) for value in contract["source_prompt_ids"]]
    if (
        len(prompts) != prompt_count
        or len(source_prompt_ids) != prompt_count
        or len(set(source_prompt_ids)) != prompt_count
        or base.sha256(prompts_path) != contract["prompt_file_sha256"]
        or source_prompt_ids != inputs["source_prompt_ids"][:prompt_count]
    ):
        raise ValueError("v179 prompt hash, cardinality, or source mapping drift")

    if provisional:
        if v178_paired_path is None:
            raise ValueError("provisional v179 requires provisional v178 metrics")
        reused = _validate_provisional_v178(v178_paired_path, prompt_count)
        reused_comparison = reused.pop("_comparison")
        if [
            int(row["source_index"])
            for row in reused_comparison["prompt_items"]
        ] != source_prompt_ids:
            raise ValueError("v178/v179 provisional source prompt mapping differs")
    else:
        paired_path = Path(contract.get("v178_paired_result", ""))
        if (
            not paired_path.is_file()
            or base.sha256(paired_path)
            != contract.get("v178_paired_result_sha256")
        ):
            raise ValueError("formal v179 is not bound to passing v178 metrics")
        v178_published_path = Path(
            contract.get("v178_published_manifest", "")
        )
        if (
            not v178_published_path.is_file()
            or base.sha256(v178_published_path)
            != contract.get("v178_published_manifest_sha256")
        ):
            raise ValueError("formal v179 is not bound to audited v178 videos")
        reused, _, v178_contract = _validate_v178_gate(
            paired_path,
            Path(inputs["v178_input_manifest"]),
            v178_published_path.parent,
        )
        if (
            v178_contract.get("prompt_file_sha256")
            != inputs["prompt_file_sha256"]
            or v178_contract.get("source_prompt_ids")
            != inputs["source_prompt_ids"]
        ):
            raise ValueError("formal v178/v179 prompt provenance differs")
        v178_paired_path = paired_path

        reused_comparison_path = Path(
            reused["input_provenance"]["comparison_manifest"]
        )
        if (
            not reused_comparison_path.is_file()
            or base.sha256(reused_comparison_path)
            != reused["input_provenance"]["comparison_manifest_sha256"]
        ):
            raise ValueError("formal v178 comparison provenance drift")
        reused_comparison = json.loads(
            reused_comparison_path.read_text(encoding="utf-8")
        )

    reused_video_rows = {
        row["key"]: row for row in reused_comparison.get("methods") or ()
    }
    if set(("all_recent", "matched")) - set(reused_video_rows):
        raise ValueError("v178 reused video cells are absent")

    published_rows = {row["key"]: row for row in published["methods"]}
    if set(GENERATED_METHODS) - set(published_rows):
        raise ValueError("v179 generated methods are absent")
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    methods = []
    for method in GENERATED_METHODS:
        row = published_rows[method]
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
        methods.append(
            {
                "key": method,
                "role": row["role"],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
            }
        )

    assert v178_paired_path is not None
    payload = {
        "version": 2,
        "experiment": comparison_experiment,
        "profile_contract": "v177",
        "prompt_suite": (
            f"moviegen_qwen_rccp_untouched_generation{prompt_count}_provisional"
            if provisional
            else "moviegen_qwen_rccp_untouched_generation32"
        ),
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
        "provisional": provisional,
        "attribution_decision_allowed": not provisional,
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": 0,
        "methods": methods,
        "factorial_design": contract["factorial_design"],
        "profile_top1_head": contract["profile_top1_head"],
        "reused_metric_cells": {
            method: {
                "v178_paired_result": str(v178_paired_path.resolve()),
                "v178_paired_result_sha256": base.sha256(v178_paired_path),
                "video_dir": str(reused_video_rows[method]["video_dir"]),
            }
            for method in ("all_recent", "matched")
        },
        "required_metric_runtime_fingerprint": reused[
            "metric_runtime_fingerprint"
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
        "new_videos": len(methods) * prompt_count,
        "reused_metric_cells": 2,
        "provisional": provisional,
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--provisional-count", type=int)
    parser.add_argument("--v178-paired", type=Path)
    args = parser.parse_args()
    report = prepare(
        args.run_root,
        args.comparison_root,
        provisional_count=args.provisional_count,
        v178_paired_path=args.v178_paired,
    )
    print(
        "[v179-vbench-prepare] "
        f"provisional={report['provisional']} "
        f"new_methods={report['methods']} new_videos={report['new_videos']} "
        f"reused_cells={report['reused_metric_cells']} "
        f"manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
