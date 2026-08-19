#!/usr/bin/env python3
"""Materialize one audited v188 scope as prompt-correct VBench-Long inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v188_robustness_matrix import scope_config, sha256


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v188 VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen v188 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(
    run_root: Path,
    comparison_root: Path,
    input_manifest: Path,
    scope_key: str,
) -> dict:
    frozen = json.loads(input_manifest.read_text(encoding="utf-8"))
    if frozen.get("experiment") != "v188_post_confirmation_robustness_matrix":
        raise ValueError("invalid v188 frozen input manifest")
    scope = scope_config(frozen, scope_key)
    methods = tuple(scope["methods"])
    prompt_count = int(scope["prompt_count"])

    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError(f"v188 {scope_key} generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v188_robustness_generation"
        or published.get("scope") != scope_key
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != scope_key
        or int(contract.get("prompt_count", -1)) != prompt_count
        or contract.get("prompt_indices") != list(range(prompt_count))
        or int(contract.get("num_output_frames", -1))
        != int(scope["num_output_frames"])
        or int(contract.get("seed", -1)) != int(scope["seed"])
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != methods
        or tuple(rows) != methods
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError(f"invalid or mixed v188 generation artifacts: {scope_key}")

    prompt_items = scope.get("prompt_items") or ()
    prompt_path = Path(scope["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompt_items) != prompt_count
        or prompts != [str(row["text"]) for row in prompt_items]
        or sha256(prompt_path) != scope["prompt_file_sha256"]
    ):
        raise ValueError(f"v188 prompt contract drifted: {scope_key}")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    input_video_hashes = {}
    for method in methods:
        source_row = rows[method]
        source_dir = Path(source_row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical v188 video set")
        audit_path = Path(source_row["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_row["audit_sha256"]:
            raise ValueError(f"{method}: v188 source audit drifted")
        target_dir = comparison_root / "published" / method
        method_hashes = {}
        for index in range(prompt_count):
            source_path = source_dir / f"{index:06d}.mp4"
            digest = sha256(source_path)
            mode = link_or_validate(source_path, target_dir / f"{index:06d}-0.mp4")
            link_counts[mode] += 1
            method_hashes[source_path.name] = digest
        input_video_hashes[method] = method_hashes
        config = frozen["method_templates"][method]
        comparison_methods.append(
            {
                "key": method,
                "role": config["role"],
                "schedule": config.get("schedule"),
                "operator": config.get("operator"),
                "execution": source_row["execution"],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": source_row["audit_sha256"],
            }
        )

    payload = {
        "version": 1,
        "experiment": f"v188_{scope_key}_vbench",
        "scope": scope_key,
        "purpose": scope["purpose"],
        "confirmatory_extension": True,
        "prompt_suite": "v187_unseen128_outcome_blind_disjoint_partition",
        "prompt_count": prompt_count,
        "prompt_file_sha256": scope["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": scope["num_output_frames"],
        "decoded_video_contract": scope["decoded_video_contract"],
        "seed": scope["seed"],
        "selected_schedule": frozen["selected_schedule"],
        "opposite_schedule": frozen["opposite_schedule"],
        "selected_operator": frozen["selected_operator"],
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
        "analysis_contract": frozen["analysis_contract"],
        "claim_boundary": frozen["claim_boundary"],
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, payload)
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
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.input_manifest, args.scope)
    print(
        "[v188-vbench-prepare] "
        f"scope={args.scope} methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']} manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
