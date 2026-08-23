#!/usr/bin/env python3
"""Materialize one audited v192 scope as prompt-correct VBench inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v192_head_phase_robustness import METHODS, scope_config, verify
from prepare_v191_vbench_comparison import DIMENSIONS


EXPERIMENT = "v192_head_phase_robustness_vbench"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v192 VBench input: {target}")
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
        raise RuntimeError(f"frozen v192 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(
    run_root: Path,
    comparison_root: Path,
    input_manifest: Path,
    scope_key: str,
) -> dict:
    frozen = verify(input_manifest)
    scope = scope_config(frozen, scope_key)
    prompt_count = int(scope["prompt_count"])
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError(f"v192 {scope_key} generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {str(row.get("key")): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("complete") is not True
        or published.get("experiment") != "v192_head_phase_robustness_generation"
        or published.get("scope") != scope_key
        or published.get("run_kind") != "full"
        or published.get("confirmatory") is not True
        or int(published.get("prompt_count", -1)) != prompt_count
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("experiment") != "v192_head_phase_robustness_generation"
        or contract.get("scope") != scope_key
        or contract.get("run_kind") != "full"
        or contract.get("prompt_indices") != list(range(prompt_count))
        or int(contract.get("num_output_frames", -1))
        != int(scope["num_output_frames"])
        or int(contract.get("seed", -1)) != int(scope["seed"])
        or contract.get("prompt_file_sha256") != scope["prompt_file_sha256"]
        or contract.get("prompt_items") != scope["prompt_items"]
        or contract.get("decoded_video_contract") != scope["decoded_video_contract"]
        or contract.get("selected_v190_method") != frozen["selected_v190_method"]
        or contract.get("selected_operator") != frozen["selected_operator"]
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != METHODS
        or tuple(rows) != METHODS
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError(f"invalid or mixed v192 generation artifacts: {scope_key}")

    prompt_path = Path(scope["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    prompt_items = list(scope["prompt_items"])
    if (
        len(prompts) != prompt_count
        or prompts != [str(row["text"]) for row in prompt_items]
        or sha256(prompt_path) != scope["prompt_file_sha256"]
    ):
        raise ValueError(f"v192 prompt contract drifted: {scope_key}")

    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    input_video_hashes = {}
    for method in METHODS:
        source_row = rows[method]
        source_dir = Path(source_row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{scope_key}/{method}: incomplete canonical videos")
        audit_path = Path(source_row["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_row["audit_sha256"]:
            raise ValueError(f"{scope_key}/{method}: source audit drifted")
        target_dir = comparison_root / "published" / method
        hashes = {}
        for index in range(prompt_count):
            source = source_dir / f"{index:06d}.mp4"
            hashes[source.name] = sha256(source)
            links[link_or_validate(source, target_dir / f"{index:06d}-0.mp4")] += 1
        input_video_hashes[method] = hashes
        config = frozen["methods"][method]
        comparison_methods.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "phase_map_id": config.get("phase_map_id"),
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": source_row["audit_sha256"],
            }
        )

    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": scope_key,
        "scope_role": scope["role"],
        "confirmatory": True,
        "prompt_count": prompt_count,
        "prompt_file_sha256": scope["prompt_file_sha256"],
        "prompt_items": prompt_items,
        "prompt_positions_in_v191": scope["prompt_positions_in_v191"],
        "num_output_frames": int(scope["num_output_frames"]),
        "decoded_video_contract": scope["decoded_video_contract"],
        "seed": int(scope["seed"]),
        "selected_v190_method": frozen["selected_v190_method"],
        "selected_operator": frozen["selected_operator"],
        "v191_positive_metrics_to_replicate": frozen[
            "v191_positive_metrics_to_replicate"
        ],
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
        "scope": scope_key,
        "methods": len(METHODS),
        "videos": len(METHODS) * prompt_count,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    report = prepare(
        args.run_root,
        args.comparison_root,
        args.input_manifest,
        args.scope,
    )
    print(
        "[v192-vbench-prepare] "
        f"scope={report['scope']} methods={report['methods']} "
        f"videos={report['videos']} links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
