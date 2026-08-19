#!/usr/bin/env python3
"""Materialize audited v187 videos as prompt-correct VBench-Long inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v187_unseen128_confirmation import METHODS, PROMPT_COUNT, SEED


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
            raise RuntimeError(f"refusing mixed v187 VBench input: {target}")
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
        raise RuntimeError(f"frozen v187 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(run_root: Path, comparison_root: Path, input_manifest: Path) -> dict:
    frozen = json.loads(input_manifest.read_text(encoding="utf-8"))
    if (
        frozen.get("experiment")
        != "v187_unseen128_phase_operator_confirmation"
        or frozen.get("scope") != "confirmatory_unseen128"
        or frozen.get("confirmatory") is not True
        or tuple(frozen.get("method_order") or ()) != METHODS
        or int(frozen.get("prompt_count", -1)) != PROMPT_COUNT
        or int(frozen.get("seed", -1)) != SEED
    ):
        raise ValueError("invalid v187 frozen input manifest")

    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v187 confirm128 generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v187_unseen128_phase_operator_generation"
        or published.get("scope") != "confirm128"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != "confirm128"
        or contract.get("confirmatory") is not True
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
        or int(contract.get("seed", -1)) != SEED
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != METHODS
        or tuple(rows) != METHODS
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError("invalid or mixed v187 generation artifacts")

    prompt_items = frozen.get("prompt_items") or ()
    prompt_path = Path(frozen["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompt_items) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in prompt_items]
        or sha256(prompt_path) != frozen["prompt_file_sha256"]
    ):
        raise ValueError("v187 prompt contract drifted")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    input_video_hashes = {}
    for method in METHODS:
        source_row = rows[method]
        source_dir = Path(source_row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical video set")
        audit_path = Path(source_row["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_row["audit_sha256"]:
            raise ValueError(f"{method}: source audit drifted")
        target_dir = comparison_root / "published" / method
        method_hashes = {}
        for index in range(PROMPT_COUNT):
            source_path = source_dir / f"{index:06d}.mp4"
            digest = sha256(source_path)
            mode = link_or_validate(
                source_path,
                target_dir / f"{index:06d}-0.mp4",
            )
            link_counts[mode] += 1
            method_hashes[source_path.name] = digest
        input_video_hashes[method] = method_hashes
        method_config = frozen["methods"][method]
        comparison_methods.append(
            {
                "key": method,
                "role": method_config["role"],
                "schedule": method_config.get("schedule"),
                "operator": method_config.get("operator"),
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": source_row["audit_sha256"],
            }
        )

    payload = {
        "version": 1,
        "experiment": "v187_unseen128_phase_operator_vbench",
        "confirmatory": True,
        "prompt_suite": "moviegen_unseen_source_indices_128_255",
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": 120,
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": SEED,
        "selected_v186_method": frozen["selected_v186_method"],
        "selected_schedule": frozen["selected_schedule"],
        "selected_operator": frozen["selected_operator"],
        "development_reference": {
            "selected_candidate_status": frozen["v186_provenance"].get(
                "selected_candidate_status"
            ),
            "selection_rule": frozen["v186_provenance"].get(
                "decision_selection_rule"
            ),
        },
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
        "claim_boundary": (
            "Paired 128-prompt confirmation on one Self-Forcing model. "
            "Cross-model and longer-duration transfer remain separate claims."
        ),
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(METHODS),
        "videos": len(METHODS) * PROMPT_COUNT,
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.input_manifest)
    print(
        "[v187-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']} manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
