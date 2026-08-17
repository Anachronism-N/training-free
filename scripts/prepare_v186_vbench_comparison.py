#!/usr/bin/env python3
"""Materialize the mixed reused/generated VBench-Long inputs for v186."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v186_phase_operator_screen import (
    GENERATED_METHODS,
    METHODS,
    PROMPT_COUNT,
)


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
            raise RuntimeError(f"refusing mixed v186 VBench input: {target}")
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
        raise RuntimeError(f"frozen v186 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _load_generated(run_root: Path, input_manifest: Path) -> dict[str, dict]:
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v186 screen32 generation must be audited first")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v186_phase_conditioned_operator_generation"
        or published.get("scope") != "screen32"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != "screen32"
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != GENERATED_METHODS
    ):
        raise ValueError("invalid or mixed v186 generated artifacts")
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if tuple(rows) != GENERATED_METHODS or not all(row.get("ok") for row in rows.values()):
        raise ValueError("v186 generated method audit is incomplete")
    return rows


def prepare(run_root: Path, comparison_root: Path, input_manifest: Path) -> dict:
    frozen = json.loads(input_manifest.read_text(encoding="utf-8"))
    if (
        frozen.get("experiment") != "v186_phase_conditioned_operator_screen"
        or tuple(frozen.get("method_order") or ()) != METHODS
        or tuple(frozen.get("generated_methods") or ()) != GENERATED_METHODS
        or int(frozen.get("prompt_count", -1)) != PROMPT_COUNT
    ):
        raise ValueError("invalid v186 frozen input manifest")
    generated = _load_generated(run_root, input_manifest)
    prompt_items = frozen.get("prompt_items") or ()
    prompt_path = Path(frozen["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompt_items) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in prompt_items]
        or sha256(prompt_path) != frozen["prompt_file_sha256"]
    ):
        raise ValueError("v186 prompt contract drifted")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    input_video_hashes = {}
    for method in METHODS:
        method_config = frozen["methods"][method]
        if method in GENERATED_METHODS:
            source_row = generated[method]
            source_dir = Path(source_row["video_dir"])
            source_evidence = {
                "kind": "v186_generated",
                "audit": source_row["audit"],
                "audit_sha256": source_row["audit_sha256"],
            }
        else:
            source_dir = Path(method_config["source_video_dir"])
            source_evidence = {
                "kind": "v184_reused",
                "source_method": method_config["source_method"],
                "audit": method_config["source_audit"],
                "audit_sha256": method_config["source_audit_sha256"],
            }
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical video set")
        audit_path = Path(source_evidence["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_evidence["audit_sha256"]:
            raise ValueError(f"{method}: source audit drifted")
        target_dir = comparison_root / "published" / method
        method_hashes = {}
        for index in range(PROMPT_COUNT):
            source_path = source_dir / f"{index:06d}.mp4"
            digest = sha256(source_path)
            if method not in GENERATED_METHODS and digest != method_config[
                "source_video_sha256"
            ][source_path.name]:
                raise ValueError(f"{method}: reused video hash drift: {source_path.name}")
            mode = link_or_validate(
                source_path,
                target_dir / f"{index:06d}-0.mp4",
            )
            link_counts[mode] += 1
            method_hashes[source_path.name] = digest
        input_video_hashes[method] = method_hashes
        comparison_methods.append(
            {
                "key": method,
                "role": (
                    "local_control"
                    if method == "all_recent"
                    else "random_operator_reference"
                    if method == "phase_reservoir"
                    else "deterministic_operator_candidate"
                ),
                "schedule": method_config["schedule"],
                "operator": method_config["operator"],
                "middle_storage_capacity": method_config[
                    "middle_storage_capacity"
                ],
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_evidence": source_evidence,
            }
        )

    payload = {
        "version": 1,
        "experiment": "v186_phase_conditioned_operator_vbench_screen32",
        "development_only": True,
        "prompt_suite": "moviegen_qwen_systematic32_reused_from_v184",
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": list(prompt_items),
        "num_output_frames": 120,
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": 0,
        "selected_v184_method": frozen["selected_v184_method"],
        "selected_schedule": frozen["selected_schedule"],
        "methods": comparison_methods,
        "input_video_sha256": input_video_hashes,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": sha256(input_manifest),
            "generated_published_manifest": str(
                (run_root / "published_manifest.json").resolve()
            ),
            "generated_published_manifest_sha256": sha256(
                run_root / "published_manifest.json"
            ),
        },
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
        "[v186-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']} manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
