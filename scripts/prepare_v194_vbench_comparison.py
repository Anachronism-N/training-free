#!/usr/bin/env python3
"""Materialize the audited v194 transfer grid as prompt-correct VBench inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v194_cf_checkpoint_transfer import METHODS, verify

EXPERIMENT = "v194_causal_checkpoint_transfer_vbench"


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
            raise RuntimeError(f"refusing mixed v194 VBench input: {target}")
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
        raise RuntimeError(f"frozen v194 VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(run_root: Path, comparison_root: Path, input_manifest: Path) -> dict:
    frozen = verify(input_manifest)
    prompt_count = int(frozen["prompt_count"])
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v194 generation must be audited before VBench preparation")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {str(row.get("key")): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("complete") is not True
        or published.get("experiment") != "v194_causal_checkpoint_transfer_generation"
        or published.get("run_kind") != "full"
        or published.get("confirmatory") is not True
        or int(published.get("prompt_count", -1)) != prompt_count
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("experiment") != "v194_causal_checkpoint_transfer_generation"
        or contract.get("run_kind") != "full"
        or contract.get("prompt_indices") != list(range(prompt_count))
        or int(contract.get("num_output_frames", -1))
        != int(frozen["num_output_frames"])
        or int(contract.get("seed", -1)) != int(frozen["seed"])
        or contract.get("prompt_file_sha256") != frozen["prompt_file_sha256"]
        or contract.get("prompt_items") != frozen["prompt_items"]
        or contract.get("decoded_video_contract") != frozen["decoded_video_contract"]
        or contract.get("checkpoint_sha256") != frozen["checkpoint"]["sha256"]
        or contract.get("checkpoint_state_key") != "generator"
        or int(contract.get("common_model_local_attn_size", -1)) != 21
        or contract.get("input_manifest_sha256") != sha256(input_manifest)
        or tuple(contract.get("methods") or ()) != METHODS
        or tuple(rows) != METHODS
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError("invalid or mixed v194 generation artifacts")

    prompt_path = Path(frozen["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompts) != prompt_count
        or prompts != [str(row["text"]) for row in frozen["prompt_items"]]
        or sha256(prompt_path) != frozen["prompt_file_sha256"]
    ):
        raise ValueError("v194 prompt contract drifted")

    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    source_file_sizes = {}
    for method in METHODS:
        source_row = rows[method]
        source_dir = Path(source_row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        if {path.name for path in source_dir.glob("*.mp4")} != expected:
            raise ValueError(f"{method}: incomplete canonical videos")
        audit_path = Path(source_row["audit"])
        if not audit_path.is_file() or sha256(audit_path) != source_row["audit_sha256"]:
            raise ValueError(f"{method}: source audit drifted")
        target_dir = comparison_root / "published" / method
        sizes = {}
        for index in range(prompt_count):
            source = source_dir / f"{index:06d}.mp4"
            if source.stat().st_size <= 0:
                raise ValueError(f"empty v194 video: {source}")
            sizes[source.name] = int(source.stat().st_size)
            links[link_or_validate(source, target_dir / f"{index:06d}-0.mp4")] += 1
        source_file_sizes[method] = sizes
        config = frozen["methods"][method]
        comparison_methods.append(
            {
                "key": method,
                "role": config["role"],
                "runtime": config["runtime"],
                "operator": config.get("operator"),
                "phase_map_id": config.get("phase_map_id"),
                "read_frame_equivalents": config.get("read_frame_equivalents"),
                "source_video_dir": str(source_dir.resolve()),
                "video_dir": str(target_dir.resolve()),
                "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": source_row["audit_sha256"],
            }
        )

    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "confirmatory": True,
        "transfer_axis": frozen["transfer_axis"],
        "prompt_count": prompt_count,
        "prompt_file_sha256": frozen["prompt_file_sha256"],
        "prompt_items": frozen["prompt_items"],
        "prompt_positions_in_v192": frozen["prompt_positions_in_v192"],
        "num_output_frames": int(frozen["num_output_frames"]),
        "decoded_video_contract": frozen["decoded_video_contract"],
        "seed": int(frozen["seed"]),
        "candidate": frozen["candidate"],
        "local_control": frozen["local_control"],
        "native_control": frozen["native_control"],
        "positive_metrics_to_transfer": frozen["positive_metrics_to_transfer"],
        "selected_v190_method": frozen["selected_v190_method"],
        "selected_operator": frozen["selected_operator"],
        "checkpoint_sha256": frozen["checkpoint"]["sha256"],
        "methods": comparison_methods,
        "source_video_sizes": source_file_sizes,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": sha256(input_manifest),
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "generation_contract": str(contract_path.resolve()),
            "generation_contract_sha256": sha256(contract_path),
        },
        "provenance_note": (
            "The decoded source audit and hardlink/samefile checks bind every video. "
            "Whole MP4 SHA hashing is omitted to keep the automatic audit bounded."
        ),
        "claim_boundary": frozen["claim_boundary"],
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, payload)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "methods": len(METHODS),
        "videos": len(METHODS) * prompt_count,
        "link_counts": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.input_manifest)
    print(
        "[v194-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']}"
    )


if __name__ == "__main__":
    main()
