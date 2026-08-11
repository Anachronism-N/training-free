#!/usr/bin/env python3
"""Materialize prompt-correct VBench-Long inputs for a v174 scope."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


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
            raise RuntimeError(f"refusing mixed VBench input: {target}")
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
        raise RuntimeError(f"frozen VBench manifest differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(run_root: Path, comparison_root: Path) -> dict:
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v174 generation must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v174_cache_compat_generation"
        or contract.get("experiment") != "v174_cache_compat_generation"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("invalid or mixed v174 generation artifacts")
    methods = tuple(row["key"] for row in published["methods"])
    if methods != tuple(contract["methods"]):
        raise ValueError("v174 method order differs from generation contract")
    prompt_count = int(contract["prompt_count"])
    prompts_path = Path(contract["prompt_file"])
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    if (
        len(prompts) != 128
        or sha256(prompts_path) != contract["prompt_file_sha256"]
    ):
        raise ValueError("v174 prompt suite hash drift")

    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    comparison_methods = []
    for row in published["methods"]:
        method = str(row["key"])
        source_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(prompt_count)}
        observed = {path.name for path in source_dir.glob("*.mp4")}
        if observed != expected:
            raise ValueError(f"{method}: incomplete canonical video set")
        target_dir = comparison_root / "published" / method
        for index in range(prompt_count):
            mode = link_or_validate(
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
        "experiment": f"v174_cache_compat_vbench_{run_root.name}",
        "prompt_suite": "moviegen_128_qwen_rewrite",
        "prompt_count": prompt_count,
        "prompt_file_sha256": contract["prompt_file_sha256"],
        "prompt_items": [
            {"index": index, "source_index": index, "text": prompts[index]}
            for index in range(prompt_count)
        ],
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": 0,
        "methods": comparison_methods,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "published_manifest": str(published_path.resolve()),
            "published_manifest_sha256": sha256(published_path),
            "experiment_contract": str(contract_path.resolve()),
            "experiment_contract_sha256": sha256(contract_path),
        },
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
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root)
    print(
        "[v174-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"links={report['link_counts']} manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
