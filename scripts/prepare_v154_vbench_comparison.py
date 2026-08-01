#!/usr/bin/env python3
"""Materialize prompt-correct VBench inputs for the v154 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPERIMENT = "v154_history_critical_moviebench16"
COMPARISON_EXPERIMENT = "v154_history_critical_vbench16"
PROMPT_COUNT = 16
METHODS = (
    "sf_native",
    "ours_qk_top4",
    "ours_qk_bottom4_control",
    "ours_qk_random4_control",
    "ours_all_recent8_control",
    "ours_all_prototype4_control",
    "ours_legacy_membership",
    "ours_legacy_reference",
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def link_or_validate(source: Path, target: Path) -> str:
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"missing source video: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed VBench input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source)
        return "symlink"


def write_frozen_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen VBench manifest differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def prepare(
    run_root: Path,
    comparison_root: Path,
    prompt_manifest_path: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    comparison_root = comparison_root.resolve()
    published_path = run_root / "published_manifest.json"
    contract_path = run_root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError("v154 generation must be audited before VBench prepare")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(
        prompt_manifest_path.read_text(encoding="utf-8")
    )
    method_rows = published.get("methods") or []
    method_keys = tuple(row.get("key") for row in method_rows)
    if (
        not published.get("ok")
        or published.get("experiment") != EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or method_keys != METHODS
        or contract.get("experiment") != EXPERIMENT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("invalid or mixed v154 published artifacts")
    prompt_items = prompt_manifest.get("items") or []
    if (
        prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
        or int(prompt_manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(prompt_items) != PROMPT_COUNT
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest.get("prompt_file_sha256")
    ):
        raise ValueError("v154 prompt manifest differs from generation contract")

    expected_source = {
        f"{index:06d}.mp4" for index in range(PROMPT_COUNT)
    }
    expected_target = {
        comparison_name(index) for index in range(PROMPT_COUNT)
    }
    comparison_methods = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for row in method_rows:
        method = str(row["key"])
        source_dir = Path(str(row["video_dir"])).resolve()
        observed = {path.name for path in source_dir.glob("*.mp4")}
        if observed != expected_source:
            raise ValueError(f"{method}: incomplete v154 published video set")
        target_dir = comparison_root / "published" / method
        extras = (
            {path.name for path in target_dir.glob("*.mp4")}
            if target_dir.is_dir()
            else set()
        ) - expected_target
        if extras:
            raise ValueError(f"{method}: unexpected VBench inputs: {extras}")
        for index in range(PROMPT_COUNT):
            mode = link_or_validate(
                source_dir / f"{index:06d}.mp4",
                target_dir / comparison_name(index),
            )
            link_counts[mode] += 1
        actual = {path.name for path in target_dir.glob("*.mp4")}
        if actual != expected_target:
            raise RuntimeError(f"{method}: VBench materialization incomplete")
        comparison_methods.append(
            {
                "key": method,
                "role": row.get("role"),
                "source_video_dir": str(source_dir),
                "video_dir": str(target_dir.resolve()),
            }
        )

    payload = {
        "version": 1,
        "experiment": COMPARISON_EXPERIMENT,
        "prompt_suite": prompt_manifest["suite"],
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": prompt_manifest["prompt_file_sha256"],
        "prompt_items": [
            {
                "index": index,
                "source_index": int(row["source_index"]),
                "text": str(row["text"]),
            }
            for index, row in enumerate(prompt_items)
        ],
        "num_output_frames": int(contract["num_output_frames"]),
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": int(contract["seed"]),
        "methods": comparison_methods,
        "vbench_long_dimensions": list(DIMENSIONS),
        "source": {
            "published_manifest": str(published_path),
            "published_manifest_sha256": sha256(published_path),
            "experiment_contract": str(contract_path),
            "experiment_contract_sha256": sha256(contract_path),
            "prompt_manifest": str(prompt_manifest_path.resolve()),
            "prompt_manifest_sha256": sha256(prompt_manifest_path),
        },
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    digest = write_frozen_json(manifest_path, payload)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "methods": len(comparison_methods),
        "videos": len(comparison_methods) * PROMPT_COUNT,
        "link_counts": link_counts,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / "v154_history_critical_moviebench16" / "full8",
    )
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=root / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    args = parser.parse_args()
    if args.comparison_root is None:
        args.comparison_root = args.run_root / "vbench_comparison"
    return args


def main() -> None:
    args = parse_args()
    report = prepare(
        args.run_root,
        args.comparison_root,
        args.prompt_manifest,
    )
    print(
        "[v154-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"manifest_sha256={report['manifest_sha256']} "
        f"links={report['link_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
