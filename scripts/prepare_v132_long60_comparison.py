#!/usr/bin/env python3
"""Assemble the completed v132 SF/main 60-second run for VBench-Long."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from prepare_v132_ablation_comparison import (
    CORE_DIMENSIONS,
    SEMANTIC_DIMENSIONS,
    materialize,
    prompt_items,
    Source,
)
from run_v100_fast_selection_1video import sha256, write_frozen


EXPERIMENT = "v132_moviebench128_60s"
COMPARISON_EXPERIMENT = "v132_main_60s_comparison"
SOURCE_CANDIDATE = "prototype_retrieval1_age24"
SOURCE_METHODS = ("sf_native", "ours_prototype_retrieval1_age24")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--run-root", type=Path, default=env_path("V132_LONG60_ROOT"))
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=env_path("V132_LONG60_COMPARISON_ROOT"),
    )
    parser.add_argument("--prompts", type=Path, default=env_path("V132_PROMPTS"))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    digest = hashlib.sha256(SOURCE_CANDIDATE.encode("ascii")).hexdigest()[:12]
    args.run_root = (
        args.run_root
        or args.repo_root
        / "runs"
        / EXPERIMENT
        / f"ours1_{digest}"
    ).resolve()
    args.comparison_root = (
        args.comparison_root
        or args.repo_root / "runs" / "v132_main_60s_comparison"
    ).resolve()
    args.prompts = (
        args.prompts
        or Path(
            "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
            "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
        )
    ).resolve()
    return args


def main() -> None:
    args = parse_args()
    published_path = args.run_root / "published_manifest.json"
    contract_path = args.run_root / "contracts" / "experiment.json"
    for path in (args.prompts, published_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    published = load_json(published_path)
    contract = load_json(contract_path)
    failures = []
    if published.get("experiment") != EXPERIMENT or not published.get("ok"):
        failures.append("published experiment/audit")
    if [row.get("key") for row in published.get("methods", [])] != list(
        SOURCE_METHODS
    ):
        failures.append("published method order")
    if published.get("prompt_count") != 128:
        failures.append("published prompt count")
    if published.get("prompt_file_sha256") != sha256(args.prompts):
        failures.append("published prompt hash")
    if contract.get("num_output_frames") != 240:
        failures.append("contract output frames")
    if contract.get("decoded_video_contract") != {
        "frames": 957,
        "fps": 16,
        "duration_seconds": 59.8125,
    }:
        failures.append("contract decoded video")
    if published.get("experiment_contract_sha256") != sha256(contract_path):
        failures.append("contract hash")
    if failures:
        raise RuntimeError("invalid v132 long60 source: " + ", ".join(failures))

    # materialize() only requires roots with published/<source_key>.
    args.v125_root = args.run_root
    sources = (
        Source("sf_native", args.run_root, "sf_native", "same_backbone_baseline"),
        Source(
            "ours_main",
            args.run_root,
            "ours_prototype_retrieval1_age24",
            "selected_binary_memory",
        ),
    )
    rows = [materialize(args, source) for source in sources]
    manifest = {
        "version": 1,
        "experiment": COMPARISON_EXPERIMENT,
        "prompt_suite": "AMA MovieGen-128 Qwen Rewrite",
        "prompt_count": 128,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
        "prompt_items": prompt_items(args.prompts),
        "num_output_frames": 240,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "reseed_per_prompt": True,
        "pf_required": False,
        "methods": rows,
        "metric_profiles": {
            "core": list(CORE_DIMENSIONS),
            "semantic_extension": list(SEMANTIC_DIMENSIONS),
            "full": list(CORE_DIMENSIONS[:-1])
            + list(SEMANTIC_DIMENSIONS)
            + ["overall_consistency"],
        },
        "source_manifest": {
            "path": str(published_path),
            "sha256": sha256(published_path),
        },
        "source_contract": {
            "path": str(contract_path),
            "sha256": sha256(contract_path),
        },
    }
    path = args.comparison_root / "comparison_manifest.json"
    digest = write_frozen(path, manifest)
    print(
        f"[v132-long60-comparison] methods=2 videos=256 "
        f"sha256={digest} path={path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
