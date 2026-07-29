#!/usr/bin/env python3
"""Prepare a diverse-16 blind review from the completed v129 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_v100_fast_selection_1video import write_frozen
from run_v120_moviebench32_main import link_or_validate
from run_v132_binary_memory_ablation import SCREEN16


DEFAULT_METHODS = (
    "sf_native",
    "deep_forcing",
    "ours_prototype_retrieval_age24",
    "ours_confidence_motion",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=root / "runs" / "v129_paper_comparison_30s",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "runs" / "v132_blind_review16",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def write_frozen_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"frozen text differs: {path}")
    else:
        path.write_bytes(data)
    return digest


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    comparison_root = args.comparison_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v129_no_pf_paper_comparison_30s":
        raise SystemExit(f"unexpected v129 comparison: {manifest_path}")
    rows = {str(row["key"]): row for row in manifest["methods"]}
    unknown = sorted(set(args.methods) - set(rows))
    if unknown or len(args.methods) != len(set(args.methods)):
        raise SystemExit(f"invalid blind-review methods: {unknown}")
    prompt_rows = manifest.get("prompt_items", [])
    if len(prompt_rows) != 128:
        raise SystemExit("v129 manifest does not contain all prompt items")

    staging = output_root / "source_subset" / "published"
    inventory = []
    for method in args.methods:
        source_dir = Path(str(rows[method]["video_dir"]))
        target_dir = staging / method
        for subset_index, source_index in enumerate(SCREEN16):
            source = source_dir / f"{source_index:06d}-0.mp4"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = target_dir / f"{subset_index:06d}-0.mp4"
            mode = link_or_validate(source, target)
            inventory.append(
                {
                    "method": method,
                    "subset_index": subset_index,
                    "source_index": source_index,
                    "source": str(source),
                    "target": str(target),
                    "link_mode": mode,
                    "size": source.stat().st_size,
                }
            )
    subset_prompts = "\n".join(
        str(prompt_rows[index]["text"]) for index in SCREEN16
    ) + "\n"
    prompt_path = output_root / "source_subset" / "prompts.txt"
    prompt_sha = write_frozen_text(prompt_path, subset_prompts)
    write_frozen(
        output_root / "source_subset" / "selection_manifest.json",
        {
            "version": 1,
            "source_comparison_manifest": str(manifest_path),
            "source_prompt_indices": list(SCREEN16),
            "methods": list(args.methods),
            "subset_prompt_sha256": prompt_sha,
            "inventory": inventory,
        },
    )
    command = [
        sys.executable,
        str(root / "scripts" / "prepare_blind_review.py"),
        "--run-root",
        str(staging),
        "--methods",
        *args.methods,
        "--prompts",
        str(prompt_path),
        "--output",
        str(output_root / "public"),
        "--private-output",
        str(output_root / "private"),
        "--prompt-count",
        str(len(SCREEN16)),
        "--seed",
        str(args.seed),
    ]
    print("[v132-blind] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
