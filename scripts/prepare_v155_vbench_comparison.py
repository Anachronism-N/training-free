#!/usr/bin/env python3
"""Materialize prompt-correct VBench inputs for v155."""
from __future__ import annotations

import argparse
from pathlib import Path

import prepare_v154_vbench_comparison as base


EXPERIMENT = "v155_profile_aligned_moviebench16"
COMPARISON_EXPERIMENT = "v155_profile_aligned_vbench16"
PROMPT_COUNT = 16
METHODS = (
    "sf_native",
    "ours_qk_top4_reservoir4",
    "ours_qk_bottom4_reservoir4_control",
    "ours_qk_random4_reservoir4_control",
    "ours_all_reservoir4_control",
    "ours_qk_top4_prototype4_reference",
    "ours_all_recent8_reference",
)
CORE_DIMENSIONS = base.DIMENSIONS
SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
)
# The eight core dimensions plus the semantic extension reproduce the complete
# VBench paper-table contract. overall_consistency is already in the core set.
DIMENSIONS = (*CORE_DIMENSIONS, *SEMANTIC_DIMENSIONS)
comparison_name = base.comparison_name
sha256 = base.sha256


def configure_base() -> None:
    base.EXPERIMENT = EXPERIMENT
    base.COMPARISON_EXPERIMENT = COMPARISON_EXPERIMENT
    base.PROMPT_COUNT = PROMPT_COUNT
    base.METHODS = METHODS
    base.DIMENSIONS = DIMENSIONS


def prepare(run_root: Path, comparison_root: Path, prompt_manifest: Path):
    configure_base()
    return base.prepare(run_root, comparison_root, prompt_manifest)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / EXPERIMENT / "full7",
    )
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=root / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    args = parser.parse_args()
    comparison_root = args.comparison_root or args.run_root / "vbench_comparison"
    report = prepare(args.run_root, comparison_root, args.prompt_manifest)
    print(
        "[v155-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"manifest_sha256={report['manifest_sha256']} "
        f"links={report['link_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
