#!/usr/bin/env python3
"""Materialize prompt-correct VBench inputs for v159."""
from __future__ import annotations

import argparse
from pathlib import Path

import prepare_v154_vbench_comparison as base


EXPERIMENT = "v159_motion_coherent_reservoir_moviebench16"
COMPARISON_EXPERIMENT = "v159_motion_coherent_reservoir_vbench16"
PROMPT_COUNT = 16
METHODS = (
    "sf_native",
    "ours_interleaved10_reservoir2_motionpair1",
    "ours_interleaved10_motionpair2",
    "ours_middle10_reservoir2_motionpair1",
    "ours_interleaved10_reservoir4_reference",
    "ours_middle10_reservoir4_reference",
    "ours_all_reservoir4_reference",
    "ours_all_recent8_reference",
)
PRIMARY = METHODS[1]
MECHANISM_CANDIDATES = METHODS[1:4]
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
DIMENSIONS = (*CORE_DIMENSIONS, *SEMANTIC_DIMENSIONS)
CORE_EVALUATION_DIMENSIONS = (*CORE_DIMENSIONS, "temporal_style")
comparison_name = base.comparison_name


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
        default=root / "runs" / EXPERIMENT / "full8",
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
        "[v159-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"manifest_sha256={report['manifest_sha256']} "
        f"links={report['link_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
