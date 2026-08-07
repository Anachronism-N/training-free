#!/usr/bin/env python3
"""Materialize prompt-correct VBench inputs for v164."""
from __future__ import annotations

import argparse
from pathlib import Path

import prepare_v154_vbench_comparison as base


EXPERIMENT = "v164_direction_freshness_moviebench16"
COMPARISON_EXPERIMENT = "v164_direction_freshness_vbench16"
PROMPT_COUNT = 16
METHODS = (
    "sf_native",
    "ours_middle10_reservoir2_directionmatch1",
    "ours_middle10_reservoir2_directionfresh1",
    "ours_middle10_reservoir2_statemotionpair1_reference",
    "ours_middle10_reservoir4_reference",
    "ours_all_recent8_reference",
)
DIMENSIONS = (*base.DIMENSIONS, "temporal_style")
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
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=root / "runs" / EXPERIMENT / "full8" / "vbench_comparison",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=root / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    args = parser.parse_args()
    report = prepare(args.run_root, args.comparison_root, args.prompt_manifest)
    print(
        "[v164-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"manifest_sha256={report['manifest_sha256']} "
        f"links={report['link_counts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
