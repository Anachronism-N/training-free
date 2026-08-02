#!/usr/bin/env python3
"""Unblind and summarize the paired v157 human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v154_blind_review as base
import prepare_v154_blind_review as prepare_base
from prepare_v157_vbench_comparison import (
    LAYER_CANDIDATES,
    METHODS,
    PROMPT_COUNT,
)


PRIMARY = "ours_layer_interleaved10_reservoir4"
COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
REQUIRED_CONTROLS = tuple(
    method for method in LAYER_CANDIDATES if method != PRIMARY
) + (
    "ours_all_reservoir4_reference",
    "ours_all_recent8_reference",
)


def configure_base() -> None:
    prepare_base.METHODS = METHODS
    prepare_base.PROMPT_COUNT = PROMPT_COUNT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.PRIMARY = PRIMARY
    base.COMPARATORS = COMPARATORS
    base.REQUIRED_CONTROLS = REQUIRED_CONTROLS
    base.BOOTSTRAP_SEED = 1572026


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = base.load_completed_rows(args.review_sheet, args.blind_key)
    report = base.analyze(rows)
    report["experiment"] = "v157_layer_gated_moviebench16_blind_review"
    report["provisional_primary"] = PRIMARY
    report["claim_boundary"] = (
        "The predeclared interleaved route is the human-review primary. If "
        "VBench selects a different layer route, that route requires a new "
        "confirmatory paired review and cannot be promoted post hoc."
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v157_blind_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = base.render_markdown(report).replace(
        "# v154 Blind Review Analysis", "# v157 Blind Review Analysis", 1
    )
    (args.output_root / "v157_blind_review_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        f"[v157-blind-analysis] PASS human_gate="
        f"{report['human_promotion_gate']}"
    )


if __name__ == "__main__":
    main()
