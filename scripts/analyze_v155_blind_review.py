#!/usr/bin/env python3
"""Unblind and summarize the paired v155 human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v154_blind_review as base
import prepare_v154_blind_review as prepare_base
from prepare_v155_vbench_comparison import METHODS, PROMPT_COUNT


PRIMARY = "ours_qk_top4_reservoir4"
COMPARATORS = tuple(method for method in METHODS if method != PRIMARY)
REQUIRED_CONTROLS = (
    "ours_qk_bottom4_reservoir4_control",
    "ours_qk_random4_reservoir4_control",
)


def configure_base() -> None:
    prepare_base.METHODS = METHODS
    prepare_base.PROMPT_COUNT = PROMPT_COUNT
    base.METHODS = METHODS
    base.PROMPT_COUNT = PROMPT_COUNT
    base.PRIMARY = PRIMARY
    base.COMPARATORS = COMPARATORS
    base.REQUIRED_CONTROLS = REQUIRED_CONTROLS
    base.BOOTSTRAP_SEED = 1552026


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = base.load_completed_rows(args.review_sheet, args.blind_key)
    report = base.analyze(rows)
    report["experiment"] = "v155_profile_aligned_moviebench16_blind_review"
    report["claim_boundary"] = (
        "The paired human screen tests membership and cache-policy transfer; "
        "it does not by itself establish a paper-scale generation result."
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v155_blind_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = base.render_markdown(report).replace(
        "# v154 Blind Review Analysis", "# v155 Blind Review Analysis", 1
    )
    (args.output_root / "v155_blind_review_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        f"[v155-blind-analysis] PASS human_gate="
        f"{report['human_promotion_gate']}"
    )


if __name__ == "__main__":
    main()
