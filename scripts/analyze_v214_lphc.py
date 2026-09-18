#!/usr/bin/env python3
"""Analyze the 96-prompt extension without regenerating or relabeling v213."""
import argparse
from pathlib import Path

import v214_lphc_protocol as p
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze, costs, render, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = p.output_root(args.run_root)
    rows, diagnostics, source, manifest = load_validated_inputs(out, protocol=p)
    report = analyze(rows, diagnostics, protocol=p)
    report["source"] = source
    report["costs"] = costs(manifest["jobs"], protocol=p)
    report["extension"] = {
        "v213_sources_excluded": list(p.previous.SOURCE_INDICES),
        "historically_unseen_prompts_claimed": False,
        "v213_32_plus_v214_96_cover_128": True,
        "combined_128_report_included": False,
        "boundary": "Same prespecified matrix and seed rule on disjoint prompts; still development evidence, not an independently locked final-method confirmation.",
    }
    for row in report["review_queue"]:
        row["videos"] = {
            m: str(out / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
            for m in (row["candidate"], row["control"])}
    root = out / "evaluation/analysis"
    stem = "v214_remaining96"
    p.frozen_json(root / f"{stem}.json", report)
    p.write_frozen(root / f"{stem}.md", render(report).encode())
    old.write_comparisons(root / f"{stem}.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
