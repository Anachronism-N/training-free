#!/usr/bin/env python3
"""Retain official-formula Quality, fixed-DD diagnosis and both selector controls."""
import argparse
from pathlib import Path

import v215_lphc_protocol as p
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze, costs, render, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = p.output_root(args.run_root)
    rows, diagnostics, source, manifest = load_validated_inputs(out, protocol=p)
    report = analyze(rows, diagnostics, protocol=p)
    report.update(source=source, costs=costs(manifest["jobs"], protocol=p))
    report["note"] = "Exploratory post-v213 redesign, no automatic paper claim. Head-preserving retrieval is not a validated head taxonomy. Shared random e1 control is descriptor-independent."
    for row in report["review_queue"]:
        row["videos"] = {m: str(out / f"evaluation/vbench_comparison/published/{m}/{row['prompt_index']:06d}-0.mp4")
                         for m in (row["candidate"], row["control"])}
    root = out / "evaluation/analysis"
    p.frozen_json(root / "v215_selector_phase.json", report)
    p.write_frozen(root / "v215_selector_phase.md", render(report).encode())
    old.write_comparisons(root / "v215_selector_phase.csv", report["comparisons"])
    print(render(report))


if __name__ == "__main__":
    main()
