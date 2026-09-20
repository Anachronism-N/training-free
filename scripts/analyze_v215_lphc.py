#!/usr/bin/env python3
"""Retain official-formula Quality, fixed-DD diagnosis and both selector controls."""
import argparse
from pathlib import Path

import v215_lphc_protocol as p
from analyze_v212_lphc import load_validated_inputs
from analyze_v213_lphc import analyze, costs, render, old
from vbench_quality_contract import reject_known_invalid_dynamic_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation-repair", action="store_true")
    args = parser.parse_args()
    out = p.output_root(args.run_root)
    protocol = p
    if args.evaluation_repair:
        from v218_evaluation_repair import Protocol
        protocol = Protocol(out)
    rows, diagnostics, source, manifest = load_validated_inputs(out, protocol=protocol)
    reject_known_invalid_dynamic_runtime(manifest["vbench_fingerprint"])
    report = analyze(rows, diagnostics, protocol=protocol)
    report.update(source=source, costs=costs(manifest["jobs"], protocol=p))
    if args.evaluation_repair:
        report["evaluation_repair"] = protocol.repair
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
