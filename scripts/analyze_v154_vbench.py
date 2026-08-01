#!/usr/bin/env python3
"""Summarize v154 VBench-Long effects with an explicit motion trade-off."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PRIMARY = "ours_qk_top4"
MEMBERSHIP_CONTROLS = (
    "ours_qk_bottom4_control",
    "ours_qk_random4_control",
)
METHODS = (
    "sf_native",
    PRIMARY,
    *MEMBERSHIP_CONTROLS,
    "ours_all_recent8_control",
    "ours_all_prototype4_control",
    "ours_legacy_membership",
    "ours_legacy_reference",
)
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "overall_consistency",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def _mean(row: dict[str, float], keys: tuple[str, ...]) -> float:
    return statistics.mean(float(row[key]) for key in keys)


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    if set(rows) != set(METHODS):
        raise ValueError(f"unexpected v154 methods: {tuple(rows)}")
    if tuple(payload.get("dimensions") or ()) != DIMENSIONS:
        raise ValueError("unexpected v154 VBench dimensions")
    if payload.get("missing"):
        raise ValueError(f"v154 VBench summary is incomplete: {payload['missing']}")
    derived = {}
    for method in METHODS:
        row = rows[method]
        if set(row) != set(DIMENSIONS) or any(row[key] is None for key in DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench scores")
        derived[method] = {
            "history_consistency": _mean(
                row,
                (
                    "subject_consistency",
                    "background_consistency",
                    "overall_consistency",
                ),
            ),
            "visual_quality": _mean(row, ("aesthetic_quality", "imaging_quality")),
            "temporal_quality": _mean(
                row, ("temporal_flickering", "motion_smoothness")
            ),
            "dynamic_degree": float(row["dynamic_degree"]),
        }
    differences = {}
    for method in METHODS:
        if method == PRIMARY:
            continue
        differences[method] = {
            key: derived[PRIMARY][key] - derived[method][key]
            for key in derived[PRIMARY]
        }
    control_gates = {}
    for control in MEMBERSHIP_CONTROLS:
        delta = differences[control]
        control_gates[control] = {
            "history_consistency_improves": delta["history_consistency"] > 0,
            "visual_quality_noninferior": delta["visual_quality"] >= -0.01,
            "temporal_quality_noninferior": delta["temporal_quality"] >= -0.005,
            "dynamic_degree_noninferior": delta["dynamic_degree"] >= -0.03,
        }
        control_gates[control]["passes"] = all(
            control_gates[control].values()
        )
    return {
        "version": 1,
        "experiment": "v154_history_critical_moviebench16_vbench",
        "primary": PRIMARY,
        "derived_scores": derived,
        "primary_minus_comparator": differences,
        "membership_control_gates": control_gates,
        "metric_promotion_gate": all(
            row["passes"] for row in control_gates.values()
        ),
        "claim_boundary": (
            "This is a 16-prompt aggregate screen without paired confidence "
            "intervals. It can reject a candidate but cannot establish the "
            "final paper claim without the blind review and 128-prompt run."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v154 VBench Analysis",
        "",
        f"Metric promotion gate: **{report['metric_promotion_gate']}**",
        "",
        (
            "| Comparator | History consistency | Visual quality | "
            "Temporal quality | Dynamic degree |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in report["primary_minus_comparator"].items():
        lines.append(
            f"| {method} | {row['history_consistency']:+.5f} | "
            f"{row['visual_quality']:+.5f} | {row['temporal_quality']:+.5f} | "
            f"{row['dynamic_degree']:+.5f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(payload)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "v154_vbench_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "v154_vbench_analysis.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(f"[v154-vbench-analysis] gate={report['metric_promotion_gate']}")


if __name__ == "__main__":
    main()
