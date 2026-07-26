#!/usr/bin/env python3
"""Create a decision-oriented report for the v98 history-polarity screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    "sf_native",
    "pf_native",
    "pf_explicit_parity",
    "pf_aw_hybrid_merge",
    "history_polarity_hybrid_merge",
    "history_polarity_stride_merge",
    "history_polarity_hybrid_merge_v78",
    "positive_rate_half_hybrid_merge",
)
COMPARISONS = (
    (
        "implementation_parity",
        "pf_native",
        "pf_explicit_parity",
        "The explicit PF route should reproduce PF before binary conclusions.",
    ),
    (
        "classifier_gap_to_pf_oracle",
        "pf_aw_hybrid_merge",
        "history_polarity_hybrid_merge",
        "Isolate the cost of PF-independent head discovery.",
    ),
    (
        "hybrid_support_memory",
        "history_polarity_stride_merge",
        "history_polarity_hybrid_merge",
        "Test whether combining periodic and sparse support is useful.",
    ),
    (
        "trusted_write_admission",
        "history_polarity_hybrid_merge",
        "history_polarity_hybrid_merge_v78",
        "Test reliability/novelty-gated middle writes.",
    ),
    (
        "polarity_vs_sign_fraction",
        "positive_rate_half_hybrid_merge",
        "history_polarity_hybrid_merge",
        "Compare normalized net support with majority-sign classification.",
    ),
    (
        "proposed_vs_pf",
        "pf_native",
        "history_polarity_hybrid_merge",
        "Primary quality gap to the strongest borrowed baseline.",
    ),
    (
        "proposed_vs_sf",
        "sf_native",
        "history_polarity_hybrid_merge",
        "Primary gain over native Self-Forcing.",
    ),
)
COMPREHENSIVE_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "m5_temporal_flickering",
    "m8_max_long_range_sim",
    "composite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--map-manifest", required=True, type=Path)
    parser.add_argument("--policy-audit", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_temporal(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = finite(row.get("temporal_jump"))
            if value is None:
                continue
            method = Path(row["video"]).parent.name
            values.setdefault(method, []).append(value)
    return {
        method: statistics.fmean(items)
        for method, items in values.items()
        if items
    }


def load_metrics(
    comprehensive_path: Path,
    vbench_path: Path,
    temporal_path: Path,
) -> dict[str, dict[str, float]]:
    comprehensive = json.loads(
        comprehensive_path.read_text(encoding="utf-8")
    ).get("per_method", {})
    vbench = json.loads(vbench_path.read_text(encoding="utf-8")).get(
        "methods", {}
    )
    temporal = load_temporal(temporal_path)
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        row: dict[str, float] = {}
        for name in COMPREHENSIVE_METRICS:
            value = finite(comprehensive.get(method, {}).get(name))
            if value is not None:
                row[name] = value
        for name, raw in vbench.get(method, {}).items():
            value = finite(raw)
            if value is not None:
                row[f"vbench_{name}"] = value
        if method in temporal:
            row["temporal_jump"] = temporal[method]
        result[method] = row
    return result


def compare(
    rows: dict[str, dict[str, float]],
    baseline: str,
    candidate: str,
) -> dict[str, float]:
    shared = sorted(set(rows[baseline]) & set(rows[candidate]))
    return {
        metric: rows[candidate][metric] - rows[baseline][metric]
        for metric in shared
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    metric_names = sorted(
        {
            metric
            for row in payload["metrics"].values()
            for metric in row
        }
    )
    lines = [
        "# v98 History-Polarity Decision Report",
        "",
        "## Integrity Gates",
        "",
        f"- Policy trace audit: `{payload['gates']['policy_trace_audit']}`",
        f"- Neutral label contract: `{payload['gates']['neutral_label_contract']}`",
        f"- PF parity max metric delta: "
        f"`{payload['gates']['pf_parity_max_abs_delta']}`",
        f"- PF parity metric gate (<= 0.02): "
        f"`{payload['gates']['pf_parity_metric_gate']}`",
        "- Visual usability gate: `pending blind human review`",
        "",
        "## Method Metrics",
        "",
        "| Method | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join("---:" for _ in metric_names) + "|",
    ]
    for method in METHODS:
        row = payload["metrics"][method]
        values = [
            "n/a" if metric not in row else f"{row[metric]:.6f}"
            for metric in metric_names
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")

    lines.extend(["", "## Controlled Comparisons", ""])
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                f"### {comparison['name']}",
                "",
                comparison["purpose"],
                "",
                f"- Baseline: `{comparison['baseline']}`",
                f"- Candidate: `{comparison['candidate']}`",
                "- Deltas (candidate - baseline): "
                f"`{json.dumps(comparison['deltas'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Rule",
            "",
            "Do not select a final method from aggregate metrics alone. First "
            "reject any cell with polygon noise or other catastrophic artifacts. "
            "Then require the PF parity control to be visually and numerically "
            "close. Among usable cells, prefer the smallest PF quality gap that "
            "retains the PF-independent polarity classifier; add trusted writes "
            "only when their controlled delta and blind review are positive.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_metrics(
        args.comprehensive, args.vbench, args.temporal_jump
    )
    comparisons = []
    for name, baseline, candidate, purpose in COMPARISONS:
        comparisons.append(
            {
                "name": name,
                "baseline": baseline,
                "candidate": candidate,
                "purpose": purpose,
                "deltas": compare(rows, baseline, candidate),
            }
        )

    parity = comparisons[0]["deltas"]
    parity_max = max((abs(value) for value in parity.values()), default=None)
    map_manifest = json.loads(args.map_manifest.read_text(encoding="utf-8"))
    policy_audit = json.loads(args.policy_audit.read_text(encoding="utf-8"))
    neutral_contract = (
        int(map_manifest.get("support_label", -1)) == 10
        and int(map_manifest.get("suppress_label", -1)) == 11
        and set(map_manifest.get("reserved_pf_labels", [])) == {-1, 1, 2}
        and map_manifest.get("claims", {}).get(
            "pf_labels_used_for_primary_classifier"
        )
        is False
    )
    payload = {
        "version": 1,
        "method": "v98_history_polarity_decision_analysis",
        "metrics": rows,
        "comparisons": comparisons,
        "gates": {
            "policy_trace_audit": bool(
                policy_audit.get("strict_pass", False)
            ),
            "neutral_label_contract": neutral_contract,
            "pf_parity_max_abs_delta": parity_max,
            "pf_parity_metric_gate": (
                parity_max is not None and parity_max <= 0.02
            ),
            "visual_usability_gate": "pending_blind_human_review",
        },
        "selection": {
            "automatic_winner": None,
            "reason": "human artifact review is a hard gate",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V98Analysis] "
        f"policy={payload['gates']['policy_trace_audit']} "
        f"neutral={neutral_contract} parity_max={parity_max}",
        flush=True,
    )


if __name__ == "__main__":
    main()
