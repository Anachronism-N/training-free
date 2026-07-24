#!/usr/bin/env python3
"""Analyze v92 binary read topology, classification, and archive factors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "composite",
)
EXPECTED_METHODS = (
    "pf",
    "v78",
    "pf_binary_read",
    "pf_binary_read_v78",
    "prompt_pfcount_read",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read",
    "prompt_kmeans_read_v78",
    "prompt_replica_read_v78",
    "prompt_consensus_read_v78",
    "prompt_inverse_read_v78",
    "prompt_random_read_v78",
    "remote_read_v78",
    "role_score_read_v78",
    "pf_read_prompt_priority",
    "prompt_read_prompt_priority",
    "prompt_read_v78_coverage",
    "pf_binary_read_v78_coverage",
)
COMPARISONS = (
    ("pf_tri_to_binary", "pf", "pf_binary_read"),
    ("pf_binary_add_transition", "pf_binary_read", "pf_binary_read_v78"),
    ("prompt_membership_read", "pf_binary_read", "prompt_pfcount_read"),
    (
        "prompt_membership_read_transition",
        "pf_binary_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_add_transition",
        "prompt_pfcount_read",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_vs_natural_partition",
        "prompt_pfcount_read_v78",
        "prompt_kmeans_read_v78",
    ),
    (
        "prompt_vs_inverse",
        "prompt_inverse_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_vs_random",
        "prompt_random_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_vs_remote",
        "remote_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_vs_role_score",
        "role_score_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_primary_vs_replica",
        "prompt_replica_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_primary_vs_consensus",
        "prompt_consensus_read_v78",
        "prompt_pfcount_read_v78",
    ),
    (
        "prompt_priority_on_pf_read",
        "v78",
        "pf_read_prompt_priority",
    ),
    (
        "prompt_priority_on_prompt_read",
        "prompt_pfcount_read_v78",
        "prompt_read_prompt_priority",
    ),
    (
        "coverage_on_prompt_read",
        "prompt_pfcount_read_v78",
        "prompt_read_v78_coverage",
    ),
    (
        "coverage_on_pf_binary_read",
        "pf_binary_read_v78",
        "pf_binary_read_v78_coverage",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument("--label-manifest", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_temporal(path: Path) -> dict[str, dict[str, float | int]]:
    rows: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = Path(row["video"]).parent.name
            value = _finite(row.get("temporal_jump"))
            if value is not None:
                rows[method].append(value)
    return {
        method: {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }
        for method, values in sorted(rows.items())
    }


def _trace_method(path: str) -> str:
    name = Path(path).name
    suffix = ".transition.jsonl"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(name).stem


def _load_coherence(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for summary in payload.get("summaries", []):
        output[_trace_method(str(summary.get("trace", "")))] = {
            "acceptance_rate": summary.get("acceptance_rate"),
            "coherence": summary.get("coherence", {}),
            "status": summary.get("status"),
        }
    return output


def _delta(
    methods: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, float | int]],
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": {},
    }
    for metric in METRICS:
        left = _finite(methods[baseline].get(metric))
        right = _finite(methods[candidate].get(metric))
        row["delta"][metric] = (
            right - left if left is not None and right is not None else None
        )
    left_jump = temporal.get(baseline, {}).get("mean")
    right_jump = temporal.get(candidate, {}).get("mean")
    row["delta"]["temporal_jump"] = (
        float(right_jump) - float(left_jump)
        if left_jump is not None and right_jump is not None
        else None
    )
    return row


def analyze(
    comprehensive: dict[str, Any],
    temporal: dict[str, dict[str, float | int]],
    coherence: dict[str, dict[str, Any]],
    label_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    methods = comprehensive.get("per_method") or {}
    missing = sorted(set(EXPECTED_METHODS) - set(methods))
    if missing:
        raise ValueError(f"missing v92 methods: {missing}")

    comparisons = {
        name: _delta(methods, temporal, baseline, candidate)
        for name, baseline, candidate in COMPARISONS
    }
    ranking = []
    for name in EXPECTED_METHODS:
        values = methods[name]
        trace = coherence.get(name) or {}
        age_spread = (
            (trace.get("coherence") or {}).get("age_spread") or {}
        ).get("mean")
        ranking.append(
            {
                "method": name,
                "dino": _finite(values.get("m1_dino_consistency")),
                "min_dino": _finite(values.get("m1_min_stability")),
                "background": _finite(values.get("m7_background_consistency")),
                "composite": _finite(values.get("composite")),
                "temporal_jump": temporal.get(name, {}).get("mean"),
                "acceptance_rate": trace.get("acceptance_rate"),
                "age_spread": age_spread,
            }
        )
    ranking.sort(
        key=lambda row: (
            row["dino"] is not None,
            row["dino"] if row["dino"] is not None else -math.inf,
        ),
        reverse=True,
    )

    prompt_vs_inverse = comparisons["prompt_vs_inverse"]["delta"][
        "m1_dino_consistency"
    ]
    prompt_vs_random = comparisons["prompt_vs_random"]["delta"][
        "m1_dino_consistency"
    ]
    prompt_vs_pf = comparisons["prompt_membership_read_transition"]["delta"][
        "m1_dino_consistency"
    ]
    classification_gate = {
        "beats_inverse_dino": (
            prompt_vs_inverse is not None and prompt_vs_inverse > 0
        ),
        "beats_random_dino": (
            prompt_vs_random is not None and prompt_vs_random > 0
        ),
        "within_0p005_of_pf_binary_dino": (
            prompt_vs_pf is not None and prompt_vs_pf >= -0.005
        ),
    }
    classification_gate["metric_passed"] = all(classification_gate.values())

    return {
        "comparisons": comparisons,
        "ranking": ranking,
        "classification_gate": classification_gate,
        "label_manifest": label_manifest,
        "temporal_jump": temporal,
        "coherence": coherence,
        "claim_note": (
            "The metric gate is necessary but not sufficient. Human review must "
            "reject duplicated subjects, scene leakage, motion loss, or physics errors."
        ),
    }


def _fmt(value: Any, digits: int = 5) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v92 Prompt-Contrastive Binary Cache Analysis",
        "",
        "## Factor comparisons",
        "",
        "| Factor | Baseline | Candidate | Delta DINO | Delta min DINO | Delta BG | Delta jump |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["comparisons"].items():
        delta = row["delta"]
        lines.append(
            f"| {name} | {row['baseline']} | {row['candidate']} | "
            f"{_fmt(delta.get('m1_dino_consistency'))} | "
            f"{_fmt(delta.get('m1_min_stability'))} | "
            f"{_fmt(delta.get('m7_background_consistency'))} | "
            f"{_fmt(delta.get('temporal_jump'))} |"
        )
    gate = payload["classification_gate"]
    lines.extend(
        [
            "",
            "## Classification gate",
            "",
            f"- Beats inverse on DINO: `{gate['beats_inverse_dino']}`",
            f"- Beats random on DINO: `{gate['beats_random_dino']}`",
            f"- Within 0.005 of PF-binary DINO: `{gate['within_0p005_of_pf_binary_dino']}`",
            f"- Metric gate passed: `{gate['metric_passed']}`",
            "",
            "This gate does not override blind artifact or motion review.",
            "",
            "## Ranking",
            "",
            "| Method | DINO | min DINO | BG | Jump | Acceptance | Age spread |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["ranking"]:
        lines.append(
            f"| {row['method']} | {_fmt(row['dino'])} | "
            f"{_fmt(row['min_dino'])} | {_fmt(row['background'])} | "
            f"{_fmt(row['temporal_jump'])} | "
            f"{_fmt(row['acceptance_rate'])} | {_fmt(row['age_spread'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    temporal = _load_temporal(args.temporal_jump)
    coherence = _load_coherence(args.trace_summary)
    label_manifest = (
        json.loads(args.label_manifest.read_text(encoding="utf-8"))
        if args.label_manifest is not None and args.label_manifest.is_file()
        else None
    )
    payload = analyze(comprehensive, temporal, coherence, label_manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[v92-analysis] "
        f"methods={len(payload['ranking'])} "
        f"classification_gate={payload['classification_gate']['metric_passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
