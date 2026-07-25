#!/usr/bin/env python3
"""Analyze v96 threshold classification and binary-cache factorization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = (
    "pf",
    "pf_binary_cyclic",
    "pf_binary_merge",
    "pf_binary_recent",
    "cfg_cyclic",
    "cfg_merge",
    "semantic_cyclic",
    "semantic_merge",
    "consensus_cyclic",
    "consensus_merge",
    "consensus_recent",
    "consensus_merge_v78",
    "consensus_cyclic_v78",
    "random_merge",
    "inverse_merge",
    "pf_binary_merge_v78",
)
COMPARISONS = (
    ("pf_binary_merge_vs_cyclic", "pf_binary_cyclic", "pf_binary_merge"),
    ("pf_binary_recent_vs_cyclic", "pf_binary_cyclic", "pf_binary_recent"),
    ("cfg_merge_vs_cyclic", "cfg_cyclic", "cfg_merge"),
    ("semantic_merge_vs_cyclic", "semantic_cyclic", "semantic_merge"),
    ("consensus_merge_vs_cyclic", "consensus_cyclic", "consensus_merge"),
    ("consensus_recent_vs_cyclic", "consensus_cyclic", "consensus_recent"),
    ("consensus_vs_random_merge", "random_merge", "consensus_merge"),
    ("consensus_vs_inverse_merge", "inverse_merge", "consensus_merge"),
    ("consensus_vs_pf_membership_merge", "pf_binary_merge", "consensus_merge"),
    ("consensus_merge_v78", "consensus_merge", "consensus_merge_v78"),
    ("consensus_cyclic_v78", "consensus_cyclic", "consensus_cyclic_v78"),
    ("pf_binary_merge_v78", "pf_binary_merge", "pf_binary_merge_v78"),
)
COMPREHENSIVE_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "composite",
)
QUALITY_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m7_background_consistency",
    "composite",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--head-profile", required=True, type=Path)
    parser.add_argument("--transition-traces", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_temporal(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = finite(row.get("temporal_jump"))
            if score is not None:
                values[Path(row["video"]).parent.name].append(score)
    return {
        method: statistics.fmean(scores)
        for method, scores in values.items()
    }


def metric_rows(
    comprehensive_payload: dict,
    temporal: dict[str, float],
    vbench_payload: dict,
) -> dict[str, dict[str, Any]]:
    comprehensive = comprehensive_payload.get("per_method") or {}
    vbench = vbench_payload.get("methods") or {}
    missing = {
        "comprehensive": sorted(set(METHODS) - set(comprehensive)),
        "temporal": sorted(set(METHODS) - set(temporal)),
        "vbench": sorted(set(METHODS) - set(vbench)),
    }
    if any(missing.values()):
        raise ValueError(f"incomplete v96 metrics: {missing}")
    result = {}
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        for metric in COMPREHENSIVE_METRICS:
            row[metric] = finite(comprehensive[method].get(metric))
        row["temporal_jump"] = finite(temporal[method])
        for dimension, value in vbench[method].items():
            row[f"vbench_{dimension}"] = finite(value)
        result[method] = row
    return result


def delta(baseline: dict, candidate: dict) -> dict[str, float | None]:
    result = {}
    for metric in sorted((set(baseline) | set(candidate)) - {"method"}):
        left = finite(baseline.get(metric))
        right = finite(candidate.get(metric))
        result[metric] = (
            right - left if left is not None and right is not None else None
        )
    return result


def wins(baseline: dict, candidate: dict) -> dict:
    changes = delta(baseline, candidate)
    available = [
        metric for metric in QUALITY_METRICS
        if changes.get(metric) is not None
    ]
    won = [
        metric for metric in available
        if float(changes[metric]) > 0.0
    ]
    jump = changes.get("temporal_jump")
    return {
        "quality_wins": len(won),
        "quality_available": len(available),
        "quality_majority": (
            bool(available) and len(won) > len(available) / 2
        ),
        "won_metrics": won,
        "temporal_jump_improved": jump is not None and jump < 0.0,
    }


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.5f}"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# v96 binary head/cache analysis",
        "",
        "## Head-discovery evidence",
        "",
        f"- classification accepted: `{payload['head_profile']['accepted']}`",
        (
            "- Stable/PF-Anchor Jaccard: "
            f"`{payload['head_profile']['stable_anchor_jaccard']:.4f}`"
        ),
        (
            "- Wave QK positive rate: "
            f"`{payload['head_profile']['wave_positive_rate']:.4f}`"
        ),
        (
            "- Wave QK sign-switch rate: "
            f"`{payload['head_profile']['wave_sign_switch_rate']:.4f}`"
        ),
        "",
        "## Ranking by DINO",
        "",
        "| Method | DINO | Min DINO | BG | Composite | Jump | VBench subject | VBench BG | Aesthetic | Imaging | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ranking_by_dino"]:
        lines.append(
            f"| {row['method']} | {fmt(row.get('m1_dino_consistency'))} | "
            f"{fmt(row.get('m1_min_stability'))} | "
            f"{fmt(row.get('m7_background_consistency'))} | "
            f"{fmt(row.get('composite'))} | "
            f"{fmt(row.get('temporal_jump'))} | "
            f"{fmt(row.get('vbench_subject_consistency'))} | "
            f"{fmt(row.get('vbench_background_consistency'))} | "
            f"{fmt(row.get('vbench_aesthetic_quality'))} | "
            f"{fmt(row.get('vbench_imaging_quality'))} | "
            f"{fmt(row.get('vbench_dynamic_degree'))} |"
        )
    lines.extend(
        [
            "",
            "## Controlled comparisons",
            "",
            "| Comparison | Baseline | Candidate | Delta DINO | Delta BG | Delta jump | Quality majority |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in payload["comparisons"].items():
        change = item["delta"]
        lines.append(
            f"| {name} | {item['baseline']} | {item['candidate']} | "
            f"{fmt(change.get('m1_dino_consistency'))} | "
            f"{fmt(change.get('m7_background_consistency'))} | "
            f"{fmt(change.get('temporal_jump'))} | "
            f"{item['wins']['quality_majority']} |"
        )
    lines.extend(
        [
            "",
            "## Decision fields",
            "",
            (
                "- PF-binary should use merge over cyclic: "
                f"`{payload['decisions']['pf_binary_merge_over_cyclic']}`"
            ),
            (
                "- Prompt-consensus should use merge over cyclic: "
                f"`{payload['decisions']['consensus_merge_over_cyclic']}`"
            ),
            (
                "- Prompt-consensus merge beats random and inverse: "
                f"`{payload['decisions']['classification_controls_pass']}`"
            ),
            (
                "- Transition trace guard passed: "
                f"`{payload['decisions']['transition_trace_guard']}`"
            ),
            "",
            "These are automated technical gates, not the paper-method decision. "
            "Freeze blinded human review before using metric rankings.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    vbench = json.loads(args.vbench.read_text(encoding="utf-8"))
    head = json.loads(args.head_profile.read_text(encoding="utf-8"))
    traces = json.loads(args.transition_traces.read_text(encoding="utf-8"))
    rows = metric_rows(
        comprehensive, load_temporal(args.temporal_jump), vbench
    )
    comparisons = {}
    for name, baseline, candidate in COMPARISONS:
        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta(rows[baseline], rows[candidate]),
            "wins": wins(rows[baseline], rows[candidate]),
        }
    consensus_map = head["maps"]["prompt_consensus_threshold"]
    overlap = consensus_map["vs_pf_anchor"]
    wave = head["pf_temporal_statistics"]["wave"]
    trace_count = int(
        traces.get("trace_count", len(traces.get("summaries", [])))
    )
    trace_pass = bool(
        traces.get(
            "strict_pass",
            all(
                item.get("status") == "nominal"
                for item in traces.get("summaries", [])
            ),
        )
    )
    decisions = {
        "pf_binary_merge_over_cyclic": comparisons[
            "pf_binary_merge_vs_cyclic"
        ]["wins"]["quality_majority"],
        "consensus_merge_over_cyclic": comparisons[
            "consensus_merge_vs_cyclic"
        ]["wins"]["quality_majority"],
        "classification_controls_pass": (
            comparisons["consensus_vs_random_merge"]["wins"][
                "quality_majority"
            ]
            and comparisons["consensus_vs_inverse_merge"]["wins"][
                "quality_majority"
            ]
        ),
        "transition_trace_guard": trace_count == 3 and trace_pass,
    }
    payload = {
        "head_profile": {
            "accepted": bool(head.get("accepted")),
            "stable_count": consensus_map["stable_count"],
            "responsive_count": consensus_map["responsive_count"],
            "stable_anchor_jaccard": overlap["stable_anchor_jaccard"],
            "wave_positive_rate": wave["positive_rate"],
            "wave_sign_switch_rate": wave["sign_switch_rate"],
            "wave_dominant_period": wave["dominant_period"],
        },
        "methods": rows,
        "ranking_by_dino": sorted(
            rows.values(),
            key=lambda row: row.get("m1_dino_consistency") or -math.inf,
            reverse=True,
        ),
        "comparisons": comparisons,
        "decisions": decisions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[v96-analysis] "
        f"head_accepted={payload['head_profile']['accepted']} "
        f"merge_pf={decisions['pf_binary_merge_over_cyclic']} "
        f"merge_consensus={decisions['consensus_merge_over_cyclic']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
