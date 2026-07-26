#!/usr/bin/env python3
"""Analyze v97 threshold, PF-merge, and cache-mechanism experiments."""

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
    "prompt_tau_0p0_merge",
    "prompt_tau_0p5_merge",
    "prompt_tau_1p0_merge",
    "prompt_tau_1p5_merge",
    "prompt_tau_2p0_merge",
    "prompt_tau_1p0_cyclic",
    "prompt_tau_1p0_recent",
    "prompt_tau_1p0_random_merge",
    "prompt_tau_1p0_reversed_merge",
    "sign_rpos_0p5_stride_merge",
    "pf_ar_stride_merge",
    "pf_aw_stride_merge",
    "pf_native",
    "pf_anchor_extended_recent",
    "pf_wave_extended_recent",
    "pf_veil_extended_recent",
)
COMPARISONS = (
    ("tau1_merge_vs_cyclic", "prompt_tau_1p0_cyclic", "prompt_tau_1p0_merge"),
    ("tau1_merge_vs_recent", "prompt_tau_1p0_recent", "prompt_tau_1p0_merge"),
    (
        "tau1_vs_random",
        "prompt_tau_1p0_random_merge",
        "prompt_tau_1p0_merge",
    ),
    (
        "tau1_vs_reversed",
        "prompt_tau_1p0_reversed_merge",
        "prompt_tau_1p0_merge",
    ),
    (
        "prompt_tau1_vs_sign",
        "sign_rpos_0p5_stride_merge",
        "prompt_tau_1p0_merge",
    ),
    ("pf_aw_vs_ar", "pf_ar_stride_merge", "pf_aw_stride_merge"),
    (
        "pf_anchor_mechanism",
        "pf_anchor_extended_recent",
        "pf_native",
    ),
    ("pf_wave_mechanism", "pf_wave_extended_recent", "pf_native"),
    ("pf_veil_mechanism", "pf_veil_extended_recent", "pf_native"),
)
THRESHOLD_METHODS = (
    ("0.0", "prompt_tau_0p0_merge", "prompt_tau_0"),
    ("0.5", "prompt_tau_0p5_merge", "prompt_tau_0p5"),
    ("1.0", "prompt_tau_1p0_merge", "prompt_tau_1"),
    ("1.5", "prompt_tau_1p5_merge", "prompt_tau_1p5"),
    ("2.0", "prompt_tau_2p0_merge", "prompt_tau_2"),
)
HIGHER_IS_BETTER = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m7_background_consistency",
    "composite",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
COMPREHENSIVE_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "composite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--policy-traces", required=True, type=Path)
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
            value = finite(row.get("temporal_jump"))
            if value is not None:
                values[Path(row["video"]).parent.name].append(value)
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
        print(f"WARNING: incomplete v97 metrics (using partial data): {missing}")
    result = {}
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        for metric in COMPREHENSIVE_METRICS:
            row[metric] = finite(comprehensive.get(method, {}).get(metric))
        row["temporal_jump"] = finite(temporal.get(method))
        for dimension, value in vbench.get(method, {}).items():
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


def comparison(baseline: dict, candidate: dict) -> dict[str, Any]:
    changes = delta(baseline, candidate)
    available = [
        metric
        for metric in HIGHER_IS_BETTER
        if changes.get(metric) is not None
    ]
    wins = [
        metric for metric in available if float(changes[metric]) > 0.0
    ]
    jump = changes.get("temporal_jump")
    return {
        "baseline": baseline["method"],
        "candidate": candidate["method"],
        "delta": changes,
        "quality_wins": len(wins),
        "quality_available": len(available),
        "quality_majority": (
            bool(available) and len(wins) > len(available) / 2
        ),
        "won_metrics": wins,
        "temporal_jump_improved": jump is not None and jump < 0.0,
    }


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.5f}"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# v97 Threshold and PF-Merge Analysis",
        "",
        "## Classification Evidence",
        "",
        (
            "- Two-component GMM preferred to one: "
            f"`{payload['classification']['two_components_preferred_to_one']}`"
        ),
        (
            "- Two-component GMM preferred to three: "
            f"`{payload['classification']['two_components_preferred_to_three']}`"
        ),
        (
            "- Policy trace strict pass: "
            f"`{payload['classification']['policy_trace_strict_pass']}`"
        ),
        (
            "- Prompt classification controls pass: "
            f"`{payload['decisions']['prompt_classification_controls_pass']}`"
        ),
        "",
        "## Manual Threshold Sweep",
        "",
        "| Tau | Stable | Responsive | DINO | Min DINO | BG | Composite | Jump | VBench subject |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']} | {row['stable_count']} | "
            f"{row['responsive_count']} | "
            f"{fmt(row.get('m1_dino_consistency'))} | "
            f"{fmt(row.get('m1_min_stability'))} | "
            f"{fmt(row.get('m7_background_consistency'))} | "
            f"{fmt(row.get('composite'))} | "
            f"{fmt(row.get('temporal_jump'))} | "
            f"{fmt(row.get('vbench_subject_consistency'))} |"
        )
    lines.extend(
        [
            "",
            "## Controlled Comparisons",
            "",
            "| Comparison | Baseline | Candidate | Delta DINO | Delta BG | Delta jump | Quality majority |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in payload["comparisons"].items():
        lines.append(
            f"| {name} | {item['baseline']} | {item['candidate']} | "
            f"{fmt(item['delta'].get('m1_dino_consistency'))} | "
            f"{fmt(item['delta'].get('m7_background_consistency'))} | "
            f"{fmt(item['delta'].get('temporal_jump'))} | "
            f"{item['quality_majority']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gates",
            "",
            (
                "- Prefer compressed responsive cache over cyclic/recent: "
                f"`{payload['decisions']['responsive_merge_supported']}`"
            ),
            (
                "- Prompt score direction beats matched controls: "
                f"`{payload['decisions']['prompt_classification_controls_pass']}`"
            ),
            (
                "- PF merge preferred by metrics: "
                f"`{payload['decisions']['preferred_pf_merge']}`"
            ),
            (
                "- PF mechanism whose replacement most hurts quality: "
                f"`{payload['decisions']['most_important_pf_class']}`"
            ),
            "",
            "These gates are diagnostic. The paper claim must also require "
            "frozen blinded review and replication on the 128-prompt suite.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    vbench = json.loads(args.vbench.read_text(encoding="utf-8"))
    classification = json.loads(
        args.classification.read_text(encoding="utf-8")
    )
    policy_traces = json.loads(
        args.policy_traces.read_text(encoding="utf-8")
    )
    rows = metric_rows(
        comprehensive,
        load_temporal(args.temporal_jump),
        vbench,
    )
    comparisons = {
        name: comparison(rows[baseline], rows[candidate])
        for name, baseline, candidate in COMPARISONS
    }

    threshold_sweep = []
    for threshold, method, map_name in THRESHOLD_METHODS:
        map_info = classification["maps"][map_name]
        counts = {
            int(key): int(value)
            for key, value in map_info["label_counts"].items()
        }
        threshold_sweep.append(
            {
                "threshold": threshold,
                "method": method,
                "stable_count": counts.get(1, 0),
                "responsive_count": counts.get(-1, 0),
                **{
                    key: value
                    for key, value in rows[method].items()
                    if key != "method"
                },
            }
        )

    controls_pass = (
        comparisons["tau1_vs_random"]["quality_majority"]
        and comparisons["tau1_vs_reversed"]["quality_majority"]
    )
    responsive_merge_supported = (
        comparisons["tau1_merge_vs_cyclic"]["quality_majority"]
        and comparisons["tau1_merge_vs_recent"]["quality_majority"]
    )
    pf_merge_comparison = comparisons["pf_aw_vs_ar"]
    if (
        pf_merge_comparison["quality_wins"]
        > pf_merge_comparison["quality_available"] / 2
    ):
        preferred_pf_merge = "anchor_wave_vs_veil"
    elif (
        pf_merge_comparison["quality_wins"]
        < pf_merge_comparison["quality_available"] / 2
    ):
        preferred_pf_merge = "anchor_vs_wave_veil"
    else:
        preferred_pf_merge = "inconclusive"

    mechanism_harm = {}
    for class_name in ("anchor", "wave", "veil"):
        item = comparisons[f"pf_{class_name}_mechanism"]
        # Comparison is ablated baseline -> native candidate. More native wins
        # means replacing this class's middle cache was more harmful.
        mechanism_harm[class_name] = int(item["quality_wins"])
    most_important = max(mechanism_harm, key=mechanism_harm.get)

    gmm_gates = classification.get("gmm_gates") or {}
    payload = {
        "version": 1,
        "method": "v97_threshold_pf_merge_analysis",
        "classification": {
            "score_csv_sha256": classification["score_csv_sha256"],
            "manual_thresholds": classification["manual_thresholds"],
            "automatic_thresholds": classification[
                "automatic_thresholds"
            ],
            "two_components_preferred_to_one": bool(
                gmm_gates.get("two_components_preferred_to_one")
            ),
            "two_components_preferred_to_three": bool(
                gmm_gates.get("two_components_preferred_to_three")
            ),
            "policy_trace_strict_pass": bool(
                policy_traces.get("strict_pass")
            ),
        },
        "methods": rows,
        "ranking_by_dino": sorted(
            rows.values(),
            key=lambda row: row.get("m1_dino_consistency") or -math.inf,
            reverse=True,
        ),
        "threshold_sweep": threshold_sweep,
        "comparisons": comparisons,
        "pf_mechanism_native_wins": mechanism_harm,
        "decisions": {
            "responsive_merge_supported": responsive_merge_supported,
            "prompt_classification_controls_pass": controls_pass,
            "prompt_bimodality_supported": bool(
                gmm_gates.get("two_components_preferred_to_one")
                and gmm_gates.get("two_components_preferred_to_three")
            ),
            "preferred_pf_merge": preferred_pf_merge,
            "most_important_pf_class": most_important,
            "policy_trace_strict_pass": bool(
                policy_traces.get("strict_pass")
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V97Analysis] "
        f"controls_pass={controls_pass} "
        f"responsive_merge={responsive_merge_supported} "
        f"pf_merge={preferred_pf_merge} "
        f"pf_class={most_important}",
        flush=True,
    )


if __name__ == "__main__":
    main()
