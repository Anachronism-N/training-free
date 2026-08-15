#!/usr/bin/env python3
"""Paired development analysis for the v182 Coverage-operator screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base


METRICS = base.METRICS
STRUCTURED = (
    "strict5_landmark",
    "strict5_prototype",
    "strict5_retrieval",
)
STRICT_METHODS = ("strict5_reservoir", *STRUCTURED)
PRIMARY = ("official_quality_score", "identity_background", "dynamic_degree")


def pareto_front(means: dict[str, dict[str, float]]) -> list[str]:
    front = []
    for candidate in STRICT_METHODS:
        dominated = False
        for other in STRICT_METHODS:
            if candidate == other:
                continue
            no_worse = all(means[other][metric] >= means[candidate][metric] for metric in PRIMARY)
            better = any(means[other][metric] > means[candidate][metric] for metric in PRIMARY)
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def comparison(
    rows: dict,
    candidate: str,
    control: str,
    metric: str,
    prompt_count: int,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(prompt_count)
    ]
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row["key"] for row in manifest["methods"])
    expected = ("all_recent", *STRICT_METHODS)
    if methods != expected:
        raise ValueError(f"v182 method order drifted: {methods}")
    prompt_count = int(manifest["prompt_count"])
    if prompt_count != 16 or tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v182 paired analysis requires a complete five-method screen16")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)

    comparisons = []
    index = 0
    for candidate in STRICT_METHODS:
        for control in ("all_recent", "strict5_reservoir"):
            if candidate == control:
                continue
            for metric in METRICS:
                comparisons.append(
                    comparison(
                        rows,
                        candidate,
                        control,
                        metric,
                        prompt_count,
                        1822026 + index,
                    )
                )
                index += 1
    base.bh(comparisons)

    means = {
        method: {
            metric: float(np.mean([rows[(method, prompt)][metric] for prompt in range(prompt_count)]))
            for metric in METRICS
        }
        for method in methods
    }
    lookup = {
        (row["candidate"], row["control"], row["metric"]): row
        for row in comparisons
    }
    front = pareto_front(means)
    candidate_status = {}
    for candidate in STRUCTURED:
        vs_reservoir = {
            metric: lookup[(candidate, "strict5_reservoir", metric)]["mean_delta"]
            for metric in PRIMARY
        }
        vs_recent = {
            metric: lookup[(candidate, "all_recent", metric)]["mean_delta"]
            for metric in PRIMARY
        }
        noninferior = (
            vs_reservoir["official_quality_score"] >= -0.005
            and vs_reservoir["identity_background"] >= -0.005
            and vs_reservoir["dynamic_degree"] >= -0.02
            and vs_recent["official_quality_score"] >= -0.005
            and vs_recent["identity_background"] >= -0.005
            and vs_recent["dynamic_degree"] >= -0.02
        )
        useful_gain = (
            vs_reservoir["official_quality_score"] > 0.002
            or vs_reservoir["identity_background"] > 0.002
        )
        candidate_status[candidate] = {
            "on_primary_pareto_front": candidate in front,
            "noninferior_to_reservoir_and_recent": bool(noninferior),
            "useful_gain_over_reservoir": bool(useful_gain),
            "promote_to_reprofile": bool(candidate in front and noninferior and useful_gain),
            "mean_delta_vs_reservoir": vs_reservoir,
            "mean_delta_vs_all_recent": vs_recent,
        }
    promoted = [
        candidate for candidate in STRUCTURED if candidate_status[candidate]["promote_to_reprofile"]
    ]

    review_rows = []
    for prompt in range(prompt_count):
        for candidate in STRUCTURED:
            quality = rows[(candidate, prompt)]["official_quality_score"] - rows[("strict5_reservoir", prompt)]["official_quality_score"]
            identity = rows[(candidate, prompt)]["identity_background"] - rows[("strict5_reservoir", prompt)]["identity_background"]
            dynamic = rows[(candidate, prompt)]["dynamic_degree"] - rows[("strict5_reservoir", prompt)]["dynamic_degree"]
            review_rows.append(
                {
                    "prompt_index": prompt,
                    "candidate": candidate,
                    "quality_delta": quality,
                    "identity_background_delta": identity,
                    "dynamic_degree_delta": dynamic,
                    "conflict_score": abs(quality - identity) + max(0.0, -dynamic),
                }
            )
    review_queue = sorted(
        review_rows,
        key=lambda row: (-row["conflict_score"], row["prompt_index"], row["candidate"]),
    )[:4]
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "development_only": True,
        "prompt_count": prompt_count,
        "methods": list(methods),
        "method_means": means,
        "comparisons": comparisons,
        "primary_pareto_front": front,
        "candidate_status": candidate_status,
        "promoted_for_operator_specific_reprofiling": promoted,
        "decision": "reprofile_structured_coverage_operator"
        if promoted
        else "no_structured_coverage_operator_promoted",
        "targeted_review_queue": review_queue,
        "claim_boundary": (
            "The 16-prompt screen is for operator selection only. A promoted operator "
            "must be reprofiled because the v177 five-head map was learned with Reservoir."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v182 Structured-Coverage Screen",
        "",
        f"Decision: `{report['decision']}`",
        f"Primary Pareto front: {', '.join(report['primary_pareto_front'])}",
        "",
        "| Candidate | Pareto | Noninferior | Useful gain | Reprofile |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate, row in report["candidate_status"].items():
        lines.append(
            f"| {candidate} | {row['on_primary_pareto_front']} | "
            f"{row['noninferior_to_reservoir_and_recent']} | "
            f"{row['useful_gain_over_reservoir']} | {row['promote_to_reprofile']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(manifest, summary, args.parts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v182-decision] {report['decision']}")


if __name__ == "__main__":
    main()
