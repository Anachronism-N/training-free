#!/usr/bin/env python3
"""Paired prompt-level mechanism analysis for a frozen v174 scope."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import analyze_v165_final_decision as base
from vbench_quality_contract import exclusive_scores, official_quality_score


METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
    "dynamic_degree",
)


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    wins = sum(value > 0.0 for value in nonzero)
    n = len(nonzero)
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n)


def bh(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    total = len(order)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = total - reverse_rank + 1
        running = min(running, rows[index]["p_value"] * total / rank)
        rows[index]["q_value"] = min(1.0, running)


def load_prompt_rows(parts_root: Path, summary: dict, methods: tuple[str, ...], prompt_count: int) -> dict:
    dimensions = tuple(summary["dimensions"])
    rows = {(method, prompt): {} for method in methods for prompt in range(prompt_count)}
    for method in methods:
        for dimension in dimensions:
            clips = base.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
            )
            raw_values = [value for prompt in range(prompt_count) for value in clips[prompt]]
            summary_value = base.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = base.scale_factor(
                float(np.mean(raw_values)),
                summary_value,
                name=f"{method}:{dimension}",
            )
            for prompt in range(prompt_count):
                rows[(method, prompt)][dimension] = factor * float(np.mean(clips[prompt]))
    return rows


def derived_rows(raw: dict, methods: tuple[str, ...], prompt_count: int) -> dict:
    result = {}
    for method in methods:
        for prompt in range(prompt_count):
            row = raw[(method, prompt)]
            exclusive = exclusive_scores(row)
            result[(method, prompt)] = {
                **exclusive,
                "official_quality_score": official_quality_score(row),
            }
    return result


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row["key"] for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v174 paired analysis received an incomplete summary")
    if "matched" not in methods or "swapped" not in methods:
        raise ValueError("v174 requires matched and swapped methods")
    raw = load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = derived_rows(raw, methods, prompt_count)
    random_methods = tuple(method for method in methods if method.startswith("random_count_matched_"))
    if not random_methods:
        raise ValueError("v174 requires at least one count-matched random control")

    controls = ["swapped", *random_methods]
    comparisons = []
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(METRICS):
            deltas = [
                rows[("matched", prompt)][metric] - rows[(control, prompt)][metric]
                for prompt in range(prompt_count)
            ]
            comparisons.append(
                {
                    "comparison": f"matched_minus_{control}",
                    "control": control,
                    "metric": metric,
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                    "bootstrap_ci95": bootstrap_ci(
                        deltas,
                        seed=1742026 + control_index * 101 + metric_index,
                    ),
                    "p_value": sign_p(deltas),
                    "per_prompt_delta": deltas,
                }
            )
    for metric_index, metric in enumerate(METRICS):
        deltas = []
        for prompt in range(prompt_count):
            random_mean = float(
                np.mean([rows[(method, prompt)][metric] for method in random_methods])
            )
            deltas.append(rows[("matched", prompt)][metric] - random_mean)
        comparisons.append(
            {
                "comparison": "matched_minus_random_ensemble",
                "control": "random_ensemble",
                "metric": metric,
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                "bootstrap_ci95": bootstrap_ci(
                    deltas,
                    seed=1749026 + metric_index,
                ),
                "p_value": sign_p(deltas),
                "per_prompt_delta": deltas,
            }
        )
    bh(comparisons)

    primary = [
        row
        for row in comparisons
        if row["control"] in {"swapped", "random_ensemble"}
        and row["metric"] in {"official_quality_score", "identity_background"}
    ]
    screen_gate = all(row["mean_delta"] > 0.0 for row in primary)
    confirmation_gate = (
        prompt_count == 128
        and all(row["bootstrap_ci95"][0] > 0.0 for row in primary)
        and all(row["q_value"] <= 0.10 for row in primary)
        and all(row["win_fraction"] >= 0.55 for row in primary)
    )
    dynamic_rows = [
        row
        for row in comparisons
        if row["control"] in {"swapped", "random_ensemble"}
        and row["metric"] == "dynamic_degree"
    ]
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "prompt_count": prompt_count,
        "methods": list(methods),
        "random_controls": list(random_methods),
        "comparisons": comparisons,
        "screen_promotion_gate": bool(screen_gate),
        "classifier_confirmation_gate": bool(confirmation_gate),
        "dynamic_nonregression_observed": all(
            row["mean_delta"] >= -0.02 for row in dynamic_rows
        ),
        "claim_boundary": (
            "Only the 128-prompt matched-vs-swapped/random paired gate can "
            "support the cache-compatibility classifier claim. Uniform "
            "controls and aggregate VBench scores are operator ablations."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v174 Paired Cache-Compatibility Analysis",
        "",
        f"Prompts: {report['prompt_count']}",
        f"Screen gate: {report['screen_promotion_gate']}",
        f"Classifier confirmation gate: {report['classifier_confirmation_gate']}",
        f"Dynamic non-regression: {report['dynamic_nonregression_observed']}",
        "",
        "| Comparison | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["control"] not in {"swapped", "random_ensemble"}:
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v174-paired] "
        f"screen_gate={report['screen_promotion_gate']} "
        f"confirmation_gate={report['classifier_confirmation_gate']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
