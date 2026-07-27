#!/usr/bin/env python3
"""Build paired MovieBench-16 comparisons for the v116 candidate screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


METRICS = {
    "composite": ("Composite", 1),
    "m1_dino_consistency": ("DINO", 1),
    "m2_drift_slope": ("Drift slope", 1),
    "m3_motion_smoothness": ("Motion acceleration", -1),
    "m4_arcface_id_sim": ("ArcFace", 1),
    "m5_temporal_flickering": ("Flicker", -1),
    "m6_clip_text_alignment": ("CLIP", 1),
    "m7_background_consistency": ("Background", 1),
    "m8_loop_score": ("Loop", -1),
}
VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(int(seed))
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(max(100, int(samples)))
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def load_manifest(run_root: Path) -> tuple[list[str], int]:
    path = run_root / "published_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise ValueError(f"published manifest is not successful: {path}")
    prompt_count = int(payload.get("prompt_count", 0))
    methods = [str(row["key"]) for row in payload.get("methods", [])]
    if prompt_count != 16 or not methods or len(methods) != len(set(methods)):
        raise ValueError("v116 manifest must contain unique methods and 16 prompts")
    return methods, prompt_count


def load_auxiliary(
    run_root: Path,
    methods: list[str],
    prompt_count: int,
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict]]:
    per_prompt: dict[str, dict[int, dict[str, float]]] = {}
    aggregates: dict[str, dict] = {}
    expected = set(range(prompt_count))
    for method in methods:
        path = run_root / "metrics" / "auxiliary" / method / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload.get("per_method", {}).get(method)
        if not isinstance(aggregate, dict):
            raise ValueError(f"{method}: missing auxiliary aggregate")
        observed: dict[int, dict[str, float]] = {}
        for row in payload.get("per_video", {}).values():
            if not isinstance(row, dict) or row.get("method") != method:
                continue
            prompt_index = int(row["prompt_index"])
            metrics = {
                key: value
                for key in METRICS
                if (value := finite(row.get("metrics", {}).get(key))) is not None
            }
            if prompt_index in observed:
                raise ValueError(f"{method}: duplicate prompt {prompt_index}")
            observed[prompt_index] = metrics
        if set(observed) != expected:
            raise ValueError(
                f"{method}: auxiliary prompt coverage mismatch; "
                f"missing={sorted(expected - set(observed))} "
                f"extra={sorted(set(observed) - expected)}"
            )
        per_prompt[method] = observed
        aggregates[method] = aggregate
    return per_prompt, aggregates


def paired_rows(
    per_prompt: dict[str, dict[int, dict[str, float]]],
    *,
    reference: str,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_rows = per_prompt[reference]
    for method, method_rows in per_prompt.items():
        for metric, (label, direction) in METRICS.items():
            paired = []
            for prompt_index in sorted(reference_rows):
                left = method_rows[prompt_index].get(metric)
                right = reference_rows[prompt_index].get(metric)
                if left is None or right is None:
                    continue
                raw_delta = float(left) - float(right)
                paired.append(
                    (prompt_index, raw_delta, raw_delta * int(direction))
                )
            if not paired:
                continue
            improvements = [value[2] for value in paired]
            raw_deltas = [value[1] for value in paired]
            tolerance = 1e-12
            wins = sum(value > tolerance for value in improvements)
            losses = sum(value < -tolerance for value in improvements)
            ties = len(improvements) - wins - losses
            ci_low, ci_high = bootstrap_mean_ci(
                improvements,
                seed=20260727 + sum(ord(char) for char in method + metric),
                samples=bootstrap_samples,
            )
            rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "metric_label": label,
                    "higher_is_better": direction > 0,
                    "paired_prompts": len(paired),
                    "mean_raw_delta": statistics.fmean(raw_deltas),
                    "mean_improvement": statistics.fmean(improvements),
                    "median_improvement": statistics.median(improvements),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
    return rows


def load_vbench(run_root: Path, methods: list[str]) -> dict[str, dict]:
    path = run_root / "metrics" / "vbench_long_summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("methods")
    if not isinstance(rows, dict):
        raise ValueError(f"invalid VBench summary: {path}")
    missing = sorted(set(methods) - set(rows))
    if missing:
        raise ValueError(f"VBench summary misses methods: {missing}")
    return {
        method: {
            dimension: finite(rows[method].get(dimension))
            for dimension in VBENCH_DIMENSIONS
        }
        for method in methods
    }


def method_summary(
    methods: list[str],
    aggregates: dict[str, dict],
    vbench: dict[str, dict],
    *,
    reference: str,
) -> list[dict[str, Any]]:
    rows = []
    reference_aux = aggregates[reference]
    reference_vbench = vbench.get(reference, {})
    for method in methods:
        row: dict[str, Any] = {"method": method}
        for metric in METRICS:
            value = finite(aggregates[method].get(metric))
            baseline = finite(reference_aux.get(metric))
            row[metric] = value
            row[f"{metric}_delta"] = (
                None
                if value is None or baseline is None
                else value - baseline
            )
        for dimension in VBENCH_DIMENSIONS:
            value = vbench.get(method, {}).get(dimension)
            baseline = reference_vbench.get(dimension)
            row[f"vbench_{dimension}"] = value
            row[f"vbench_{dimension}_delta"] = (
                None
                if value is None or baseline is None
                else value - baseline
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    number = finite(value)
    return "n/a" if number is None else f"{number:.6f}"


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    reference: str,
    has_vbench: bool,
) -> None:
    lines = [
        "# v116 candidate metric analysis",
        "",
        f"Paired reference: `{reference}`. Positive paired improvement always "
        "means better after metric-direction normalization.",
        "",
        "## Candidate overview",
        "",
        "| Method | Composite | Δ composite | DINO | Δ DINO | "
        "Motion accel. | Dynamic degree | Subject | Background |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {fmt(row.get('composite'))} | "
            f"{fmt(row.get('composite_delta'))} | "
            f"{fmt(row.get('m1_dino_consistency'))} | "
            f"{fmt(row.get('m1_dino_consistency_delta'))} | "
            f"{fmt(row.get('m3_motion_smoothness'))} | "
            f"{fmt(row.get('vbench_dynamic_degree'))} | "
            f"{fmt(row.get('vbench_subject_consistency'))} | "
            f"{fmt(row.get('vbench_background_consistency'))} |"
        )
    lines.extend(
        [
            "",
            "## Paired auxiliary deltas",
            "",
            "| Method | Metric | N | Mean improvement | Median | 95% CI | W/T/L |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row["method"] == reference:
            continue
        lines.append(
            f"| {row['method']} | {row['metric_label']} | "
            f"{row['paired_prompts']} | {row['mean_improvement']:.6f} | "
            f"{row['median_improvement']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- `m3_motion_smoothness` measures acceleration smoothness, not motion "
            "magnitude; use VBench dynamic degree to reject frozen methods.",
            "- DINO/ArcFace gains accompanied by lower dynamic degree or visible "
            "scale enlargement are not accepted as improvements.",
            "- A 16-prompt confidence interval is diagnostic, not final-paper "
            "significance. Confirm the selected method on MovieBench-128.",
        ]
    )
    if not has_vbench:
        lines.append(
            "- VBench summary was not present; rerun this analyzer after VBench "
            "collection to populate motion and quality columns."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--reference", default="landmark_recent8")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    methods, prompt_count = load_manifest(run_root)
    if args.reference not in methods:
        raise SystemExit(
            f"reference {args.reference!r} is not in manifest methods"
        )
    per_prompt, aggregates = load_auxiliary(
        run_root,
        methods,
        prompt_count,
    )
    paired = paired_rows(
        per_prompt,
        reference=args.reference,
        bootstrap_samples=args.bootstrap_samples,
    )
    vbench = load_vbench(run_root, methods)
    summaries = method_summary(
        methods,
        aggregates,
        vbench,
        reference=args.reference,
    )
    output = run_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "run_root": str(run_root),
        "reference": args.reference,
        "prompt_count": prompt_count,
        "methods": methods,
        "vbench_available": bool(vbench),
        "summaries": summaries,
        "paired": paired,
    }
    (output / "v116_candidate_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "v116_candidate_summary.csv", summaries)
    write_csv(output / "v116_candidate_paired.csv", paired)
    write_markdown(
        output / "v116_candidate_metrics.md",
        summaries,
        paired,
        reference=args.reference,
        has_vbench=bool(vbench),
    )
    print(
        f"[v116-analysis] methods={len(methods)} prompts={prompt_count} "
        f"vbench={bool(vbench)} output={output}"
    )


if __name__ == "__main__":
    main()
