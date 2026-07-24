#!/usr/bin/env python3
"""Combine v90 paired-seed, quality, jump, and cache-coherence results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


PAIRS = (("pf", "v78", 0),) + tuple(
    (f"pf_s{seed}", f"v78_s{seed}", seed) for seed in (1, 2, 3)
)
METRICS = (
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
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "nonnegative": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "nonnegative": sum(value >= 0.0 for value in values),
    }


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
            "age_over_effective_max": summary.get(
                "age_over_effective_max",
                {},
            ),
            "status": summary.get("status"),
        }
    return output


def analyze(
    comprehensive: dict[str, Any],
    temporal: dict[str, dict[str, float | int]],
    coherence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    methods = comprehensive.get("per_method") or {}
    missing = sorted(
        {
            name
            for left, right, _ in PAIRS
            for name in (left, right)
            if name not in methods
        }
    )
    if missing:
        raise ValueError(f"missing paired methods: {missing}")

    paired = []
    differences: dict[str, list[float]] = defaultdict(list)
    for pf_name, v78_name, seed in PAIRS:
        row: dict[str, Any] = {
            "seed": seed,
            "pf": pf_name,
            "v78": v78_name,
            "differences": {},
        }
        for metric in METRICS:
            left = _finite(methods[pf_name].get(metric))
            right = _finite(methods[v78_name].get(metric))
            difference = (
                right - left if left is not None and right is not None else None
            )
            row["differences"][metric] = difference
            if difference is not None:
                differences[metric].append(difference)
        pf_jump = temporal.get(pf_name, {}).get("mean")
        v78_jump = temporal.get(v78_name, {}).get("mean")
        jump_difference = (
            float(v78_jump) - float(pf_jump)
            if pf_jump is not None and v78_jump is not None
            else None
        )
        row["differences"]["temporal_jump"] = jump_difference
        if jump_difference is not None:
            differences["temporal_jump"].append(jump_difference)
        paired.append(row)

    baseline_names = {name for pair in PAIRS for name in pair[:2]}
    baseline_names.update(
        {"pf_binary_balanced", "learned_balanced"}
    )
    candidates = []
    for name, values in methods.items():
        if name in baseline_names:
            continue
        candidates.append(
            {
                "method": name,
                "dino": _finite(values.get("m1_dino_consistency")),
                "min_dino": _finite(values.get("m1_min_stability")),
                "background": _finite(values.get("m7_background_consistency")),
                "composite": _finite(values.get("composite")),
                "temporal_jump": temporal.get(name, {}).get("mean"),
                "trace": coherence.get(name),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["dino"] is not None,
            item["dino"] if item["dino"] is not None else -math.inf,
        ),
        reverse=True,
    )

    dino_summary = _summary(differences["m1_dino_consistency"])
    return {
        "paired": paired,
        "paired_summary": {
            metric: _summary(values)
            for metric, values in sorted(differences.items())
        },
        "v78_identity_gate": {
            "mean_dino_positive": (
                dino_summary["mean"] is not None
                and float(dino_summary["mean"]) > 0.0
            ),
            "at_least_three_nonnegative_seeds": (
                int(dino_summary["nonnegative"]) >= 3
            ),
            "passed": (
                dino_summary["mean"] is not None
                and float(dino_summary["mean"]) > 0.0
                and int(dino_summary["nonnegative"]) >= 3
            ),
        },
        "candidates": candidates,
        "temporal_jump": temporal,
        "coherence": coherence,
    }


def _fmt(value: Any, digits: int = 5) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v90 Combined Analysis",
        "",
        "## Paired PF-v78 results",
        "",
        "| Seed | Delta DINO | Delta min DINO | Delta BG | Delta jump |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paired"]:
        differences = row["differences"]
        lines.append(
            f"| {row['seed']} | "
            f"{_fmt(differences.get('m1_dino_consistency'))} | "
            f"{_fmt(differences.get('m1_min_stability'))} | "
            f"{_fmt(differences.get('m7_background_consistency'))} | "
            f"{_fmt(differences.get('temporal_jump'))} |"
        )
    dino = payload["paired_summary"]["m1_dino_consistency"]
    lines.extend(
        [
            "",
            f"- Mean paired Delta DINO: `{_fmt(dino['mean'])}`",
            f"- Non-negative seeds: `{dino['nonnegative']}/{dino['count']}`",
            f"- Identity gate passed: `{payload['v78_identity_gate']['passed']}`",
            "",
            "## Seed-0 factorization candidates",
            "",
            "| Method | DINO | min DINO | BG | Jump | Acceptance | Age spread |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["candidates"]:
        trace = item.get("trace") or {}
        coherence = trace.get("coherence") or {}
        age_spread = (coherence.get("age_spread") or {}).get("mean")
        lines.append(
            f"| {item['method']} | {_fmt(item['dino'])} | "
            f"{_fmt(item['min_dino'])} | {_fmt(item['background'])} | "
            f"{_fmt(item['temporal_jump'])} | "
            f"{_fmt(trace.get('acceptance_rate'))} | "
            f"{_fmt(age_spread)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    temporal = _load_temporal(args.temporal_jump)
    coherence = _load_coherence(args.trace_summary)
    payload = analyze(comprehensive, temporal, coherence)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        f"[v90-analysis] paired={len(payload['paired'])} "
        f"candidates={len(payload['candidates'])} "
        f"gate={payload['v78_identity_gate']['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
