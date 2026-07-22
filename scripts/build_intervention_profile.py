#!/usr/bin/env python3
"""Build a layer/head intervention profile from paired counterfactual metrics.

The input is a long-form CSV. Each intervention row declares the inclusive
layer/head range that received historical memory. Native rows are paired by
``prompt_id`` and ``seed``. Utility is equal-weight percentile rank aggregation
over oriented metric deltas; no semantic head labels are assigned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


IDENTITY_METRICS = ("dino", "min_dino", "arcface", "vbench_subject")
DYNAMICS_METRICS = ("motion", "vbench_dynamic")
LOWER_IS_BETTER = ("loop", "flicker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--baseline-cell", default="sf_native")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument(
        "--metrics",
        default=",".join((*IDENTITY_METRICS, *DYNAMICS_METRICS, *LOWER_IS_BETTER)),
    )
    return parser.parse_args()


def _number(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "").strip()
    if not raw:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("prompt_id", "").strip(), row.get("seed", "").strip()


def _percentile_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank / (len(values) - 1)
    return result


def build_profile(
    rows: list[dict[str, str]],
    *,
    baseline_cell: str,
    num_layers: int,
    num_heads: int,
    metrics: list[str],
    min_samples: int,
) -> dict:
    baselines = {
        _pair_key(row): row
        for row in rows
        if row.get("cell", "").strip() == baseline_cell
    }
    observations: list[dict] = []
    for row in rows:
        if row.get("cell", "").strip() == baseline_cell:
            continue
        baseline = baselines.get(_pair_key(row))
        if baseline is None:
            continue
        deltas = {}
        for metric in metrics:
            value = _number(row, metric)
            reference = _number(baseline, metric)
            if value is None or reference is None:
                continue
            delta = (value - reference) / max(abs(reference), 1e-8)
            if metric in LOWER_IS_BETTER:
                delta = -delta
            deltas[metric] = delta
        if not deltas:
            continue
        observation = {
            "row": row,
            "deltas": deltas,
            "metric_ranks": {},
        }
        observations.append(observation)

    for metric in metrics:
        available = [obs for obs in observations if metric in obs["deltas"]]
        ranks = _percentile_ranks([obs["deltas"][metric] for obs in available])
        for obs, rank in zip(available, ranks):
            obs["metric_ranks"][metric] = rank

    buckets: dict[tuple[int, int, str, int], list[tuple[float, float]]] = defaultdict(list)
    for observation in observations:
        row = observation["row"]
        ranks = list(observation["metric_ranks"].values())
        utility = sum(ranks) / len(ranks)
        oriented_delta = sum(observation["deltas"].values()) / len(
            observation["deltas"]
        )
        layer_start = int(row["layer_start"])
        layer_end = int(row["layer_end"])
        head_start = int(row["head_start"])
        head_end = int(row["head_end"])
        memory_mode = row.get("memory_mode", "noisy").strip() or "noisy"
        call_index = int(row.get("attention_call_index", "-1") or -1)
        if not (0 <= layer_start <= layer_end < num_layers):
            raise ValueError(f"invalid layer range in row: {row}")
        if not (0 <= head_start <= head_end < num_heads):
            raise ValueError(f"invalid head range in row: {row}")
        for layer in range(layer_start, layer_end + 1):
            for head in range(head_start, head_end + 1):
                buckets[(layer, head, memory_mode, call_index)].append(
                    (utility, oriented_delta)
                )

    entries = []
    for (layer, head, memory_mode, call_index), values in sorted(buckets.items()):
        utilities = [value[0] for value in values]
        signs = [value[1] >= 0.0 for value in values]
        positive_fraction = sum(signs) / len(signs)
        sign_consistency = abs(2.0 * positive_fraction - 1.0)
        sample_factor = min(1.0, len(values) / max(1, min_samples))
        entries.append(
            {
                "layer": layer,
                "head": head,
                "memory_mode": memory_mode,
                "attention_call_index": call_index,
                "utility": sum(utilities) / len(utilities),
                "reliability": sign_consistency * sample_factor,
                "samples": len(values),
                "positive_fraction": positive_fraction,
            }
        )
    return {
        "version": 1,
        "method": "paired_counterfactual_rank_aggregation",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "default_utility": 0.0,
        "metrics": metrics,
        "baseline_cell": baseline_cell,
        "paired_observations": len(observations),
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "cell",
        "prompt_id",
        "seed",
        "layer_start",
        "layer_end",
        "head_start",
        "head_end",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    profile = build_profile(
        rows,
        baseline_cell=args.baseline_cell,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        metrics=metrics,
        min_samples=args.min_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[profile] observations={profile['paired_observations']} "
        f"entries={len(profile['entries'])} output={args.output_json}"
    )


if __name__ == "__main__":
    main()
