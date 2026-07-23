#!/usr/bin/env python3
"""Validate and summarize PyramidKV cache-transition JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "max": max(finite),
    }


def _load(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(event, dict):
                failures.append(f"line {line_number}: event is not an object")
                continue
            events.append(event)
    return events, failures


def summarize(path: Path, expected_layers: int) -> dict[str, Any]:
    events, failures = _load(path)
    reasons: Counter[str] = Counter()
    labels: dict[int, list[bool]] = defaultdict(list)
    layers: set[int] = set()
    branches: Counter[str] = Counter()
    accepted_by_layer: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    accepted_by_branch: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    reliability: list[float] = []
    shock: list[float] = []
    denoise: list[float] = []
    ages: list[int] = []
    accepted = 0
    total = 0
    modes: Counter[str] = Counter()

    if not events:
        failures.append("trace is empty")
    for index, event in enumerate(events):
        if event.get("event") != "cache_transition":
            failures.append(f"event {index}: unexpected type {event.get('event')!r}")
            continue
        layer = int(event.get("layer", -1))
        if layer < 0:
            failures.append(f"event {index}: invalid layer")
        layers.add(layer)
        branch = str(event.get("branch", "missing"))
        branches[branch] += 1
        modes[str(event.get("mode", "missing"))] += 1

        mask = event.get("commit_mask", [])
        event_reasons = event.get("reasons", [])
        event_labels = event.get("head_labels", [])
        event_reliability = event.get("reliability", [])
        event_shock = event.get("shock", [])
        event_denoise = event.get("denoise_disagreement", [])
        event_ages = event.get("age_before", [])
        lengths = {
            len(mask),
            len(event_reasons),
            len(event_labels),
            len(event_reliability),
            len(event_shock),
            len(event_denoise),
            len(event_ages),
        }
        if len(lengths) != 1 or not mask:
            failures.append(f"event {index}: per-head field length mismatch")
            continue
        if int(event.get("accepted", -1)) != sum(bool(value) for value in mask):
            failures.append(f"event {index}: accepted count does not match mask")
        if int(event.get("total", -1)) != len(mask):
            failures.append(f"event {index}: total count does not match mask")
        numeric = [
            *event_reliability,
            *event_shock,
            *event_denoise,
            *event_ages,
        ]
        if any(not math.isfinite(float(value)) for value in numeric):
            failures.append(f"event {index}: non-finite metric")

        reasons.update(str(reason) for reason in event_reasons)
        for label, committed in zip(event_labels, mask):
            labels[int(label)].append(bool(committed))
        reliability.extend(float(value) for value in event_reliability)
        shock.extend(float(value) for value in event_shock)
        denoise.extend(float(value) for value in event_denoise)
        ages.extend(int(value) for value in event_ages)
        accepted += sum(bool(value) for value in mask)
        total += len(mask)
        accepted_by_layer[layer][0] += sum(bool(value) for value in mask)
        accepted_by_layer[layer][1] += len(mask)
        accepted_by_branch[branch][0] += sum(bool(value) for value in mask)
        accepted_by_branch[branch][1] += len(mask)

    missing_layers = sorted(set(range(expected_layers)) - layers)
    if missing_layers:
        failures.append(f"missing layers: {missing_layers}")
    if reliability and (min(reliability) < 0.0 or max(reliability) > 1.0):
        failures.append("reliability is outside [0, 1]")
    if modes == Counter({"audit": len(events)}) and accepted != total:
        failures.append("audit mode rejected at least one middle-cache update")

    return {
        "trace": str(path),
        "status": "failed" if failures else "nominal",
        "events": len(events),
        "layers": sorted(layers),
        "branches": dict(sorted(branches.items())),
        "modes": dict(sorted(modes.items())),
        "accepted": accepted,
        "total": total,
        "acceptance_rate": accepted / total if total else None,
        "reasons": dict(sorted(reasons.items())),
        "reliability": _stats(reliability),
        "shock": _stats(shock),
        "denoise_disagreement": _stats(denoise),
        "age_before": _stats(ages),
        "acceptance_by_label": {
            str(label): {
                "accepted": sum(values),
                "total": len(values),
                "rate": sum(values) / len(values),
            }
            for label, values in sorted(labels.items())
        },
        "acceptance_by_layer": {
            str(layer): {
                "accepted": counts[0],
                "total": counts[1],
                "rate": counts[0] / counts[1],
            }
            for layer, counts in sorted(accepted_by_layer.items())
        },
        "acceptance_by_branch": {
            branch: {
                "accepted": counts[0],
                "total": counts[1],
                "rate": counts[0] / counts[1],
            }
            for branch, counts in sorted(accepted_by_branch.items())
        },
        "failures": sorted(set(failures)),
    }


def _write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cache Transition Trace Summary",
        "",
        "| Trace | Status | Events | Acceptance | Median reliability | "
        "Median shock | Median denoise disagreement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        acceptance = item["acceptance_rate"]
        lines.append(
            f"| {Path(item['trace']).name} | {item['status']} | {item['events']} | "
            f"{acceptance:.4f} | "
            f"{item['reliability']['median']} | {item['shock']['median']} | "
            f"{item['denoise_disagreement']['median']} |"
            if acceptance is not None
            else f"| {Path(item['trace']).name} | {item['status']} | "
            f"{item['events']} | n/a | n/a | n/a | n/a |"
        )
    for item in summaries:
        lines.extend(
            [
                "",
                f"## {Path(item['trace']).name}",
                f"- Reasons: `{json.dumps(item['reasons'], sort_keys=True)}`",
                f"- Acceptance by label: "
                f"`{json.dumps(item['acceptance_by_label'], sort_keys=True)}`",
            ]
        )
        lines.extend(f"- FAILURE: {failure}" for failure in item["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summaries = [
        summarize(path, expected_layers=args.expected_layers) for path in args.traces
    ]
    payload = {"summaries": summaries}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.output_md:
        _write_markdown(summaries, args.output_md)
    return int(args.strict and any(item["failures"] for item in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
