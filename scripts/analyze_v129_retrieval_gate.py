#!/usr/bin/env python3
"""Summarize confidence-gated retrieval decisions from v129 policy traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_METHODS = (
    "ours_prototype_retrieval_conf_recent",
    "ours_prototype_retrieval_conf_motion",
)
TRACE_NAME = re.compile(r"^(ours_.+)__p(\d{3})\.policy\.jsonl$")
SIMILARITY_SWEEP = (0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80)
MARGIN_SWEEP = (0.0, 0.0025, 0.005, 0.01, 0.02)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            key: None
            for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")
        }

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def retrieval_state(event: dict[str, Any]) -> dict[str, Any] | None:
    strategies = event.get("strategies")
    if not isinstance(strategies, list):
        return None
    matches = [
        row
        for row in strategies
        if isinstance(row, dict)
        and row.get("name") == "SemanticRetrievalStrategy"
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("multiple SemanticRetrievalStrategy entries")
    state = matches[0].get("state")
    if not isinstance(state, dict):
        raise ValueError("SemanticRetrievalStrategy has no debug state")
    retrieval = state.get("last_retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("SemanticRetrievalStrategy has no retrieval state")
    return retrieval


def trace_inventory(trace_root: Path) -> dict[str, list[tuple[int, Path]]]:
    inventory: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in trace_root.glob("*.policy.jsonl"):
        match = TRACE_NAME.fullmatch(path.name)
        if match is None:
            continue
        method, prompt_index = match.group(1), int(match.group(2))
        if method in EXPECTED_METHODS:
            inventory[method].append((prompt_index, path))
    for method in EXPECTED_METHODS:
        rows = sorted(inventory.get(method, []))
        indices = [index for index, _ in rows]
        if indices != list(range(128)):
            raise RuntimeError(
                f"{method}: expected trace prompts [0,128), found {indices}"
            )
        inventory[method] = rows
    return inventory


def summarize_method(
    method: str,
    traces: list[tuple[int, Path]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reason_counts: Counter[str] = Counter()
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    by_branch: dict[str, Counter[str]] = defaultdict(Counter)
    by_prompt: dict[int, Counter[str]] = defaultdict(Counter)
    top1_values: list[float] = []
    top2_values: list[float] = []
    margin_values: list[float] = []
    selected_ages: list[float] = []
    sweep_observations: list[dict[str, float | None]] = []
    thresholds: set[tuple[float, float, bool]] = set()
    records = 0
    retrieval_records = 0
    selected_records = 0
    low_confidence_records = 0
    malformed: list[str] = []
    cache_violations: Counter[str] = Counter()

    for prompt_index, path in traces:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                records += 1
                try:
                    event = json.loads(raw)
                    if event.get("event") != "middle_selection":
                        raise ValueError(
                            f"unexpected event={event.get('event')!r}"
                        )
                    for violation in event.get(
                        "cache_contract_violations", []
                    ):
                        cache_violations[str(violation)] += 1
                    retrieval = retrieval_state(event)
                    if retrieval is None:
                        continue
                    retrieval_records += 1
                    reason = str(retrieval.get("reason", "unknown"))
                    selected = retrieval.get("selected")
                    has_selection = isinstance(selected, list) and bool(
                        selected
                    )
                    if has_selection:
                        selected_records += 1
                    if reason in {"similarity_gate", "margin_gate"}:
                        low_confidence_records += 1
                    reason_counts[reason] += 1
                    by_layer[int(event["layer"])][reason] += 1
                    by_branch[str(event["branch"])][reason] += 1
                    by_prompt[prompt_index][reason] += 1
                    top1 = finite(retrieval.get("top1_similarity"))
                    top2 = finite(retrieval.get("top2_similarity"))
                    margin = finite(retrieval.get("margin"))
                    if top1 is not None:
                        top1_values.append(top1)
                    if top2 is not None:
                        top2_values.append(top2)
                    if margin is not None:
                        margin_values.append(margin)
                    if isinstance(selected, list):
                        for row in selected:
                            if isinstance(row, dict):
                                age = finite(row.get("age"))
                                if age is not None:
                                    selected_ages.append(age)
                    thresholds.add(
                        (
                            float(retrieval.get("min_similarity", -0.25)),
                            float(retrieval.get("min_margin", 0.0)),
                            bool(
                                retrieval.get(
                                    "abstain_on_low_confidence", False
                                )
                            ),
                        )
                    )
                    if top1 is not None:
                        sweep_observations.append(
                            {
                                "top1": top1,
                                "margin": margin,
                            }
                        )
                except Exception as error:
                    malformed.append(
                        f"{path.name}:{line_number}: {error}"
                    )
    if malformed:
        raise RuntimeError(
            f"{method}: malformed traces: {malformed[:10]}"
        )
    if not retrieval_records:
        raise RuntimeError(f"{method}: no retrieval records")
    if len(thresholds) != 1:
        raise RuntimeError(
            f"{method}: inconsistent retrieval thresholds {thresholds}"
        )
    threshold = next(iter(thresholds))
    scored_records = len(sweep_observations)

    sweep_rows = []
    for similarity in SIMILARITY_SWEEP:
        for margin_floor in MARGIN_SWEEP:
            accepted = sum(
                1
                for row in sweep_observations
                if float(row["top1"]) >= similarity
                and (
                    row["margin"] is None
                    or float(row["margin"]) >= margin_floor
                )
            )
            total = len(sweep_observations)
            sweep_rows.append(
                {
                    "method": method,
                    "min_similarity": similarity,
                    "min_margin": margin_floor,
                    "eligible_records": total,
                    "posthoc_accept_count": accepted,
                    "posthoc_accept_rate": (
                        accepted / total if total else None
                    ),
                }
            )

    return (
        {
            "method": method,
            "trace_count": len(traces),
            "record_count": records,
            "retrieval_record_count": retrieval_records,
            "scored_candidate_count": scored_records,
            "selected_record_count": selected_records,
            "selected_rate": selected_records / retrieval_records,
            "selected_rate_when_scored": (
                selected_records / scored_records
                if scored_records
                else None
            ),
            "low_confidence_abstain_count": low_confidence_records,
            "low_confidence_abstain_rate": (
                low_confidence_records / retrieval_records
            ),
            "low_confidence_abstain_rate_when_scored": (
                low_confidence_records / scored_records
                if scored_records
                else None
            ),
            "configured_gate": {
                "min_similarity": threshold[0],
                "min_margin": threshold[1],
                "abstain_on_low_confidence": threshold[2],
            },
            "reason_counts": dict(sorted(reason_counts.items())),
            "top1_similarity": quantiles(top1_values),
            "top2_similarity": quantiles(top2_values),
            "margin": quantiles(margin_values),
            "selected_age": quantiles(selected_ages),
            "cache_contract_violations": dict(
                sorted(cache_violations.items())
            ),
            "by_layer": {
                str(layer): dict(sorted(counts.items()))
                for layer, counts in sorted(by_layer.items())
            },
            "by_branch": {
                branch: dict(sorted(counts.items()))
                for branch, counts in sorted(by_branch.items())
            },
            "by_prompt": {
                str(prompt): dict(sorted(counts.items()))
                for prompt, counts in sorted(by_prompt.items())
            },
        },
        sweep_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = trace_inventory(args.run_root.resolve() / "traces")
    summaries = []
    sweep_rows = []
    for method in EXPECTED_METHODS:
        summary, method_sweep = summarize_method(
            method,
            inventory[method],
        )
        summaries.append(summary)
        sweep_rows.extend(method_sweep)
    output = {
        "version": 1,
        "run_root": str(args.run_root.resolve()),
        "methods": summaries,
        "interpretation": {
            "selected_rate": (
                "fraction of traced suppressive-head reads that retained "
                "the retrieval frame"
            ),
            "posthoc_sweep": (
                "diagnostic only; it reclassifies logged scores and does "
                "not change generated videos"
            ),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "retrieval_gate_summary.json"
    csv_path = args.output_root / "retrieval_gate_summary.csv"
    sweep_path = args.output_root / "retrieval_gate_threshold_sweep.csv"
    md_path = args.output_root / "retrieval_gate_summary.md"
    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "method",
            "trace_count",
            "retrieval_record_count",
            "scored_candidate_count",
            "selected_record_count",
            "selected_rate",
            "selected_rate_when_scored",
            "low_confidence_abstain_count",
            "low_confidence_abstain_rate",
            "low_confidence_abstain_rate_when_scored",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fieldnames})
    with sweep_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "method",
            "min_similarity",
            "min_margin",
            "eligible_records",
            "posthoc_accept_count",
            "posthoc_accept_rate",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)
    lines = [
        "# v129 retrieval gate diagnostics",
        "",
        "| Method | Traces | Retrieval reads | Scored | Selected | "
        "Selected / scored | Abstained / scored |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['trace_count']} | "
            f"{row['retrieval_record_count']} | "
            f"{row['scored_candidate_count']} | "
            f"{row['selected_record_count']} | "
            f"{percentage(row['selected_rate_when_scored'])} | "
            f"{percentage(row['low_confidence_abstain_rate_when_scored'])} |"
        )
    lines.extend(
        [
            "",
            "The threshold sweep is post-hoc diagnostics only. Use it to "
            "interpret gate activity, not to relabel a completed video run.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[v129-gate-analysis] methods={len(summaries)} "
        f"summary={json_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
