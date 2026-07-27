#!/usr/bin/env python3
"""Summarize v111/v112 role-event feature and cache-decision traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "min": min(finite),
        "median": median(finite),
        "mean": mean(finite),
        "max": max(finite),
    }


def _selected_t(decision: dict[str, Any]) -> int | None:
    if "candidate_t" in decision:
        return int(decision["candidate_t"])
    pair = decision.get("candidate_pair")
    if isinstance(pair, list) and len(pair) == 2:
        return int(pair[1])
    return None


def analyze_cell(
    role_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    feature_motion: dict[str, list[float]] = defaultdict(list)
    feature_semantic: dict[str, list[float]] = defaultdict(list)
    feature_records: Counter[str] = Counter()
    feature_seen: set[tuple[Any, ...]] = set()
    for event in _read_jsonl(role_path):
        if event.get("event") != "role_event_features":
            continue
        context_key = str(event["context_key"])
        key = (
            int(event["layer"]),
            str(event.get("branch", "unknown")),
            int(event["frame_start_t"]),
            context_key,
        )
        if key in feature_seen:
            continue
        feature_seen.add(key)
        feature_records[context_key] += 1
        feature_motion[context_key].extend(
            float(value) for value in event.get("motion_scores", [])
        )
        feature_semantic[context_key].extend(
            float(value)
            for value in event.get("adjacent_semantic_similarity", [])
        )

    decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_seen: set[tuple[Any, ...]] = set()
    for event in _read_jsonl(policy_path):
        if event.get("event") != "middle_selection":
            continue
        for strategy in event.get("strategies", []):
            state = strategy.get("state")
            if not isinstance(state, dict):
                continue
            context_key = state.get("context_key")
            decision = state.get("last_decision")
            if not context_key or not isinstance(decision, dict) or not decision:
                continue
            key = (
                int(event["layer"]),
                str(event.get("branch", "unknown")),
                str(context_key),
                int(decision.get("frame_start_t", -1)),
            )
            if key in decision_seen:
                continue
            decision_seen.add(key)
            decisions[str(context_key)].append(
                {
                    "strategy": str(decision.get("strategy", "")),
                    "accepted": bool(decision.get("accepted", False)),
                    "reason": str(decision.get("reason", "unknown")),
                    "selected_t": _selected_t(decision),
                    "bank_size": len(
                        decision.get(
                            "bank_after",
                            decision.get("pairs_after", []),
                        )
                    ),
                    "motion": decision.get("motion"),
                    "semantic": decision.get(
                        "semantic",
                        decision.get("coherence"),
                    ),
                    "novelty": decision.get("novelty"),
                }
            )

    contexts = {}
    all_contexts = sorted(
        set(feature_records)
        | set(decisions)
    )
    for context_key in all_contexts:
        rows = decisions[context_key]
        accepted = [row for row in rows if row["accepted"]]
        selected = [
            int(row["selected_t"])
            for row in accepted
            if row["selected_t"] is not None
        ]
        modulo = Counter(value % 6 for value in selected)
        contexts[context_key] = {
            "feature_records": int(feature_records[context_key]),
            "motion_features": _stats(feature_motion[context_key]),
            "semantic_features": _stats(feature_semantic[context_key]),
            "decision_records": len(rows),
            "accepted": len(accepted),
            "acceptance_rate": (
                len(accepted) / len(rows) if rows else None
            ),
            "reasons": dict(sorted(Counter(
                row["reason"] for row in rows
            ).items())),
            "bank_size": _stats(
                [float(row["bank_size"]) for row in rows]
            ),
            "accepted_motion": _stats([
                float(row["motion"])
                for row in accepted
                if row["motion"] is not None
            ]),
            "accepted_semantic": _stats([
                float(row["semantic"])
                for row in accepted
                if row["semantic"] is not None
            ]),
            "accepted_novelty": _stats([
                float(row["novelty"])
                for row in accepted
                if row["novelty"] is not None
            ]),
            "selected_frame_modulo_6": {
                str(key): value for key, value in sorted(modulo.items())
            },
            "period6_dominant_fraction": (
                max(modulo.values()) / len(selected)
                if selected
                else None
            ),
        }
    failures = []
    if not feature_seen:
        failures.append("no role_event_features records")
    if not decisions:
        failures.append("no role-event decisions in policy trace")
    return {
        "role_trace": str(role_path),
        "policy_trace": str(policy_path),
        "contexts": contexts,
        "failures": failures,
        "ok": not failures,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Role-event cache trace summary",
        "",
        "| Cell | Context | Decisions | Accept rate | Motion mean | "
        "Semantic mean | Period-6 dominant |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell, report in payload["cells"].items():
        for context_key, row in report["contexts"].items():
            acceptance = row["acceptance_rate"]
            motion = row["accepted_motion"]["mean"]
            semantic = row["accepted_semantic"]["mean"]
            period = row["period6_dominant_fraction"]
            values = [
                "n/a" if acceptance is None else f"{acceptance:.3f}",
                "n/a" if motion is None else f"{motion:.5f}",
                "n/a" if semantic is None else f"{semantic:.5f}",
                "n/a" if period is None else f"{period:.3f}",
            ]
            lines.append(
                f"| {cell} | {context_key} | {row['decision_records']} | "
                + " | ".join(values)
                + " |"
            )
    if payload["failures"]:
        lines.extend(["", "Failures:"])
        lines.extend(f"- {value}" for value in payload["failures"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    trace_root = args.run_root / "traces"
    role_paths = sorted(trace_root.glob("*.role_event.jsonl"))
    cells: dict[str, Any] = {}
    failures: list[str] = []
    for role_path in role_paths:
        cell = role_path.name.removesuffix(".role_event.jsonl")
        policy_path = trace_root / f"{cell}.policy.jsonl"
        if not policy_path.is_file():
            failures.append(f"{cell}: missing policy trace")
            continue
        report = analyze_cell(role_path, policy_path)
        cells[cell] = report
        failures.extend(f"{cell}: {value}" for value in report["failures"])
    if not role_paths:
        failures.append(f"no role-event traces under {trace_root}")
    payload = {
        "version": 1,
        "run_root": str(args.run_root),
        "cells": cells,
        "failures": failures,
        "ok": not failures,
    }
    output_json = (
        args.output_json
        or args.run_root / "diagnostics" / "role_event_summary.json"
    )
    output_md = (
        args.output_md
        or args.run_root / "diagnostics" / "role_event_summary.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"[role-event-summary] cells={len(cells)} "
        f"failures={len(failures)} output={output_json}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
