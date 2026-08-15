#!/usr/bin/env python3
"""Summarize which historical frames each v182 Coverage operator reads."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SELECTION_RULES = {
    "reservoir": "deterministic-seeded uniform reservoir over eligible history",
    "landmark": "online semantic coherence and novelty landmark coreset",
    "prototype": "motion-aware compression into temporal segment medoids",
    "retrieval": "current-query relevance and diversity retrieval from a bounded archive",
}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def summarize_method(run_root: Path, method: str, config: dict) -> dict:
    paths = sorted((run_root / "traces" / method).glob("*.policy.jsonl"))
    ages: list[float] = []
    middle_counts: list[int] = []
    selected_frames = set()
    decisions = Counter()
    retrieval_reasons = Counter()
    per_head = defaultdict(lambda: {"records": 0, "ages": [], "frames": set()})
    records = 0
    for path in paths:
        for row in iter_jsonl(path):
            if int(row.get("label", -1)) != 21:
                continue
            records += 1
            sync_t = int(row.get("sync_t", 0))
            frame_ids = [int(value) for value in row.get("union_frame_ids") or []]
            middle_counts.append(len(frame_ids))
            key = f"L{int(row['layer'])}H{int(row['head'])}"
            per_head[key]["records"] += 1
            for frame_id in frame_ids:
                age = max(0, sync_t - frame_id)
                ages.append(float(age))
                selected_frames.add(frame_id)
                per_head[key]["ages"].append(float(age))
                per_head[key]["frames"].add(frame_id)
            for strategy in row.get("strategies") or []:
                state = strategy.get("state") or {}
                decision = state.get("last_decision") or {}
                retrieval = state.get("last_retrieval") or {}
                if decision.get("reason"):
                    decisions[str(decision["reason"])] += 1
                if retrieval.get("reason"):
                    retrieval_reasons[str(retrieval["reason"])] += 1

    def age_summary(values: list[float]) -> dict:
        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)) if values else None,
            "p25": percentile(values, 0.25),
            "median": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "max": max(values) if values else None,
        }

    head_rows = {}
    for head, row in sorted(per_head.items()):
        head_rows[head] = {
            "records": row["records"],
            "unique_frame_ids": len(row["frames"]),
            "selected_age": age_summary(row["ages"]),
        }
    policy = config["coverage_policy"]
    return {
        "policy": policy,
        "selection_rule": SELECTION_RULES[policy],
        "uses_random_selection": policy == "reservoir",
        "trace_files": len(paths),
        "selected_records": records,
        "middle_frame_count_mean": (
            float(statistics.fmean(middle_counts)) if middle_counts else 0.0
        ),
        "middle_frame_count_max": max(middle_counts, default=0),
        "unique_selected_frame_ids": len(selected_frames),
        "selected_age": age_summary(ages),
        "decision_reasons": dict(decisions),
        "retrieval_reasons": dict(retrieval_reasons),
        "per_head": head_rows,
        "middle_read_capacity": config["middle_read_capacity"],
        "middle_storage_capacity": config["middle_storage_capacity"],
    }


def render(report: dict) -> str:
    lines = [
        "# v182 Structured-Coverage Trace Summary",
        "",
        "| Method | Policy | Random | Read/store | Mean middle | Median age | Max age |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["methods"].items():
        age = row["selected_age"]
        lines.append(
            f"| {method} | {row['policy']} | {row['uses_random_selection']} | "
            f"{row['middle_read_capacity']}/{row['middle_storage_capacity']} | "
            f"{row['middle_frame_count_mean']:.3f} | "
            f"{age['median'] if age['median'] is not None else 'n/a'} | "
            f"{age['max'] if age['max'] is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "This report is a mechanism diagnostic. It does not rank video quality.",
            "Retrieval reads four frames but keeps a twelve-frame archive; report its storage overhead separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v182_structured_coverage_screen":
        raise ValueError("trace analyzer received the wrong manifest")
    report = {
        "version": 1,
        "experiment": "v182_structured_coverage_trace_analysis",
        "development_only": True,
        "methods": {
            method: summarize_method(args.run_root, method, manifest["methods"][method])
            for method in manifest["method_order"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v182-traces] methods={len(report['methods'])} output={args.output}")


if __name__ == "__main__":
    main()
