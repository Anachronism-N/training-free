#!/usr/bin/env python3
"""Replay stale direction-tie margins on frozen v164 candidate traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v164_direction_freshness_trace as common
import analyze_v165_direction_stale_tie_trace as v165


SOURCE_METHOD = common.DIRECTION_MATCH
PROMPT_COUNT = 16
MARGINS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
STALE_TIE_AGE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def compatible_events(paths: list[Path]) -> list[dict]:
    events = []
    for path in paths:
        for row in common.load_representative(path):
            retrieval = row["strategy"].get("state", {}).get(
                "last_retrieval",
                {},
            )
            candidates = list(retrieval.get("candidates", []))
            passing = [
                candidate
                for candidate in candidates
                if candidate.get("state_pass") is True
                and candidate.get("direction_pass") is True
                and candidate.get("direction_similarity") is not None
                and candidate.get("compatibility") is not None
            ]
            if not passing:
                continue
            baseline = max(
                passing,
                key=lambda item: (
                    float(item["direction_similarity"]),
                    int(item["pair"][1]),
                ),
            )
            events.append(
                {
                    "prompt_index": common.prompt_index(path),
                    "current_t": int(row["sync_t"]),
                    "candidates": passing,
                    "baseline": baseline,
                }
            )
    return events


def replay(events: list[dict], *, margin: float) -> dict:
    selected_ages = []
    direction_losses = []
    age_gains = []
    changed_prompts = set()
    tie_applied_count = 0
    changed_count = 0
    for event in events:
        selected = v165.expected_selection(
            event["candidates"],
            current_t=event["current_t"],
            margin=margin,
        )
        pair = selected["selected"]
        if pair is None:
            raise ValueError("compatible replay event produced no selection")
        selected_ages.append(float(event["current_t"] - int(pair[1])))
        tie_applied_count += int(selected["tie_applied"])
        if selected["changed"]:
            changed_count += 1
            changed_prompts.add(int(event["prompt_index"]))
            direction_losses.append(float(selected["direction_loss"]))
            age_gains.append(float(selected["age_gain"]))
    return {
        "margin": margin,
        "stale_tie_age": STALE_TIE_AGE,
        "event_count": len(events),
        "tie_applied_count": tie_applied_count,
        "changed_count": changed_count,
        "changed_prompt_count": len(changed_prompts),
        "changed_prompts": sorted(changed_prompts),
        "selected_age": common.distribution(selected_ages),
        "direction_loss_on_change": common.distribution(direction_losses),
        "age_gain_on_change": common.distribution(age_gains),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v165 Frozen-Trace Margin Replay",
        "",
        f"Compatible v164 retrievals: **{report['compatible_event_count']}**",
        "",
        "| Margin | Changed | Changed prompts | Selected age mean | "
        "Selected age p95 | Direction loss mean | Direction loss p95 | "
        "Age gain mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["margins"]:
        lines.append(
            f"| {row['margin']:.3f} | {row['changed_count']} | "
            f"{row['changed_prompt_count']} | "
            f"{row['selected_age']['mean']:.4f} | "
            f"{row['selected_age']['p95']:.4f} | "
            f"{row['direction_loss_on_change']['mean']:.4f} | "
            f"{row['direction_loss_on_change']['p95']:.4f} | "
            f"{row['age_gain_on_change']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This replay uses only frozen v164 candidate traces. It does not",
            "use v165 videos or quality metrics and is not an evaluation result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(args.trace_dir.glob(f"{SOURCE_METHOD}__p*.policy.jsonl"))
    if len(paths) != PROMPT_COUNT:
        raise ValueError(
            f"expected {PROMPT_COUNT} frozen traces, found {len(paths)}"
        )
    if [common.prompt_index(path) for path in paths] != list(range(PROMPT_COUNT)):
        raise ValueError("v164 replay prompt coverage mismatch")
    events = compatible_events(paths)
    if not events:
        raise ValueError("no compatible v164 direction events")
    margin_rows = [replay(events, margin=margin) for margin in MARGINS]
    report = {
        "version": 1,
        "experiment": "v165_margin_replay",
        "source_method": SOURCE_METHOD,
        "source_trace_count": len(paths),
        "compatible_event_count": len(events),
        "baseline_selected_age": common.distribution(
            [
                float(
                    event["current_t"] - int(event["baseline"]["pair"][1])
                )
                for event in events
            ]
        ),
        "margins": margin_rows,
        "selected_margins": [0.03, 0.05],
        "selection_reason": (
            "0.03 changes a conservative subset with small direction loss; "
            "0.05 tests a moderate operating point without approaching the "
            "95 changes produced by v164's global freshness penalty"
        ),
        "claim_boundary": (
            "offline mechanism calibration only; no generated-video metric used"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(
        json.dumps(
            {
                "events": len(events),
                "selected": [
                    row
                    for row in margin_rows
                    if row["margin"] in report["selected_margins"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
