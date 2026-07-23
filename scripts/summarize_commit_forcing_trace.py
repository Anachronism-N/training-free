#!/usr/bin/env python3
"""Validate and summarize Commit Forcing JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    return True


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "max": max(finite),
    }


def load_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: JSON record is not an object")
                continue
            if not _finite_numbers(record):
                errors.append(f"line {line_number}: non-finite numeric value")
            events.append(record)
    return events, errors


def summarize(path: Path) -> dict[str, Any]:
    events, parse_errors = load_events(path)
    event_counts = Counter(item.get("event", "missing") for item in events)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_video[int(event.get("video_index", -1))].append(event)

    failures = list(parse_errors)
    warnings: list[str] = []
    relative_corrections: list[float] = []
    outcome_disagreement: list[float] = []
    reliabilities: list[float] = []
    selected_distances: list[int] = []
    selected_kind_counts: Counter[str] = Counter()
    renoise_mode_counts: Counter[str] = Counter()
    summary_merge_methods: Counter[str] = Counter()
    nominal_timestep_counts: Counter[int] = Counter()
    rejection_reasons: Counter[str] = Counter()
    selected_summary_support: list[int] = []
    motion_ratios: list[float] = []
    motion_gated_selections = 0
    max_bank_size = 0

    if not events:
        failures.append("trace is empty")
    if event_counts["video_start"] == 0:
        failures.append("no video_start event")
    if event_counts["block_reliability"] == 0:
        failures.append("no block_reliability event")
    if event_counts["commit_accepted"] == 0:
        failures.append("no commit_accepted event")

    for event in events:
        event_type = event.get("event")
        if event_type == "block_reliability":
            reliabilities.extend(event.get("reliability", []))
        elif event_type == "block_motion":
            motion_ratios.extend(event.get("motion_ratio", []))
        elif event_type == "commit_accepted":
            bank_frames = event.get("bank_frames", [])
            max_bank_size = max(max_bank_size, len(bank_frames))
        elif event_type == "summary_merge":
            summary_merge_methods[
                str(event.get("merge_method", "missing"))
            ] += 1
        elif event_type == "reference_selection":
            if bool(event.get("motion_gated", False)):
                motion_gated_selections += 1
        elif event_type == "commit_rejected":
            rejection_reasons[str(event.get("reason", "missing"))] += 1
        elif event_type == "correction":
            relative_corrections.append(
                float(event.get("relative_correction", float("nan")))
            )
            nominal_timestep_counts[
                int(event.get("nominal_timestep", -1))
            ] += 1
            frames = event.get("selected_frames", [])
            kinds = event.get("selected_kinds", [])
            if not frames:
                failures.append(
                    f"video {event.get('video_index')} correction has no reference"
                )
            if len(frames) != len(kinds):
                failures.append(
                    f"video {event.get('video_index')} reference field length mismatch"
                )
            selected_kind_counts.update(str(item) for item in kinds)
            renoise_mode_counts[
                str(event.get("renoise_mode", "fresh"))
            ] += 1
            supports = event.get("selected_support", [])
            selected_summary_support.extend(
                int(support)
                for kind, support in zip(kinds, supports)
                if str(kind) == "summary"
            )
            current_frame = int(event.get("current_frame", -1))
            selected_distances.extend(current_frame - int(item) for item in frames)
        elif event_type == "correction_outcome":
            outcome_disagreement.append(
                float(
                    event.get(
                        "relative_reference_to_native",
                        float("nan"),
                    )
                )
            )

    for video_index, video_events in sorted(by_video.items()):
        if video_index < 0:
            continue
        starts = [item for item in video_events if item.get("event") == "video_start"]
        if not starts:
            failures.append(f"video {video_index}: missing video_start")
            continue
        config = starts[-1].get("config", {})
        trigger_mode = config.get("trigger_mode")
        mode = config.get("reference_mode")
        renoise_mode = config.get("renoise_mode", "fresh")
        capacity = int(config.get("reference_capacity", 0))
        corrections = [
            item for item in video_events if item.get("event") == "correction"
        ]
        if trigger_mode == "always" and not corrections:
            failures.append(
                f"video {video_index}: always trigger produced no correction"
            )
        if trigger_mode == "always" and corrections:
            expected_timesteps = {
                int(item) for item in config.get("timesteps", [])
            }
            observed_timesteps = {
                int(item.get("nominal_timestep", -1))
                for item in corrections
            }
            missing_timesteps = expected_timesteps - observed_timesteps
            if missing_timesteps:
                failures.append(
                    f"video {video_index}: missing nominal correction "
                    f"timesteps {sorted(missing_timesteps)}"
                )
        if renoise_mode == "trajectory":
            fallback_corrections = [
                item
                for item in corrections
                if item.get("renoise_mode") != "trajectory"
                or item.get("renoise_fallback") is not None
            ]
            if fallback_corrections:
                failures.append(
                    f"video {video_index}: trajectory re-noise fell back "
                    f"{len(fallback_corrections)} time(s)"
                )
        for correction in corrections:
            kinds = set(correction.get("selected_kinds", []))
            if mode == "origin" and kinds - {"origin"}:
                failures.append(
                    f"video {video_index}: origin mode selected {sorted(kinds)}"
                )
            if mode == "trusted" and kinds - {"trusted", "recent"}:
                failures.append(
                    f"video {video_index}: trusted mode selected {sorted(kinds)}"
                )
        for accepted in (
            item for item in video_events if item.get("event") == "commit_accepted"
        ):
            if capacity and len(accepted.get("bank_frames", [])) > capacity:
                failures.append(
                    f"video {video_index}: bank exceeded capacity {capacity}"
                )

    correction_stats = _stats(relative_corrections)
    outcome_stats = _stats(outcome_disagreement)
    if event_counts["correction"] == 0:
        warnings.append("no correction events; valid only for an abstaining trigger")
    elif (
        correction_stats["median"] is not None
        and float(correction_stats["median"]) < 1e-4
    ):
        failures.append("median relative correction is below 1e-4")
    if selected_distances and min(selected_distances) <= 0:
        failures.append("a correction selected a non-historical reference")
    if event_counts["correction_outcome"] != event_counts["correction"]:
        failures.append("correction and correction_outcome counts do not match")
    if reliabilities and (min(reliabilities) < 0 or max(reliabilities) > 1):
        failures.append("reliability is outside [0, 1]")

    return {
        "trace": str(path),
        "events": len(events),
        "videos": len([item for item in by_video if item >= 0]),
        "event_counts": dict(sorted(event_counts.items())),
        "correction_relative_rms": correction_stats,
        "reference_to_native_disagreement": outcome_stats,
        "frame_reliability": _stats(reliabilities),
        "motion_ratio": _stats(motion_ratios),
        "reference_distance_frames": _stats(selected_distances),
        "selected_summary_support": _stats(selected_summary_support),
        "selected_kind_counts": dict(sorted(selected_kind_counts.items())),
        "renoise_mode_counts": dict(sorted(renoise_mode_counts.items())),
        "summary_merge_methods": dict(sorted(summary_merge_methods.items())),
        "motion_gated_selections": motion_gated_selections,
        "nominal_timestep_counts": {
            str(key): value for key, value in sorted(nominal_timestep_counts.items())
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "max_bank_size": max_bank_size,
        "warnings": sorted(set(warnings)),
        "failures": sorted(set(failures)),
        "status": "failed" if failures else "nominal",
    }


def write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Commit Forcing Trace Summary",
        "",
        "| Trace | Status | Videos | Corrections | Median delta/input | "
        "Median reliability | Max bank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        correction = item["correction_relative_rms"]["median"]
        reliability = item["frame_reliability"]["median"]
        lines.append(
            f"| {Path(item['trace']).name} | {item['status']} | "
            f"{item['videos']} | {item['event_counts'].get('correction', 0)} | "
            f"{correction if correction is not None else 'n/a'} | "
            f"{reliability if reliability is not None else 'n/a'} | "
            f"{item['max_bank_size']} |"
        )
    for item in summaries:
        if not item["failures"] and not item["warnings"]:
            continue
        lines.extend(["", f"## {Path(item['trace']).name}"])
        lines.extend(f"- FAILURE: {message}" for message in item["failures"])
        lines.extend(f"- WARNING: {message}" for message in item["warnings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.traces]
    payload = {"traces": summaries}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        write_markdown(summaries, args.output_md)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 2 if args.strict and any(item["failures"] for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
