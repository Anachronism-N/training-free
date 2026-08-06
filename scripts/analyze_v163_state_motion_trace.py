#!/usr/bin/env python3
"""Audit whether v163 state-conditioned motion retrieval changed cache reads."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


PRIMARY = "ours_middle10_reservoir2_statemotion1_strict"
PROMPT_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def prompt_index(path: Path) -> int:
    marker = "__p"
    if marker not in path.stem:
        raise ValueError(f"cannot parse prompt index from {path.name}")
    return int(path.stem.split(marker, 1)[1].split(".", 1)[0])


def load_representative(path: Path) -> list[dict]:
    candidates = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            if (
                row.get("event") != "middle_selection"
                or row.get("branch") != "cond"
                or int(row.get("label", -1)) != 10
            ):
                continue
            strategy = next(
                (
                    item
                    for item in row.get("strategies", [])
                    if item.get("name") == "CoherentMotionStrategy"
                ),
                None,
            )
            if strategy is None:
                continue
            candidates.append(
                {
                    "line_number": line_number,
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "sync_t": int(row["sync_t"]),
                    "strategy": strategy,
                }
            )
    if not candidates:
        raise ValueError(f"no state-motion trace rows in {path}")
    representative = min((row["layer"], row["head"]) for row in candidates)
    rows = [
        row
        for row in candidates
        if (row["layer"], row["head"]) == representative
    ]
    unique = {}
    for row in rows:
        key = int(row["sync_t"])
        if key in unique:
            raise ValueError(f"duplicate representative read t={key} in {path}")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def analyze_prompt(path: Path) -> dict:
    rows = load_representative(path)
    failures = []
    archive_sizes = []
    selected_ages = []
    selected_count = 0
    abstain_count = 0
    multi_candidate_count = 0
    selected_not_newest_count = 0
    negative_direction_rejections = 0
    direction_available_count = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        item = row["strategy"]
        state = item.get("state", {})
        if (
            state.get("state_match") is not True
            or int(state.get("state_archive_capacity", -1)) != 4
            or int(state.get("state_max_read_age", -1)) != 24
        ):
            failures.append(
                f"line {row['line_number']}: frozen state-match config mismatch"
            )
        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        read_ids = [int(value) for value in item.get("frame_ids", [])]
        if len(read_ids) not in {0, 2} or (
            len(read_ids) == 2 and read_ids[0] + 1 != read_ids[1]
        ):
            failures.append(
                f"line {row['line_number']}: non-atomic read {read_ids}"
            )
        retrieval = state.get("last_retrieval", {})
        if not retrieval:
            continue
        reason = str(retrieval.get("reason", "missing"))
        reasons[reason] += 1
        if retrieval.get("direction_available"):
            direction_available_count += 1
        candidates = retrieval.get("candidates", [])
        if len(candidates) >= 2:
            multi_candidate_count += 1
        negative_direction_rejections += sum(
            item.get("direction_similarity") is not None
            and float(item["direction_similarity"]) < 0.0
            and item.get("direction_pass") is False
            for item in candidates
        )
        selected = retrieval.get("selected", [])
        if selected:
            selected_count += 1
            if len(selected) != 1 or len(selected[0]) != 2:
                failures.append(
                    f"line {row['line_number']}: malformed selection {selected}"
                )
                continue
            selected_pair = [int(value) for value in selected[0]]
            if selected_pair != read_ids:
                failures.append(
                    f"line {row['line_number']}: selected/read mismatch "
                    f"{selected_pair} != {read_ids}"
                )
            age = int(row["sync_t"]) - selected_pair[1]
            selected_ages.append(float(age))
            if age > 24 or age < 0:
                failures.append(
                    f"line {row['line_number']}: selected age {age} outside [0,24]"
                )
            eligible_pairs = [
                [int(value) for value in candidate["pair"]]
                for candidate in candidates
            ]
            if eligible_pairs and selected_pair[1] != max(
                pair[1] for pair in eligible_pairs
            ):
                selected_not_newest_count += 1
        else:
            abstain_count += 1
            if read_ids:
                failures.append(
                    f"line {row['line_number']}: abstain emitted frames {read_ids}"
                )
    return {
        "prompt_index": prompt_index(path),
        "trace": str(path),
        "representative_layer": rows[0]["layer"],
        "representative_head": rows[0]["head"],
        "read_count": len(rows),
        "archive_size_max": max(archive_sizes, default=0),
        "selected_count": selected_count,
        "abstain_count": abstain_count,
        "multi_candidate_count": multi_candidate_count,
        "selected_not_newest_count": selected_not_newest_count,
        "negative_direction_rejections": negative_direction_rejections,
        "direction_available_count": direction_available_count,
        "selected_age_median": (
            statistics.median(selected_ages) if selected_ages else None
        ),
        "selected_age_p95": percentile(selected_ages, 0.95),
        "selected_age_max": max(selected_ages, default=None),
        "reason_counts": dict(sorted(reasons.items())),
        "failures": failures,
    }


def write_markdown(path: Path, report: dict) -> None:
    aggregate = report["aggregate"]
    lines = [
        "# v163 State-Matched Motion Trace Audit",
        "",
        f"Mechanism gate: **{report['mechanism_gate']}**",
        "",
        f"- Prompts: {aggregate['prompt_count']}",
        f"- Multi-candidate reads: {aggregate['multi_candidate_count']}",
        f"- Selected non-newest compatible pair: {aggregate['selected_not_newest_count']}",
        f"- State/direction abstentions: {aggregate['state_direction_abstain_count']}",
        f"- Negative-direction candidate rejections: {aggregate['negative_direction_rejections']}",
        f"- Atomic-pair violations: {aggregate['atomic_or_contract_failure_count']}",
        f"- Selected age p95: {aggregate['selected_age_p95']}",
        "",
        "This gate verifies execution and choice diversity only. It does not establish video-quality improvement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(args.trace_dir.glob(f"{PRIMARY}__p*.policy.jsonl"))
    if len(paths) != PROMPT_COUNT:
        raise ValueError(
            f"expected {PROMPT_COUNT} primary traces, found {len(paths)}"
        )
    prompts = [analyze_prompt(path) for path in paths]
    if [row["prompt_index"] for row in prompts] != list(range(PROMPT_COUNT)):
        raise ValueError("v163 trace prompt coverage mismatch")
    all_selected_ages = [
        float(row["selected_age_p95"])
        for row in prompts
        if row["selected_age_p95"] is not None
    ]
    reason_counts: Counter[str] = Counter()
    for row in prompts:
        reason_counts.update(row["reason_counts"])
    contract_failures = sum(len(row["failures"]) for row in prompts)
    state_direction_abstain = sum(
        int(row["reason_counts"].get("state_or_direction_gate", 0))
        for row in prompts
    )
    aggregate = {
        "prompt_count": len(prompts),
        "read_count": sum(row["read_count"] for row in prompts),
        "archive_size_max": max(row["archive_size_max"] for row in prompts),
        "selected_count": sum(row["selected_count"] for row in prompts),
        "abstain_count": sum(row["abstain_count"] for row in prompts),
        "multi_candidate_count": sum(
            row["multi_candidate_count"] for row in prompts
        ),
        "selected_not_newest_count": sum(
            row["selected_not_newest_count"] for row in prompts
        ),
        "state_direction_abstain_count": state_direction_abstain,
        "negative_direction_rejections": sum(
            row["negative_direction_rejections"] for row in prompts
        ),
        "direction_available_count": sum(
            row["direction_available_count"] for row in prompts
        ),
        "selected_age_p95": percentile(all_selected_ages, 0.95),
        "selected_age_max": max(
            (
                row["selected_age_max"]
                for row in prompts
                if row["selected_age_max"] is not None
            ),
            default=None,
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "atomic_or_contract_failure_count": contract_failures,
    }
    mechanism_gate = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and (
            aggregate["selected_not_newest_count"] > 0
            or aggregate["state_direction_abstain_count"] > 0
        )
        and aggregate["atomic_or_contract_failure_count"] == 0
    )
    report = {
        "version": 1,
        "experiment": "v163_state_matched_motion_trace",
        "primary": PRIMARY,
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "multiple eligible choices exist; state matching changes at least "
            "one choice or abstains; all reads remain atomic and within age"
        ),
        "aggregate": aggregate,
        "prompts": prompts,
        "claim_boundary": (
            "mechanism execution is separate from video-quality promotion"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    if contract_failures:
        raise SystemExit(
            f"v163 trace contract failed with {contract_failures} violations"
        )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
