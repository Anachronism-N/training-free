#!/usr/bin/env python3
"""Audit v163 state-motion freshness controls from policy traces."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


PROMPT_COUNT = 16
LEGACY_SELECTED_AGE_P95 = 22.0
METHOD_SPECS = {
    "ours_middle10_reservoir2_stateage12motionpair1": {
        "read_max_age": 12,
        "recency_weight": 0.0,
        "selection_mode": "legacy_lexicographic",
    },
    "ours_middle10_reservoir2_statebalancedmotionpair1": {
        "read_max_age": 24,
        "recency_weight": 0.25,
        "selection_mode": "recency_regularized",
    },
}


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
    selected = [
        row
        for row in candidates
        if (row["layer"], row["head"]) == representative
    ]
    unique = {}
    for row in selected:
        sync_t = int(row["sync_t"])
        if sync_t in unique:
            raise ValueError(f"duplicate representative read t={sync_t} in {path}")
        unique[sync_t] = row
    return [unique[key] for key in sorted(unique)]


def _pair(value: object) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 1:
        return None
    pair = value[0]
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    return [int(pair[0]), int(pair[1])]


def analyze_prompt(path: Path, spec: dict) -> dict:
    rows = load_representative(path)
    failures = []
    archive_sizes = []
    selected_ages: list[float] = []
    compatibility_gains: list[float] = []
    age_gaps: list[float] = []
    selected_count = 0
    abstain_count = 0
    multi_candidate_count = 0
    changed_from_legacy_count = 0
    selected_newest_count = 0
    negative_direction_rejections = 0
    reasons: Counter[str] = Counter()
    required_retrieval = {
        "query_t",
        "eligible_before_age",
        "eligible",
        "state_max_read_age",
        "state_recency_weight",
        "selection_mode",
        "direction_available",
        "candidates",
        "selected",
        "legacy_selected",
        "newest_passing",
        "selection_changed_from_legacy",
        "selected_age",
        "selected_is_newest_passing",
        "selected_compatibility",
        "selected_score",
        "selected_vs_newest_compatibility_gain",
        "selected_vs_newest_age_gap",
        "reason",
    }
    for row in rows:
        line = row["line_number"]
        item = row["strategy"]
        state = item.get("state", {})
        if (
            state.get("state_match") is not True
            or int(state.get("state_archive_capacity", -1)) != 4
            or int(state.get("state_max_read_age", -1))
            != int(spec["read_max_age"])
            or not math.isclose(
                float(state.get("state_recency_weight", -1.0)),
                float(spec["recency_weight"]),
                abs_tol=1e-12,
            )
        ):
            failures.append(f"line {line}: frozen state-match config mismatch")
        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        if len(stored) > 4:
            failures.append(f"line {line}: motion archive exceeds four pairs")
        read_ids = [int(value) for value in item.get("frame_ids", [])]
        if len(read_ids) not in {0, 2} or (
            len(read_ids) == 2 and read_ids[0] + 1 != read_ids[1]
        ):
            failures.append(f"line {line}: non-atomic read {read_ids}")
        retrieval = state.get("last_retrieval", {})
        if not retrieval:
            continue
        missing = sorted(required_retrieval - set(retrieval))
        if missing:
            failures.append(f"line {line}: retrieval fields missing {missing}")
            continue
        if (
            int(retrieval["query_t"]) != int(row["sync_t"])
            or int(retrieval["state_max_read_age"]) != int(spec["read_max_age"])
            or not math.isclose(
                float(retrieval["state_recency_weight"]),
                float(spec["recency_weight"]),
                abs_tol=1e-12,
            )
            or retrieval["selection_mode"] != spec["selection_mode"]
        ):
            failures.append(f"line {line}: retrieval config/debug mismatch")
        reason = str(retrieval.get("reason", "missing"))
        reasons[reason] += 1
        candidates = retrieval.get("candidates", [])
        passing = [
            candidate
            for candidate in candidates
            if candidate.get("state_pass") is True
            and candidate.get("direction_pass") is True
        ]
        if len(candidates) >= 2:
            multi_candidate_count += 1
        if any(
            int(candidate.get("age", -1)) < 0
            or int(candidate.get("age", -1)) > int(spec["read_max_age"])
            for candidate in candidates
        ):
            failures.append(f"line {line}: candidate escaped read-age gate")
        negative_direction_rejections += sum(
            candidate.get("direction_similarity") is not None
            and float(candidate["direction_similarity"]) < 0.0
            and candidate.get("direction_pass") is False
            for candidate in candidates
        )
        selected = _pair(retrieval.get("selected"))
        legacy = _pair(retrieval.get("legacy_selected"))
        newest = _pair(retrieval.get("newest_passing"))
        if retrieval.get("selected") and selected is None:
            failures.append(f"line {line}: malformed selected pair")
            continue
        if selected is None:
            abstain_count += 1
            if read_ids:
                failures.append(f"line {line}: abstain emitted frames {read_ids}")
            if passing:
                failures.append(f"line {line}: abstained despite passing candidate")
            continue
        selected_count += 1
        if selected != read_ids:
            failures.append(
                f"line {line}: selected/read mismatch {selected} != {read_ids}"
            )
        age = int(row["sync_t"]) - selected[1]
        selected_ages.append(float(age))
        if age != int(retrieval["selected_age"]):
            failures.append(f"line {line}: selected-age debug mismatch")
        if age < 0 or age > int(spec["read_max_age"]):
            failures.append(f"line {line}: selected age {age} outside contract")
        selected_candidate = next(
            (
                candidate
                for candidate in passing
                if [int(value) for value in candidate["pair"]] == selected
            ),
            None,
        )
        if selected_candidate is None:
            failures.append(f"line {line}: selected pair is not a passing candidate")
        elif not math.isclose(
            float(retrieval["selected_score"]),
            float(selected_candidate["selection_score"]),
            abs_tol=2e-6,
        ):
            failures.append(f"line {line}: selected score does not match candidate")
        changed = legacy is not None and selected != legacy
        if changed:
            changed_from_legacy_count += 1
        if bool(retrieval["selection_changed_from_legacy"]) != changed:
            failures.append(f"line {line}: legacy-change debug mismatch")
        is_newest = newest is not None and selected == newest
        if is_newest:
            selected_newest_count += 1
        if bool(retrieval["selected_is_newest_passing"]) != is_newest:
            failures.append(f"line {line}: newest-selection debug mismatch")
        if spec["selection_mode"] == "legacy_lexicographic" and changed:
            failures.append(f"line {line}: zero-weight selection changed legacy")
        if spec["selection_mode"] == "recency_regularized" and passing:
            best_score = max(
                float(candidate["selection_score"]) for candidate in passing
            )
            if (
                selected_candidate is not None
                and float(selected_candidate["selection_score"])
                < best_score - 2e-6
            ):
                failures.append(f"line {line}: regularized argmax mismatch")
        gain = retrieval.get("selected_vs_newest_compatibility_gain")
        gap = retrieval.get("selected_vs_newest_age_gap")
        if gain is not None:
            compatibility_gains.append(float(gain))
        if gap is not None:
            age_gaps.append(float(gap))
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
        "changed_from_legacy_count": changed_from_legacy_count,
        "selected_newest_count": selected_newest_count,
        "negative_direction_rejections": negative_direction_rejections,
        "selected_ages": selected_ages,
        "selected_age_median": (
            statistics.median(selected_ages) if selected_ages else None
        ),
        "selected_age_p95": percentile(selected_ages, 0.95),
        "selected_age_max": max(selected_ages, default=None),
        "compatibility_gain_median": (
            statistics.median(compatibility_gains)
            if compatibility_gains
            else None
        ),
        "selected_vs_newest_age_gap_median": (
            statistics.median(age_gaps) if age_gaps else None
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "failures": failures,
    }


def analyze_method(trace_dir: Path, method: str, spec: dict) -> dict:
    paths = sorted(trace_dir.glob(f"{method}__p*.policy.jsonl"))
    if len(paths) != PROMPT_COUNT:
        raise ValueError(f"expected {PROMPT_COUNT} {method} traces, found {len(paths)}")
    prompts = [analyze_prompt(path, spec) for path in paths]
    if [row["prompt_index"] for row in prompts] != list(range(PROMPT_COUNT)):
        raise ValueError(f"{method} prompt coverage mismatch")
    ages = [age for row in prompts for age in row["selected_ages"]]
    reasons: Counter[str] = Counter()
    for row in prompts:
        reasons.update(row["reason_counts"])
    failures = [
        f"prompt {row['prompt_index']}: {failure}"
        for row in prompts
        for failure in row["failures"]
    ]
    aggregate = {
        "prompt_count": len(prompts),
        "read_count": sum(row["read_count"] for row in prompts),
        "archive_size_max": max(row["archive_size_max"] for row in prompts),
        "selected_count": sum(row["selected_count"] for row in prompts),
        "abstain_count": sum(row["abstain_count"] for row in prompts),
        "multi_candidate_count": sum(
            row["multi_candidate_count"] for row in prompts
        ),
        "changed_from_legacy_count": sum(
            row["changed_from_legacy_count"] for row in prompts
        ),
        "selected_newest_count": sum(
            row["selected_newest_count"] for row in prompts
        ),
        "negative_direction_rejections": sum(
            row["negative_direction_rejections"] for row in prompts
        ),
        "selected_age_median": statistics.median(ages) if ages else None,
        "selected_age_p95": percentile(ages, 0.95),
        "selected_age_max": max(ages, default=None),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": len(failures),
    }
    mechanism_gate = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["contract_failure_count"] == 0
        and (
            spec["selection_mode"] == "legacy_lexicographic"
            or aggregate["changed_from_legacy_count"] > 0
        )
    )
    freshness_gate = bool(
        aggregate["selected_age_p95"] is not None
        and aggregate["selected_age_p95"] < LEGACY_SELECTED_AGE_P95
        and aggregate["selected_age_max"] <= int(spec["read_max_age"])
    )
    return {
        "spec": spec,
        "mechanism_gate": mechanism_gate,
        "freshness_gate": freshness_gate,
        "aggregate": aggregate,
        "failures": failures,
        "prompts": prompts,
    }


def markdown(report: dict) -> str:
    lines = [
        "# v163 Recency Trace Audit",
        "",
        f"Contract gate: **{report['contract_gate']}**",
        f"At least one freshness mechanism passes: **{report['freshness_gate']}**",
        "",
        "| Method | Mechanism | Freshness | Changed from v161 | Age median | Age p95 | Age max | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["methods"].items():
        aggregate = row["aggregate"]
        lines.append(
            f"| {method} | {row['mechanism_gate']} | {row['freshness_gate']} | "
            f"{aggregate['changed_from_legacy_count']} | "
            f"{aggregate['selected_age_median']} | "
            f"{aggregate['selected_age_p95']} | "
            f"{aggregate['selected_age_max']} | "
            f"{aggregate['contract_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "This report proves execution and freshness behavior only. Video-quality promotion is handled by the automatic metric gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    methods = {
        method: analyze_method(args.trace_dir, method, spec)
        for method, spec in METHOD_SPECS.items()
    }
    contract_gate = all(row["mechanism_gate"] for row in methods.values())
    freshness_gate = any(row["freshness_gate"] for row in methods.values())
    report = {
        "version": 1,
        "experiment": "v163_recency_regularized_state_motion_trace",
        "legacy_selected_age_p95": LEGACY_SELECTED_AGE_P95,
        "contract_gate": contract_gate,
        "freshness_gate": freshness_gate,
        "methods": methods,
        "claim_boundary": (
            "trace gates establish execution and retrieval-age control, not video quality"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        markdown(report),
        encoding="utf-8",
    )
    if not contract_gate:
        raise SystemExit("v163 trace mechanism/contract gate failed")
    print(
        json.dumps(
            {
                method: row["aggregate"]
                for method, row in methods.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
