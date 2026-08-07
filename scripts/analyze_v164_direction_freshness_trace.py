#!/usr/bin/env python3
"""Audit v164 direction-only retrieval, freshness scoring, and fallback."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


DIRECTION_MATCH = "ours_middle10_reservoir2_directionmatch1"
DIRECTION_FRESH = "ours_middle10_reservoir2_directionfresh1"
METHODS = (DIRECTION_MATCH, DIRECTION_FRESH)
PROMPT_COUNT = 16
EXPECTED_RECENCY_WEIGHT = {
    DIRECTION_MATCH: 0.0,
    DIRECTION_FRESH: 0.25,
}
TOLERANCE = 2e-5


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


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values, default=None),
        "p05": percentile(values, 0.05),
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
    }


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
        raise ValueError(f"no direction-retrieval rows in {path}")
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


def close(left: object, right: float) -> bool:
    return left is not None and abs(float(left) - float(right)) <= TOLERANCE


def normalized_pair(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return [int(value[0]), int(value[1])]


def expected_selection(
    candidates: list[dict],
    *,
    method: str,
) -> tuple[list[int] | None, bool]:
    passing = [
        candidate
        for candidate in candidates
        if candidate.get("state_pass") is True
        and candidate.get("direction_pass") is True
        and candidate.get("compatibility") is not None
    ]
    if passing:
        if method == DIRECTION_FRESH:
            selected = max(
                passing,
                key=lambda item: (
                    float(item["selection_score"]),
                    float(item["compatibility"]),
                    int(item["pair"][1]),
                ),
            )
        else:
            selected = max(
                passing,
                key=lambda item: (
                    float(item["direction_similarity"]),
                    int(item["pair"][1]),
                ),
            )
        return normalized_pair(selected["pair"]), False
    if candidates:
        newest = max(candidates, key=lambda item: int(item["pair"][1]))
        return normalized_pair(newest["pair"]), True
    return None, False


def analyze_prompt(path: Path, *, method: str) -> dict:
    rows = load_representative(path)
    recency_weight = EXPECTED_RECENCY_WEIGHT[method]
    failures = []
    archive_sizes = []
    selected_ages: list[float] = []
    state_similarities: list[float] = []
    direction_similarities: list[float] = []
    candidate_scores: list[float] = []
    reason_counts: Counter[str] = Counter()
    retrieval_count = 0
    multi_candidate_count = 0
    compatible_selection_count = 0
    direction_rejection_count = 0
    fallback_count = 0
    read_budget_violation_count = 0
    selected_not_newest_age_eligible_count = 0
    freshness_changed_count = 0
    for row in rows:
        item = row["strategy"]
        state = item.get("state", {})
        expected_state = {
            "state_match": True,
            "state_archive_capacity": 4,
            "state_max_read_age": 24,
            "state_min_similarity": -1.0,
            "state_min_direction_similarity": 0.1,
            "state_selection_order": ["direction_similarity", "recency"],
            "state_recency_weight": recency_weight,
            "state_similarity_weight": 0.0,
            "state_fallback_to_newest": True,
        }
        actual_state = {key: state.get(key) for key in expected_state}
        if actual_state != expected_state:
            failures.append(
                f"line {row['line_number']}: state contract {actual_state} "
                f"!= {expected_state}"
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
        retrieval_count += 1
        reason_counts[str(retrieval.get("reason", "missing"))] += 1
        candidates = list(retrieval.get("candidates", []))
        if int(retrieval.get("eligible", -1)) != len(candidates):
            failures.append(
                f"line {row['line_number']}: eligible/candidate mismatch"
            )
        if len(candidates) >= 2:
            multi_candidate_count += 1
        for candidate in candidates:
            pair = normalized_pair(candidate.get("pair"))
            if pair is None or pair[0] + 1 != pair[1]:
                failures.append(
                    f"line {row['line_number']}: malformed candidate pair "
                    f"{candidate.get('pair')}"
                )
                continue
            age = int(candidate.get("age", -1))
            if age < 0 or age > 24:
                failures.append(
                    f"line {row['line_number']}: candidate age {age} invalid"
                )
            state_similarity = candidate.get("state_similarity")
            if state_similarity is not None:
                state_similarities.append(float(state_similarity))
            direction_similarity = candidate.get("direction_similarity")
            if direction_similarity is None:
                if candidate.get("compatibility") is not None:
                    failures.append(
                        f"line {row['line_number']}: directionless candidate "
                        "has a compatibility score"
                    )
                continue
            direction_value = float(direction_similarity)
            direction_similarities.append(direction_value)
            expected_pass = direction_value >= 0.1
            if candidate.get("direction_pass") is not expected_pass:
                failures.append(
                    f"line {row['line_number']}: direction gate mismatch"
                )
            if not expected_pass:
                direction_rejection_count += 1
            expected_score = direction_value - recency_weight * age / 24.0
            if not close(candidate.get("compatibility"), direction_value):
                failures.append(
                    f"line {row['line_number']}: state leaked into "
                    "direction compatibility"
                )
            if not close(candidate.get("selection_score"), expected_score):
                failures.append(
                    f"line {row['line_number']}: score mismatch "
                    f"{candidate.get('selection_score')} != {expected_score}"
                )
            candidate_scores.append(expected_score)
        expected_pair, expected_fallback = expected_selection(
            candidates,
            method=method,
        )
        selected_rows = retrieval.get("selected", [])
        selected_pair = (
            normalized_pair(selected_rows[0])
            if isinstance(selected_rows, list) and len(selected_rows) == 1
            else None
        )
        if expected_pair != selected_pair:
            failures.append(
                f"line {row['line_number']}: selected {selected_pair} != "
                f"recomputed {expected_pair}"
            )
        if bool(retrieval.get("fallback_used")) != expected_fallback:
            failures.append(
                f"line {row['line_number']}: fallback flag mismatch"
            )
        if expected_fallback:
            fallback_count += 1
        else:
            compatible_selection_count += int(selected_pair is not None)
        if candidates and (
            selected_pair is None
            or retrieval.get("read_budget_preserved") is not True
        ):
            read_budget_violation_count += 1
            failures.append(
                f"line {row['line_number']}: age-eligible pair budget lost"
            )
        if selected_pair is None:
            if read_ids:
                failures.append(
                    f"line {row['line_number']}: empty selection read {read_ids}"
                )
            continue
        if selected_pair != read_ids:
            failures.append(
                f"line {row['line_number']}: selected/read mismatch "
                f"{selected_pair} != {read_ids}"
            )
        selected_age = int(row["sync_t"]) - selected_pair[1]
        selected_ages.append(float(selected_age))
        if selected_age < 0 or selected_age > 24:
            failures.append(
                f"line {row['line_number']}: selected age {selected_age} invalid"
            )
        if candidates and selected_pair[1] != max(
            int(candidate["pair"][1]) for candidate in candidates
        ):
            selected_not_newest_age_eligible_count += 1
        if retrieval.get("selection_changed_from_legacy") is True:
            freshness_changed_count += 1
    return {
        "prompt_index": prompt_index(path),
        "trace": str(path),
        "representative_layer": rows[0]["layer"],
        "representative_head": rows[0]["head"],
        "read_count": len(rows),
        "retrieval_count": retrieval_count,
        "archive_size_max": max(archive_sizes, default=0),
        "multi_candidate_count": multi_candidate_count,
        "compatible_selection_count": compatible_selection_count,
        "direction_rejection_count": direction_rejection_count,
        "fallback_count": fallback_count,
        "read_budget_violation_count": read_budget_violation_count,
        "selected_not_newest_age_eligible_count": (
            selected_not_newest_age_eligible_count
        ),
        "freshness_changed_count": freshness_changed_count,
        "selected_ages": selected_ages,
        "state_similarities": state_similarities,
        "direction_similarities": direction_similarities,
        "candidate_scores": candidate_scores,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def aggregate_method(prompts: list[dict], *, method: str) -> dict:
    selected_ages = [
        value for prompt in prompts for value in prompt["selected_ages"]
    ]
    state_similarities = [
        value for prompt in prompts for value in prompt["state_similarities"]
    ]
    direction_similarities = [
        value
        for prompt in prompts
        for value in prompt["direction_similarities"]
    ]
    candidate_scores = [
        value for prompt in prompts for value in prompt["candidate_scores"]
    ]
    reasons: Counter[str] = Counter()
    for prompt in prompts:
        reasons.update(prompt["reason_counts"])
    aggregate = {
        "method": method,
        "prompt_count": len(prompts),
        "read_count": sum(prompt["read_count"] for prompt in prompts),
        "retrieval_count": sum(
            prompt["retrieval_count"] for prompt in prompts
        ),
        "archive_size_max": max(
            prompt["archive_size_max"] for prompt in prompts
        ),
        "multi_candidate_count": sum(
            prompt["multi_candidate_count"] for prompt in prompts
        ),
        "compatible_selection_count": sum(
            prompt["compatible_selection_count"] for prompt in prompts
        ),
        "direction_rejection_count": sum(
            prompt["direction_rejection_count"] for prompt in prompts
        ),
        "fallback_count": sum(
            prompt["fallback_count"] for prompt in prompts
        ),
        "read_budget_violation_count": sum(
            prompt["read_budget_violation_count"] for prompt in prompts
        ),
        "selected_not_newest_age_eligible_count": sum(
            prompt["selected_not_newest_age_eligible_count"]
            for prompt in prompts
        ),
        "freshness_changed_count": sum(
            prompt["freshness_changed_count"] for prompt in prompts
        ),
        "selected_age": distribution(selected_ages),
        "state_similarity": distribution(state_similarities),
        "direction_similarity": distribution(direction_similarities),
        "candidate_score": distribution(candidate_scores),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": sum(
            len(prompt["failures"]) for prompt in prompts
        ),
    }
    aggregate["mechanism_gate"] = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["compatible_selection_count"] > 0
        and aggregate["read_budget_violation_count"] == 0
        and aggregate["contract_failure_count"] == 0
        and (
            method != DIRECTION_FRESH
            or aggregate["freshness_changed_count"] > 0
        )
    )
    return aggregate


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v164 Direction/Freshness Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Compatible reads | Direction rejects | Fallbacks | Freshness changes | Age p95 | Budget violations | Contract failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(row["mechanism_gate"]),
                    str(row["compatible_selection_count"]),
                    str(row["direction_rejection_count"]),
                    str(row["fallback_count"]),
                    str(row["freshness_changed_count"]),
                    str(row["selected_age"]["p95"]),
                    str(row["read_budget_violation_count"]),
                    str(row["contract_failure_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The audit recomputes every candidate score and selected pair. ",
            "It establishes mechanism execution only, not video-quality improvement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_reports = {}
    for method in METHODS:
        paths = sorted(args.trace_dir.glob(f"{method}__p*.policy.jsonl"))
        if len(paths) != PROMPT_COUNT:
            raise ValueError(
                f"expected {PROMPT_COUNT} traces for {method}, found {len(paths)}"
            )
        prompts = [analyze_prompt(path, method=method) for path in paths]
        if [row["prompt_index"] for row in prompts] != list(range(PROMPT_COUNT)):
            raise ValueError(f"prompt coverage mismatch for {method}")
        method_reports[method] = {
            "aggregate": aggregate_method(prompts, method=method),
            "prompts": prompts,
        }
    mechanism_gate = all(
        method_reports[method]["aggregate"]["mechanism_gate"]
        for method in METHODS
    )
    report = {
        "version": 1,
        "experiment": "v164_direction_freshness_trace",
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "both methods exercise multi-candidate direction retrieval with "
            "atomic equal-budget reads; DirectionFresh must change at least "
            "one direction-only choice"
        ),
        "methods": method_reports,
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
    failure_count = sum(
        method_reports[method]["aggregate"]["contract_failure_count"]
        for method in METHODS
    )
    if failure_count:
        raise SystemExit(
            f"v164 trace contract failed with {failure_count} violations"
        )
    print(
        json.dumps(
            {
                method: method_reports[method]["aggregate"]
                for method in METHODS
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
