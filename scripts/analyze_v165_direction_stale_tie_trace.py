#!/usr/bin/env python3
"""Audit v165 stale-aware direction-margin retrieval from saved traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common


DIRECTION_TIE_003 = "ours_middle10_reservoir2_dirstaletie003"
DIRECTION_TIE_005 = "ours_middle10_reservoir2_dirstaletie005"
METHODS = (DIRECTION_TIE_003, DIRECTION_TIE_005)
PROMPT_COUNT = 16
EXPECTED_MARGIN = {
    DIRECTION_TIE_003: 0.03,
    DIRECTION_TIE_005: 0.05,
}
STALE_TIE_AGE = 12
MAX_READ_AGE = 24
TOLERANCE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def close(left: object, right: float) -> bool:
    return left is not None and abs(float(left) - float(right)) <= TOLERANCE


def expected_selection(
    candidates: list[dict],
    *,
    current_t: int,
    margin: float,
) -> dict:
    passing = [
        candidate
        for candidate in candidates
        if candidate.get("state_pass") is True
        and candidate.get("direction_pass") is True
        and candidate.get("direction_similarity") is not None
        and candidate.get("compatibility") is not None
    ]
    if not passing:
        newest = (
            max(candidates, key=lambda item: int(item["pair"][1]))
            if candidates
            else None
        )
        return {
            "selected": (
                None
                if newest is None
                else common.normalized_pair(newest["pair"])
            ),
            "direction_best": None,
            "direction_best_age": None,
            "tie_candidates": [],
            "tie_applied": False,
            "changed": False,
            "direction_loss": None,
            "age_gain": None,
            "fallback": newest is not None,
        }
    direction_best = max(
        passing,
        key=lambda item: (
            float(item["direction_similarity"]),
            int(item["pair"][1]),
        ),
    )
    best_direction = float(direction_best["direction_similarity"])
    best_age = int(current_t) - int(direction_best["pair"][1])
    tie_candidates = [
        item
        for item in passing
        if best_direction - float(item["direction_similarity"])
        <= margin + 1e-12
    ]
    tie_applied = best_age > STALE_TIE_AGE and len(tie_candidates) > 1
    selected = (
        max(
            tie_candidates,
            key=lambda item: (
                int(item["pair"][1]),
                float(item["direction_similarity"]),
            ),
        )
        if tie_applied
        else direction_best
    )
    changed = selected["pair"] != direction_best["pair"]
    return {
        "selected": common.normalized_pair(selected["pair"]),
        "direction_best": common.normalized_pair(direction_best["pair"]),
        "direction_best_age": best_age,
        "tie_candidates": [
            common.normalized_pair(item["pair"]) for item in tie_candidates
        ],
        "tie_applied": tie_applied,
        "changed": changed,
        "direction_loss": best_direction
        - float(selected["direction_similarity"]),
        "age_gain": int(selected["pair"][1])
        - int(direction_best["pair"][1]),
        "fallback": False,
    }


def analyze_prompt(path: Path, *, method: str) -> dict:
    rows = common.load_representative(path)
    margin = EXPECTED_MARGIN[method]
    failures = []
    archive_sizes = []
    selected_ages: list[float] = []
    direction_losses: list[float] = []
    age_gains: list[float] = []
    state_similarities: list[float] = []
    direction_similarities: list[float] = []
    reason_counts: Counter[str] = Counter()
    retrieval_count = 0
    multi_candidate_count = 0
    compatible_selection_count = 0
    direction_rejection_count = 0
    fallback_count = 0
    read_budget_violation_count = 0
    tie_applied_count = 0
    changed_count = 0
    for row in rows:
        item = row["strategy"]
        state = item.get("state", {})
        expected_state = {
            "state_match": True,
            "state_archive_capacity": 4,
            "state_max_read_age": MAX_READ_AGE,
            "state_min_similarity": -1.0,
            "state_min_direction_similarity": 0.1,
            "state_selection_order": ["direction_similarity", "recency"],
            "state_recency_weight": 0.0,
            "state_similarity_weight": 0.0,
            "state_fallback_to_newest": True,
            "state_direction_tie_margin": margin,
            "state_stale_tie_age": STALE_TIE_AGE,
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
            pair = common.normalized_pair(candidate.get("pair"))
            if pair is None or pair[0] + 1 != pair[1]:
                failures.append(
                    f"line {row['line_number']}: malformed candidate pair "
                    f"{candidate.get('pair')}"
                )
                continue
            age = int(candidate.get("age", -1))
            if age < 0 or age > MAX_READ_AGE:
                failures.append(
                    f"line {row['line_number']}: candidate age {age} invalid"
                )
            state_similarity = candidate.get("state_similarity")
            if state_similarity is not None:
                state_similarities.append(float(state_similarity))
            direction_similarity = candidate.get("direction_similarity")
            if direction_similarity is None:
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
            if not close(candidate.get("compatibility"), direction_value):
                failures.append(
                    f"line {row['line_number']}: state leaked into "
                    "direction compatibility"
                )
            if not close(candidate.get("selection_score"), direction_value):
                failures.append(
                    f"line {row['line_number']}: unexpected weighted score"
                )
        expected = expected_selection(
            candidates,
            current_t=int(row["sync_t"]),
            margin=margin,
        )
        selected_rows = retrieval.get("selected", [])
        selected_pair = (
            common.normalized_pair(selected_rows[0])
            if isinstance(selected_rows, list) and len(selected_rows) == 1
            else None
        )
        if selected_pair != expected["selected"]:
            failures.append(
                f"line {row['line_number']}: selected {selected_pair} != "
                f"recomputed {expected['selected']}"
            )
        if bool(retrieval.get("fallback_used")) != expected["fallback"]:
            failures.append(
                f"line {row['line_number']}: fallback flag mismatch"
            )
        actual_best = retrieval.get("direction_best", [])
        actual_best = actual_best[0] if len(actual_best) == 1 else None
        if common.normalized_pair(actual_best) != expected["direction_best"]:
            failures.append(
                f"line {row['line_number']}: direction baseline mismatch"
            )
        if retrieval.get("direction_best_age") != expected["direction_best_age"]:
            failures.append(
                f"line {row['line_number']}: direction baseline age mismatch"
            )
        actual_ties = sorted(
            common.normalized_pair(pair)
            for pair in retrieval.get("direction_tie_candidates", [])
        )
        expected_ties = sorted(expected["tie_candidates"])
        if actual_ties != expected_ties:
            failures.append(
                f"line {row['line_number']}: direction tie set mismatch"
            )
        if int(retrieval.get("direction_tie_candidate_count", -1)) != len(
            expected_ties
        ):
            failures.append(
                f"line {row['line_number']}: direction tie count mismatch"
            )
        if bool(retrieval.get("direction_tie_applied")) != expected["tie_applied"]:
            failures.append(
                f"line {row['line_number']}: direction tie flag mismatch"
            )
        if bool(retrieval.get("selection_changed_from_legacy")) != expected[
            "changed"
        ]:
            failures.append(
                f"line {row['line_number']}: changed flag mismatch"
            )
        if expected["direction_loss"] is not None and not close(
            retrieval.get("selected_direction_loss"),
            expected["direction_loss"],
        ):
            failures.append(
                f"line {row['line_number']}: direction loss mismatch"
            )
        if retrieval.get("selected_age_gain_vs_direction_best") != expected[
            "age_gain"
        ]:
            failures.append(
                f"line {row['line_number']}: age gain mismatch"
            )
        fallback_count += int(expected["fallback"])
        compatible_selection_count += int(
            expected["selected"] is not None and not expected["fallback"]
        )
        tie_applied_count += int(expected["tie_applied"])
        changed_count += int(expected["changed"])
        if expected["changed"]:
            direction_losses.append(float(expected["direction_loss"]))
            age_gains.append(float(expected["age_gain"]))
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
        if selected_age < 0 or selected_age > MAX_READ_AGE:
            failures.append(
                f"line {row['line_number']}: selected age {selected_age} invalid"
            )
    return {
        "prompt_index": common.prompt_index(path),
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
        "tie_applied_count": tie_applied_count,
        "changed_count": changed_count,
        "selected_ages": selected_ages,
        "direction_losses": direction_losses,
        "age_gains": age_gains,
        "state_similarities": state_similarities,
        "direction_similarities": direction_similarities,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def aggregate_method(prompts: list[dict], *, method: str) -> dict:
    def flatten(key: str) -> list[float]:
        return [value for prompt in prompts for value in prompt[key]]

    reasons: Counter[str] = Counter()
    for prompt in prompts:
        reasons.update(prompt["reason_counts"])
    aggregate = {
        "method": method,
        "margin": EXPECTED_MARGIN[method],
        "prompt_count": len(prompts),
        "read_count": sum(prompt["read_count"] for prompt in prompts),
        "retrieval_count": sum(prompt["retrieval_count"] for prompt in prompts),
        "archive_size_max": max(prompt["archive_size_max"] for prompt in prompts),
        "multi_candidate_count": sum(
            prompt["multi_candidate_count"] for prompt in prompts
        ),
        "compatible_selection_count": sum(
            prompt["compatible_selection_count"] for prompt in prompts
        ),
        "direction_rejection_count": sum(
            prompt["direction_rejection_count"] for prompt in prompts
        ),
        "fallback_count": sum(prompt["fallback_count"] for prompt in prompts),
        "read_budget_violation_count": sum(
            prompt["read_budget_violation_count"] for prompt in prompts
        ),
        "tie_applied_count": sum(
            prompt["tie_applied_count"] for prompt in prompts
        ),
        "changed_count": sum(prompt["changed_count"] for prompt in prompts),
        "selected_age": common.distribution(flatten("selected_ages")),
        "direction_loss_on_change": common.distribution(
            flatten("direction_losses")
        ),
        "age_gain_on_change": common.distribution(flatten("age_gains")),
        "state_similarity": common.distribution(flatten("state_similarities")),
        "direction_similarity": common.distribution(
            flatten("direction_similarities")
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": sum(
            len(prompt["failures"]) for prompt in prompts
        ),
    }
    loss_max = aggregate["direction_loss_on_change"]["max"]
    aggregate["mechanism_gate"] = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["compatible_selection_count"] > 0
        and aggregate["tie_applied_count"] > 0
        and aggregate["changed_count"] > 0
        and aggregate["read_budget_violation_count"] == 0
        and aggregate["contract_failure_count"] == 0
        and loss_max is not None
        and float(loss_max) <= EXPECTED_MARGIN[method] + TOLERANCE
    )
    return aggregate


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v165 Direction Stale-Tie Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Tie uses | Changed | Direction loss p95 | "
        "Age gain mean | Selected age p95 | Fallbacks | Failures |",
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
                    str(row["tie_applied_count"]),
                    str(row["changed_count"]),
                    str(row["direction_loss_on_change"]["p95"]),
                    str(row["age_gain_on_change"]["mean"]),
                    str(row["selected_age"]["p95"]),
                    str(row["fallback_count"]),
                    str(row["contract_failure_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The audit independently recomputes the direction maximum, stale",
            "gate, near-equivalent candidate set, newest tie choice, and read.",
            "It establishes mechanism execution only, not video quality.",
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
        "experiment": "v165_direction_stale_tie_trace",
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "both margins exercise stale-only near-equivalent direction "
            "selection with atomic equal-budget reads and bounded direction loss"
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
            f"v165 trace contract failed with {failure_count} violations"
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
