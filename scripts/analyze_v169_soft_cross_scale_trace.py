#!/usr/bin/env python3
"""Independently recompute every v169 retrieval and cache read."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common
import analyze_v168_cross_scale_consensus_trace as v168
import v169_soft_cross_scale_contract as contract


PROMPT_COUNT = 16
MAX_READ_AGE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def analyze_prompt(
    path: Path,
    *,
    method: str,
    rows: list[dict] | None = None,
) -> dict:
    rows = common.load_representative(path) if rows is None else rows
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    mode = contract.EXPECTED_MODE[method]
    failures: list[str] = []
    reason_counts: Counter[str] = Counter()
    archive_sizes: list[int] = []
    selected_ages: list[float] = []
    query_scores: list[float] = []
    bottleneck_scores: list[float] = []
    retrieval_count = 0
    multi_candidate_count = 0
    fallback_count = 0
    old_recall_count = 0
    changed_from_v166_count = 0
    read_budget_violation_count = 0

    for row in rows:
        strategy = row["strategy"]
        state = strategy.get("state", {})
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
            "state_direction_tie_margin": 0.0,
            "state_stale_tie_age": 0,
            "state_motion_signature_mode": mode,
        }
        if {key: state.get(key) for key in expected_state} != expected_state:
            failures.append(f"line {row['line_number']}: state contract mismatch")

        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        stored_pairs = {
            tuple(value) for raw in stored if (value := v168.pair(raw)) is not None
        }
        if len(stored) > 4:
            failures.append(f"line {row['line_number']}: archive exceeds capacity")
        read_ids = [int(value) for value in strategy.get("frame_ids", [])]
        if len(read_ids) not in {0, 2} or (
            len(read_ids) == 2 and read_ids[0] + 1 != read_ids[1]
        ):
            failures.append(f"line {row['line_number']}: non-atomic read {read_ids}")

        retrieval = state.get("last_retrieval", {})
        if not retrieval:
            continue
        retrieval_count += 1
        reason_counts[str(retrieval.get("selection_reason", "missing"))] += 1
        if retrieval.get("selection_mode") != mode:
            failures.append(f"line {row['line_number']}: selection mode mismatch")
        if retrieval.get("state_filter_mode") != "none":
            failures.append(f"line {row['line_number']}: state filter leaked into v169")
        if retrieval.get("motion_deficit_gate_enabled") is not False:
            failures.append(f"line {row['line_number']}: deficit gate leaked into v169")

        candidates = list(retrieval.get("candidates", []))
        if int(retrieval.get("eligible", -1)) != len(candidates):
            failures.append(f"line {row['line_number']}: eligible/candidate mismatch")
        expected = contract.expected_selection(candidates, method=method)
        multi_candidate_count += int(len(expected["rows"]) >= 2)

        for candidate in candidates:
            candidate_pair = v168.pair(candidate.get("pair"))
            if candidate_pair is None or candidate_pair[0] + 1 != candidate_pair[1]:
                failures.append(f"line {row['line_number']}: malformed candidate pair")
                continue
            if tuple(candidate_pair) not in stored_pairs:
                failures.append(
                    f"line {row['line_number']}: candidate absent from archive"
                )
            age = int(row["sync_t"]) - candidate_pair[1]
            if int(candidate.get("age", -1)) != age or not (0 <= age <= MAX_READ_AGE):
                failures.append(f"line {row['line_number']}: invalid candidate age")
            scores = contract.candidate_scores(candidate)
            checks = {
                "local_magnitude_similarity": scores["local_magnitude"],
                "context_magnitude_similarity": scores["context_magnitude"],
                "magnitude_similarity": scores["magnitude"],
                "multiscale_direction_similarity": scores["multiscale_direction"],
                "motion_signature_score": scores["score"],
                "compatibility": scores["score"],
                "selection_score": scores["score"],
                "local_motion_component": scores["local_component"],
                "context_motion_component": scores["context_component"],
                "query_weighted_motion_score": scores["query_weighted_score"],
                "bottleneck_motion_score": scores["bottleneck_score"],
            }
            for key, expected_value in checks.items():
                if not contract.close(candidate.get(key), expected_value):
                    failures.append(f"line {row['line_number']}: {key} mismatch")
            actual_weights = candidate.get("query_weighted_component_weights", {})
            for key in ("local", "context"):
                if not contract.close(actual_weights.get(key), scores["weights"][key]):
                    failures.append(f"line {row['line_number']}: {key} weight mismatch")
            if candidate.get("state_pass") is not scores["state_pass"]:
                failures.append(f"line {row['line_number']}: state gate mismatch")
            if candidate.get("direction_pass") is not scores["direction_pass"]:
                failures.append(f"line {row['line_number']}: direction gate mismatch")
            if scores["query_weighted_score"] is not None:
                query_scores.append(float(scores["query_weighted_score"]))
            if scores["bottleneck_score"] is not None:
                bottleneck_scores.append(float(scores["bottleneck_score"]))

        selected = v168.first_pair(retrieval.get("selected", []))
        if selected != expected["selected"]:
            failures.append(
                f"line {row['line_number']}: selected {selected} != "
                f"{expected['selected']}"
            )
        query_expected = contract.expected_selection(
            candidates,
            method=contract.QUERY_WEIGHTED,
        )
        bottleneck_expected = contract.expected_selection(
            candidates,
            method=contract.BOTTLENECK,
        )
        expected_pairs = {
            "motion_signature_selected": expected["baseline"],
            "newest_passing": expected["newest"],
            "query_weighted_selected": query_expected["custom"],
            "bottleneck_selected": bottleneck_expected["custom"],
        }
        for key, expected_pair in expected_pairs.items():
            if v168.first_pair(retrieval.get(key, [])) != expected_pair:
                failures.append(f"line {row['line_number']}: {key} mismatch")
        if retrieval.get("selection_reason") != expected["reason"]:
            failures.append(f"line {row['line_number']}: selection reason mismatch")
        if bool(retrieval.get("fallback_used")) != expected["fallback"]:
            failures.append(f"line {row['line_number']}: fallback flag mismatch")
        fallback_count += int(expected["fallback"])
        changed = bool(
            selected is not None
            and expected["baseline"] is not None
            and selected != expected["baseline"]
        )
        if bool(retrieval.get("selection_changed_from_motion_signature")) != changed:
            failures.append(f"line {row['line_number']}: v166-change flag mismatch")
        changed_from_v166_count += int(changed)
        if selected is not None and expected["newest"] is not None:
            old_recall_count += int(selected != expected["newest"])
            selected_age = int(row["sync_t"]) - selected[1]
            selected_ages.append(float(selected_age))
            if not 0 <= selected_age <= MAX_READ_AGE:
                failures.append(f"line {row['line_number']}: selected age invalid")
        if selected is None:
            if read_ids:
                failures.append(
                    f"line {row['line_number']}: empty selection read frames"
                )
        elif selected != read_ids:
            failures.append(f"line {row['line_number']}: selected/read mismatch")
        if candidates and (
            selected is None or retrieval.get("read_budget_preserved") is not True
        ):
            read_budget_violation_count += 1
            failures.append(f"line {row['line_number']}: compatible read budget lost")

    return {
        "prompt_index": common.prompt_index(path),
        "trace": str(path),
        "representative_layer": rows[0]["layer"],
        "representative_head": rows[0]["head"],
        "read_count": len(rows),
        "retrieval_count": retrieval_count,
        "archive_size_max": max(archive_sizes, default=0),
        "multi_candidate_count": multi_candidate_count,
        "fallback_count": fallback_count,
        "old_recall_count": old_recall_count,
        "changed_from_v166_count": changed_from_v166_count,
        "read_budget_violation_count": read_budget_violation_count,
        "selected_ages": selected_ages,
        "query_weighted_scores": query_scores,
        "bottleneck_scores": bottleneck_scores,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def aggregate_method(prompts: list[dict], *, method: str) -> dict:
    reasons: Counter[str] = Counter()
    for prompt in prompts:
        reasons.update(prompt["reason_counts"])
    aggregate = {
        "method": method,
        "mode": contract.EXPECTED_MODE[method],
        "prompt_count": len(prompts),
        "read_count": sum(row["read_count"] for row in prompts),
        "retrieval_count": sum(row["retrieval_count"] for row in prompts),
        "archive_size_max": max(row["archive_size_max"] for row in prompts),
        "multi_candidate_count": sum(row["multi_candidate_count"] for row in prompts),
        "fallback_count": sum(row["fallback_count"] for row in prompts),
        "old_recall_count": sum(row["old_recall_count"] for row in prompts),
        "changed_from_v166_count": sum(
            row["changed_from_v166_count"] for row in prompts
        ),
        "read_budget_violation_count": sum(
            row["read_budget_violation_count"] for row in prompts
        ),
        "selected_age": common.distribution(
            [value for row in prompts for value in row["selected_ages"]]
        ),
        "query_weighted_score": common.distribution(
            [value for row in prompts for value in row["query_weighted_scores"]]
        ),
        "bottleneck_score": common.distribution(
            [value for row in prompts for value in row["bottleneck_scores"]]
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": sum(len(row["failures"]) for row in prompts),
    }
    aggregate["mechanism_gate"] = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["old_recall_count"] > 0
        and aggregate["changed_from_v166_count"] > 0
        and aggregate["read_budget_violation_count"] == 0
        and aggregate["contract_failure_count"] == 0
    )
    return aggregate


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v169 Soft Cross-scale Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Old recalls | Changed vs v166 | Fallbacks | "
        "Age median | Budget violations | Contract failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in contract.METHODS:
        row = report["methods"][method]["aggregate"]
        lines.append(
            f"| {method} | {row['mechanism_gate']} | "
            f"{row['old_recall_count']} | {row['changed_from_v166_count']} | "
            f"{row['fallback_count']} | {row['selected_age'].get('median')} | "
            f"{row['read_budget_violation_count']} | "
            f"{row['contract_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "The audit recomputes primitive component scores, query weights, "
            "both counterfactual selectors and the actual atomic read. It is "
            "mechanism evidence, not a video-quality claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_reports = {}
    for method in contract.METHODS:
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
        for method in contract.METHODS
    )
    report = {
        "version": 1,
        "experiment": "v169_soft_cross_scale_trace",
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "exact score and selection recomputation; both methods must "
            "change v166 choices, recall old pairs and preserve atomic reads"
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
    failures = sum(
        method_reports[method]["aggregate"]["contract_failure_count"]
        for method in contract.METHODS
    )
    if failures:
        raise SystemExit(f"v169 trace contract failed with {failures} violations")
    if not mechanism_gate:
        raise SystemExit("v169 mechanism branches did not all execute")
    print(
        json.dumps(
            {
                method: method_reports[method]["aggregate"]
                for method in contract.METHODS
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
