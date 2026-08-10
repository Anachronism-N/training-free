#!/usr/bin/env python3
"""Recompute and audit every v168 cross-scale retrieval decision."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common


PARETO_MOTION = "ours_middle10_reservoir2_multiscalepareto1"
CONSENSUS_MOTION = "ours_middle10_reservoir2_multiscaleconsensus1"
METHODS = (PARETO_MOTION, CONSENSUS_MOTION)
EXPECTED_MODE = {
    PARETO_MOTION: "pareto_multiscale_magnitude",
    CONSENSUS_MOTION: "consensus_multiscale_magnitude",
}
PROMPT_COUNT = 16
MAX_READ_AGE = 24
TOLERANCE = 3e-5
NUMERIC_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def close(left: object, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def pair(value: object) -> list[int] | None:
    return common.normalized_pair(value)


def first_pair(value: object) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 1:
        return None
    return pair(value[0])


def magnitude_similarity(left: object, right: object) -> float:
    left_value = max(0.0, float(left))
    right_value = max(0.0, float(right))
    if left_value <= 1e-8 and right_value <= 1e-8:
        return 1.0
    if left_value <= 1e-8 or right_value <= 1e-8:
        return 0.0
    return min(left_value, right_value) / max(left_value, right_value)


def recompute_candidate(candidate: dict) -> dict:
    local_direction = candidate.get("local_direction_similarity")
    context_direction = candidate.get("context_direction_similarity")
    directions = [
        float(value)
        for value in (local_direction, context_direction)
        if value is not None
    ]
    local_magnitude = (
        magnitude_similarity(
            candidate["query_local_magnitude"],
            candidate["candidate_local_magnitude"],
        )
        if local_direction is not None
        else None
    )
    context_magnitude = (
        magnitude_similarity(
            candidate["query_context_magnitude_per_step"],
            candidate["candidate_context_magnitude_per_step"],
        )
        if context_direction is not None
        else None
    )
    magnitudes = [
        value
        for value in (local_magnitude, context_magnitude)
        if value is not None
    ]
    multiscale_direction = (
        sum(directions) / len(directions) if directions else None
    )
    magnitude = (
        magnitudes[0]
        if len(magnitudes) == 1
        else (magnitudes[0] * magnitudes[1]) ** 0.5
        if len(magnitudes) == 2
        else None
    )
    score = (
        multiscale_direction * magnitude
        if multiscale_direction is not None and magnitude is not None
        else None
    )
    local_component = (
        float(local_direction) * float(local_magnitude)
        if local_direction is not None and local_magnitude is not None
        else None
    )
    context_component = (
        float(context_direction) * float(context_magnitude)
        if context_direction is not None and context_magnitude is not None
        else None
    )
    state_similarity = candidate.get("state_similarity")
    state_pass = (
        state_similarity is not None and float(state_similarity) >= -1.0
    )
    direction_pass = (
        multiscale_direction is None or multiscale_direction >= 0.1
    )
    return {
        "multiscale_direction": multiscale_direction,
        "local_magnitude": local_magnitude,
        "context_magnitude": context_magnitude,
        "magnitude": magnitude,
        "score": score,
        "local_component": local_component,
        "context_component": context_component,
        "state_pass": state_pass,
        "direction_pass": direction_pass,
        "passing": state_pass and direction_pass and score is not None,
    }


def expected_selection(candidates: list[dict], *, method: str) -> dict:
    rows = []
    for candidate in candidates:
        values = recompute_candidate(candidate)
        if values["passing"]:
            rows.append((candidate, values))
    if not rows:
        newest = (
            max(candidates, key=lambda item: int(item["pair"][1]))
            if candidates
            else None
        )
        return {
            "selected": None if newest is None else pair(newest["pair"]),
            "fallback": newest is not None,
            "motion": None,
            "newest": None,
            "local_best": None,
            "context_best": None,
            "agreement": None,
            "conflict": False,
            "pareto_pass": None,
            "local_delta": None,
            "context_delta": None,
            "reason": "no_passing_candidate",
            "rows": rows,
        }
    newest = max(rows, key=lambda item: int(item[0]["pair"][1]))
    motion = max(
        rows,
        key=lambda item: (
            float(item[1]["score"]),
            float(item[1]["score"]),
            int(item[0]["pair"][1]),
        ),
    )
    local_rows = [
        item for item in rows if item[1]["local_component"] is not None
    ]
    context_rows = [
        item for item in rows if item[1]["context_component"] is not None
    ]
    local_best = (
        max(
            local_rows,
            key=lambda item: (
                float(item[1]["local_component"]),
                int(item[0]["pair"][1]),
            ),
        )
        if local_rows
        else None
    )
    context_best = (
        max(
            context_rows,
            key=lambda item: (
                float(item[1]["context_component"]),
                int(item[0]["pair"][1]),
            ),
        )
        if context_rows
        else None
    )
    agreement = (
        pair(local_best[0]["pair"]) == pair(context_best[0]["pair"])
        if local_best is not None and context_best is not None
        else None
    )
    local_delta = (
        float(motion[1]["local_component"])
        - float(newest[1]["local_component"])
        if motion[1]["local_component"] is not None
        and newest[1]["local_component"] is not None
        else None
    )
    context_delta = (
        float(motion[1]["context_component"])
        - float(newest[1]["context_component"])
        if motion[1]["context_component"] is not None
        and newest[1]["context_component"] is not None
        else None
    )
    pareto_pass = bool(
        pair(motion[0]["pair"]) == pair(newest[0]["pair"])
        or (
            local_delta is not None
            and context_delta is not None
            and local_delta >= -NUMERIC_TOLERANCE
            and context_delta >= -NUMERIC_TOLERANCE
        )
    )
    if method == PARETO_MOTION:
        selected = motion if pareto_pass else newest
        reason = (
            "pareto_newest_motion_winner"
            if pair(motion[0]["pair"]) == pair(newest[0]["pair"])
            else "pareto_motion_recall"
            if pareto_pass
            else "pareto_newest_dominance_reject"
        )
    else:
        selected = local_best if agreement else newest
        reason = (
            "scale_component_unavailable_newest"
            if agreement is None
            else "scale_conflict_newest"
            if not agreement
            else "scale_consensus_newest"
            if pair(selected[0]["pair"]) == pair(newest[0]["pair"])
            else "scale_consensus_recall"
        )
    return {
        "selected": pair(selected[0]["pair"]),
        "fallback": False,
        "motion": pair(motion[0]["pair"]),
        "newest": pair(newest[0]["pair"]),
        "local_best": (
            None if local_best is None else pair(local_best[0]["pair"])
        ),
        "context_best": (
            None if context_best is None else pair(context_best[0]["pair"])
        ),
        "agreement": agreement,
        "conflict": agreement is False,
        "pareto_pass": pareto_pass,
        "local_delta": local_delta,
        "context_delta": context_delta,
        "reason": reason,
        "rows": rows,
    }


def analyze_prompt(path: Path, *, method: str) -> dict:
    rows = common.load_representative(path)
    mode = EXPECTED_MODE[method]
    failures: list[str] = []
    archive_sizes: list[int] = []
    selected_ages: list[float] = []
    local_components: list[float] = []
    context_components: list[float] = []
    reason_counts: Counter[str] = Counter()
    retrieval_count = 0
    multi_candidate_count = 0
    fallback_count = 0
    old_recall_count = 0
    conflict_count = 0
    agreement_count = 0
    pareto_recall_count = 0
    pareto_reject_count = 0
    changed_from_motion_count = 0
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
        actual_state = {key: state.get(key) for key in expected_state}
        if actual_state != expected_state:
            failures.append(
                f"line {row['line_number']}: state contract mismatch"
            )
        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        stored_pairs = {tuple(pair(value)) for value in stored if pair(value)}
        if len(stored) > 4:
            failures.append(
                f"line {row['line_number']}: archive exceeds capacity"
            )
        read_ids = [int(value) for value in strategy.get("frame_ids", [])]
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
        reason_counts[str(retrieval.get("selection_reason", "missing"))] += 1
        if retrieval.get("selection_mode") != mode:
            failures.append(
                f"line {row['line_number']}: selection mode mismatch"
            )
        if retrieval.get("state_filter_mode") != "none":
            failures.append(
                f"line {row['line_number']}: state filter leaked into v168"
            )
        if retrieval.get("motion_deficit_gate_enabled") is not False:
            failures.append(
                f"line {row['line_number']}: deficit gate leaked into v168"
            )
        candidates = list(retrieval.get("candidates", []))
        if int(retrieval.get("eligible", -1)) != len(candidates):
            failures.append(
                f"line {row['line_number']}: eligible/candidate mismatch"
            )
        if len(candidates) >= 2:
            multi_candidate_count += 1
        recomputed_by_pair = {}
        for candidate in candidates:
            candidate_pair = pair(candidate.get("pair"))
            if candidate_pair is None or candidate_pair[0] + 1 != candidate_pair[1]:
                failures.append(
                    f"line {row['line_number']}: malformed candidate pair"
                )
                continue
            if tuple(candidate_pair) not in stored_pairs:
                failures.append(
                    f"line {row['line_number']}: candidate absent from archive"
                )
            expected_age = int(row["sync_t"]) - candidate_pair[1]
            if int(candidate.get("age", -1)) != expected_age or not (
                0 <= expected_age <= MAX_READ_AGE
            ):
                failures.append(
                    f"line {row['line_number']}: invalid candidate age"
                )
            values = recompute_candidate(candidate)
            recomputed_by_pair[tuple(candidate_pair)] = values
            checks = {
                "local_magnitude_similarity": values["local_magnitude"],
                "context_magnitude_similarity": values[
                    "context_magnitude"
                ],
                "magnitude_similarity": values["magnitude"],
                "multiscale_direction_similarity": values[
                    "multiscale_direction"
                ],
                "motion_signature_score": values["score"],
                "compatibility": values["score"],
                "selection_score": values["score"],
                "local_motion_component": values["local_component"],
                "context_motion_component": values["context_component"],
            }
            for key, expected in checks.items():
                if not close(candidate.get(key), expected):
                    failures.append(
                        f"line {row['line_number']}: {key} mismatch"
                    )
            if candidate.get("state_pass") is not values["state_pass"]:
                failures.append(
                    f"line {row['line_number']}: state gate mismatch"
                )
            if candidate.get("direction_pass") is not values["direction_pass"]:
                failures.append(
                    f"line {row['line_number']}: direction gate mismatch"
                )
            if values["local_component"] is not None:
                local_components.append(float(values["local_component"]))
            if values["context_component"] is not None:
                context_components.append(float(values["context_component"]))
        expected = expected_selection(candidates, method=method)
        passing = expected["rows"]
        for component_key, trace_rank_key in (
            ("local_component", "local_component_rank"),
            ("context_component", "context_component_rank"),
        ):
            ranked = sorted(
                [item for item in passing if item[1][component_key] is not None],
                key=lambda item: (
                    float(item[1][component_key]),
                    int(item[0]["pair"][1]),
                ),
                reverse=True,
            )
            expected_ranks = {
                tuple(pair(item[0]["pair"])): rank
                for rank, item in enumerate(ranked, start=1)
            }
            for candidate in candidates:
                candidate_pair = pair(candidate.get("pair"))
                if candidate_pair is not None and candidate.get(
                    trace_rank_key
                ) != expected_ranks.get(tuple(candidate_pair)):
                    failures.append(
                        f"line {row['line_number']}: {trace_rank_key} mismatch"
                    )
        selected = first_pair(retrieval.get("selected", []))
        if selected != expected["selected"]:
            failures.append(
                f"line {row['line_number']}: selected {selected} != "
                f"{expected['selected']}"
            )
        expected_pairs = {
            "motion_signature_selected": expected["motion"],
            "pareto_candidate": expected["motion"],
            "newest_passing": expected["newest"],
            "local_component_best": expected["local_best"],
            "context_component_best": expected["context_best"],
        }
        for key, expected_pair in expected_pairs.items():
            actual_pair = first_pair(retrieval.get(key, []))
            if actual_pair != expected_pair:
                failures.append(
                    f"line {row['line_number']}: {key} mismatch"
                )
        if retrieval.get("scale_argmax_agreement") is not expected["agreement"]:
            failures.append(
                f"line {row['line_number']}: agreement flag mismatch"
            )
        if bool(retrieval.get("cross_scale_conflict")) != expected["conflict"]:
            failures.append(
                f"line {row['line_number']}: conflict flag mismatch"
            )
        if retrieval.get("pareto_pass") is not expected["pareto_pass"]:
            failures.append(
                f"line {row['line_number']}: Pareto flag mismatch"
            )
        deltas = retrieval.get("pareto_component_delta", {})
        if not close(deltas.get("local"), expected["local_delta"]):
            failures.append(
                f"line {row['line_number']}: local Pareto delta mismatch"
            )
        if not close(deltas.get("context"), expected["context_delta"]):
            failures.append(
                f"line {row['line_number']}: context Pareto delta mismatch"
            )
        if float(retrieval.get("component_numeric_tolerance", -1.0)) != (
            NUMERIC_TOLERANCE
        ):
            failures.append(
                f"line {row['line_number']}: numeric tolerance mismatch"
            )
        if not expected["fallback"] and retrieval.get(
            "selection_reason"
        ) != expected["reason"]:
            failures.append(
                f"line {row['line_number']}: selection reason mismatch"
            )
        if bool(retrieval.get("fallback_used")) != expected["fallback"]:
            failures.append(
                f"line {row['line_number']}: fallback flag mismatch"
            )
        fallback_count += int(expected["fallback"])
        conflict_count += int(expected["conflict"])
        agreement_count += int(expected["agreement"] is True)
        pareto_recall_count += int(
            expected["pareto_pass"] is True
            and expected["motion"] is not None
            and expected["newest"] is not None
            and expected["motion"] != expected["newest"]
        )
        pareto_reject_count += int(expected["pareto_pass"] is False)
        changed = bool(
            selected is not None
            and expected["motion"] is not None
            and selected != expected["motion"]
        )
        if bool(
            retrieval.get("selection_changed_from_motion_signature")
        ) != changed:
            failures.append(
                f"line {row['line_number']}: motion-change flag mismatch"
            )
        changed_from_motion_count += int(changed)
        if selected is not None and expected["newest"] is not None:
            old_recall_count += int(selected != expected["newest"])
            selected_age = int(row["sync_t"]) - selected[1]
            selected_ages.append(float(selected_age))
            if not 0 <= selected_age <= MAX_READ_AGE:
                failures.append(
                    f"line {row['line_number']}: selected age invalid"
                )
        if selected is None:
            if read_ids:
                failures.append(
                    f"line {row['line_number']}: empty selection read frames"
                )
        elif selected != read_ids:
            failures.append(
                f"line {row['line_number']}: selected/read mismatch"
            )
        if candidates and (
            selected is None
            or retrieval.get("read_budget_preserved") is not True
        ):
            read_budget_violation_count += 1
            failures.append(
                f"line {row['line_number']}: compatible read budget lost"
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
        "fallback_count": fallback_count,
        "old_recall_count": old_recall_count,
        "conflict_count": conflict_count,
        "agreement_count": agreement_count,
        "pareto_recall_count": pareto_recall_count,
        "pareto_reject_count": pareto_reject_count,
        "changed_from_motion_count": changed_from_motion_count,
        "read_budget_violation_count": read_budget_violation_count,
        "selected_ages": selected_ages,
        "local_components": local_components,
        "context_components": context_components,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def aggregate_method(prompts: list[dict], *, method: str) -> dict:
    reasons: Counter[str] = Counter()
    for prompt in prompts:
        reasons.update(prompt["reason_counts"])
    aggregate = {
        "method": method,
        "mode": EXPECTED_MODE[method],
        "prompt_count": len(prompts),
        "read_count": sum(row["read_count"] for row in prompts),
        "retrieval_count": sum(row["retrieval_count"] for row in prompts),
        "archive_size_max": max(row["archive_size_max"] for row in prompts),
        "multi_candidate_count": sum(
            row["multi_candidate_count"] for row in prompts
        ),
        "fallback_count": sum(row["fallback_count"] for row in prompts),
        "old_recall_count": sum(row["old_recall_count"] for row in prompts),
        "cross_scale_conflict_count": sum(
            row["conflict_count"] for row in prompts
        ),
        "scale_argmax_agreement_count": sum(
            row["agreement_count"] for row in prompts
        ),
        "pareto_recall_count": sum(
            row["pareto_recall_count"] for row in prompts
        ),
        "pareto_reject_count": sum(
            row["pareto_reject_count"] for row in prompts
        ),
        "changed_from_v166_motion_count": sum(
            row["changed_from_motion_count"] for row in prompts
        ),
        "read_budget_violation_count": sum(
            row["read_budget_violation_count"] for row in prompts
        ),
        "selected_age": common.distribution(
            [value for row in prompts for value in row["selected_ages"]]
        ),
        "local_component": common.distribution(
            [value for row in prompts for value in row["local_components"]]
        ),
        "context_component": common.distribution(
            [
                value
                for row in prompts
                for value in row["context_components"]
            ]
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": sum(
            len(row["failures"]) for row in prompts
        ),
    }
    shared_gate = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["read_budget_violation_count"] == 0
        and aggregate["contract_failure_count"] == 0
        and aggregate["changed_from_v166_motion_count"] > 0
    )
    if method == PARETO_MOTION:
        branch_gate = bool(
            aggregate["pareto_recall_count"] > 0
            and aggregate["pareto_reject_count"] > 0
            and aggregate["old_recall_count"] > 0
        )
    else:
        branch_gate = bool(
            aggregate["scale_argmax_agreement_count"] > 0
            and aggregate["cross_scale_conflict_count"] > 0
            and aggregate["old_recall_count"] > 0
        )
    aggregate["mechanism_gate"] = shared_gate and branch_gate
    return aggregate


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v168 Cross-scale Consensus Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Old recalls | Pareto recalls | Pareto rejects | "
        "Scale agreements | Scale conflicts | Changed vs v166 | Violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]["aggregate"]
        lines.append(
            f"| {method} | {row['mechanism_gate']} | "
            f"{row['old_recall_count']} | {row['pareto_recall_count']} | "
            f"{row['pareto_reject_count']} | "
            f"{row['scale_argmax_agreement_count']} | "
            f"{row['cross_scale_conflict_count']} | "
            f"{row['changed_from_v166_motion_count']} | "
            f"{row['contract_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "This audit independently recomputes every component, rank, "
            "counterfactual and selected atomic pair. It proves mechanism "
            "execution only; video-quality promotion remains separate.",
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
                f"expected {PROMPT_COUNT} traces for {method}, found "
                f"{len(paths)}"
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
        "experiment": "v168_cross_scale_consensus_trace",
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "exact component/rank/selection recomputation; both Pareto "
            "accept/reject and scale agreement/conflict branches execute; "
            "all reads remain atomic and budget preserving"
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
            f"v168 trace contract failed with {failure_count} violations"
        )
    if not mechanism_gate:
        raise SystemExit("v168 mechanism branches did not all execute")
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
