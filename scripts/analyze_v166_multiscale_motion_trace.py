#!/usr/bin/env python3
"""Audit v166 multi-scale motion-signature retrieval from saved traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common


MULTISCALE_DIRECTION = "ours_middle10_reservoir2_multiscaledir1"
MULTISCALE_MOTION = "ours_middle10_reservoir2_multiscalemotion1"
METHODS = (MULTISCALE_DIRECTION, MULTISCALE_MOTION)
EXPECTED_MODE = {
    MULTISCALE_DIRECTION: "multiscale_direction",
    MULTISCALE_MOTION: "multiscale_magnitude",
}
PROMPT_COUNT = 16
MAX_READ_AGE = 24
TOLERANCE = 3e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def close(left: object, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE


def available(candidate: dict, names: tuple[str, ...]) -> list[float]:
    return [
        float(candidate[name])
        for name in names
        if candidate.get(name) is not None
    ]


def magnitude_similarity(left: object, right: object) -> float:
    left_value = max(0.0, float(left))
    right_value = max(0.0, float(right))
    if left_value <= 1e-8 and right_value <= 1e-8:
        return 1.0
    if left_value <= 1e-8 or right_value <= 1e-8:
        return 0.0
    return min(left_value, right_value) / max(left_value, right_value)


def recompute_candidate(candidate: dict, *, mode: str) -> dict:
    directions = available(
        candidate,
        ("local_direction_similarity", "context_direction_similarity"),
    )
    local_magnitude = (
        magnitude_similarity(
            candidate["query_local_magnitude"],
            candidate["candidate_local_magnitude"],
        )
        if candidate.get("local_direction_similarity") is not None
        else None
    )
    context_magnitude = (
        magnitude_similarity(
            candidate["query_context_magnitude_per_step"],
            candidate["candidate_context_magnitude_per_step"],
        )
        if candidate.get("context_direction_similarity") is not None
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
        multiscale_direction
        if mode == "multiscale_direction"
        else multiscale_direction * magnitude
        if multiscale_direction is not None and magnitude is not None
        else None
    )
    compatibility = score
    selection_score = compatibility
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
        "compatibility": compatibility,
        "selection_score": selection_score,
        "state_pass": state_pass,
        "direction_pass": direction_pass,
        "passing": state_pass and direction_pass and score is not None,
    }


def expected_selection(candidates: list[dict], *, mode: str) -> dict:
    scored = []
    for candidate in candidates:
        recomputed = recompute_candidate(candidate, mode=mode)
        if recomputed["passing"]:
            scored.append((candidate, recomputed))
    if scored:
        selected, values = max(
            scored,
            key=lambda item: (
                float(item[1]["selection_score"]),
                float(item[1]["compatibility"]),
                int(item[0]["pair"][1]),
            ),
        )
        return {
            "pair": common.normalized_pair(selected["pair"]),
            "fallback": False,
            "score": values["score"],
        }
    newest = (
        max(candidates, key=lambda item: int(item["pair"][1]))
        if candidates
        else None
    )
    return {
        "pair": (
            None if newest is None else common.normalized_pair(newest["pair"])
        ),
        "fallback": newest is not None,
        "score": None,
    }


def expected_legacy_selection(candidates: list[dict]) -> dict:
    passing = []
    for candidate in candidates:
        state_similarity = candidate.get("state_similarity")
        direction_similarity = candidate.get("direction_similarity")
        if (
            state_similarity is not None
            and float(state_similarity) >= -1.0
            and direction_similarity is not None
            and float(direction_similarity) >= 0.1
        ):
            passing.append(candidate)
    if passing:
        selected = max(
            passing,
            key=lambda item: (
                float(item["direction_similarity"]),
                int(item["pair"][1]),
            ),
        )
        return {
            "pair": common.normalized_pair(selected["pair"]),
            "passing_pair": common.normalized_pair(selected["pair"]),
            "fallback": False,
        }
    newest = (
        max(candidates, key=lambda item: int(item["pair"][1]))
        if candidates
        else None
    )
    return {
        "pair": (
            None if newest is None else common.normalized_pair(newest["pair"])
        ),
        "passing_pair": None,
        "fallback": newest is not None,
    }


def analyze_prompt(path: Path, *, method: str) -> dict:
    rows = common.load_representative(path)
    mode = EXPECTED_MODE[method]
    failures: list[str] = []
    archive_sizes: list[int] = []
    selected_ages: list[float] = []
    direction_scores: list[float] = []
    magnitude_scores: list[float] = []
    signature_scores: list[float] = []
    reason_counts: Counter[str] = Counter()
    retrieval_count = 0
    multi_candidate_count = 0
    changed_from_legacy_count = 0
    fallback_count = 0
    legacy_fallback_count = 0
    direction_rejection_count = 0
    read_budget_violation_count = 0
    component_count = 0
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
            "state_direction_tie_margin": 0.0,
            "state_stale_tie_age": 0,
            "state_motion_signature_mode": mode,
        }
        actual_state = {key: state.get(key) for key in expected_state}
        if actual_state != expected_state:
            failures.append(
                f"line {row['line_number']}: state contract "
                f"{actual_state} != {expected_state}"
            )
        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        stored_pairs = set()
        if len(stored) > 4:
            failures.append(
                f"line {row['line_number']}: archive exceeds capacity"
            )
        for value in stored:
            pair = common.normalized_pair(value)
            if pair is None or pair[0] + 1 != pair[1]:
                failures.append(
                    f"line {row['line_number']}: malformed stored pair {value}"
                )
            else:
                stored_pairs.add(tuple(pair))
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
        if retrieval.get("selection_mode") != mode:
            failures.append(
                f"line {row['line_number']}: selection mode mismatch"
            )
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
                    f"line {row['line_number']}: malformed pair "
                    f"{candidate.get('pair')}"
                )
                continue
            if tuple(pair) not in stored_pairs:
                failures.append(
                    f"line {row['line_number']}: candidate {pair} absent "
                    "from archive"
                )
            age = int(candidate.get("age", -1))
            expected_age = int(row["sync_t"]) - pair[1]
            if age != expected_age or age < 0 or age > MAX_READ_AGE:
                failures.append(
                    f"line {row['line_number']}: invalid age {age}; "
                    f"expected {expected_age}"
                )
            values = recompute_candidate(candidate, mode=mode)
            if not close(
                candidate.get("multiscale_direction_similarity"),
                values["multiscale_direction"],
            ):
                failures.append(
                    f"line {row['line_number']}: direction mean mismatch"
                )
            if not close(candidate.get("magnitude_similarity"), values["magnitude"]):
                failures.append(
                    f"line {row['line_number']}: magnitude mean mismatch"
                )
            if not close(
                candidate.get("local_magnitude_similarity"),
                values["local_magnitude"],
            ):
                failures.append(
                    f"line {row['line_number']}: local magnitude mismatch"
                )
            if not close(
                candidate.get("context_magnitude_similarity"),
                values["context_magnitude"],
            ):
                failures.append(
                    f"line {row['line_number']}: context magnitude mismatch"
                )
            if not close(candidate.get("motion_signature_score"), values["score"]):
                failures.append(
                    f"line {row['line_number']}: signature score mismatch"
                )
            if not close(
                candidate.get("compatibility"),
                values["compatibility"],
            ):
                failures.append(
                    f"line {row['line_number']}: compatibility mismatch"
                )
            if not close(
                candidate.get("selection_score"),
                values["selection_score"],
            ):
                failures.append(
                    f"line {row['line_number']}: selection score mismatch"
                )
            if candidate.get("state_pass") is not values["state_pass"]:
                failures.append(
                    f"line {row['line_number']}: state gate mismatch"
                )
            if candidate.get("direction_pass") is not values["direction_pass"]:
                failures.append(
                    f"line {row['line_number']}: direction gate mismatch"
                )
            direction_rejection_count += int(not values["direction_pass"])
            if values["multiscale_direction"] is not None:
                direction_scores.append(values["multiscale_direction"])
                component_count += 1
            if values["magnitude"] is not None:
                magnitude_scores.append(values["magnitude"])
            if values["score"] is not None:
                signature_scores.append(values["score"])
        expected = expected_selection(candidates, mode=mode)
        selected_rows = retrieval.get("selected", [])
        selected_pair = (
            common.normalized_pair(selected_rows[0])
            if isinstance(selected_rows, list) and len(selected_rows) == 1
            else None
        )
        if selected_pair != expected["pair"]:
            failures.append(
                f"line {row['line_number']}: selected {selected_pair} != "
                f"recomputed {expected['pair']}"
            )
        if bool(retrieval.get("fallback_used")) != expected["fallback"]:
            failures.append(
                f"line {row['line_number']}: fallback flag mismatch"
            )
        fallback_count += int(expected["fallback"])
        legacy_rows = retrieval.get("legacy_selected", [])
        legacy_pair = (
            common.normalized_pair(legacy_rows[0])
            if isinstance(legacy_rows, list) and len(legacy_rows) == 1
            else None
        )
        legacy_passing_rows = retrieval.get("legacy_passing_selected", [])
        legacy_passing_pair = (
            common.normalized_pair(legacy_passing_rows[0])
            if isinstance(legacy_passing_rows, list)
            and len(legacy_passing_rows) == 1
            else None
        )
        expected_legacy = expected_legacy_selection(candidates)
        if legacy_pair != expected_legacy["pair"]:
            failures.append(
                f"line {row['line_number']}: legacy {legacy_pair} != "
                f"recomputed {expected_legacy['pair']}"
            )
        if legacy_passing_pair != expected_legacy["passing_pair"]:
            failures.append(
                f"line {row['line_number']}: legacy passing "
                f"{legacy_passing_pair} != recomputed "
                f"{expected_legacy['passing_pair']}"
            )
        if bool(retrieval.get("legacy_fallback_used")) != expected_legacy[
            "fallback"
        ]:
            failures.append(
                f"line {row['line_number']}: legacy fallback flag mismatch"
            )
        legacy_fallback_count += int(expected_legacy["fallback"])
        changed = bool(
            selected_pair is not None
            and legacy_pair is not None
            and selected_pair != legacy_pair
        )
        if bool(retrieval.get("selection_changed_from_legacy")) != changed:
            failures.append(
                f"line {row['line_number']}: legacy-change flag mismatch"
            )
        changed_from_legacy_count += int(changed)
        if candidates and (
            selected_pair is None
            or retrieval.get("read_budget_preserved") is not True
        ):
            read_budget_violation_count += 1
            failures.append(
                f"line {row['line_number']}: eligible pair budget lost"
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
                f"line {row['line_number']}: invalid selected age "
                f"{selected_age}"
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
        "changed_from_legacy_count": changed_from_legacy_count,
        "fallback_count": fallback_count,
        "legacy_fallback_count": legacy_fallback_count,
        "direction_rejection_count": direction_rejection_count,
        "read_budget_violation_count": read_budget_violation_count,
        "component_count": component_count,
        "selected_ages": selected_ages,
        "direction_scores": direction_scores,
        "magnitude_scores": magnitude_scores,
        "signature_scores": signature_scores,
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
        "mode": EXPECTED_MODE[method],
        "prompt_count": len(prompts),
        "read_count": sum(prompt["read_count"] for prompt in prompts),
        "retrieval_count": sum(prompt["retrieval_count"] for prompt in prompts),
        "archive_size_max": max(prompt["archive_size_max"] for prompt in prompts),
        "multi_candidate_count": sum(
            prompt["multi_candidate_count"] for prompt in prompts
        ),
        "changed_from_legacy_count": sum(
            prompt["changed_from_legacy_count"] for prompt in prompts
        ),
        "fallback_count": sum(prompt["fallback_count"] for prompt in prompts),
        "legacy_fallback_count": sum(
            prompt["legacy_fallback_count"] for prompt in prompts
        ),
        "direction_rejection_count": sum(
            prompt["direction_rejection_count"] for prompt in prompts
        ),
        "read_budget_violation_count": sum(
            prompt["read_budget_violation_count"] for prompt in prompts
        ),
        "component_count": sum(prompt["component_count"] for prompt in prompts),
        "selected_age": common.distribution(flatten("selected_ages")),
        "multiscale_direction": common.distribution(flatten("direction_scores")),
        "magnitude_match": common.distribution(flatten("magnitude_scores")),
        "motion_signature": common.distribution(flatten("signature_scores")),
        "reason_counts": dict(sorted(reasons.items())),
        "contract_failure_count": sum(
            len(prompt["failures"]) for prompt in prompts
        ),
    }
    aggregate["mechanism_gate"] = bool(
        aggregate["archive_size_max"] >= 2
        and aggregate["multi_candidate_count"] > 0
        and aggregate["changed_from_legacy_count"] > 0
        and aggregate["component_count"] > 0
        and aggregate["read_budget_violation_count"] == 0
        and aggregate["contract_failure_count"] == 0
    )
    return aggregate


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v166 Multi-scale Motion Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Changed vs legacy | Direction p50 | "
        "Magnitude p50 | Selected age p95 | Fallbacks | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(row["mechanism_gate"]),
                    str(row["changed_from_legacy_count"]),
                    str(row["multiscale_direction"]["median"]),
                    str(row["magnitude_match"]["median"]),
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
            "This audit recomputes magnitude matches, aggregate scores,",
            "gates, counterfactual/final selections, and atomic reads from",
            "the logged cosine and norm primitives. It proves execution,",
            "not quality.",
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
                f"expected {PROMPT_COUNT} traces for {method}, "
                f"found {len(paths)}"
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
        "experiment": "v166_multiscale_motion_trace",
        "mechanism_gate": mechanism_gate,
        "gate_definition": (
            "magnitude/aggregate-score/gate/selection recomputation from "
            "logged cosine and norm primitives, exercised multi-candidate "
            "ranking, changed legacy choices, and atomic equal-budget reads"
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
            f"v166 trace contract failed with {failure_count} violations"
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
