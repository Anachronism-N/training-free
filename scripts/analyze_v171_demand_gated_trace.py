#!/usr/bin/env python3
"""Audit all active-layer v171 demand-gated policy traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import analyze_v168_cross_scale_consensus_trace as v168
import v171_demand_gated_contract as contract


MAX_READ_AGE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_active_layer_rows(path: Path) -> dict[int, list[dict]]:
    by_layer: dict[int, dict[int, dict]] = {}
    observed_heads: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
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
            layer = int(row["layer"])
            head = int(row["head"])
            observed_heads.add(head)
            if head not in contract.TRACE_HEADS:
                raise ValueError(f"unexpected traced head {head} in {path}")
            sync_t = int(row["sync_t"])
            layer_rows = by_layer.setdefault(layer, {})
            if sync_t in layer_rows:
                raise ValueError(
                    f"duplicate layer={layer} head={head} t={sync_t} in {path}"
                )
            layer_rows[sync_t] = {
                "line_number": line_number,
                "layer": layer,
                "head": head,
                "sync_t": sync_t,
                "strategy": strategy,
                "cache_contract_pass": row.get("cache_contract_pass"),
                "cache_contract_violations": list(
                    row.get("cache_contract_violations", [])
                ),
            }
    if set(by_layer) != set(contract.ACTIVE_LAYERS):
        raise ValueError(
            f"active-layer coverage mismatch in {path}: {sorted(by_layer)}"
        )
    if observed_heads != set(contract.TRACE_HEADS):
        raise ValueError(
            f"trace-head coverage mismatch in {path}: {sorted(observed_heads)}"
        )
    return {
        layer: [rows[key] for key in sorted(rows)]
        for layer, rows in sorted(by_layer.items())
    }


def _check_deficit(deficit: dict, *, label: str, failures: list[str]) -> None:
    ready = bool(deficit.get("ready", False))
    triggered = bool(deficit.get("triggered", False))
    if triggered and not ready:
        failures.append(f"{label}: deficit triggered before ready")
    if ready:
        local_ratio = float(deficit["local_ratio"])
        context_ratio = float(deficit["context_ratio"])
        expected = local_ratio < 1.0 and context_ratio < 1.0
        if triggered != expected:
            failures.append(f"{label}: deficit trigger rule mismatch")
    if deficit and deficit.get("rule") != "both_scales_below_online_median":
        failures.append(f"{label}: unexpected deficit rule")


def analyze_rows(rows: list[dict], *, method: str, prompt_index: int) -> dict:
    mode = contract.EXPECTED_MODE[method]
    failures: list[str] = []
    reason_counts: Counter[str] = Counter()
    archive_sizes: list[int] = []
    selected_ages: list[float] = []
    retrieval_count = 0
    ready_count = 0
    triggered_count = 0
    healthy_count = 0
    multi_candidate_count = 0
    changed_count = 0
    healthy_changed_count = 0
    old_recall_count = 0
    fallback_count = 0
    read_budget_violation_count = 0
    cache_contract_failure_count = 0

    for row in rows:
        label = f"p{prompt_index}:l{row['layer']}:t{row['sync_t']}"
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
            failures.append(f"{label}: state contract mismatch")
        if row["cache_contract_pass"] is not True or row[
            "cache_contract_violations"
        ]:
            cache_contract_failure_count += 1
            failures.append(f"{label}: cache ownership/budget contract failed")

        stored = state.get("pair_frame_ids", [])
        archive_sizes.append(len(stored))
        stored_pairs = {
            tuple(pair)
            for raw in stored
            if (pair := v168.pair(raw)) is not None
        }
        if len(stored) > 4:
            failures.append(f"{label}: archive exceeds four atomic pairs")
        read_ids = [int(value) for value in strategy.get("frame_ids", [])]
        if len(read_ids) not in {0, 2} or (
            len(read_ids) == 2 and read_ids[0] + 1 != read_ids[1]
        ):
            failures.append(f"{label}: non-atomic read {read_ids}")

        retrieval = state.get("last_retrieval", {})
        if not retrieval:
            continue
        retrieval_count += 1
        if retrieval.get("selection_mode") != mode:
            failures.append(f"{label}: selector mode mismatch")
        if retrieval.get("state_filter_mode") != "none":
            failures.append(f"{label}: state-rank filter leaked into v171")
        deficit = dict(retrieval.get("motion_deficit") or {})
        _check_deficit(deficit, label=label, failures=failures)
        ready = bool(deficit.get("ready", False))
        triggered = bool(deficit.get("triggered", False))
        ready_count += int(ready)
        triggered_count += int(triggered)
        healthy_count += int(not triggered)
        if retrieval.get("motion_deficit_gate_enabled") is not True:
            failures.append(f"{label}: motion deficit gate disabled")
        if retrieval.get("demand_gate_enabled") is not True:
            failures.append(f"{label}: demand gate disabled")
        if bool(retrieval.get("demand_gate_triggered")) != triggered:
            failures.append(f"{label}: demand gate trigger mismatch")
        if bool(retrieval.get("motion_deficit_gate_triggered")) != triggered:
            failures.append(f"{label}: deficit gate trigger mismatch")
        if not contract.close(
            retrieval.get("baseline_local_magnitude_target"),
            deficit.get("local_median"),
        ):
            failures.append(f"{label}: local baseline target mismatch")
        if not contract.close(
            retrieval.get("baseline_context_magnitude_target_per_step"),
            deficit.get("context_median_per_step"),
        ):
            failures.append(f"{label}: context baseline target mismatch")

        candidates = list(retrieval.get("candidates", []))
        if int(retrieval.get("eligible", -1)) != len(candidates):
            failures.append(f"{label}: eligible/candidate count mismatch")
        expected = contract.expected_selection(
            candidates,
            method=method,
            motion_deficit=deficit,
        )
        multi_candidate_count += int(len(expected["rows"]) >= 2)
        for candidate in candidates:
            candidate_pair = v168.pair(candidate.get("pair"))
            if candidate_pair is None or candidate_pair[0] + 1 != candidate_pair[1]:
                failures.append(f"{label}: malformed candidate pair")
                continue
            if tuple(candidate_pair) not in stored_pairs:
                failures.append(f"{label}: candidate absent from archive")
            age = int(row["sync_t"]) - int(candidate_pair[1])
            if int(candidate.get("age", -1)) != age or not (0 <= age <= 24):
                failures.append(f"{label}: invalid candidate age")
            scores = contract.candidate_scores(
                candidate,
                motion_deficit=deficit,
            )
            checks = {
                "multiscale_direction_similarity": scores[
                    "multiscale_direction"
                ],
                "magnitude_similarity": scores["magnitude"],
                "motion_signature_score": scores["score"],
                "query_weighted_motion_score": scores[
                    "query_weighted_score"
                ],
                "baseline_local_magnitude_target": scores[
                    "baseline_local_target"
                ],
                "baseline_context_magnitude_target_per_step": scores[
                    "baseline_context_target"
                ],
                "baseline_local_magnitude_similarity": scores[
                    "baseline_local_magnitude"
                ],
                "baseline_context_magnitude_similarity": scores[
                    "baseline_context_magnitude"
                ],
                "baseline_magnitude_similarity": scores[
                    "baseline_magnitude"
                ],
                "deficit_baseline_motion_score": scores[
                    "deficit_baseline_score"
                ],
            }
            for key, expected_value in checks.items():
                if not contract.close(candidate.get(key), expected_value):
                    failures.append(f"{label}: {key} mismatch")
            if candidate.get("state_pass") is not scores["state_pass"]:
                failures.append(f"{label}: state gate mismatch")
            if candidate.get("direction_pass") is not scores["direction_pass"]:
                failures.append(f"{label}: direction gate mismatch")

        selected = v168.first_pair(retrieval.get("selected", []))
        expected_pairs = {
            "selected": expected["selected"],
            "motion_signature_selected": expected["baseline"],
            "query_weighted_selected": expected["query_weighted"],
            "deficit_baseline_selected": expected["deficit_baseline"],
            "newest_passing": expected["newest"],
        }
        if selected != expected["selected"]:
            failures.append(
                f"{label}: selected {selected} != {expected['selected']}"
            )
        for key, pair in expected_pairs.items():
            actual = (
                selected
                if key == "selected"
                else v168.first_pair(retrieval.get(key, []))
            )
            if actual != pair:
                failures.append(f"{label}: {key} mismatch")
        if retrieval.get("selection_reason") != expected["reason"]:
            failures.append(f"{label}: selection reason mismatch")
        if bool(retrieval.get("fallback_used")) != expected["fallback"]:
            failures.append(f"{label}: fallback mismatch")
        fallback_count += int(expected["fallback"])
        changed = bool(
            selected is not None
            and expected["baseline"] is not None
            and selected != expected["baseline"]
        )
        if bool(retrieval.get("selection_changed_from_motion_signature")) != changed:
            failures.append(f"{label}: v166 change flag mismatch")
        changed_count += int(changed)
        healthy_changed_count += int(changed and not triggered)
        if selected is not None and expected["newest"] is not None:
            old_recall_count += int(selected != expected["newest"])
            selected_age = int(row["sync_t"]) - int(selected[1])
            selected_ages.append(float(selected_age))
        if selected is None:
            if read_ids:
                failures.append(f"{label}: empty selection read frames")
        elif list(selected) != read_ids:
            failures.append(f"{label}: selected/read mismatch")
        if candidates and (
            selected is None or retrieval.get("read_budget_preserved") is not True
        ):
            read_budget_violation_count += 1
            failures.append(f"{label}: compatible read budget lost")
        reason_counts[str(retrieval.get("selection_reason", "missing"))] += 1

    return {
        "prompt_index": prompt_index,
        "layer": rows[0]["layer"],
        "head": rows[0]["head"],
        "read_count": len(rows),
        "retrieval_count": retrieval_count,
        "archive_size_max": max(archive_sizes, default=0),
        "ready_count": ready_count,
        "triggered_count": triggered_count,
        "healthy_count": healthy_count,
        "multi_candidate_count": multi_candidate_count,
        "changed_from_v166_count": changed_count,
        "healthy_changed_count": healthy_changed_count,
        "old_recall_count": old_recall_count,
        "fallback_count": fallback_count,
        "read_budget_violation_count": read_budget_violation_count,
        "cache_contract_failure_count": cache_contract_failure_count,
        "selected_ages": selected_ages,
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[index],
    }


def aggregate(rows: list[dict], *, method: str) -> dict:
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["reason_counts"])
    result = {
        "method": method,
        "mode": contract.EXPECTED_MODE[method],
        "row_count": len(rows),
        "read_count": sum(row["read_count"] for row in rows),
        "retrieval_count": sum(row["retrieval_count"] for row in rows),
        "archive_size_max": max(row["archive_size_max"] for row in rows),
        "ready_count": sum(row["ready_count"] for row in rows),
        "triggered_count": sum(row["triggered_count"] for row in rows),
        "healthy_count": sum(row["healthy_count"] for row in rows),
        "multi_candidate_count": sum(
            row["multi_candidate_count"] for row in rows
        ),
        "changed_from_v166_count": sum(
            row["changed_from_v166_count"] for row in rows
        ),
        "healthy_changed_count": sum(
            row["healthy_changed_count"] for row in rows
        ),
        "old_recall_count": sum(row["old_recall_count"] for row in rows),
        "fallback_count": sum(row["fallback_count"] for row in rows),
        "read_budget_violation_count": sum(
            row["read_budget_violation_count"] for row in rows
        ),
        "cache_contract_failure_count": sum(
            row["cache_contract_failure_count"] for row in rows
        ),
        "contract_failure_count": sum(len(row["failures"]) for row in rows),
        "selected_age": distribution(
            [value for row in rows for value in row["selected_ages"]]
        ),
        "reason_counts": dict(sorted(reasons.items())),
    }
    result["mechanism_gate"] = bool(
        result["archive_size_max"] >= 2
        and result["triggered_count"] > 0
        and result["healthy_count"] > 0
        and result["multi_candidate_count"] > 0
        and result["changed_from_v166_count"] > 0
        and result["healthy_changed_count"] == 0
        and result["old_recall_count"] > 0
        and result["read_budget_violation_count"] == 0
        and result["cache_contract_failure_count"] == 0
        and result["contract_failure_count"] == 0
    )
    return result


def analyze_method(trace_dir: Path, *, method: str) -> dict:
    paths = sorted(trace_dir.glob(f"{method}__p*.policy.jsonl"))
    if len(paths) != contract.PROMPT_COUNT:
        raise ValueError(
            f"expected {contract.PROMPT_COUNT} traces for {method}, found {len(paths)}"
        )
    layer_rows = {layer: [] for layer in contract.ACTIVE_LAYERS}
    prompts = []
    for path in paths:
        prompt_index = int(path.name.split("__p", 1)[1].split(".", 1)[0])
        by_layer = load_active_layer_rows(path)
        prompt_layers = {}
        for layer, rows in by_layer.items():
            result = analyze_rows(
                rows,
                method=method,
                prompt_index=prompt_index,
            )
            layer_rows[layer].append(result)
            prompt_layers[str(layer)] = result
        prompts.append(
            {
                "prompt_index": prompt_index,
                "trace": str(path.resolve()),
                "layers": prompt_layers,
            }
        )
    if [row["prompt_index"] for row in prompts] != list(
        range(contract.PROMPT_COUNT)
    ):
        raise ValueError(f"prompt coverage mismatch for {method}")
    flattened = [row for layer in contract.ACTIVE_LAYERS for row in layer_rows[layer]]
    return {
        "aggregate": aggregate(flattened, method=method),
        "layers": {
            str(layer): aggregate(rows, method=method)
            for layer, rows in layer_rows.items()
        },
        "prompts": prompts,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v171 Full Active-layer Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        "| Method | Gate | Reads | Triggered | Changed | Healthy changed | Old recalls | Budget/cache errors | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in contract.CANDIDATES:
        row = report["methods"][method]["aggregate"]
        errors = (
            row["read_budget_violation_count"]
            + row["cache_contract_failure_count"]
        )
        lines.append(
            f"| {method} | {row['mechanism_gate']} | {row['read_count']} | "
            f"{row['triggered_count']} | {row['changed_from_v166_count']} | "
            f"{row['healthy_changed_count']} | {row['old_recall_count']} | "
            f"{errors} | {row['contract_failure_count']} |"
        )
    lines.extend(
        [
            "",
            (
                "The gate validates execution and exact selector replay only. "
                "It does not establish quality improvement or a head taxonomy."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = {
        method: analyze_method(args.trace_dir, method=method)
        for method in contract.CANDIDATES
    }
    mechanism_gate = all(
        row["aggregate"]["mechanism_gate"] for row in methods.values()
    )
    report = {
        "version": 1,
        "experiment": "v171_demand_gated_full_layer_trace",
        "mechanism_gate": mechanism_gate,
        "trace_contract": {
            "layers": list(contract.ACTIVE_LAYERS),
            "heads": list(contract.TRACE_HEADS),
            "prompt_count": contract.PROMPT_COUNT,
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    if not mechanism_gate:
        raise SystemExit("v171 full-layer mechanism gate failed")
    print(
        json.dumps(
            {method: row["aggregate"] for method, row in methods.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
