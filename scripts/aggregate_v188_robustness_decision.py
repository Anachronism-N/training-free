#!/usr/bin/env python3
"""Aggregate the three frozen v188 scopes without selecting by visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "replica64_seed20000": "replication_confirmed",
    "long60_seed10000_32": "long_horizon_confirmed",
    "mechanism32_seed10000": "phase_specificity_supported",
}


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope = str(payload.get("scope", ""))
    if (
        scope not in EXPECTED
        or payload.get("confirmatory_extension") is not True
        or EXPECTED[scope] not in payload
    ):
        raise ValueError(f"invalid v188 scope analysis: {path}")
    return payload


def aggregate(reports: list[dict]) -> dict:
    by_scope = {str(report["scope"]): report for report in reports}
    if set(by_scope) != set(EXPECTED) or len(reports) != len(EXPECTED):
        raise ValueError("v188 aggregate requires exactly one report per frozen scope")
    schedules = {report["selected_schedule"] for report in reports}
    operators = {report["selected_operator"] for report in reports}
    if len(schedules) != 1 or len(operators) != 1:
        raise ValueError("v188 scope reports mix schedule/operator contracts")
    gates = {
        scope: bool(by_scope[scope][gate]) for scope, gate in EXPECTED.items()
    }
    if all(gates.values()):
        recommendation = "advance_phase_structured_memory_to_cross_model"
    elif gates["replica64_seed20000"] and gates["long60_seed10000_32"]:
        recommendation = "retain_effect_drop_phase_mechanism_claim"
    elif gates["replica64_seed20000"]:
        recommendation = "retain_30s_effect_continue_long_horizon_optimization"
    else:
        recommendation = "stop_frozen_phase_structured_memory_method"

    queue = []
    seen_sources = set()
    for scope in EXPECTED:
        for row in by_scope[scope].get("targeted_review_queue") or ():
            source = int(row["source_index"])
            if source in seen_sources:
                continue
            seen_sources.add(source)
            queue.append({"scope": scope, **row})
    queue.sort(key=lambda row: (-float(row.get("priority", 0.0)), row["source_index"]))
    review_required = recommendation == "advance_phase_structured_memory_to_cross_model"
    return {
        "version": 1,
        "experiment": "v188_post_confirmation_robustness_matrix",
        "selected_schedule": schedules.pop(),
        "selected_operator": operators.pop(),
        "scope_gates": gates,
        "recommendation": recommendation,
        "manual_review_required": review_required,
        "targeted_review_queue": queue[:6] if review_required else [],
        "manual_review_cap": 6,
        "next_step": (
            "Port the frozen schedule/operator contract to a second causal video "
            "diffusion backbone before adding ABA or further cache tricks."
            if recommendation == "advance_phase_structured_memory_to_cross_model"
            else "Follow the recommendation literally; do not rescue a failed gate by cherry-picking videos."
        ),
        "claim_boundary": (
            "A positive v188 decision establishes single-model seed and duration "
            "robustness plus phase attribution. Cross-model and prompt-switch claims "
            "remain untested."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v188 Robustness Decision",
        "",
        f"Recommendation: `{report['recommendation']}`",
        f"Schedule/operator: `{report['selected_schedule']}` / `{report['selected_operator']}`",
        "",
        "| Scope | Passed |",
        "|---|---:|",
    ]
    for scope, passed in report["scope_gates"].items():
        lines.append(f"| {scope} | {passed} |")
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--long60", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate([load(args.replica), load(args.long60), load(args.mechanism)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v188-decision] {report['recommendation']}")


if __name__ == "__main__":
    main()
