#!/usr/bin/env python3
"""Replay v169 selectors on frozen v166 MultiScaleMotion traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common
import analyze_v168_cross_scale_consensus_trace as v168
import v169_soft_cross_scale_contract as contract


SOURCE_METHOD = "ours_middle10_reservoir2_multiscalemotion1"
PROMPT_COUNT = 16


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=(
            root / "runs" / "v166_multiscale_motion_moviebench16" / "full8" / "traces"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "runs"
            / "v169_soft_cross_scale_moviebench16"
            / "offline_counterfactual.json"
        ),
    )
    return parser.parse_args()


def analyze_method(paths: list[Path], *, method: str) -> dict:
    totals: Counter[str] = Counter()
    selected_ages: list[float] = []
    prompts = []
    for path in paths:
        counts: Counter[str] = Counter()
        prompt_ages: list[float] = []
        for row in common.load_representative(path):
            retrieval = row["strategy"].get("state", {}).get("last_retrieval", {})
            if not retrieval:
                continue
            candidates = list(retrieval.get("candidates", []))
            expected = contract.expected_selection(candidates, method=method)
            source_selected = v168.first_pair(retrieval.get("selected", []))
            counts["retrievals"] += 1
            counts["passing_decisions"] += int(bool(expected["rows"]))
            counts["multi_candidate"] += int(len(expected["rows"]) >= 2)
            counts["fallback"] += int(expected["fallback"])
            counts["changed_from_v166"] += int(
                source_selected is not None and expected["selected"] != source_selected
            )
            counts["old_recall"] += int(
                expected["selected"] is not None
                and expected["newest"] is not None
                and expected["selected"] != expected["newest"]
            )
            v168_expected = v168.expected_selection(
                candidates,
                method=v168.PARETO_MOTION,
            )
            changed = bool(
                source_selected is not None and expected["selected"] != source_selected
            )
            counts["changed_on_scale_conflict"] += int(
                changed and v168_expected["conflict"]
            )
            counts["changed_on_scale_agreement"] += int(
                changed and v168_expected["agreement"] is True
            )
            if expected["selected"] is not None:
                age = int(row["sync_t"]) - int(expected["selected"][1])
                prompt_ages.append(float(age))
                selected_ages.append(float(age))
        totals.update(counts)
        prompts.append(
            {
                "prompt_index": common.prompt_index(path),
                "counts": dict(sorted(counts.items())),
                "selected_age": common.distribution(prompt_ages),
            }
        )
    passing = totals["passing_decisions"]
    change_rate = totals["changed_from_v166"] / max(passing, 1)
    gate = bool(
        totals["retrievals"] > 0
        and totals["multi_candidate"] > 0
        and totals["changed_from_v166"] > 0
        and totals["old_recall"] > 0
        and totals["changed_on_scale_conflict"] > 0
        and change_rate <= 0.20
    )
    return {
        "method": method,
        "mode": contract.EXPECTED_MODE[method],
        "counts": dict(sorted(totals.items())),
        "change_rate_among_passing": change_rate,
        "selected_age": common.distribution(selected_ages),
        "controlled_change_gate": gate,
        "prompts": prompts,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v169 Offline Counterfactual",
        "",
        f"Controlled-change gate: **{report['controlled_change_gate']}**",
        "",
        "| Method | Passing | Changed vs v166 | Change rate | Old recall | "
        "Conflict changes | Agreement changes | Age median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in contract.METHODS:
        row = report["methods"][method]
        counts = row["counts"]
        lines.append(
            f"| {method} | {counts.get('passing_decisions', 0)} | "
            f"{counts.get('changed_from_v166', 0)} | "
            f"{row['change_rate_among_passing']:.4f} | "
            f"{counts.get('old_recall', 0)} | "
            f"{counts.get('changed_on_scale_conflict', 0)} | "
            f"{counts.get('changed_on_scale_agreement', 0)} | "
            f"{row['selected_age'].get('median')} |"
        )
    lines.extend(
        [
            "",
            "The gate only proves that both selectors make bounded, genuine "
            "changes while preserving old-history reads. It does not predict "
            "video quality.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = sorted(args.trace_dir.glob(f"{SOURCE_METHOD}__p*.policy.jsonl"))
    if len(paths) != PROMPT_COUNT:
        raise ValueError(
            f"expected {PROMPT_COUNT} frozen v166 traces, found {len(paths)}"
        )
    if [common.prompt_index(path) for path in paths] != list(range(PROMPT_COUNT)):
        raise ValueError("v166 trace prompt coverage mismatch")
    methods = {
        method: analyze_method(paths, method=method) for method in contract.METHODS
    }
    gate = all(row["controlled_change_gate"] for row in methods.values())
    report = {
        "version": 1,
        "experiment": "v169_offline_counterfactual",
        "source_method": SOURCE_METHOD,
        "source_trace_dir": str(args.trace_dir.resolve()),
        "prompt_count": PROMPT_COUNT,
        "methods": methods,
        "controlled_change_gate": gate,
        "claim_boundary": (
            "counterfactual selection coverage only; no video-quality claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    if not gate:
        raise SystemExit("v169 offline controlled-change gate failed")
    print(
        json.dumps(
            {method: row["counts"] for method, row in methods.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
