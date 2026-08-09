#!/usr/bin/env python3
"""Replay v168 selectors on frozen v166 MultiScaleMotion traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import analyze_v164_direction_freshness_trace as common
import analyze_v168_cross_scale_consensus_trace as v168


SOURCE_METHOD = "ours_middle10_reservoir2_multiscalemotion1"
PROMPT_COUNT = 16


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=(
            root
            / "runs"
            / "v166_multiscale_motion_moviebench16"
            / "full8"
            / "traces"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "runs"
            / "v168_cross_scale_consensus_moviebench16"
            / "offline_counterfactual.json"
        ),
    )
    return parser.parse_args()


def analyze_method(paths: list[Path], *, method: str) -> dict:
    totals: Counter[str] = Counter()
    prompts = []
    for path in paths:
        counts: Counter[str] = Counter()
        for row in common.load_representative(path):
            retrieval = (
                row["strategy"].get("state", {}).get("last_retrieval", {})
            )
            if not retrieval:
                continue
            candidates = list(retrieval.get("candidates", []))
            expected = v168.expected_selection(candidates, method=method)
            source_selected = v168.first_pair(retrieval.get("selected", []))
            counts["retrievals"] += 1
            counts["multi_candidate"] += int(len(candidates) >= 2)
            counts["old_recall"] += int(
                expected["selected"] is not None
                and expected["newest"] is not None
                and expected["selected"] != expected["newest"]
            )
            counts["changed_from_v166"] += int(
                source_selected is not None
                and expected["selected"] != source_selected
            )
            counts["pareto_old_accept"] += int(
                expected["pareto_pass"] is True
                and expected["motion"] is not None
                and expected["motion"] != expected["newest"]
            )
            counts["pareto_reject"] += int(
                expected["pareto_pass"] is False
            )
            counts["scale_agreement"] += int(
                expected["agreement"] is True
            )
            counts["scale_conflict"] += int(expected["conflict"])
            counts["fallback"] += int(expected["fallback"])
        totals.update(counts)
        prompts.append(
            {
                "prompt_index": common.prompt_index(path),
                "counts": dict(sorted(counts.items())),
            }
        )
    branch_gate = bool(
        totals["retrievals"] > 0
        and totals["multi_candidate"] > 0
        and totals["changed_from_v166"] > 0
        and totals["old_recall"] > 0
        and totals["pareto_old_accept"] > 0
        and totals["pareto_reject"] > 0
        and totals["scale_agreement"] > 0
        and totals["scale_conflict"] > 0
    )
    return {
        "method": method,
        "counts": dict(sorted(totals.items())),
        "branch_coverage_gate": branch_gate,
        "prompts": prompts,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v168 Offline Counterfactual",
        "",
        f"Branch coverage gate: **{report['branch_coverage_gate']}**",
        "",
        "| Method | Old recall | Changed vs v166 | Pareto accept | "
        "Pareto reject | Scale agreement | Scale conflict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in v168.METHODS:
        counts = report["methods"][method]["counts"]
        lines.append(
            f"| {method} | {counts.get('old_recall', 0)} | "
            f"{counts.get('changed_from_v166', 0)} | "
            f"{counts.get('pareto_old_accept', 0)} | "
            f"{counts.get('pareto_reject', 0)} | "
            f"{counts.get('scale_agreement', 0)} | "
            f"{counts.get('scale_conflict', 0)} |"
        )
    lines.extend(
        [
            "",
            "This is deterministic branch-coverage evidence from frozen "
            "v166 traces. It does not estimate v168 video quality.",
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
        method: analyze_method(paths, method=method) for method in v168.METHODS
    }
    gate = all(row["branch_coverage_gate"] for row in methods.values())
    report = {
        "version": 1,
        "experiment": "v168_offline_counterfactual",
        "source_method": SOURCE_METHOD,
        "source_trace_dir": str(args.trace_dir.resolve()),
        "prompt_count": PROMPT_COUNT,
        "methods": methods,
        "branch_coverage_gate": gate,
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
        raise SystemExit("v168 offline branch coverage gate failed")
    print(
        json.dumps(
            {
                method: row["counts"] for method, row in methods.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
