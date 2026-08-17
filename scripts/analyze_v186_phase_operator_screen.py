#!/usr/bin/env python3
"""Paired VBench-Long analysis for the v186 operator screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from prepare_v186_phase_operator_screen import (
    GENERATED_METHODS,
    METHODS,
    PROMPT_COUNT,
    STORAGE_FFE,
)


LOCAL_CONTROL = "all_recent"
RANDOM_REFERENCE = "phase_reservoir"
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)


def pareto_front(means: dict[str, dict[str, float]]) -> list[str]:
    front = []
    for candidate, row in means.items():
        dominated = False
        for other, other_row in means.items():
            if other == candidate:
                continue
            weakly_better = all(
                other_row[metric] >= row[metric] for metric in PRIMARY_METRICS
            )
            strictly_better = any(
                other_row[metric] > row[metric] for metric in PRIMARY_METRICS
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front)


def _contrast(
    rows: dict,
    candidate: str,
    control: str,
    role: str,
    metric: str,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(PROMPT_COUNT)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "comparison_role": role,
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
        "inferential_role": "development_only",
    }


def _comparison_rows(comparisons: list[dict], candidate: str, control: str) -> dict:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == candidate and row["control"] == control
    }


def select_candidate(promoted: list[str], statuses: dict[str, dict]) -> str | None:
    if not promoted:
        return None
    return min(
        promoted,
        key=lambda candidate: (
            STORAGE_FFE[candidate],
            -statuses[candidate]["deltas_vs_reservoir"]["identity_background"],
            -statuses[candidate]["deltas_vs_reservoir"]["temporal_mechanics"],
            -statuses[candidate]["deltas_vs_reservoir"]["official_quality_score"],
            -statuses[candidate]["deltas_vs_reservoir"]["dynamic_degree"],
            candidate,
        ),
    )


def _targeted_review(
    manifest: dict,
    rows: dict,
    selected: str | None,
    limit: int = 4,
) -> list[dict]:
    candidates = [selected] if selected else list(GENERATED_METHODS)
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    queue = []
    for candidate in candidates:
        for prompt in range(PROMPT_COUNT):
            identity = (
                rows[(candidate, prompt)]["identity_background"]
                - rows[(RANDOM_REFERENCE, prompt)]["identity_background"]
            )
            dynamic = (
                rows[(candidate, prompt)]["dynamic_degree"]
                - rows[(RANDOM_REFERENCE, prompt)]["dynamic_degree"]
            )
            quality = (
                rows[(candidate, prompt)]["official_quality_score"]
                - rows[(RANDOM_REFERENCE, prompt)]["official_quality_score"]
            )
            temporal = (
                rows[(candidate, prompt)]["temporal_mechanics"]
                - rows[(RANDOM_REFERENCE, prompt)]["temporal_mechanics"]
            )
            conflict = (
                (identity > 0.0 and dynamic < 0.0)
                or (dynamic > 0.0 and identity < 0.0)
                or abs(quality) >= 0.5
            )
            if not conflict:
                continue
            priority = 25.0 * abs(identity) + abs(dynamic) + 0.1 * abs(quality)
            item = manifest["prompt_items"][prompt]
            queue.append(
                {
                    "candidate": candidate,
                    "prompt_index": prompt,
                    "source_index": int(item["source_index"]),
                    "prompt": item["text"],
                    "identity_delta_vs_reservoir": float(identity),
                    "dynamic_delta_vs_reservoir": float(dynamic),
                    "quality_delta_vs_reservoir": float(quality),
                    "temporal_delta_vs_reservoir": float(temporal),
                    "priority": float(priority),
                    "videos": {
                        method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                        for method in (LOCAL_CONTROL, RANDOM_REFERENCE, candidate)
                    },
                }
            )
    return sorted(
        queue,
        key=lambda row: (-row["priority"], row["candidate"], row["prompt_index"]),
    )[:limit]


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row.get("key") for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment")
        != "v186_phase_conditioned_operator_vbench_screen32"
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(manifest.get("prompt_items") or ()) != PROMPT_COUNT
        or tuple(summary.get("methods") or {}) != METHODS
        or summary.get("missing")
    ):
        raise ValueError("v186 paired analysis requires a complete frozen screen")
    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    comparisons = []
    pairs = [
        (method, LOCAL_CONTROL, "operator_schedule_vs_recent")
        for method in METHODS
        if method != LOCAL_CONTROL
    ] + [
        (method, RANDOM_REFERENCE, "deterministic_vs_random_operator")
        for method in GENERATED_METHODS
    ]
    for pair_index, (candidate, control, role) in enumerate(pairs):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                _contrast(
                    rows,
                    candidate,
                    control,
                    role,
                    metric,
                    seed=1862026 + pair_index * 101 + metric_index,
                )
            )
    base.bh(comparisons)

    method_means = {
        method: {
            metric: float(
                np.mean(
                    [rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)]
                )
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }
    front = pareto_front(
        {
            method: {
                metric: method_means[method][metric] for metric in PRIMARY_METRICS
            }
            for method in METHODS
        }
    )
    statuses = {}
    promoted = []
    for candidate in GENERATED_METHODS:
        recent = _comparison_rows(comparisons, candidate, LOCAL_CONTROL)
        reservoir = _comparison_rows(comparisons, candidate, RANDOM_REFERENCE)
        preserves_v184_actuation = bool(
            recent["official_quality_score"]["mean_delta"] >= 0.0
            and recent["identity_background"]["mean_delta"] >= -0.001
            and recent["dynamic_degree"]["mean_delta"] >= 0.02
            and recent["temporal_mechanics"]["mean_delta"] >= -0.002
        )
        noninferior_to_reservoir = bool(
            reservoir["official_quality_score"]["mean_delta"] >= -0.10
            and reservoir["identity_background"]["mean_delta"] >= -0.0005
            and reservoir["dynamic_degree"]["mean_delta"] >= -0.02
            and reservoir["temporal_mechanics"]["mean_delta"] >= -0.001
        )
        improves_explanatory_axis = bool(
            reservoir["identity_background"]["mean_delta"] >= 0.0005
            or reservoir["temporal_mechanics"]["mean_delta"] >= 0.001
            or reservoir["official_quality_score"]["mean_delta"] >= 0.10
        )
        on_front = candidate in front
        promote = bool(
            preserves_v184_actuation
            and noninferior_to_reservoir
            and improves_explanatory_axis
            and on_front
        )
        if promote:
            promoted.append(candidate)
        statuses[candidate] = {
            "on_primary_pareto_front": on_front,
            "preserves_v184_actuation": preserves_v184_actuation,
            "noninferior_to_reservoir": noninferior_to_reservoir,
            "improves_explanatory_axis": improves_explanatory_axis,
            "promote_to_fresh128": promote,
            "middle_storage_capacity": STORAGE_FFE[candidate],
            "deltas_vs_recent": {
                metric: recent[metric]["mean_delta"] for metric in PRIMARY_METRICS
            },
            "deltas_vs_reservoir": {
                metric: reservoir[metric]["mean_delta"] for metric in PRIMARY_METRICS
            },
        }

    selected = select_candidate(promoted, statuses)
    if selected is not None:
        recommendation = "advance_deterministic_operator_to_fresh128"
    else:
        recommendation = "no_deterministic_operator_advance"
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "selected_schedule": manifest["selected_schedule"],
        "methods": list(METHODS),
        "method_means": method_means,
        "comparisons": comparisons,
        "primary_pareto_front": front,
        "candidate_status": statuses,
        "promoted_to_fresh128": promoted,
        "selected_for_fresh128": selected,
        "selection_rule": (
            "Require preservation versus all-Recent and non-inferiority plus "
            "one explanatory-axis improvement versus Reservoir. Prefer equal "
            "four-frame storage, then identity, temporal, quality, and dynamic."
        ),
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": False,
        "targeted_review_queue": _targeted_review(manifest, rows, selected),
        "claim_boundary": (
            "The development32 screen may select one operator for a fresh "
            "confirmatory benchmark. It cannot establish final superiority, "
            "cross-model transfer, or head specialization."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v186 Phase-Conditioned Coverage Operator Screen",
        "",
        f"Schedule: `{report['selected_schedule']}`",
        f"Recommendation: `{report['recommendation']}`",
        "Promoted: " + (", ".join(report["promoted_to_fresh128"]) or "none"),
        "Selected: " + (report["selected_for_fresh128"] or "none"),
        "",
        "| Candidate | dQ/R | dID/R | dDyn/R | dTemp/R | Storage | Pareto | Promote |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, status in report["candidate_status"].items():
        delta = status["deltas_vs_reservoir"]
        lines.append(
            f"| {candidate} | {delta['official_quality_score']:.6f} | "
            f"{delta['identity_background']:.6f} | {delta['dynamic_degree']:.6f} | "
            f"{delta['temporal_mechanics']:.6f} | "
            f"{status['middle_storage_capacity']} | "
            f"{status['on_primary_pareto_front']} | "
            f"{status['promote_to_fresh128']} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.comparison_root / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(manifest, summary, args.parts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        f"[v186-decision] {report['recommendation']} "
        f"selected={report['selected_for_fresh128'] or 'none'}"
    )


if __name__ == "__main__":
    main()
