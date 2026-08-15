#!/usr/bin/env python3
"""Paired VBench-Long analysis for the v184 denoising-phase screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from prepare_v184_denoise_phase_screen import METHODS, PROMPT_COUNT


CONTROL = "all_recent"
PHASE_CANDIDATES = (
    "coverage_early1",
    "coverage_early2",
    "coverage_late2",
)
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)


def pareto_front(means: dict[str, dict[str, float]]) -> list[str]:
    metrics = PRIMARY_METRICS
    front = []
    for candidate, row in means.items():
        dominated = False
        for other, other_row in means.items():
            if other == candidate:
                continue
            weakly_better = all(other_row[metric] >= row[metric] for metric in metrics)
            strictly_better = any(other_row[metric] > row[metric] for metric in metrics)
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


def _targeted_review(
    manifest: dict,
    rows: dict,
    promoted: list[str],
    limit: int = 4,
) -> list[dict]:
    candidates = promoted or list(PHASE_CANDIDATES)
    video_dirs = {
        row["key"]: row["video_dir"] for row in manifest["methods"]
    }
    queue = []
    for candidate in candidates:
        for prompt in range(PROMPT_COUNT):
            identity = (
                rows[(candidate, prompt)]["identity_background"]
                - rows[(CONTROL, prompt)]["identity_background"]
            )
            dynamic = (
                rows[(candidate, prompt)]["dynamic_degree"]
                - rows[(CONTROL, prompt)]["dynamic_degree"]
            )
            quality = (
                rows[(candidate, prompt)]["official_quality_score"]
                - rows[(CONTROL, prompt)]["official_quality_score"]
            )
            temporal = (
                rows[(candidate, prompt)]["temporal_mechanics"]
                - rows[(CONTROL, prompt)]["temporal_mechanics"]
            )
            conflict = dynamic > 0.0 and (identity < 0.0 or temporal < 0.0)
            if not conflict:
                continue
            priority = abs(dynamic) + 20.0 * abs(identity) + 0.1 * abs(quality)
            item = manifest["prompt_items"][prompt]
            queue.append(
                {
                    "candidate": candidate,
                    "prompt_index": prompt,
                    "source_index": int(item["source_index"]),
                    "prompt": item["text"],
                    "identity_delta": float(identity),
                    "dynamic_delta": float(dynamic),
                    "quality_delta": float(quality),
                    "temporal_delta": float(temporal),
                    "priority": float(priority),
                    "videos": {
                        method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                        for method in (CONTROL, candidate)
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
        != "v184_denoise_phase_coverage_vbench_screen32"
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(manifest.get("prompt_items") or ()) != PROMPT_COUNT
        or tuple(summary.get("methods") or {}) != METHODS
        or summary.get("missing")
    ):
        raise ValueError("v184 paired analysis requires a complete frozen screen")
    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    comparisons = []
    pairs = [
        (candidate, CONTROL, "schedule_vs_recent")
        for candidate in METHODS
        if candidate != CONTROL
    ] + [
        ("coverage_early2", "coverage_late2", "early_vs_late_equal_dose"),
        ("coverage_early2", "coverage_early1", "early_dose_increment"),
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
                    seed=1842026 + pair_index * 101 + metric_index,
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
            method: {metric: method_means[method][metric] for metric in PRIMARY_METRICS}
            for method in METHODS
        }
    )
    statuses = {}
    promoted = []
    for candidate in PHASE_CANDIDATES:
        metrics = _comparison_rows(comparisons, candidate, CONTROL)
        directional_gate = bool(
            metrics["official_quality_score"]["mean_delta"] >= 0.0
            and metrics["identity_background"]["mean_delta"] >= -0.001
            and metrics["dynamic_degree"]["mean_delta"] >= 0.02
            and metrics["temporal_mechanics"]["mean_delta"] >= -0.002
        )
        on_front = candidate in front
        promote = directional_gate and on_front
        if promote:
            promoted.append(candidate)
        statuses[candidate] = {
            "on_primary_pareto_front": on_front,
            "directional_gate": directional_gate,
            "promote_to_operator_screen": promote,
            "deltas_vs_recent": {
                metric: metrics[metric]["mean_delta"] for metric in PRIMARY_METRICS
            },
        }

    all_coverage = _comparison_rows(comparisons, "all_coverage_noisy", CONTROL)
    phase_has_motion = any(
        statuses[candidate]["deltas_vs_recent"]["dynamic_degree"] >= 0.02
        for candidate in PHASE_CANDIDATES
    )
    coverage_is_actuator = bool(
        all_coverage["dynamic_degree"]["mean_delta"] >= 0.02
    )
    if promoted:
        recommendation = "advance_phase_schedule_to_operator_screen"
    elif coverage_is_actuator and phase_has_motion:
        recommendation = "design_online_motion_deficit_gate"
    elif coverage_is_actuator:
        recommendation = "phase_sparsification_failed_revisit_operator"
    else:
        recommendation = "stop_coverage_schedule"

    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "methods": list(METHODS),
        "method_means": method_means,
        "comparisons": comparisons,
        "primary_pareto_front": front,
        "candidate_status": statuses,
        "promoted_to_operator_screen": promoted,
        "recommendation": recommendation,
        "all_coverage_is_motion_actuator": coverage_is_actuator,
        "manual_review_required_for_recommendation": False,
        "targeted_review_queue": _targeted_review(
            manifest,
            rows,
            promoted,
        ),
        "claim_boundary": (
            "The screen can select a denoising schedule for a controlled "
            "deterministic-operator screen. It cannot establish a final benchmark "
            "result, a universal timestep mechanism, or static head specialization."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v184 Denoising-Phase Coverage Screen",
        "",
        f"Recommendation: `{report['recommendation']}`",
        "Promoted: " + (", ".join(report["promoted_to_operator_screen"]) or "none"),
        f"Primary Pareto front: {', '.join(report['primary_pareto_front'])}",
        "",
        "| Candidate | dQuality | dIdentity | dDynamic | dTemporal | Pareto | Promote |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, status in report["candidate_status"].items():
        delta = status["deltas_vs_recent"]
        lines.append(
            f"| {candidate} | {delta['official_quality_score']:.6f} | "
            f"{delta['identity_background']:.6f} | {delta['dynamic_degree']:.6f} | "
            f"{delta['temporal_mechanics']:.6f} | "
            f"{status['on_primary_pareto_front']} | {status['promote_to_operator_screen']} |"
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
        f"[v184-decision] {report['recommendation']} "
        f"promoted={','.join(report['promoted_to_operator_screen']) or 'none'}"
    )


if __name__ == "__main__":
    main()
