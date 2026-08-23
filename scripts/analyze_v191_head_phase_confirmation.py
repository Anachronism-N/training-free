#!/usr/bin/env python3
"""Confirm the frozen v190 Head x Phase method on unseen MovieGen prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from analyze_v190_head_phase_causal_screen import (
    TEMPORAL_FEATURES,
    dynamic_metric_validity,
    load_temporal_rows,
    temporal_guard,
)
from prepare_v191_head_phase_confirmation import METHODS, PROMPT_COUNT, SEED
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract


CANDIDATE = "head_phase_joint"
LOCAL_CONTROL = "all_recent"
NATIVE_CONTROL = "sf_native"
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)
NONINFERIORITY_MARGINS = {
    LOCAL_CONTROL: {
        "official_quality_score": -0.15,
        "identity_background": -0.001,
        "dynamic_degree": -0.02,
        "temporal_mechanics": -0.002,
    },
    NATIVE_CONTROL: {
        "official_quality_score": -0.20,
        "identity_background": -0.0015,
        "dynamic_degree": -0.02,
        "temporal_mechanics": -0.003,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contrast(
    rows: dict,
    *,
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
    }


def comparison_metrics(
    comparisons: list[dict], candidate: str, control: str
) -> dict[str, dict]:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == candidate and row["control"] == control
    }


def pareto_front(means: dict[str, dict[str, float]]) -> list[str]:
    front = []
    for candidate, row in means.items():
        dominated = False
        for other, other_row in means.items():
            if other == candidate:
                continue
            no_worse = all(
                other_row[metric] >= row[metric] for metric in PRIMARY_METRICS
            )
            strictly_better = any(
                other_row[metric] > row[metric] for metric in PRIMARY_METRICS
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front)


def lower(rows: dict[str, dict], metric: str) -> float:
    return float(rows[metric]["bootstrap_ci95"][0])


def noninferiority_gate(
    rows: dict[str, dict],
    *,
    control: str,
    dynamic_validity: dict,
    means: dict,
) -> dict:
    margins = NONINFERIORITY_MARGINS[control]
    metric_pass = {
        metric: lower(rows, metric) >= margin
        for metric, margin in margins.items()
        if metric != "dynamic_degree"
    }
    if dynamic_validity["informative"]:
        dynamic_pass = lower(rows, "dynamic_degree") >= margins["dynamic_degree"]
        dynamic_rule = "paired_ci_lower_ge_margin"
    elif dynamic_validity["ceiling_nonregression_only"]:
        dynamic_pass = bool(
            means[CANDIDATE]["dynamic_degree"] >= 1.0 - 1e-12
            and means[control]["dynamic_degree"] >= 1.0 - 1e-12
        )
        dynamic_rule = "all_one_ceiling_nonregression"
    else:
        dynamic_pass = False
        dynamic_rule = "constant_non_ceiling_metric_rejected"
    metric_pass["dynamic_degree"] = bool(dynamic_pass)
    return {
        "control": control,
        "margins": margins,
        "metric_pass": metric_pass,
        "dynamic_rule": dynamic_rule,
        "pass": all(metric_pass.values()),
    }


def positive_effect_gate(rows: dict[str, dict], dynamic_validity: dict) -> dict:
    eligible = (
        "official_quality_score",
        "identity_background",
        "temporal_mechanics",
    )
    ci_lower_gt_zero = {metric: lower(rows, metric) > 0.0 for metric in eligible}
    ci_lower_gt_zero["dynamic_degree"] = bool(
        dynamic_validity["informative"]
        and lower(rows, "dynamic_degree") > 0.0
    )
    return {
        "ci_lower_gt_zero": ci_lower_gt_zero,
        "pass": any(ci_lower_gt_zero.values()),
        "dynamic_is_eligible": dynamic_validity["informative"],
    }


def development_sign_consistency(manifest: dict, local: dict[str, dict]) -> dict:
    status = (
        manifest.get("v190_provenance", {})
        .get("selected_status", {})
        .get("baseline_deltas", {})
    )
    metrics = {}
    for metric in PRIMARY_METRICS:
        development = status.get(metric)
        confirmation = local[metric]["mean_delta"]
        metrics[metric] = {
            "development_delta": development,
            "confirmation_delta": confirmation,
            "same_sign": (
                None
                if development is None
                else bool(
                    (float(development) == 0.0 and confirmation == 0.0)
                    or (float(development) > 0.0 and confirmation > 0.0)
                    or (float(development) < 0.0 and confirmation < 0.0)
                )
            ),
        }
    observed = [row["same_sign"] for row in metrics.values() if row["same_sign"] is not None]
    return {
        "metrics": metrics,
        "observed_count": len(observed),
        "consistent_count": sum(bool(value) for value in observed),
        "descriptive_only": True,
    }


def targeted_review(
    manifest: dict,
    rows: dict,
    temporal_rows: dict,
    guards: tuple[dict, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    video_dirs = {str(row["key"]): str(row["video_dir"]) for row in manifest["methods"]}
    flags = {}
    for guard in guards:
        for row in guard.get("flagged_prompts", ()):
            flags.setdefault(int(row["prompt_index"]), set()).update(row["flags"])
    ranked = []
    for prompt in range(PROMPT_COUNT):
        deltas = {
            metric: rows[(CANDIDATE, prompt)][metric]
            - rows[(LOCAL_CONTROL, prompt)][metric]
            for metric in PRIMARY_METRICS
        }
        current = temporal_rows[(CANDIDATE, prompt)]
        recent = temporal_rows[(LOCAL_CONTROL, prompt)]
        temporal_disagreement = abs(
            math.log(
                (current["flow_speed_median"] + 1e-8)
                / (recent["flow_speed_median"] + 1e-8)
            )
        ) + 0.25 * abs(
            math.log(
                (current["temporal_jump"] + 1e-8)
                / (recent["temporal_jump"] + 1e-8)
            )
        )
        priority = (
            10.0 * bool(flags.get(prompt))
            + 100.0 * abs(deltas["identity_background"])
            + 20.0 * abs(deltas["temporal_mechanics"])
            + abs(deltas["dynamic_degree"])
            + 0.05 * abs(deltas["official_quality_score"])
            + temporal_disagreement
        )
        ranked.append((priority, prompt, deltas, temporal_disagreement))
    queue = []
    for priority, prompt, deltas, disagreement in sorted(ranked, reverse=True)[:limit]:
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "priority": float(priority),
                "deltas_vs_all_recent": deltas,
                "temporal_disagreement": float(disagreement),
                "automatic_flags": sorted(flags.get(prompt, ())),
                "videos": {
                    method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return queue


def analyze(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    *,
    temporal_rows: dict[tuple[str, int], dict[str, float]],
) -> dict:
    methods = tuple(str(row.get("key")) for row in manifest.get("methods") or ())
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != "v191_unseen128_head_phase_vbench"
        or manifest.get("confirmatory") is not True
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("seed", -1)) != SEED
        or len(prompt_items) != PROMPT_COUNT
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != list(range(128, 256))
        or tuple(summary.get("methods") or {}) != METHODS
        or summary.get("missing")
    ):
        raise ValueError("v191 analysis requires the complete frozen unseen128 scope")

    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    dynamic_validity = dynamic_metric_validity(
        rows, methods=METHODS, prompt_count=PROMPT_COUNT
    )
    pairs = (
        (CANDIDATE, LOCAL_CONTROL, "primary_equal_budget"),
        (CANDIDATE, NATIVE_CONTROL, "external_native_reference"),
        (LOCAL_CONTROL, NATIVE_CONTROL, "cache_runtime_context"),
    )
    comparisons = []
    for pair_index, (candidate, control, role) in enumerate(pairs):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                contrast(
                    rows,
                    candidate=candidate,
                    control=control,
                    role=role,
                    metric=metric,
                    seed=1912026 + pair_index * 101 + metric_index,
                )
            )
    base.bh(comparisons)

    means = {
        method: {
            metric: float(
                np.mean([rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)])
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }
    front = pareto_front(
        {
            method: {metric: means[method][metric] for metric in PRIMARY_METRICS}
            for method in METHODS
        }
    )
    local = comparison_metrics(comparisons, CANDIDATE, LOCAL_CONTROL)
    native = comparison_metrics(comparisons, CANDIDATE, NATIVE_CONTROL)
    local_noninferiority = noninferiority_gate(
        local,
        control=LOCAL_CONTROL,
        dynamic_validity=dynamic_validity,
        means=means,
    )
    native_noninferiority = noninferiority_gate(
        native,
        control=NATIVE_CONTROL,
        dynamic_validity=dynamic_validity,
        means=means,
    )
    positive = positive_effect_gate(local, dynamic_validity)
    recent_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=LOCAL_CONTROL,
        prompt_count=PROMPT_COUNT,
    )
    native_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=NATIVE_CONTROL,
        prompt_count=PROMPT_COUNT,
    )
    gates = {
        "equal_budget_noninferiority": local_noninferiority["pass"],
        "equal_budget_positive_effect": positive["pass"],
        "native_noninferiority": native_noninferiority["pass"],
        "candidate_on_primary_pareto_front": CANDIDATE in front,
        "temporal_safety_vs_equal_budget": recent_guard["automatic_safety_pass"],
        "temporal_safety_vs_native": native_guard["automatic_safety_pass"],
    }
    confirmed = all(gates.values())
    recommendation = (
        "freeze_head_phase_method_for_seed_length_and_cross_model_replication"
        if confirmed
        else "do_not_freeze_v191_head_phase_method"
    )
    motion_claim = bool(
        dynamic_validity["informative"]
        and lower(local, "dynamic_degree") > 0.0
    )
    queue = (
        targeted_review(
            manifest,
            rows,
            temporal_rows,
            (recent_guard, native_guard),
        )
        if confirmed
        else []
    )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "confirmatory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_source_index_range": [128, 255],
        "seed": SEED,
        "selected_v190_method": manifest["selected_v190_method"],
        "selected_operator": manifest["selected_operator"],
        "methods": list(METHODS),
        "method_means": means,
        "comparisons": comparisons,
        "primary_pareto_front": front,
        "metric_validity": {"dynamic_degree": dynamic_validity},
        "equal_budget_noninferiority": local_noninferiority,
        "native_noninferiority": native_noninferiority,
        "positive_effect_vs_equal_budget": positive,
        "automatic_temporal_guards": {
            LOCAL_CONTROL: recent_guard,
            NATIVE_CONTROL: native_guard,
        },
        "confirmation_gates": gates,
        "head_phase_effect_confirmed": confirmed,
        "motion_improvement_claim_supported": motion_claim,
        "development_sign_consistency": development_sign_consistency(
            manifest, local
        ),
        "candidate_delta_vs_all_recent": {
            metric: local[metric]["mean_delta"] for metric in PRIMARY_METRICS
        },
        "candidate_delta_vs_sf_native": {
            metric: native[metric]["mean_delta"] for metric in PRIMARY_METRICS
        },
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": confirmed,
        "targeted_review_queue": queue,
        "frozen_gate_note": (
            "The confidence-bound non-inferiority margins and positive-effect rule "
            "were fixed before v191 generation. They are development tolerances, "
            "not universal equivalence margins."
        ),
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v191 Unseen-128 Head x Phase Confirmation",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Confirmed: `{report['head_phase_effect_confirmed']}`",
        f"- Operator: `{report['selected_operator']}`",
        f"- Dynamic Degree informative: "
        f"`{report['metric_validity']['dynamic_degree']['informative']}`",
        f"- Motion-improvement claim supported: "
        f"`{report['motion_improvement_claim_supported']}`",
        f"- Manual review videos: `{len(report['targeted_review_queue'])}`",
        "",
        "| Contrast | dQuality | dIdentity | dDynamic | dTemporal |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Joint - all-Recent", "candidate_delta_vs_all_recent"),
        ("Joint - SF native", "candidate_delta_vs_sf_native"),
    ):
        row = report[key]
        lines.append(
            f"| {label} | {row['official_quality_score']:.6f} | "
            f"{row['identity_background']:.6f} | {row['dynamic_degree']:.6f} | "
            f"{row['temporal_mechanics']:.6f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verify_temporal_contract(
        args.temporal_contract,
        manifest_path,
        args.temporal_csv,
    )
    temporal_rows = load_temporal_rows(
        args.temporal_csv,
        methods=METHODS,
        prompt_count=PROMPT_COUNT,
    )
    report = analyze(
        manifest,
        summary,
        args.parts_root,
        temporal_rows=temporal_rows,
    )
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "vbench_summary": str(args.summary.resolve()),
        "vbench_summary_sha256": sha256(args.summary),
        "temporal_diagnostics": str(args.temporal_csv.resolve()),
        "temporal_diagnostics_sha256": sha256(args.temporal_csv),
        "temporal_contract": str(args.temporal_contract.resolve()),
        "temporal_contract_sha256": sha256(args.temporal_contract),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v191-analysis] "
        f"recommendation={report['recommendation']} "
        f"confirmed={str(report['head_phase_effect_confirmed']).lower()}"
    )


if __name__ == "__main__":
    main()
