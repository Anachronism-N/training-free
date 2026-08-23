#!/usr/bin/env python3
"""Analyze v192 seed replication, 60-second persistence, and final gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from analyze_v190_head_phase_causal_screen import (
    dynamic_metric_validity,
    load_temporal_rows,
    temporal_guard,
)
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v192_head_phase_robustness import (
    CANDIDATE,
    LOCAL_CONTROL,
    METHODS,
    SCOPE_SPECS,
    verify as verify_input,
)
from prepare_v192_vbench_comparison import EXPERIMENT


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
PERSISTENCE_MARGINS = {
    "official_quality_score": -0.25,
    "identity_background": -0.0025,
    "dynamic_degree": -0.04,
    "temporal_mechanics": -0.004,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    role: str,
    metric: str,
    window: str,
    prompt_count: int,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(prompt_count)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "comparison_role": role,
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "window": window,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def _rows_by_metric(
    comparisons: list[dict], candidate: str, control: str, window: str
) -> dict[str, dict]:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == candidate
        and row["control"] == control
        and row["window"] == window
    }


def _lower(rows: dict[str, dict], metric: str) -> float:
    return float(rows[metric]["bootstrap_ci95"][0])


def _method_means(rows: dict, prompt_count: int) -> dict:
    return {
        method: {
            metric: float(
                np.mean([rows[(method, prompt)][metric] for prompt in range(prompt_count)])
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }


def noninferiority_gate(
    rows: dict[str, dict],
    *,
    control: str,
    dynamic_validity: dict,
    means: dict,
) -> dict:
    margins = NONINFERIORITY_MARGINS[control]
    metric_pass = {
        metric: _lower(rows, metric) >= margin
        for metric, margin in margins.items()
        if metric != "dynamic_degree"
    }
    if dynamic_validity["informative"]:
        dynamic_pass = _lower(rows, "dynamic_degree") >= margins["dynamic_degree"]
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


def _load_window_rows(
    parts_root: Path,
    summary: dict,
    prompt_count: int,
    *,
    clips_per_video: int,
    start: int,
    end: int,
) -> dict:
    if not 0 <= start < end <= clips_per_video:
        raise ValueError(f"invalid v192 clip window [{start}, {end})")
    rows = {
        (method, prompt): {} for method in METHODS for prompt in range(prompt_count)
    }
    for method in METHODS:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=prompt_count,
                clips_per_video=clips_per_video,
            )
            raw_values = [
                value for prompt in range(prompt_count) for value in clips[prompt]
            ]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(raw_values)),
                summary_value,
                name=f"{method}:{dimension}",
            )
            for prompt in range(prompt_count):
                rows[(method, prompt)][dimension] = factor * float(
                    np.mean(clips[prompt][start:end])
                )
    return base.derived_rows(rows, METHODS, prompt_count)


def _load_scope_rows(
    scope: str,
    parts_root: Path,
    summary: dict,
    prompt_count: int,
) -> dict[str, dict]:
    if scope == "seed2026_30s_128":
        raw = base.load_prompt_rows(parts_root, summary, METHODS, prompt_count)
        return {"full": base.derived_rows(raw, METHODS, prompt_count)}
    clips = int(SCOPE_SPECS[scope]["num_output_frames"]) // 8
    return {
        "full": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            clips_per_video=clips,
            start=0,
            end=clips,
        ),
        "early_half": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            clips_per_video=clips,
            start=0,
            end=clips // 2,
        ),
        "late_half": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            clips_per_video=clips,
            start=clips // 2,
            end=clips,
        ),
    }


def persistence_gate(
    window_rows: dict[str, dict],
    *,
    prompt_count: int,
) -> dict:
    rows = []
    for control_index, control in enumerate((LOCAL_CONTROL, NATIVE_CONTROL)):
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            values = [
                (
                    window_rows["late_half"][(CANDIDATE, prompt)][metric]
                    - window_rows["late_half"][(control, prompt)][metric]
                )
                - (
                    window_rows["early_half"][(CANDIDATE, prompt)][metric]
                    - window_rows["early_half"][(control, prompt)][metric]
                )
                for prompt in range(prompt_count)
            ]
            array = np.asarray(values, dtype=np.float64)
            interval = base.bootstrap_ci(
                values,
                seed=1928000 + control_index * 101 + metric_index,
            )
            margin = PERSISTENCE_MARGINS[metric]
            rows.append(
                {
                    "control": control,
                    "metric": metric,
                    "late_minus_early_effect": float(array.mean()),
                    "bootstrap_ci95": interval,
                    "margin": margin,
                    "pass": float(interval[0]) >= margin,
                    "per_prompt_delta": values,
                }
            )
    return {"rows": rows, "pass": all(row["pass"] for row in rows)}


def _targeted_review(
    manifest: dict,
    rows: dict,
    temporal_rows: dict,
    guards: tuple[dict, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    prompt_count = int(manifest["prompt_count"])
    video_dirs = {str(row["key"]): str(row["video_dir"]) for row in manifest["methods"]}
    flags = {}
    for guard in guards:
        for row in guard.get("flagged_prompts", ()):
            flags.setdefault(int(row["prompt_index"]), set()).update(row["flags"])
    ranked = []
    for prompt in range(prompt_count):
        deltas = {
            metric: rows[(CANDIDATE, prompt)][metric]
            - rows[(LOCAL_CONTROL, prompt)][metric]
            for metric in PRIMARY_METRICS
        }
        current = temporal_rows[(CANDIDATE, prompt)]
        recent = temporal_rows[(LOCAL_CONTROL, prompt)]
        disagreement = abs(
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
            + disagreement
        )
        ranked.append((priority, prompt, deltas, disagreement))
    queue = []
    for priority, prompt, deltas, disagreement in sorted(ranked, reverse=True)[:limit]:
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "scope": manifest["scope"],
                "prompt_index": prompt,
                "v191_prompt_index": int(item["v191_prompt_index"]),
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


def analyze_scope(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    *,
    temporal_rows: dict[tuple[str, int], dict[str, float]],
) -> dict:
    scope = str(manifest.get("scope", ""))
    spec = SCOPE_SPECS.get(scope)
    methods = tuple(str(row.get("key")) for row in manifest.get("methods") or ())
    if spec is None:
        raise ValueError(f"unsupported v192 scope: {scope}")
    prompt_count = int(spec["prompt_count"])
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("confirmatory") is not True
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != prompt_count
        or int(manifest.get("num_output_frames", -1))
        != int(spec["num_output_frames"])
        or int(manifest.get("seed", -1)) != int(spec["seed"])
        or len(prompt_items) != prompt_count
        or [int(row.get("v191_prompt_index", -1)) for row in prompt_items]
        != list(spec["prompt_positions"])
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or tuple(summary.get("methods") or {}) != METHODS
        or tuple(summary.get("dimensions") or ()) != DIMENSIONS
        or summary.get("missing")
    ):
        raise ValueError("v192 analysis requires one complete frozen scope")

    windows = _load_scope_rows(scope, parts_root, summary, prompt_count)
    comparisons = []
    pairs = (
        (CANDIDATE, LOCAL_CONTROL, "primary_equal_budget"),
        (CANDIDATE, NATIVE_CONTROL, "external_native_reference"),
        (LOCAL_CONTROL, NATIVE_CONTROL, "cache_runtime_context"),
    )
    for window_index, (window, rows) in enumerate(windows.items()):
        for pair_index, (candidate, control, role) in enumerate(pairs):
            for metric_index, metric in enumerate(base.METRICS):
                comparisons.append(
                    _contrast(
                        rows,
                        candidate=candidate,
                        control=control,
                        role=role,
                        metric=metric,
                        window=window,
                        prompt_count=prompt_count,
                        seed=(
                            1922026
                            + window_index * 1009
                            + pair_index * 101
                            + metric_index
                        ),
                    )
                )
    base.bh(comparisons)
    means = {
        window: _method_means(rows, prompt_count) for window, rows in windows.items()
    }
    dynamic_validity = dynamic_metric_validity(
        windows["full"], methods=METHODS, prompt_count=prompt_count
    )
    ni = {}
    required_windows = ("full",) if scope == "seed2026_30s_128" else ("full", "late_half")
    for window in required_windows:
        ni[window] = {}
        for control in (LOCAL_CONTROL, NATIVE_CONTROL):
            ni[window][control] = noninferiority_gate(
                _rows_by_metric(comparisons, CANDIDATE, control, window),
                control=control,
                dynamic_validity=dynamic_validity,
                means=means[window],
            )

    recent_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=LOCAL_CONTROL,
        prompt_count=prompt_count,
    )
    native_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=NATIVE_CONTROL,
        prompt_count=prompt_count,
    )
    if scope == "seed2026_30s_128":
        targets = list(manifest["v191_positive_metrics_to_replicate"])
        local = _rows_by_metric(comparisons, CANDIDATE, LOCAL_CONTROL, "full")
        target_rows = {
            metric: {
                "mean_delta": float(local[metric]["mean_delta"]),
                "bootstrap_ci95": local[metric]["bootstrap_ci95"],
                "replicated": _lower(local, metric) > 0.0,
            }
            for metric in targets
        }
        positive = {
            "rule": "at_least_one_v191_positive_nonmotion_metric_has_ci_lower_gt_zero",
            "targets": target_rows,
            "pass": any(row["replicated"] for row in target_rows.values()),
        }
        persistence = None
        gates = {
            "equal_budget_noninferiority": ni["full"][LOCAL_CONTROL]["pass"],
            "native_noninferiority": ni["full"][NATIVE_CONTROL]["pass"],
            "v191_positive_effect_replicated": positive["pass"],
            "temporal_safety_vs_equal_budget": recent_guard["automatic_safety_pass"],
            "temporal_safety_vs_native": native_guard["automatic_safety_pass"],
        }
    else:
        eligible = ["official_quality_score", "identity_background", "temporal_mechanics"]
        if dynamic_validity["informative"]:
            eligible.append("dynamic_degree")
        directional = {}
        for window in ("full", "late_half"):
            local = _rows_by_metric(comparisons, CANDIDATE, LOCAL_CONTROL, window)
            directional[window] = {
                metric: float(local[metric]["mean_delta"]) > 0.0 for metric in eligible
            }
        positive = {
            "rule": (
                "at_least_one_quality_identity_temporal_or_informative_motion_"
                "mean_delta_vs_all_recent_is_positive"
            ),
            "directional": directional,
            "pass": any(
                passed for rows in directional.values() for passed in rows.values()
            ),
        }
        persistence = persistence_gate(windows, prompt_count=prompt_count)
        gates = {
            "full_equal_budget_noninferiority": ni["full"][LOCAL_CONTROL]["pass"],
            "full_native_noninferiority": ni["full"][NATIVE_CONTROL]["pass"],
            "late_equal_budget_noninferiority": ni["late_half"][LOCAL_CONTROL]["pass"],
            "late_native_noninferiority": ni["late_half"][NATIVE_CONTROL]["pass"],
            "positive_direction_present": positive["pass"],
            "effect_persistence": persistence["pass"],
            "temporal_safety_vs_equal_budget": recent_guard["automatic_safety_pass"],
            "temporal_safety_vs_native": native_guard["automatic_safety_pass"],
        }
    passed = all(gates.values())
    review_rows = windows.get("late_half", windows["full"])
    review_queue = (
        _targeted_review(
            manifest,
            review_rows,
            temporal_rows,
            (recent_guard, native_guard),
        )
        if passed
        else []
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": scope,
        "scope_role": manifest["scope_role"],
        "confirmatory": True,
        "prompt_count": prompt_count,
        "seed": int(spec["seed"]),
        "num_output_frames": int(spec["num_output_frames"]),
        "methods": list(METHODS),
        "selected_v190_method": manifest["selected_v190_method"],
        "selected_operator": manifest["selected_operator"],
        "windows": {
            "full": [0, int(spec["num_output_frames"]) // 8],
            **(
                {
                    "early_half": [0, 15],
                    "late_half": [15, 30],
                }
                if scope == "long60_seed10000_32"
                else {}
            ),
        },
        "method_means": means,
        "comparisons": comparisons,
        "metric_validity": {"dynamic_degree": dynamic_validity},
        "noninferiority": ni,
        "positive_effect": positive,
        "effect_persistence": persistence,
        "automatic_temporal_guards": {
            LOCAL_CONTROL: recent_guard,
            NATIVE_CONTROL: native_guard,
        },
        "robustness_gates": gates,
        "scope_pass": passed,
        "targeted_review_queue": review_queue,
        "manual_review_required_for_scope_pass": passed,
        "claim_boundary": manifest["claim_boundary"],
    }


def _find_comparison(report: dict, control: str, metric: str, window: str) -> dict:
    return next(
        row
        for row in report["comparisons"]
        if row["candidate"] == CANDIDATE
        and row["control"] == control
        and row["metric"] == metric
        and row["window"] == window
    )


def _verify_report_source(report: dict) -> None:
    source = report.get("source") or {}
    for key in (
        "comparison_manifest",
        "vbench_summary",
        "temporal_diagnostics",
        "temporal_contract",
    ):
        path = Path(str(source.get(key, "")))
        if not path.is_file() or sha256(path) != source.get(f"{key}_sha256"):
            raise ValueError(f"v192 scope evidence drifted: {report.get('scope')}/{key}")


def combine_reports(input_manifest: dict, seed_report: dict, long_report: dict) -> dict:
    if (
        seed_report.get("scope") != "seed2026_30s_128"
        or long_report.get("scope") != "long60_seed10000_32"
        or seed_report.get("scope_pass") not in (True, False)
        or long_report.get("scope_pass") not in (True, False)
        or seed_report.get("selected_v190_method")
        != input_manifest.get("selected_v190_method")
        or long_report.get("selected_v190_method")
        != input_manifest.get("selected_v190_method")
        or seed_report.get("selected_operator") != input_manifest.get("selected_operator")
        or long_report.get("selected_operator") != input_manifest.get("selected_operator")
    ):
        raise ValueError("v192 combined decision received mixed scope reports")
    _verify_report_source(seed_report)
    _verify_report_source(long_report)

    provenance = input_manifest["v191_provenance"]
    decision_path = Path(provenance["decision"])
    if not decision_path.is_file() or sha256(decision_path) != provenance["decision_sha256"]:
        raise ValueError("v192 combined decision lost its v191 prerequisite")
    v191 = json.loads(decision_path.read_text(encoding="utf-8"))
    targets = list(input_manifest["v191_positive_metrics_to_replicate"])
    pooled = {}
    for metric_index, metric in enumerate(targets):
        prior = next(
            row
            for row in v191["comparisons"]
            if row["candidate"] == CANDIDATE
            and row["control"] == LOCAL_CONTROL
            and row["metric"] == metric
        )
        current = _find_comparison(seed_report, LOCAL_CONTROL, metric, "full")
        prior_values = np.asarray(prior["per_prompt_delta"], dtype=np.float64)
        current_values = np.asarray(current["per_prompt_delta"], dtype=np.float64)
        if prior_values.shape != (128,) or current_values.shape != (128,):
            raise ValueError(f"v192 seed pairing is incomplete: {metric}")
        prompt_averages = ((prior_values + current_values) / 2.0).tolist()
        interval = base.bootstrap_ci(
            prompt_averages,
            seed=1929000 + metric_index,
        )
        pooled[metric] = {
            "v191_seed10000_mean_delta": float(prior_values.mean()),
            "v192_seed2026_mean_delta": float(current_values.mean()),
            "two_seed_prompt_average_delta": float(np.mean(prompt_averages)),
            "bootstrap_ci95_over_prompts": interval,
            "both_seed_means_positive": bool(
                prior_values.mean() > 0.0 and current_values.mean() > 0.0
            ),
            "pooled_ci_lower_gt_zero": float(interval[0]) > 0.0,
        }
    pooled_pass = any(
        row["both_seed_means_positive"] and row["pooled_ci_lower_gt_zero"]
        for row in pooled.values()
    )
    gates = {
        "new_seed_scope_pass": seed_report["scope_pass"],
        "two_seed_pooled_positive_effect": pooled_pass,
        "long60_scope_pass": long_report["scope_pass"],
    }
    passed = all(gates.values())
    recommendation = (
        "freeze_within_model_head_phase_method_for_cross_model_transfer"
        if passed
        else "do_not_advance_v192_head_phase_robustness"
    )
    seed_dynamic = _find_comparison(
        seed_report, LOCAL_CONTROL, "dynamic_degree", "full"
    )
    long_dynamic = _find_comparison(
        long_report, LOCAL_CONTROL, "dynamic_degree", "late_half"
    )
    motion_claim = bool(
        input_manifest["v191_motion_improvement_claim_supported"]
        and seed_report["metric_validity"]["dynamic_degree"]["informative"]
        and long_report["metric_validity"]["dynamic_degree"]["informative"]
        and seed_dynamic["bootstrap_ci95"][0] > 0.0
        and long_dynamic["mean_delta"] > 0.0
    )
    review_queue = []
    if passed:
        review_queue.extend(seed_report["targeted_review_queue"][:2])
        review_queue.extend(long_report["targeted_review_queue"][:2])
    return {
        "version": 1,
        "experiment": "v192_head_phase_seed_length_robustness",
        "confirmatory": True,
        "selected_v190_method": input_manifest["selected_v190_method"],
        "selected_operator": input_manifest["selected_operator"],
        "methods": list(METHODS),
        "scope_reports": {
            seed_report["scope"]: seed_report.get("source", {}),
            long_report["scope"]: long_report.get("source", {}),
        },
        "two_seed_pooled_effect": pooled,
        "combined_gates": gates,
        "within_model_seed_length_robustness_confirmed": passed,
        "motion_improvement_claim_supported": motion_claim,
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": passed,
        "targeted_review_queue": review_queue,
        "claim_boundary": input_manifest["claim_boundary"],
    }


def render_scope(report: dict) -> str:
    lines = [
        f"# v192 {report['scope']} Analysis",
        "",
        f"- Scope pass: `{report['scope_pass']}`",
        f"- Operator: `{report['selected_operator']}`",
        f"- Manual review videos: `{len(report['targeted_review_queue'])}`",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for gate, passed in report["robustness_gates"].items():
        lines.append(f"| {gate} | {passed} |")
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def render_combined(report: dict) -> str:
    lines = [
        "# v192 Seed and Length Robustness Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Confirmed: `{report['within_model_seed_length_robustness_confirmed']}`",
        f"- Motion claim supported: `{report['motion_improvement_claim_supported']}`",
        f"- Manual review videos: `{len(report['targeted_review_queue'])}`",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for gate, passed in report["combined_gates"].items():
        lines.append(f"| {gate} | {passed} |")
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    scope_parser = subparsers.add_parser("scope")
    scope_parser.add_argument("--comparison-root", type=Path, required=True)
    scope_parser.add_argument("--summary", type=Path, required=True)
    scope_parser.add_argument("--parts-root", type=Path, required=True)
    scope_parser.add_argument("--temporal-csv", type=Path, required=True)
    scope_parser.add_argument("--temporal-contract", type=Path, required=True)
    scope_parser.add_argument("--output", type=Path, required=True)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--input-manifest", type=Path, required=True)
    combine_parser.add_argument("--seed-report", type=Path, required=True)
    combine_parser.add_argument("--long-report", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "scope":
        manifest_path = args.comparison_root / "comparison_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        prompt_count = int(manifest["prompt_count"])
        verify_temporal_contract(
            args.temporal_contract,
            manifest_path,
            args.temporal_csv,
        )
        temporal_rows = load_temporal_rows(
            args.temporal_csv,
            methods=METHODS,
            prompt_count=prompt_count,
        )
        report = analyze_scope(
            manifest,
            summary,
            args.parts_root,
            temporal_rows=temporal_rows,
        )
        report["metric_runtime_fingerprint"] = metric_runtime_fingerprint(
            args.parts_root,
            METHODS,
            tuple(summary["dimensions"]),
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
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output.with_suffix(".md").write_text(
            render_scope(report), encoding="utf-8"
        )
        print(
            "[v192-scope-analysis] "
            f"scope={report['scope']} pass={str(report['scope_pass']).lower()}"
        )
        return

    input_manifest = verify_input(args.input_manifest)
    seed_report = json.loads(args.seed_report.read_text(encoding="utf-8"))
    long_report = json.loads(args.long_report.read_text(encoding="utf-8"))
    report = combine_reports(input_manifest, seed_report, long_report)
    report["source"] = {
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "seed_report": str(args.seed_report.resolve()),
        "seed_report_sha256": sha256(args.seed_report),
        "long_report": str(args.long_report.resolve()),
        "long_report_sha256": sha256(args.long_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        render_combined(report), encoding="utf-8"
    )
    print(
        "[v192-combined] "
        f"recommendation={report['recommendation']} "
        f"confirmed={str(report['within_model_seed_length_robustness_confirmed']).lower()}"
    )


if __name__ == "__main__":
    main()
