#!/usr/bin/env python3
"""Paired seed, long-horizon, and phase-counterfactual analysis for v188."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v188_robustness_matrix import BASE_METHODS, MECHANISM_METHODS


CANDIDATE = "phase_deterministic"
LOCAL = "all_recent"
RANDOM = "phase_reservoir"
NATIVE = "sf_native"
OPPOSITE = "opposite_phase_deterministic"
ALL_NOISY = "all_noisy_deterministic"
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)
CLIPS_PER_30S = 15


def _contrast(
    rows: dict,
    prompt_count: int,
    candidate: str,
    control: str,
    metric: str,
    *,
    seed: int,
    window: str = "full",
) -> dict:
    values = np.asarray(
        [
            rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
            for prompt in range(prompt_count)
        ],
        dtype=np.float64,
    )
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "window": window,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(values.tolist(), seed=seed),
        "p_value": base.sign_p(values.tolist()),
        "per_prompt_delta": values.tolist(),
    }


def _comparison_rows(
    comparisons: list[dict], candidate: str, control: str, window: str = "full"
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


def _means(rows: dict, methods: tuple[str, ...], prompt_count: int) -> dict:
    return {
        method: {
            metric: float(
                np.mean([rows[(method, prompt)][metric] for prompt in range(prompt_count)])
            )
            for metric in base.METRICS
        }
        for method in methods
    }


def _load_full_rows(parts_root: Path, summary: dict, methods: tuple[str, ...], count: int):
    raw = base.load_prompt_rows(parts_root, summary, methods, count)
    return base.derived_rows(raw, methods, count)


def _load_long_window_rows(
    parts_root: Path,
    summary: dict,
    methods: tuple[str, ...],
    prompt_count: int,
    start: int,
    end: int,
) -> dict:
    if not 0 <= start < end <= 30:
        raise ValueError(f"invalid v188 long60 clip window [{start}, {end})")
    rows = {
        (method, prompt): {} for method in methods for prompt in range(prompt_count)
    }
    for method in methods:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=prompt_count,
                clips_per_video=30,
            )
            raw_values = [value for prompt in range(prompt_count) for value in clips[prompt]]
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
    return base.derived_rows(rows, methods, prompt_count)


def _validate(manifest: dict, summary: dict) -> tuple[str, tuple[str, ...], int]:
    scope = str(manifest.get("scope", ""))
    expected = {
        "replica64_seed20000": (BASE_METHODS, 64, 120, 20000),
        "long60_seed10000_32": (BASE_METHODS, 32, 240, 10000),
        "mechanism32_seed10000": (MECHANISM_METHODS, 32, 120, 10000),
    }
    if scope not in expected:
        raise ValueError(f"unsupported v188 analysis scope: {scope}")
    methods, count, frames, seed = expected[scope]
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != f"v188_{scope}_vbench"
        or manifest.get("confirmatory_extension") is not True
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != methods
        or int(manifest.get("prompt_count", -1)) != count
        or int(manifest.get("num_output_frames", -1)) != frames
        or int(manifest.get("seed", -1)) != seed
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or len(prompt_items) != count
        or [int(row.get("index", -1)) for row in prompt_items] != list(range(count))
        or len({int(row.get("v187_index", -1)) for row in prompt_items}) != count
        or tuple(summary.get("methods") or {}) != methods
        or tuple(summary.get("dimensions") or ()) != DIMENSIONS
        or summary.get("missing")
    ):
        raise ValueError(f"v188 {scope} analysis contract is incomplete or mixed")
    return scope, methods, count


def load_v187_reference(
    comparison_root: Path,
    summary_path: Path,
    parts_root: Path,
) -> tuple[dict, dict]:
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "v187_unseen128_phase_operator_vbench"
        or tuple(row.get("key") for row in manifest.get("methods") or ()) != BASE_METHODS
        or int(manifest.get("prompt_count", -1)) != 128
        or tuple(summary.get("methods") or {}) != BASE_METHODS
        or summary.get("missing")
    ):
        raise ValueError("v188 seed replication requires complete v187 core-9 evidence")
    rows = _load_full_rows(parts_root, summary, BASE_METHODS, 128)
    return manifest, rows


def _seed_consistency(
    manifest: dict,
    rows: dict,
    v187_manifest: dict,
    v187_rows: dict,
) -> dict:
    source_items = v187_manifest.get("prompt_items") or ()
    if len(source_items) != 128:
        raise ValueError("v187 prompt mapping is incomplete")
    report = {}
    for metric in PRIMARY_METRICS:
        old_effects = []
        new_effects = []
        for local_index, item in enumerate(manifest["prompt_items"]):
            source_index = int(item["v187_index"])
            if (
                int(source_items[source_index].get("source_index", -1))
                != int(item["source_index"])
                or str(source_items[source_index].get("text")) != str(item["text"])
            ):
                raise ValueError("v188/v187 seed prompt mapping drifted")
            old_effects.append(
                v187_rows[(CANDIDATE, source_index)][metric]
                - v187_rows[(LOCAL, source_index)][metric]
            )
            new_effects.append(
                rows[(CANDIDATE, local_index)][metric]
                - rows[(LOCAL, local_index)][metric]
            )
        old = np.asarray(old_effects, dtype=np.float64)
        new = np.asarray(new_effects, dtype=np.float64)
        interaction = new - old
        if float(np.std(old)) > 0.0 and float(np.std(new)) > 0.0:
            correlation = float(np.corrcoef(old, new)[0, 1])
        else:
            correlation = None
        report[metric] = {
            "v187_seed10000_mean": float(old.mean()),
            "v188_seed20000_mean": float(new.mean()),
            "same_aggregate_sign": bool(
                (old.mean() == 0.0 and new.mean() == 0.0)
                or (old.mean() > 0.0 and new.mean() > 0.0)
                or (old.mean() < 0.0 and new.mean() < 0.0)
            ),
            "per_prompt_sign_agreement": float(
                np.mean(np.sign(old) == np.sign(new))
            ),
            "per_prompt_effect_correlation": correlation,
            "seed_interaction_mean": float(interaction.mean()),
            "seed_interaction_ci95": base.bootstrap_ci(
                interaction.tolist(), seed=1887000 + len(report)
            ),
            "two_seed_meta_mean": float(np.mean((old + new) / 2.0)),
        }
    return report


def _analyze_replica(
    manifest: dict,
    rows: dict,
    comparisons: list[dict],
    v187_reference: tuple[dict, dict] | None,
) -> dict:
    if v187_reference is None:
        raise ValueError("replica64 analysis requires the v187 reference metrics")
    recent = _comparison_rows(comparisons, CANDIDATE, LOCAL)
    reservoir = _comparison_rows(comparisons, CANDIDATE, RANDOM)
    seed_consistency = _seed_consistency(
        manifest, rows, v187_reference[0], v187_reference[1]
    )
    gates = {
        "quality_ci_lower_ge_minus_0_15": _lower(recent, "official_quality_score")
        >= -0.15,
        "identity_ci_lower_ge_minus_0_0015": _lower(recent, "identity_background")
        >= -0.0015,
        "dynamic_mean_ge_0_015": recent["dynamic_degree"]["mean_delta"] >= 0.015,
        "dynamic_ci_lower_ge_minus_0_01": _lower(recent, "dynamic_degree") >= -0.01,
        "temporal_ci_lower_ge_minus_0_003": _lower(recent, "temporal_mechanics")
        >= -0.003,
        "dynamic_aggregate_sign_replicated": seed_consistency["dynamic_degree"][
            "same_aggregate_sign"
        ],
        "at_least_three_primary_signs_replicated": sum(
            row["same_aggregate_sign"] for row in seed_consistency.values()
        )
        >= 3,
    }
    operator_noninferiority = {
        "quality_ci_lower_ge_minus_0_20": _lower(
            reservoir, "official_quality_score"
        )
        >= -0.20,
        "identity_ci_lower_ge_minus_0_001": _lower(
            reservoir, "identity_background"
        )
        >= -0.001,
        "dynamic_ci_lower_ge_minus_0_03": _lower(reservoir, "dynamic_degree")
        >= -0.03,
        "temporal_ci_lower_ge_minus_0_002": _lower(
            reservoir, "temporal_mechanics"
        )
        >= -0.002,
    }
    return {
        "seed_consistency": seed_consistency,
        "replication_gate": gates,
        "replication_confirmed": all(gates.values()),
        "operator_noninferiority_gate": operator_noninferiority,
        "operator_noninferiority_replicated": all(operator_noninferiority.values()),
    }


def _noninferiority(rows: dict[str, dict]) -> dict:
    return {
        "quality_ci_lower_ge_minus_0_20": _lower(rows, "official_quality_score")
        >= -0.20,
        "identity_ci_lower_ge_minus_0_0015": _lower(rows, "identity_background")
        >= -0.0015,
        "dynamic_ci_lower_ge_minus_0_03": _lower(rows, "dynamic_degree") >= -0.03,
        "temporal_ci_lower_ge_minus_0_003": _lower(rows, "temporal_mechanics")
        >= -0.003,
    }


def _gain(rows: dict[str, dict]) -> dict:
    return {
        "quality_mean_ge_0_10": rows["official_quality_score"]["mean_delta"] >= 0.10,
        "identity_mean_ge_0_0005": rows["identity_background"]["mean_delta"]
        >= 0.0005,
        "dynamic_mean_ge_0_02": rows["dynamic_degree"]["mean_delta"] >= 0.02,
        "temporal_mean_ge_0_001": rows["temporal_mechanics"]["mean_delta"]
        >= 0.001,
    }


def _analyze_mechanism(comparisons: list[dict]) -> dict:
    rows_by_control = {
        control: _comparison_rows(comparisons, CANDIDATE, control)
        for control in (OPPOSITE, ALL_NOISY, RANDOM, LOCAL)
    }
    noninferiority = {
        control: _noninferiority(rows) for control, rows in rows_by_control.items()
    }
    gains = {control: _gain(rows) for control, rows in rows_by_control.items()}
    counterfactual_support = {
        control: bool(
            all(noninferiority[control].values()) and any(gains[control].values())
        )
        for control in (OPPOSITE, ALL_NOISY)
    }
    return {
        "counterfactual_noninferiority": noninferiority,
        "counterfactual_explanatory_gains": gains,
        "counterfactual_support": counterfactual_support,
        "phase_specificity_supported": all(counterfactual_support.values()),
        "interpretation": (
            "The equal-dose opposite phase isolates call position; all-noisy "
            "isolates selective exposure from persistent long-history access."
        ),
    }


def _analyze_long(
    window_rows: dict[str, dict], prompt_count: int
) -> tuple[list[dict], dict]:
    comparisons = []
    controls = (LOCAL, RANDOM, NATIVE)
    for window_index, (window, rows) in enumerate(window_rows.items()):
        for control_index, control in enumerate(controls):
            for metric_index, metric in enumerate(base.METRICS):
                comparisons.append(
                    _contrast(
                        rows,
                        prompt_count,
                        CANDIDATE,
                        control,
                        metric,
                        window=window,
                        seed=1883000
                        + window_index * 1000
                        + control_index * 101
                        + metric_index,
                    )
                )
    primary = [row for row in comparisons if row["metric"] in PRIMARY_METRICS]
    base.bh(primary)
    full = _comparison_rows(comparisons, CANDIDATE, LOCAL, "full")
    late = _comparison_rows(comparisons, CANDIDATE, LOCAL, "late_half")
    gates = {
        "full_quality_ci_lower_ge_minus_0_20": _lower(
            full, "official_quality_score"
        )
        >= -0.20,
        "full_identity_ci_lower_ge_minus_0_0015": _lower(
            full, "identity_background"
        )
        >= -0.0015,
        "full_dynamic_mean_ge_0_015": full["dynamic_degree"]["mean_delta"]
        >= 0.015,
        "full_temporal_ci_lower_ge_minus_0_003": _lower(
            full, "temporal_mechanics"
        )
        >= -0.003,
        "late_quality_ci_lower_ge_minus_0_25": _lower(
            late, "official_quality_score"
        )
        >= -0.25,
        "late_identity_ci_lower_ge_minus_0_002": _lower(
            late, "identity_background"
        )
        >= -0.002,
        "late_dynamic_mean_ge_0_015": late["dynamic_degree"]["mean_delta"]
        >= 0.015,
        "late_dynamic_ci_lower_ge_minus_0_02": _lower(late, "dynamic_degree")
        >= -0.02,
        "late_temporal_ci_lower_ge_minus_0_004": _lower(
            late, "temporal_mechanics"
        )
        >= -0.004,
    }
    persistence = {}
    early_rows = window_rows["early_half"]
    late_rows = window_rows["late_half"]
    for metric in PRIMARY_METRICS:
        values = [
            (
                late_rows[(CANDIDATE, prompt)][metric]
                - late_rows[(LOCAL, prompt)][metric]
            )
            - (
                early_rows[(CANDIDATE, prompt)][metric]
                - early_rows[(LOCAL, prompt)][metric]
            )
            for prompt in range(prompt_count)
        ]
        persistence[metric] = {
            "late_minus_early_effect": float(np.mean(values)),
            "bootstrap_ci95": base.bootstrap_ci(
                values, seed=1888000 + len(persistence)
            ),
            "per_prompt_delta": values,
        }
    return comparisons, {
        "long_horizon_gate": gates,
        "long_horizon_confirmed": all(gates.values()),
        "effect_persistence": persistence,
    }


def _targeted_review(
    manifest: dict,
    rows: dict,
    methods: tuple[str, ...],
    *,
    limit: int = 6,
) -> list[dict]:
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    controls = [LOCAL]
    if OPPOSITE in methods:
        controls.extend([OPPOSITE, ALL_NOISY])
    queue = []
    for prompt in range(int(manifest["prompt_count"])):
        deltas = {
            control: {
                metric: rows[(CANDIDATE, prompt)][metric]
                - rows[(control, prompt)][metric]
                for metric in PRIMARY_METRICS
            }
            for control in controls
        }
        conflict = any(
            (row["identity_background"] > 0.0 and row["dynamic_degree"] < 0.0)
            or (row["identity_background"] < 0.0 and row["dynamic_degree"] > 0.0)
            or row["temporal_mechanics"] <= -0.01
            or abs(row["official_quality_score"]) >= 1.0
            for row in deltas.values()
        )
        if not conflict:
            continue
        priority = max(
            30.0 * abs(row["identity_background"])
            + 15.0 * abs(row["temporal_mechanics"])
            + abs(row["dynamic_degree"])
            + 0.1 * abs(row["official_quality_score"])
            for row in deltas.values()
        )
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "deltas": deltas,
                "priority": float(priority),
                "videos": {
                    method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                    for method in methods
                },
            }
        )
    return sorted(queue, key=lambda row: (-row["priority"], row["prompt_index"]))[
        :limit
    ]


def _effect_strata(manifest: dict, rows: dict, prompt_count: int) -> dict:
    """Describe heterogeneity without using strata to select the method."""

    quartile = max(1, prompt_count // 4)
    variables = {
        "baseline_dynamic": [
            float(rows[(LOCAL, prompt)]["dynamic_degree"])
            for prompt in range(prompt_count)
        ],
        "baseline_identity": [
            float(rows[(LOCAL, prompt)]["identity_background"])
            for prompt in range(prompt_count)
        ],
        "prompt_word_count": [
            float(len(str(item["text"]).split())) for item in manifest["prompt_items"]
        ],
    }
    report = {}
    for variable, values in variables.items():
        order = sorted(range(prompt_count), key=lambda index: (values[index], index))
        groups = {"low_quartile": order[:quartile], "high_quartile": order[-quartile:]}
        report[variable] = {}
        for group, prompts in groups.items():
            report[variable][group] = {
                "prompt_count": len(prompts),
                "range": [
                    float(min(values[prompt] for prompt in prompts)),
                    float(max(values[prompt] for prompt in prompts)),
                ],
                "candidate_minus_all_recent": {
                    metric: float(
                        np.mean(
                            [
                                rows[(CANDIDATE, prompt)][metric]
                                - rows[(LOCAL, prompt)][metric]
                                for prompt in prompts
                            ]
                        )
                    )
                    for metric in PRIMARY_METRICS
                },
            }
    return {
        "inferential_role": "descriptive_heterogeneity_only",
        "selection_effect": False,
        "strata": report,
    }


def analyze(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    v187_reference: tuple[dict, dict] | None = None,
) -> dict:
    scope, methods, prompt_count = _validate(manifest, summary)
    if scope == "long60_seed10000_32":
        window_rows = {
            "full": _load_long_window_rows(
                parts_root, summary, methods, prompt_count, 0, 30
            ),
            "early_half": _load_long_window_rows(
                parts_root, summary, methods, prompt_count, 0, 15
            ),
            "late_half": _load_long_window_rows(
                parts_root, summary, methods, prompt_count, 15, 30
            ),
        }
        comparisons, scope_result = _analyze_long(window_rows, prompt_count)
        rows = window_rows["full"]
        review_rows = window_rows["late_half"]
    else:
        rows = _load_full_rows(parts_root, summary, methods, prompt_count)
        review_rows = rows
        pairs = [(CANDIDATE, LOCAL), (CANDIDATE, RANDOM), (CANDIDATE, NATIVE)]
        if scope == "mechanism32_seed10000":
            pairs.extend([(CANDIDATE, OPPOSITE), (CANDIDATE, ALL_NOISY)])
        comparisons = []
        for pair_index, (candidate, control) in enumerate(pairs):
            for metric_index, metric in enumerate(base.METRICS):
                comparisons.append(
                    _contrast(
                        rows,
                        prompt_count,
                        candidate,
                        control,
                        metric,
                        seed=1881000 + pair_index * 101 + metric_index,
                    )
                )
        base.bh(comparisons)
        scope_result = (
            _analyze_replica(manifest, rows, comparisons, v187_reference)
            if scope == "replica64_seed20000"
            else _analyze_mechanism(comparisons)
        )
    review_needed = bool(
        scope_result.get("replication_confirmed")
        or scope_result.get("long_horizon_confirmed")
        or scope_result.get("phase_specificity_supported")
    )
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "scope": scope,
        "purpose": manifest["purpose"],
        "confirmatory_extension": True,
        "prompt_count": prompt_count,
        "seed": manifest["seed"],
        "num_output_frames": manifest["num_output_frames"],
        "selected_schedule": manifest["selected_schedule"],
        "opposite_schedule": manifest["opposite_schedule"],
        "selected_operator": manifest["selected_operator"],
        "methods": list(methods),
        "method_means": _means(rows, methods, prompt_count),
        "comparisons": comparisons,
        "effect_strata": _effect_strata(manifest, rows, prompt_count),
        **scope_result,
        "manual_review_required": review_needed,
        "targeted_review_queue": (
            _targeted_review(manifest, review_rows, methods) if review_needed else []
        ),
        "manual_review_cap": 6,
        "claim_boundary": manifest["claim_boundary"],
    }


def render(report: dict) -> str:
    decision = (
        report.get("replication_confirmed")
        if report["scope"].startswith("replica")
        else report.get("long_horizon_confirmed")
        if report["scope"].startswith("long60")
        else report.get("phase_specificity_supported")
    )
    lines = [
        f"# v188 {report['scope']} Paired Analysis",
        "",
        f"Scope decision: `{decision}`",
        f"Schedule/operator: `{report['selected_schedule']}` / `{report['selected_operator']}`",
        f"Manual review required: `{report['manual_review_required']}`",
        "",
        "| Contrast | Metric | Window | Mean delta | CI95 | Win |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['window']} | "
            f"{row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v187-comparison-root", type=Path)
    parser.add_argument("--v187-summary", type=Path)
    parser.add_argument("--v187-parts-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(
        (args.comparison_root / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    reference_args = (
        args.v187_comparison_root,
        args.v187_summary,
        args.v187_parts_root,
    )
    if any(value is not None for value in reference_args) and not all(
        value is not None for value in reference_args
    ):
        raise ValueError("all three v187 reference paths must be supplied together")
    reference = (
        load_v187_reference(*reference_args)
        if all(value is not None for value in reference_args)
        else None
    )
    report = analyze(manifest, summary, args.parts_root, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v188-analysis] "
        f"scope={report['scope']} review={str(report['manual_review_required']).lower()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
