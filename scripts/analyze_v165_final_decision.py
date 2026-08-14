#!/usr/bin/env python3
"""Combine v165 VBench, cache traces, and safety metrics into one decision."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from prepare_v165_vbench_comparison import DIMENSIONS, METHODS, PROMPT_COUNT
from v165_decision_contract import (
    DEVELOPMENT_GATES,
    DIRECTION_FRESH,
    DIRECTION_MATCH,
    PRIMARY,
    SF,
    STATE_MOTION,
    TIE_003,
    derive_scores,
    evaluate_development_gates,
)


CLIPS_PER_VIDEO = 15
PATH_PATTERN = re.compile(
    r"(?:^|/)split_clip/(\d{6})-0/\1-0_(\d{3})\.mp4(?:/|$)"
)
FALLBACK_PATTERN = re.compile(r"(?:^|/)(\d{6})-0_(\d{3})\.mp4(?:/|$)")
REFERENCES = (DIRECTION_MATCH, SF, TIE_003, DIRECTION_FRESH, STATE_MOTION)
SAFETY_REFERENCES = (DIRECTION_MATCH, SF)
TEMPORAL_FIELDS = (
    "late_motion_ratio",
    "temporal_jump",
    "appearance_outlier_fraction",
    "flow_accel_outlier_fraction",
    "dark_frame_fraction",
    "bright_frame_fraction",
    "low_contrast_frame_fraction",
    "edge_density_outlier_fraction",
)
COMPREHENSIVE_FIELDS = (
    "m1_dino_consistency",
    "m1_first_last_gap",
    "m1_min_stability",
    "m3_motion_smoothness",
    "m5_max_flicker",
    "m5_temporal_flickering",
    "m6_clip_text_alignment",
    "m7_background_consistency",
    "m7_background_drift",
    "m8_loop_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-parts-root", required=True, type=Path)
    parser.add_argument("--vbench-summary", required=True, type=Path)
    parser.add_argument("--temporal-csv", required=True, type=Path)
    parser.add_argument("--comprehensive-json", required=True, type=Path)
    parser.add_argument("--trace-report", required=True, type=Path)
    parser.add_argument("--published-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {number}")
    return number


def vbench_detail_value(
    value: Any,
    *,
    dimension: str,
    name: str,
) -> float:
    # VBench serializes per-clip dynamic-degree decisions as JSON booleans.
    if isinstance(value, bool):
        if dimension != "dynamic_degree":
            raise ValueError(f"{name}: boolean detail is invalid")
        return float(value)
    return finite(value, name=name)


def candidate_record_lists(value: Any) -> Iterable[list[dict[str, Any]]]:
    """Yield detailed VBench record lists without mixing duplicate views."""
    if isinstance(value, dict):
        for item in value.values():
            yield from candidate_record_lists(item)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, dict) for item in value):
            records = [
                item
                for item in value
                if "video_path" in item and "video_results" in item
            ]
            if records:
                yield records
        for item in value:
            yield from candidate_record_lists(item)


def prompt_clip(video_path: str) -> tuple[int, int]:
    normalized = str(video_path).replace("\\", "/")
    match = PATH_PATTERN.search(normalized) or FALLBACK_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"cannot recover prompt/clip from {video_path}")
    return int(match.group(1)), int(match.group(2))


def load_dimension(
    path: Path,
    dimension: str,
    *,
    prompt_count: int | None = None,
    clips_per_video: int | None = None,
) -> dict[int, list[float]]:
    prompt_count = PROMPT_COUNT if prompt_count is None else int(prompt_count)
    clips_per_video = (
        CLIPS_PER_VIDEO if clips_per_video is None else int(clips_per_video)
    )
    if prompt_count <= 0 or clips_per_video <= 0:
        raise ValueError("prompt_count and clips_per_video must be positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or dimension not in payload:
        raise ValueError(f"invalid {dimension} result: {path}")
    expected_prompts = set(range(prompt_count))
    expected_clips = set(range(clips_per_video))
    failures = []
    for candidate_index, records in enumerate(
        candidate_record_lists(payload[dimension])
    ):
        grouped: dict[int, dict[int, float]] = {}
        try:
            for record in records:
                prompt, clip = prompt_clip(str(record["video_path"]))
                value = vbench_detail_value(
                    record["video_results"],
                    dimension=dimension,
                    name=f"{dimension}:{prompt}:{clip}",
                )
                prior = grouped.setdefault(prompt, {}).get(clip)
                if prior is not None and not math.isclose(
                    prior, value, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"conflicting prompt={prompt} clip={clip}"
                    )
                grouped[prompt][clip] = value
        except ValueError as error:
            failures.append(f"candidate {candidate_index}: {error}")
            continue
        if set(grouped) != expected_prompts or any(
            set(clips) != expected_clips for clips in grouped.values()
        ):
            failures.append(
                f"candidate {candidate_index}: coverage="
                f"{sum(len(clips) for clips in grouped.values())}/"
                f"{prompt_count * clips_per_video}"
            )
            continue
        return {
            prompt: [clips[index] for index in range(clips_per_video)]
            for prompt, clips in grouped.items()
        }
    detail = "; ".join(failures[:5]) or "no per-video detail list"
    raise ValueError(f"{dimension} has no complete clip view: {detail}")


def validate_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if (
        tuple(rows) != METHODS
        or dimensions != DIMENSIONS
        or payload.get("missing")
    ):
        raise ValueError("v165 VBench summary violates the frozen grid")
    for method, row in rows.items():
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench summary")
        for dimension in DIMENSIONS:
            finite(row[dimension], name=f"summary:{method}:{dimension}")
    return payload


def scale_factor(raw_mean: float, summary_value: float, *, name: str) -> float:
    if math.isclose(raw_mean, 0.0, abs_tol=1e-12):
        if not math.isclose(summary_value, 0.0, abs_tol=1e-12):
            raise ValueError(f"{name}: zero detail mean but nonzero summary")
        return 1.0
    ratio = summary_value / raw_mean
    allowed = (0.01, 1.0, 100.0)
    nearest = min(allowed, key=lambda value: abs(value - ratio))
    if not math.isclose(ratio, nearest, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"{name}: unexpected VBench detail scale {ratio}")
    return nearest


def load_vbench_prompt_rows(
    parts_root: Path,
    summary: dict[str, Any],
) -> tuple[
    dict[tuple[str, int], dict[str, float]],
    dict[str, dict[str, float]],
]:
    rows: dict[tuple[str, int], dict[str, float]] = {
        (method, prompt): {}
        for method in METHODS
        for prompt in range(PROMPT_COUNT)
    }
    scales: dict[str, dict[str, float]] = {}
    for method in METHODS:
        scales[method] = {}
        for dimension in DIMENSIONS:
            clips = load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
            )
            raw_values = [
                value
                for prompt in range(PROMPT_COUNT)
                for value in clips[prompt]
            ]
            summary_value = finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = scale_factor(
                statistics.fmean(raw_values),
                summary_value,
                name=f"{method}:{dimension}",
            )
            scales[method][dimension] = factor
            for prompt in range(PROMPT_COUNT):
                rows[(method, prompt)][dimension] = factor * statistics.fmean(
                    clips[prompt]
                )
            reconstructed = statistics.fmean(
                rows[(method, prompt)][dimension]
                for prompt in range(PROMPT_COUNT)
            )
            if not math.isclose(
                reconstructed,
                summary_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{method}:{dimension}: prompt/detail mean does not "
                    "reconstruct the frozen summary"
                )
    return rows, scales


def derived_prompt_rows(
    raw: dict[tuple[str, int], dict[str, float]],
) -> dict[tuple[str, int], dict[str, float]]:
    result = {}
    for prompt in range(PROMPT_COUNT):
        derived = derive_scores(
            {method: raw[(method, prompt)] for method in METHODS}
        )
        for method in METHODS:
            result[(method, prompt)] = derived[method]
    return result


def bootstrap_ci(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(10000)
    )
    return [means[249], means[9749]]


def paired_comparison(
    rows: dict[tuple[str, int], dict[str, float]],
    *,
    candidate: str,
    reference: str,
    seed_offset: int,
) -> dict[str, Any]:
    result = {}
    metric_names = tuple(rows[(candidate, 0)])
    for metric_index, metric in enumerate(metric_names):
        values = [
            rows[(candidate, prompt)][metric]
            - rows[(reference, prompt)][metric]
            for prompt in range(PROMPT_COUNT)
        ]
        result[metric] = {
            "mean_delta": statistics.fmean(values),
            "median_delta": statistics.median(values),
            "positive_prompts": sum(value > 1e-12 for value in values),
            "negative_prompts": sum(value < -1e-12 for value in values),
            "bootstrap_mean_ci95": bootstrap_ci(
                values,
                seed=1652026 + 100 * seed_offset + metric_index,
            ),
            "per_prompt": [
                {"prompt_index": prompt, "delta": value}
                for prompt, value in enumerate(values)
            ],
        }
    return result


def validate_coverage(rows: dict[tuple[str, int], Any], *, label: str) -> None:
    expected = {
        (method, prompt)
        for method in METHODS
        for prompt in range(PROMPT_COUNT)
    }
    if set(rows) != expected:
        raise ValueError(
            f"{label} coverage mismatch: "
            f"missing={sorted(expected - set(rows))[:8]} "
            f"extra={sorted(set(rows) - expected)[:8]}"
        )


def load_temporal(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw["method"]), int(raw["prompt_index"]))
            if key in rows:
                raise ValueError(f"duplicate temporal row: {key}")
            rows[key] = {
                field: finite(raw[field], name=f"{key}:{field}")
                for field in TEMPORAL_FIELDS
            }
    validate_coverage(rows, label="temporal")
    return rows


def load_comprehensive(
    path: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, float]],
    dict[int, str],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_video = payload.get("per_video")
    if not isinstance(per_video, dict):
        raise ValueError("comprehensive result has no per_video mapping")
    rows = {}
    prompts = {}
    for raw in per_video.values():
        key = (str(raw["method"]), int(raw["prompt_index"]))
        if key in rows:
            raise ValueError(f"duplicate comprehensive row: {key}")
        metrics = raw.get("metrics") or {}
        rows[key] = {
            field: finite(metrics[field], name=f"{key}:{field}")
            for field in COMPREHENSIVE_FIELDS
        }
        prompt = str(raw.get("prompt", "")).strip()
        prior = prompts.setdefault(key[1], prompt)
        if not prompt or prior != prompt:
            raise ValueError(f"inconsistent prompt text for index {key[1]}")
    validate_coverage(rows, label="comprehensive")
    if set(prompts) != set(range(PROMPT_COUNT)):
        raise ValueError("incomplete prompt text coverage")
    return rows, prompts


def candidate_safety_flags(
    temporal: dict[tuple[str, int], dict[str, float]],
    comprehensive: dict[tuple[str, int], dict[str, float]],
    *,
    prompt: int,
) -> list[str]:
    current_t = temporal[(PRIMARY, prompt)]
    reference_t = [temporal[(method, prompt)] for method in SAFETY_REFERENCES]
    current_c = comprehensive[(PRIMARY, prompt)]
    reference_c = [
        comprehensive[(method, prompt)] for method in SAFETY_REFERENCES
    ]
    flags = []
    if (
        current_t["dark_frame_fraction"] > 0.02
        or current_t["bright_frame_fraction"] > 0.02
        or current_t["low_contrast_frame_fraction"] > 0.05
    ):
        flags.append("luminance_or_contrast_failure")
    if (
        current_t["edge_density_outlier_fraction"] > 0.10
        and current_t["edge_density_outlier_fraction"]
        > max(row["edge_density_outlier_fraction"] for row in reference_t)
        + 0.05
    ):
        flags.append("edge_density_failure")
    if (
        current_t["late_motion_ratio"] < 0.55
        and current_t["late_motion_ratio"]
        < min(row["late_motion_ratio"] for row in reference_t) - 0.20
    ):
        flags.append("late_motion_collapse")
    if (
        current_t["temporal_jump"]
        > 1.35 * max(row["temporal_jump"] for row in reference_t)
        and current_t["appearance_outlier_fraction"]
        > max(row["appearance_outlier_fraction"] for row in reference_t)
        + 0.02
    ):
        flags.append("temporal_discontinuity")
    if current_c["m1_dino_consistency"] < min(
        row["m1_dino_consistency"] for row in reference_c
    ) - 0.03:
        flags.append("subject_consistency_drop")
    if current_c["m7_background_drift"] > max(
        row["m7_background_drift"] for row in reference_c
    ) + 0.05:
        flags.append("background_drift")
    if current_c["m5_max_flicker"] > 1.35 * max(
        row["m5_max_flicker"] for row in reference_c
    ):
        flags.append("severe_flicker")
    return flags


def review_plan(
    comparisons: dict[str, Any],
    safety: dict[int, list[str]],
    prompts: dict[int, str],
) -> dict[str, Any]:
    match = comparisons[DIRECTION_MATCH]
    rows = []
    for prompt in range(PROMPT_COUNT):
        deltas = {
            metric: float(match[metric]["per_prompt"][prompt]["delta"])
            for metric in match
        }
        rows.append(
            {
                "prompt_index": prompt,
                "prompt": prompts[prompt],
                "automatic_flags": safety[prompt],
                "frontier_delta_mean": statistics.fmean(deltas.values()),
                "metric_disagreement": max(deltas.values())
                - min(deltas.values()),
                "deltas_vs_directionmatch": deltas,
            }
        )
    selected: dict[int, set[str]] = {}
    flag_weights = {
        "luminance_or_contrast_failure": 4,
        "temporal_discontinuity": 4,
        "subject_consistency_drop": 3,
        "background_drift": 3,
        "late_motion_collapse": 3,
        "severe_flicker": 3,
        "edge_density_failure": 1,
    }
    flagged = sorted(
        (row for row in rows if row["automatic_flags"]),
        key=lambda row: (
            -sum(flag_weights.get(flag, 1) for flag in row["automatic_flags"]),
            row["frontier_delta_mean"],
            row["prompt_index"],
        ),
    )
    for row in flagged[:2]:
        selected.setdefault(row["prompt_index"], set()).add(
            "candidate_safety_flag"
        )
    worst = min(rows, key=lambda row: (row["frontier_delta_mean"], row["prompt_index"]))
    if len(selected) < 2:
        selected.setdefault(worst["prompt_index"], set()).add(
            "worst_frontier_delta"
        )
    disagreement = max(
        rows,
        key=lambda row: (row["metric_disagreement"], -row["prompt_index"]),
    )
    if len(selected) < 2:
        selected.setdefault(disagreement["prompt_index"], set()).add(
            "largest_metric_disagreement"
        )
    if len(selected) < 2:
        next_worst = next(
            row for row in sorted(rows, key=lambda row: row["frontier_delta_mean"])
            if row["prompt_index"] not in selected
        )
        selected[next_worst["prompt_index"]] = {"second_worst_frontier_delta"}
    chosen = sorted(
        (row for row in rows if row["prompt_index"] in selected),
        key=lambda row: (
            0 if row["automatic_flags"] else 1,
            row["frontier_delta_mean"],
            row["prompt_index"],
        ),
    )[:2]
    for row in chosen:
        row["reasons"] = sorted(selected[row["prompt_index"]])
    return {
        "mode": "two_prompt_blind",
        "methods": [PRIMARY, DIRECTION_MATCH],
        "rows": chosen,
        "prompt_count": len(chosen),
        "video_count": 2 * len(chosen),
        "maximum_video_count": 4,
        "selection_is_diagnostic_only": True,
    }


def validate_mechanism(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    methods = trace.get("methods") or {}
    primary = methods.get(PRIMARY) or {}
    aggregate = primary.get("aggregate") or {}
    if (
        trace.get("mechanism_gate") is not True
        or aggregate.get("mechanism_gate") is not True
        or int(aggregate.get("changed_count", 0)) <= 0
        or int(aggregate.get("contract_failure_count", -1)) != 0
        or int(aggregate.get("read_budget_violation_count", -1)) != 0
    ):
        raise ValueError("v165 primary mechanism gate is not valid")
    return trace


def validate_published(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method_keys = tuple(row.get("key") for row in payload.get("methods", []))
    if (
        payload.get("ok") is not True
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or method_keys != METHODS
        or any(
            int(row.get("indexed_video_count", -1)) != PROMPT_COUNT
            for row in payload["methods"]
        )
    ):
        raise ValueError("invalid v165 published manifest")
    return payload


def paired_support_checks(comparisons: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        (
            "temporal_majority_vs_directionmatch",
            DIRECTION_MATCH,
            "temporal_quality",
            9,
        ),
        ("history_majority_vs_sf", SF, "history_consistency", 9),
        ("dynamic_majority_vs_sf", SF, "dynamic_degree", 9),
    )
    return [
        {
            "name": name,
            "reference": reference,
            "metric": metric,
            "positive_prompts": int(
                comparisons[reference][metric]["positive_prompts"]
            ),
            "minimum_positive_prompts": minimum,
            "pass": int(comparisons[reference][metric]["positive_prompts"])
            >= minimum,
        }
        for name, reference, metric, minimum in definitions
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summary = validate_summary(args.vbench_summary)
    trace = validate_mechanism(args.trace_report)
    published = validate_published(args.published_manifest)
    raw_vbench, scales = load_vbench_prompt_rows(
        args.vbench_parts_root,
        summary,
    )
    derived = derived_prompt_rows(raw_vbench)
    aggregate = derive_scores(summary["methods"])
    aggregate_gates, aggregate_gate = evaluate_development_gates(aggregate)
    comparisons = {
        reference: paired_comparison(
            derived,
            candidate=PRIMARY,
            reference=reference,
            seed_offset=index,
        )
        for index, reference in enumerate(REFERENCES)
    }
    support_checks = paired_support_checks(comparisons)
    temporal = load_temporal(args.temporal_csv)
    comprehensive, prompts = load_comprehensive(args.comprehensive_json)
    safety = {
        prompt: candidate_safety_flags(
            temporal,
            comprehensive,
            prompt=prompt,
        )
        for prompt in range(PROMPT_COUNT)
    }
    flagged = {prompt: flags for prompt, flags in safety.items() if flags}
    safety_gate = not flagged
    support_gate = all(bool(row["pass"]) for row in support_checks)
    candidate_gate = aggregate_gate and support_gate and safety_gate
    plan = review_plan(comparisons, safety, prompts)
    failed_aggregate = [row["name"] for row in aggregate_gates if not row["pass"]]
    if candidate_gate:
        recommendation = "targeted_review_then_heldout_confirmation"
    elif len(failed_aggregate) >= len(DEVELOPMENT_GATES) // 2:
        recommendation = "reject_stale_tie_and_design_multiscale_descriptor"
    else:
        recommendation = "targeted_review_before_keep_or_reject"
    return {
        "version": 1,
        "experiment": "v165_final_development_decision",
        "primary_candidate": PRIMARY,
        "mechanism_gate": True,
        "aggregate_development_gates": aggregate_gates,
        "aggregate_gate": aggregate_gate,
        "paired_support_checks": support_checks,
        "paired_support_gate": support_gate,
        "candidate_specific_safety_flags": [
            {"prompt_index": prompt, "flags": flags}
            for prompt, flags in sorted(flagged.items())
        ],
        "candidate_specific_safety_gate": safety_gate,
        "development_candidate_gate": candidate_gate,
        "recommendation": recommendation,
        "aggregate_scores": aggregate,
        "paired_vbench_comparisons": comparisons,
        "vbench_detail_scale_factors": scales,
        "review_plan": plan,
        "inputs": {
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": sha256(args.vbench_summary),
            "temporal_csv": str(args.temporal_csv.resolve()),
            "temporal_csv_sha256": sha256(args.temporal_csv),
            "comprehensive_json": str(args.comprehensive_json.resolve()),
            "comprehensive_json_sha256": sha256(args.comprehensive_json),
            "trace_report": str(args.trace_report.resolve()),
            "trace_report_sha256": sha256(args.trace_report),
            "published_manifest": str(args.published_manifest.resolve()),
            "published_manifest_sha256": sha256(args.published_manifest),
            "published_experiment": published["experiment"],
            "trace_experiment": trace["experiment"],
        },
        "claim_boundary": (
            "This adaptive 16-prompt analysis chooses the next development "
            "step. It is not held-out evidence, and its selected review clips "
            "cannot be reported as an unbiased human comparison."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v165 Final Development Decision",
        "",
        f"Primary candidate: `{report['primary_candidate']}`",
        "",
        f"Mechanism gate: **{report['mechanism_gate']}**  ",
        f"Aggregate gate: **{report['aggregate_gate']}**  ",
        f"Paired-support gate: **{report['paired_support_gate']}**  ",
        f"Candidate safety gate: **{report['candidate_specific_safety_gate']}**  ",
        f"Development candidate gate: **{report['development_candidate_gate']}**",
        "",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "## Frozen aggregate gates",
        "",
        "| Gate | Metric | Delta | Minimum | Pass |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["aggregate_development_gates"]:
        lines.append(
            f"| {row['name']} | {row['metric']} | {row['delta']:+.5f} | "
            f"{row['minimum_delta']:+.5f} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Paired VBench deltas",
            "",
            "| Reference | Metric | Mean | 95% CI | Positive prompts |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for reference in (DIRECTION_MATCH, SF, TIE_003):
        for metric, row in report["paired_vbench_comparisons"][reference].items():
            low, high = row["bootstrap_mean_ci95"]
            lines.append(
                f"| {reference} | {metric} | {row['mean_delta']:+.5f} | "
                f"[{low:+.5f}, {high:+.5f}] | "
                f"{row['positive_prompts']}/{PROMPT_COUNT} |"
            )
    lines.extend(["", "## Candidate-specific safety", ""])
    flags = report["candidate_specific_safety_flags"]
    if flags:
        for row in flags:
            lines.append(
                f"- Prompt {row['prompt_index']}: {', '.join(row['flags'])}"
            )
    else:
        lines.append("- No automatic candidate-specific safety flag.")
    lines.extend(["", "## Minimal review", ""])
    for row in report["review_plan"]["rows"]:
        lines.append(
            f"- Prompt {row['prompt_index']}: {', '.join(row['reasons'])}"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        "[v165-final-decision] "
        f"gate={report['development_candidate_gate']} "
        f"recommendation={report['recommendation']} "
        f"review_videos={report['review_plan']['video_count']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
