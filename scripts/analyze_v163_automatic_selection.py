#!/usr/bin/env python3
"""Select a v163 candidate automatically and cap optional review at six videos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

import analyze_v162_metric_human_calibration as calibration
from prepare_v163_vbench_comparison import METHODS, PROMPT_COUNT


ROOT = Path(__file__).resolve().parents[1]
AGE12 = "ours_middle10_reservoir2_stateage12motionpair1"
BALANCED = "ours_middle10_reservoir2_statebalancedmotionpair1"
SF = "sf_native"
LEGACY = "ours_middle10_reservoir2_statemotionpair1_reference"
FRESH = "ours_middle10_reservoir2_freshmotionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
CANDIDATES = (AGE12, BALANCED)
REFERENCES = (SF, LEGACY, FRESH, RESERVOIR)
SAFETY_REFERENCES = (LEGACY, FRESH, RESERVOIR)
TARGETS = calibration.TARGETS

# These diagnostics only localize failures and review prompts. Learned VBench
# calibration supplies the candidate-comparison signal.
TEMPORAL_FEATURES = {
    "motion_coverage_fraction": 1,
    "late_motion_ratio": 1,
    "longest_low_motion_run_fraction": -1,
    "temporal_jump": -1,
    "appearance_outlier_fraction": -1,
    "flow_accel_outlier_fraction": -1,
    "dark_frame_fraction": -1,
    "bright_frame_fraction": -1,
    "low_contrast_frame_fraction": -1,
    "edge_density_outlier_fraction": -1,
}
COMPREHENSIVE_FEATURES = {
    "m1_dino_consistency": 1,
    "m1_min_stability": 1,
    "m1_first_last_gap": -1,
    "m2_drift_slope": 1,
    "m3_motion_smoothness": -1,
    "m4_arcface_id_sim": 1,
    "m5_temporal_flickering": -1,
    "m5_max_flicker": -1,
    "m6_clip_text_alignment": 1,
    "m6_clip_text_min": 1,
    "m7_background_consistency": 1,
    "m7_background_drift": -1,
    "m8_loop_score": -1,
}
SEVERE_FLAGS = {
    "luminance_or_contrast_failure",
    "temporal_discontinuity",
    "edge_density_failure",
    "subject_consistency_drop",
    "background_drift",
    "severe_flicker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-parts-root", required=True, type=Path)
    parser.add_argument("--temporal-csv", required=True, type=Path)
    parser.add_argument("--comprehensive-json", required=True, type=Path)
    parser.add_argument("--trace-report", required=True, type=Path)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v162_automatic_calibration"
            / "analysis"
            / "v162_metric_human_calibration.json"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def validate_coverage(rows: dict[tuple[str, int], dict], *, label: str) -> None:
    expected = {
        (method, prompt)
        for method in METHODS
        for prompt in range(PROMPT_COUNT)
    }
    actual = set(rows)
    if actual != expected:
        raise ValueError(
            f"{label} coverage mismatch: missing={sorted(expected-actual)[:12]} "
            f"extra={sorted(actual-expected)[:12]}"
        )


def load_temporal(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw["method"]), int(raw["prompt_index"]))
            if key in rows:
                raise ValueError(f"duplicate temporal row: {key}")
            rows[key] = {
                feature: finite(raw.get(feature), name=f"{key}:{feature}")
                for feature in TEMPORAL_FEATURES
            }
    validate_coverage(rows, label="temporal")
    return rows


def load_comprehensive(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_video = payload.get("per_video")
    if not isinstance(per_video, dict):
        raise ValueError("comprehensive result has no per_video mapping")
    rows = {}
    for raw in per_video.values():
        key = (str(raw["method"]), int(raw["prompt_index"]))
        if key in rows:
            raise ValueError(f"duplicate comprehensive row: {key}")
        metrics = raw.get("metrics") or {}
        rows[key] = {
            feature: finite(value, name=f"{key}:{feature}")
            for feature in COMPREHENSIVE_FEATURES
            if (value := metrics.get(feature)) is not None
        }
    validate_coverage(rows, label="comprehensive")
    return rows


def robust_scale(values: list[float]) -> float:
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    if mad > 1e-9:
        return 1.4826 * mad
    return max((max(values) - min(values)) / 4.0, 1e-6)


def feature_scales(rows: dict[tuple[str, int], dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for metrics in rows.values():
        for feature, value in metrics.items():
            values.setdefault(feature, []).append(value)
    return {
        feature: robust_scale(feature_values)
        for feature, feature_values in values.items()
        if len(feature_values) >= PROMPT_COUNT
    }


def automatic_flags(
    temporal: dict[tuple[str, int], dict[str, float]],
    comprehensive: dict[tuple[str, int], dict[str, float]],
    *,
    candidate: str,
    prompt: int,
) -> list[str]:
    primary = temporal[(candidate, prompt)]
    refs = [temporal[(method, prompt)] for method in SAFETY_REFERENCES]
    flags = []
    if (
        primary["longest_low_motion_run_fraction"] > 0.20
        and primary["longest_low_motion_run_fraction"]
        > max(row["longest_low_motion_run_fraction"] for row in refs) + 0.10
    ):
        flags.append("long_low_motion_run")
    if (
        primary["late_motion_ratio"] < 0.55
        and primary["late_motion_ratio"]
        < min(row["late_motion_ratio"] for row in refs) - 0.20
    ):
        flags.append("late_motion_collapse")
    if (
        primary["temporal_jump"] > 1.35 * max(row["temporal_jump"] for row in refs)
        and primary["appearance_outlier_fraction"]
        > max(row["appearance_outlier_fraction"] for row in refs) + 0.02
    ):
        flags.append("temporal_discontinuity")
    if (
        primary["dark_frame_fraction"] > 0.02
        or primary["bright_frame_fraction"] > 0.02
        or primary["low_contrast_frame_fraction"] > 0.05
    ):
        flags.append("luminance_or_contrast_failure")
    if (
        primary["edge_density_outlier_fraction"] > 0.10
        and primary["edge_density_outlier_fraction"]
        > max(row["edge_density_outlier_fraction"] for row in refs) + 0.05
    ):
        flags.append("edge_density_failure")

    metrics = comprehensive[(candidate, prompt)]
    ref_metrics = [
        comprehensive[(method, prompt)] for method in SAFETY_REFERENCES
    ]

    def all_have(feature: str) -> bool:
        return feature in metrics and all(feature in row for row in ref_metrics)

    if (
        all_have("m1_dino_consistency")
        and metrics["m1_dino_consistency"]
        < min(row["m1_dino_consistency"] for row in ref_metrics) - 0.03
    ):
        flags.append("subject_consistency_drop")
    if (
        all_have("m7_background_drift")
        and metrics["m7_background_drift"]
        > max(row["m7_background_drift"] for row in ref_metrics) + 0.05
    ):
        flags.append("background_drift")
    if (
        all_have("m5_max_flicker")
        and metrics["m5_max_flicker"]
        > 1.35 * max(row["m5_max_flicker"] for row in ref_metrics)
    ):
        flags.append("severe_flicker")
    return flags


def bootstrap_ci(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(5000)
    ]
    means.sort()
    return [means[124], means[4874]]


def predict_delta(model: dict, delta: np.ndarray) -> float:
    weights = model.get("feature_weights") or {}
    if set(weights) != set(calibration.FEATURE_NAMES):
        raise ValueError("calibration model feature contract mismatch")
    vector = np.asarray(
        [float(weights[name]) for name in calibration.FEATURE_NAMES],
        dtype=np.float64,
    )
    return float(delta @ vector)


def prediction_table(
    features: dict[tuple[str, int], np.ndarray],
    models: dict,
) -> dict[str, Any]:
    result = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        comparisons = {}
        for reference_index, reference in enumerate(REFERENCES):
            target_rows = {}
            for target_index, target in enumerate(TARGETS):
                values = [
                    predict_delta(
                        models[target],
                        features[(candidate, prompt)]
                        - features[(reference, prompt)],
                    )
                    for prompt in range(PROMPT_COUNT)
                ]
                target_rows[target] = {
                    "mean_predicted_delta": statistics.mean(values),
                    "median_predicted_delta": statistics.median(values),
                    "positive_prompts": sum(value > 0.0 for value in values),
                    "bootstrap_mean_ci95": bootstrap_ci(
                        values,
                        seed=(
                            1632026
                            + 100 * candidate_index
                            + 10 * reference_index
                            + target_index
                        ),
                    ),
                    "per_prompt": [
                        {"prompt_index": prompt, "predicted_delta": value}
                        for prompt, value in enumerate(values)
                    ],
                }
            comparisons[reference] = target_rows
        result[candidate] = comparisons
    return result


def oriented_risk_rows(
    temporal: dict[tuple[str, int], dict[str, float]],
    comprehensive: dict[tuple[str, int], dict[str, float]],
    predictions: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    temporal_scales = feature_scales(temporal)
    comprehensive_scales = feature_scales(comprehensive)
    result = {}
    for candidate in CANDIDATES:
        rows = []
        for prompt in range(PROMPT_COUNT):
            negative = []
            for reference in REFERENCES:
                for metrics, directions, scales in (
                    (temporal, TEMPORAL_FEATURES, temporal_scales),
                    (comprehensive, COMPREHENSIVE_FEATURES, comprehensive_scales),
                ):
                    left = metrics[(candidate, prompt)]
                    right = metrics[(reference, prompt)]
                    for feature, direction in directions.items():
                        if feature not in left or feature not in right or feature not in scales:
                            continue
                        value = direction * (left[feature] - right[feature]) / scales[feature]
                        if value < 0.0:
                            negative.append(-value)
            flags = automatic_flags(
                temporal,
                comprehensive,
                candidate=candidate,
                prompt=prompt,
            )
            target_values = [
                predictions[candidate][reference][target]["per_prompt"][prompt][
                    "predicted_delta"
                ]
                for reference in REFERENCES
                for target in TARGETS
            ]
            rows.append(
                {
                    "prompt_index": prompt,
                    "automatic_flags": flags,
                    "severe_flags": sorted(set(flags) & SEVERE_FLAGS),
                    "risk_score": (
                        statistics.mean(negative) if negative else 0.0
                    )
                    + 2.0 * len(flags),
                    "prediction_disagreement": statistics.pstdev(target_values),
                }
            )
        result[candidate] = rows
    return result


def candidate_checks(
    *,
    candidate: str,
    calibration_gate: bool,
    trace: dict,
    predictions: dict[str, Any],
    risks: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_row = (trace.get("methods") or {}).get(candidate) or {}
    comparison_checks = {}
    for reference in REFERENCES:
        targets = predictions[candidate][reference]
        comparison_checks[reference] = {
            "overall_mean_positive": (
                targets["overall"]["mean_predicted_delta"] > 0.0
            ),
            "overall_prompt_majority": targets["overall"]["positive_prompts"] >= 9,
            "overall_ci_not_materially_negative": (
                targets["overall"]["bootstrap_mean_ci95"][0] >= -0.10
            ),
            "identity_noninferior": (
                targets["identity"]["mean_predicted_delta"] >= -0.05
            ),
            "background_noninferior": (
                targets["background"]["mean_predicted_delta"] >= -0.05
            ),
            "motion_mean_positive": targets["motion"]["mean_predicted_delta"] > 0.0,
        }
    severe = [row["prompt_index"] for row in risks if row["severe_flags"]]
    checks = {
        "calibration_transferred": bool(calibration_gate),
        "trace_mechanism": trace_row.get("mechanism_gate") is True,
        "trace_freshness": trace_row.get("freshness_gate") is True,
        "no_automatic_severe_failure": not severe,
        "all_reference_comparisons": all(
            all(values.values()) for values in comparison_checks.values()
        ),
    }
    score = statistics.mean(
        predictions[candidate][reference]["overall"]["mean_predicted_delta"]
        + 0.5
        * predictions[candidate][reference]["motion"]["mean_predicted_delta"]
        + 0.25
        * predictions[candidate][reference]["identity"]["mean_predicted_delta"]
        for reference in REFERENCES
    )
    return {
        "checks": checks,
        "comparison_checks": comparison_checks,
        "automatic_severe_prompt_indices": severe,
        "selection_score": score,
        "passes": all(checks.values()),
    }


def review_recommendation(
    candidates: dict[str, Any],
    predictions: dict[str, Any],
    risks: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    eligible = [candidate for candidate in CANDIDATES if candidates[candidate]["passes"]]
    if not eligible:
        return {
            "mode": "none",
            "winner": None,
            "reference": None,
            "blind_prompt_indices": [],
            "safety_extra_prompt_indices": [],
            "manual_video_count": 0,
            "reason": "no candidate passed the automatic mechanism, safety, and calibrated comparison gates",
        }
    winner = max(
        eligible,
        key=lambda candidate: (candidates[candidate]["selection_score"], candidate),
    )
    reference = min(
        REFERENCES,
        key=lambda method: predictions[winner][method]["overall"][
            "mean_predicted_delta"
        ],
    )
    overall = {
        row["prompt_index"]: row["predicted_delta"]
        for row in predictions[winner][reference]["overall"]["per_prompt"]
    }
    risk_by_prompt = {row["prompt_index"]: row for row in risks[winner]}
    weakest = min(overall, key=lambda prompt: (overall[prompt], prompt))
    median_value = statistics.median(overall.values())
    typical = min(
        (prompt for prompt in overall if prompt != weakest),
        key=lambda prompt: (abs(overall[prompt] - median_value), prompt),
    )
    blind = [weakest, typical]
    extras = [
        row["prompt_index"]
        for row in sorted(
            risks[winner],
            key=lambda row: (-row["risk_score"], row["prompt_index"]),
        )
        if row["prompt_index"] not in blind and row["automatic_flags"]
    ][:2]
    count = 2 * len(blind) + len(extras)
    if count < 4 or count > 6:
        raise ValueError("v163 review budget must remain within four to six videos")
    return {
        "mode": "conditional_blind",
        "winner": winner,
        "reference": reference,
        "blind_prompt_indices": blind,
        "safety_extra_prompt_indices": extras,
        "manual_video_count": count,
        "reason": (
            "review only the automatic winner against its strongest reference; "
            "add at most two flagged winner-only safety clips"
        ),
        "prompt_diagnostics": {
            str(prompt): risk_by_prompt[prompt] for prompt in blind + extras
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# v163 Automatic Candidate Selection",
        "",
        f"Automatic winner: **{report['review_recommendation']['winner']}**",
        f"Manual videos requested: **{report['review_recommendation']['manual_video_count']}**",
        "",
        "| Candidate | Pass | Score | Severe prompts |",
        "|---|---:|---:|---|",
    ]
    for candidate in CANDIDATES:
        row = report["candidate_gates"][candidate]
        lines.append(
            f"| {candidate} | {row['passes']} | {row['selection_score']:.4f} | "
            f"{row['automatic_severe_prompt_indices']} |"
        )
    lines.extend(
        [
            "",
            "If no candidate passes, no videos are sent to human review. This is an engineering selection protocol, not a paper human study.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    required = (
        args.temporal_csv,
        args.comprehensive_json,
        args.trace_report,
        args.calibration_report,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing v163 selection inputs: {missing}")
    calibration_report = json.loads(
        args.calibration_report.read_text(encoding="utf-8")
    )
    trace = json.loads(args.trace_report.read_text(encoding="utf-8"))
    if (
        calibration_report.get("experiment") != "v162_metric_human_calibration"
        or tuple(calibration_report.get("feature_names") or ())
        != calibration.FEATURE_NAMES
        or trace.get("experiment")
        != "v163_recency_regularized_state_motion_trace"
        or trace.get("contract_gate") is not True
    ):
        raise ValueError("v162 calibration or v163 trace contract mismatch")
    features = calibration.load_features(args.vbench_parts_root, METHODS)
    temporal = load_temporal(args.temporal_csv)
    comprehensive = load_comprehensive(args.comprehensive_json)
    predictions = prediction_table(features, calibration_report.get("models") or {})
    risks = oriented_risk_rows(temporal, comprehensive, predictions)
    gates = {
        candidate: candidate_checks(
            candidate=candidate,
            calibration_gate=bool(calibration_report.get("calibration_gate")),
            trace=trace,
            predictions=predictions,
            risks=risks[candidate],
        )
        for candidate in CANDIDATES
    }
    recommendation = review_recommendation(gates, predictions, risks)
    report = {
        "version": 1,
        "experiment": "v163_automatic_candidate_selection",
        "methods": list(METHODS),
        "candidates": list(CANDIDATES),
        "references": list(REFERENCES),
        "calibration_gate": bool(calibration_report.get("calibration_gate")),
        "predictions": predictions,
        "automatic_risk": risks,
        "candidate_gates": gates,
        "review_recommendation": recommendation,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (
                ("temporal_csv", args.temporal_csv),
                ("comprehensive_json", args.comprehensive_json),
                ("trace_report", args.trace_report),
                ("calibration_report", args.calibration_report),
            )
        },
        "vbench_feature_sha256": calibration.feature_digest(features),
        "claim_boundary": (
            "adaptive automatic selection and its optional review are development evidence only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        "[v163-selection] "
        f"winner={recommendation['winner']} "
        f"review_videos={recommendation['manual_video_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
