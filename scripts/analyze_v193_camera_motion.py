#!/usr/bin/env python3
"""Calibrate and compare v193 camera-compensated motion diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compute_v193_camera_motion import CSV_FIELDS, sha256

EXPERIMENT = "v193_camera_compensated_motion_calibration"
MOTION_IMPLEMENTATION = Path(__file__).with_name("compute_v193_camera_motion.py")
MAGNITUDE_METRIC = "residual_motion_p90_ndps_median"
COVERAGE_METRICS = (
    "residual_transition_active_fraction",
    "residual_active_area_fraction_mean",
)
MOTION_METRICS = (
    "raw_motion_ndps_median",
    "global_motion_ndps_median",
    "residual_motion_ndps_median",
    MAGNITUDE_METRIC,
    *COVERAGE_METRICS,
    "late_residual_motion_ratio",
    "longest_low_residual_run_fraction",
    "residual_accel_outlier_fraction",
    "camera_motion_fraction_median",
    "camera_fit_valid_fraction",
    "camera_fit_inlier_fraction_median",
    "camera_fit_error_nd_median",
    "residual_energy_concentration_mean",
    "residual_direction_entropy_mean",
)
DIRECTIONS = {
    "residual_motion_ndps_median": 1,
    MAGNITUDE_METRIC: 1,
    "residual_transition_active_fraction": 1,
    "residual_active_area_fraction_mean": 1,
    "late_residual_motion_ratio": 1,
    "longest_low_residual_run_fraction": -1,
    "residual_accel_outlier_fraction": -1,
}
QUALITY_MARGINS = {
    "official_quality_score": -0.15,
    "identity_background": -0.001,
    "temporal_mechanics": -0.002,
}


def _atomic_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric v193 value: {label}={value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite v193 value: {label}={value!r}")
    return result


def verify_motion_contract(
    manifest_path: Path, csv_path: Path, contract_path: Path
) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    methods = [str(row.get("key", "")) for row in manifest.get("methods") or ()]
    prompt_count = int(manifest.get("prompt_count", -1))
    if (
        contract.get("kind") != "merged"
        or int(contract.get("version", -1)) != 1
        or contract.get("methods") != methods
        or int(contract.get("prompt_count", -1)) != prompt_count
        or int(contract.get("row_count", -1)) != len(methods) * prompt_count
        or contract.get("comparison_manifest_sha256") != sha256(manifest_path)
        or Path(str(contract.get("output_csv", ""))).resolve() != csv_path.resolve()
        or contract.get("output_csv_sha256") != sha256(csv_path)
        or contract.get("implementation_sha256") != sha256(MOTION_IMPLEMENTATION)
    ):
        raise ValueError("v193 camera-motion provenance contract is invalid or drifted")
    return manifest, contract


def load_rows(
    manifest: dict, csv_path: Path
) -> dict[tuple[str, int], dict[str, float | str]]:
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    video_dirs = {
        str(row["key"]): Path(str(row["video_dir"])) for row in manifest["methods"]
    }
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("v193 camera-motion CSV schema drifted")
        for raw in reader:
            method = str(raw["method"])
            prompt = int(raw["prompt_index"])
            key = (method, prompt)
            if key in rows:
                raise ValueError(f"duplicate v193 camera-motion row: {key}")
            if method not in video_dirs or not 0 <= prompt < prompt_count:
                raise ValueError(f"invalid v193 camera-motion key: {key}")
            expected_video = video_dirs[method] / f"{prompt:06d}-0.mp4"
            actual_video = Path(raw["video"])
            if (
                int(raw["sample_index"]) != 0
                or not expected_video.is_file()
                or not actual_video.is_file()
                or not actual_video.samefile(expected_video)
            ):
                raise ValueError(
                    f"v193 row is not bound to its comparison video: {key}"
                )
            rows[key] = {
                "video": str(actual_video.resolve()),
                **{
                    metric: finite(raw[metric], label=f"{key}:{metric}")
                    for metric in MOTION_METRICS
                },
            }
    expected = {
        (method, prompt) for method in methods for prompt in range(prompt_count)
    }
    if set(rows) != expected:
        raise ValueError(
            "v193 motion row grid mismatch: "
            f"missing={sorted(expected - set(rows))[:12]} "
            f"extra={sorted(set(rows) - expected)[:12]}"
        )
    return rows


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_rank = _ranks(np.asarray(left, dtype=np.float64))
    right_rank = _ranks(np.asarray(right, dtype=np.float64))
    if float(left_rank.std()) <= 1e-15 or float(right_rank.std()) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def calibrate_metrics(
    rows: dict, *, methods: tuple[str, ...], prompt_count: int
) -> dict:
    calibration = {}
    vectors = {}
    for metric in MOTION_METRICS:
        values = np.asarray(
            [
                rows[(method, prompt)][metric]
                for method in methods
                for prompt in range(prompt_count)
            ],
            dtype=np.float64,
        )
        vectors[metric] = values.tolist()
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        value_range = float(values.max() - values.min())
        unique = int(np.unique(values).size)
        dominant_fraction = float(
            max(np.mean(values == values.min()), np.mean(values == values.max()))
        )
        numerical_floor = max(1e-12, 1e-6 * max(abs(median), 1e-6))
        informative = bool(
            value_range > numerical_floor
            and unique >= min(8, max(3, prompt_count // 16))
            and dominant_fraction < 0.98
        )
        calibration[metric] = {
            "informative": informative,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": value_range,
            "median": median,
            "mad": mad,
            "standard_deviation": float(values.std()),
            "unique_value_count": unique,
            "dominant_boundary_fraction": dominant_fraction,
            "observation_count": int(values.size),
            "gate": {
                "range_above_numerical_floor": value_range > numerical_floor,
                "enough_unique_values": unique >= min(8, max(3, prompt_count // 16)),
                "not_boundary_saturated": dominant_fraction < 0.98,
            },
        }
    redundant = []
    ordered = list(MOTION_METRICS)
    for left_index, left in enumerate(ordered):
        if not calibration[left]["informative"]:
            continue
        for right in ordered[left_index + 1 :]:
            if not calibration[right]["informative"]:
                continue
            rho = spearman(vectors[left], vectors[right])
            if abs(rho) >= 0.98:
                redundant.append({"left": left, "right": right, "spearman": rho})
    return {
        "metrics": calibration,
        "redundant_pairs_abs_spearman_ge_0p98": redundant,
        "claim_boundary": (
            "Calibration checks numerical discrimination and redundancy only; "
            "it does not validate correspondence with human motion judgments."
        ),
    }


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    wins = sum(value > 0 for value in nonzero)
    total = len(nonzero)
    upper = sum(math.comb(total, count) for count in range(wins, total + 1)) / 2**total
    lower = sum(math.comb(total, count) for count in range(wins + 1)) / 2**total
    return min(1.0, 2.0 * min(upper, lower))


def paired_contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    metric: str,
    prompt_count: int,
    seed: int,
) -> dict:
    direction = DIRECTIONS.get(metric, 1)
    raw = [
        float(rows[(candidate, prompt)][metric])
        - float(rows[(control, prompt)][metric])
        for prompt in range(prompt_count)
    ]
    oriented = [direction * value for value in raw]
    return {
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "preferred_direction": "higher" if direction > 0 else "lower",
        "mean_delta": float(np.mean(raw)),
        "mean_oriented_delta": float(np.mean(oriented)),
        "median_oriented_delta": float(np.median(oriented)),
        "oriented_win_fraction": float(np.mean(np.asarray(oriented) > 0)),
        "bootstrap_ci95_oriented": bootstrap_ci(oriented, seed=seed),
        "sign_p_two_sided": sign_p(oriented),
        "per_prompt_delta": raw,
    }


def _automatic_flags(
    rows: dict,
    *,
    candidate: str,
    control: str,
    prompt_count: int,
) -> list[dict]:
    flagged = []
    for prompt in range(prompt_count):
        current = rows[(candidate, prompt)]
        reference = rows[(control, prompt)]
        flags = []
        if (
            current["camera_fit_valid_fraction"] < 0.90
            and current["camera_fit_valid_fraction"]
            < reference["camera_fit_valid_fraction"] - 0.05
        ):
            flags.append("camera_model_unreliable")
        if (
            current["longest_low_residual_run_fraction"] > 0.20
            and current["longest_low_residual_run_fraction"]
            > reference["longest_low_residual_run_fraction"] + 0.10
        ):
            flags.append("long_local_motion_collapse")
        if (
            current["late_residual_motion_ratio"] < 0.55
            and current["late_residual_motion_ratio"]
            < reference["late_residual_motion_ratio"] - 0.20
        ):
            flags.append("late_local_motion_collapse")
        if (
            current["residual_accel_outlier_fraction"] > 0.05
            and current["residual_accel_outlier_fraction"]
            > reference["residual_accel_outlier_fraction"] + 0.02
        ):
            flags.append("local_motion_discontinuity")
        if flags:
            flagged.append({"prompt_index": prompt, "flags": flags})
    return flagged


def load_quality_context(
    path: Path | None, *, candidate: str, controls: tuple[str, ...]
) -> dict:
    if path is None:
        return {
            "available": False,
            "reason": "No paired VBench analysis was supplied.",
            "all_controls_noninferior": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons") or ()
    rows = {}
    for control in controls:
        metrics = {}
        for metric, margin in QUALITY_MARGINS.items():
            matches = [
                row
                for row in comparisons
                if row.get("candidate") == candidate
                and row.get("control") == control
                and row.get("metric") == metric
                and row.get("window", "full") == "full"
            ]
            if len(matches) != 1:
                continue
            mean = finite(
                matches[0].get("mean_delta"), label=f"quality:{control}:{metric}"
            )
            metrics[metric] = {
                "mean_delta": mean,
                "noninferiority_margin": margin,
                "noninferior": mean >= margin,
            }
        rows[control] = {
            "complete": set(metrics) == set(QUALITY_MARGINS),
            "noninferior": set(metrics) == set(QUALITY_MARGINS)
            and all(row["noninferior"] for row in metrics.values()),
            "metrics": metrics,
        }
    complete = set(rows) == set(controls) and all(
        row["complete"] for row in rows.values()
    )
    return {
        "available": complete,
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "controls": rows,
        "all_controls_noninferior": complete
        and all(row["noninferior"] for row in rows.values()),
        "margins_are_development_tolerances": True,
    }


def _review_queue(
    manifest: dict,
    rows: dict,
    *,
    candidate: str,
    controls: tuple[str, ...],
    flags: dict[int, set[str]],
    limit: int = 4,
) -> list[dict]:
    prompt_count = int(manifest["prompt_count"])
    ranked = []
    for prompt in range(prompt_count):
        magnitudes = []
        coverages = []
        for control in controls:
            magnitudes.append(
                float(rows[(candidate, prompt)][MAGNITUDE_METRIC])
                - float(rows[(control, prompt)][MAGNITUDE_METRIC])
            )
            coverages.append(
                float(rows[(candidate, prompt)]["residual_active_area_fraction_mean"])
                - float(rows[(control, prompt)]["residual_active_area_fraction_mean"])
            )
        scale_magnitude = max(
            1e-8,
            float(
                np.median(
                    [
                        rows[(method, prompt)][MAGNITUDE_METRIC]
                        for method in (candidate, *controls)
                    ]
                )
            ),
        )
        disagreement = max(abs(value) for value in magnitudes) / scale_magnitude
        disagreement += 2.0 * max(abs(value) for value in coverages)
        score = disagreement + 10.0 * bool(flags.get(prompt))
        ranked.append((score, prompt, magnitudes, coverages))
    video_dirs = {
        str(row["key"]): Path(str(row["video_dir"])) for row in manifest["methods"]
    }
    queue = []
    for score, prompt, magnitudes, coverages in sorted(ranked, reverse=True)[:limit]:
        prompt_items = manifest.get("prompt_items") or ()
        item = prompt_items[prompt] if len(prompt_items) == prompt_count else {}
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": item.get("source_index"),
                "prompt": item.get("text"),
                "priority": float(score),
                "automatic_flags": sorted(flags.get(prompt, ())),
                "magnitude_deltas": magnitudes,
                "active_area_deltas": coverages,
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in (candidate, *controls)
                },
            }
        )
    return queue


def analyze(
    manifest: dict,
    rows: dict,
    *,
    candidate: str,
    controls: tuple[str, ...],
    quality_context: dict,
) -> dict:
    methods = tuple(str(row["key"]) for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if (
        candidate not in methods
        or not controls
        or any(control not in methods for control in controls)
    ):
        raise ValueError("candidate and controls must be distinct comparison methods")
    if candidate in controls or len(set(controls)) != len(controls):
        raise ValueError("candidate/control roles overlap")
    calibration = calibrate_metrics(rows, methods=methods, prompt_count=prompt_count)
    comparisons = []
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(DIRECTIONS):
            comparisons.append(
                paired_contrast(
                    rows,
                    candidate=candidate,
                    control=control,
                    metric=metric,
                    prompt_count=prompt_count,
                    seed=193000 + 101 * control_index + metric_index,
                )
            )
    by_pair = {(row["control"], row["metric"]): row for row in comparisons}
    flags_by_control = {
        control: _automatic_flags(
            rows,
            candidate=candidate,
            control=control,
            prompt_count=prompt_count,
        )
        for control in controls
    }
    allowed_flags = max(1, math.ceil(prompt_count / 32))
    all_flags: dict[int, set[str]] = {}
    for control, items in flags_by_control.items():
        for item in items:
            all_flags.setdefault(item["prompt_index"], set()).update(
                f"{control}:{flag}" for flag in item["flags"]
            )
    calibration_ok = bool(
        calibration["metrics"][MAGNITUDE_METRIC]["informative"]
        and any(
            calibration["metrics"][metric]["informative"] for metric in COVERAGE_METRICS
        )
        and all(
            float(
                np.mean(
                    [
                        rows[(method, prompt)]["camera_fit_valid_fraction"]
                        for prompt in range(prompt_count)
                    ]
                )
            )
            >= 0.90
            for method in methods
        )
    )
    per_control = {}
    for control in controls:
        magnitude = by_pair[(control, MAGNITUDE_METRIC)]
        coverage = [by_pair[(control, metric)] for metric in COVERAGE_METRICS]
        safety = len(flags_by_control[control]) <= allowed_flags
        directional = bool(
            calibration_ok
            and magnitude["mean_oriented_delta"] > 0
            and any(row["mean_oriented_delta"] > 0 for row in coverage)
            and safety
        )
        strong = bool(
            directional
            and magnitude["bootstrap_ci95_oriented"][0] > 0
            and any(
                row["mean_oriented_delta"] > 0
                and calibration["metrics"][row["metric"]]["informative"]
                for row in coverage
            )
        )
        raw_increase = float(
            np.mean(
                [
                    rows[(candidate, prompt)]["raw_motion_ndps_median"]
                    - rows[(control, prompt)]["raw_motion_ndps_median"]
                    for prompt in range(prompt_count)
                ]
            )
        )
        residual_increase = magnitude["mean_delta"]
        per_control[control] = {
            "directional_local_motion_signal": directional,
            "strong_local_motion_signal": strong,
            "automatic_safety_pass": safety,
            "flagged_prompt_count": len(flags_by_control[control]),
            "allowed_flagged_prompt_count": allowed_flags,
            "flagged_prompts": flags_by_control[control],
            "mean_raw_motion_delta": raw_increase,
            "mean_residual_motion_delta": residual_increase,
            "camera_only_motion_increase": raw_increase > 0 and residual_increase <= 0,
        }
    directional_all = calibration_ok and all(
        row["directional_local_motion_signal"] for row in per_control.values()
    )
    strong_all = calibration_ok and all(
        row["strong_local_motion_signal"] for row in per_control.values()
    )
    if not calibration_ok:
        recommendation = "camera_compensated_motion_measurement_not_discriminative"
    elif not directional_all:
        recommendation = "candidate_local_motion_gain_not_supported"
    elif quality_context.get("available") and not quality_context.get(
        "all_controls_noninferior"
    ):
        recommendation = "candidate_local_motion_quality_tradeoff"
    elif quality_context.get("available"):
        recommendation = (
            "camera_compensated_motion_gain_with_quality_noninferiority"
            if strong_all
            else "directional_local_motion_gain_with_quality_noninferiority"
        )
    else:
        recommendation = "local_motion_signal_promising_requires_quality_pairing"
    review_queue = _review_queue(
        manifest,
        rows,
        candidate=candidate,
        controls=controls,
        flags=all_flags,
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "candidate": candidate,
        "controls": list(controls),
        "methods": list(methods),
        "prompt_count": prompt_count,
        "metric_calibration": calibration,
        "comparisons": comparisons,
        "control_status": per_control,
        "measurement_calibration_pass": calibration_ok,
        "directional_local_motion_signal_against_all_controls": directional_all,
        "strong_local_motion_signal_against_all_controls": strong_all,
        "quality_context": quality_context,
        "recommendation": recommendation,
        "manual_review_required": False,
        "targeted_review_recommended_only_after_automatic_pass": directional_all,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": review_queue if directional_all else [],
        "claim_boundary": (
            "v193 distinguishes residual local flow from global camera flow. It is a "
            "predeclared automatic diagnostic, not a validated perceptual metric. A "
            "motion claim still requires paired quality/identity evidence and targeted "
            "human calibration."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--motion-contract", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controls = tuple(
        value.strip() for value in args.controls.split(",") if value.strip()
    )
    manifest, motion_contract = verify_motion_contract(
        args.comparison_manifest, args.motion_csv, args.motion_contract
    )
    rows = load_rows(manifest, args.motion_csv)
    quality = load_quality_context(
        args.quality_report, candidate=args.candidate, controls=controls
    )
    report = analyze(
        manifest,
        rows,
        candidate=args.candidate,
        controls=controls,
        quality_context=quality,
    )
    report["source"] = {
        "comparison_manifest": str(args.comparison_manifest.resolve()),
        "comparison_manifest_sha256": sha256(args.comparison_manifest),
        "motion_csv": str(args.motion_csv.resolve()),
        "motion_csv_sha256": sha256(args.motion_csv),
        "motion_contract": str(args.motion_contract.resolve()),
        "motion_contract_sha256": sha256(args.motion_contract),
        "motion_implementation_sha256": motion_contract["implementation_sha256"],
    }
    digest = _atomic_json(args.output, report)
    print(
        f"[v193-analysis] recommendation={report['recommendation']} "
        f"calibration={str(report['measurement_calibration_pass']).lower()} "
        f"directional={str(report['directional_local_motion_signal_against_all_controls']).lower()} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
