#!/usr/bin/env python3
"""Calibrate automatic VBench signals against frozen human reviews."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prepare_v162_vbench_comparison import METHODS as V162_METHODS


ROOT = Path(__file__).resolve().parents[1]
PROMPT_COUNT = 16
CLIPS_PER_VIDEO = 15
MODEL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "overall_consistency",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
MIN_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "overall_consistency",
)
TREND_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "overall_consistency",
    "dynamic_degree",
)
FEATURE_NAMES = (
    *(f"{name}_mean" for name in MODEL_DIMENSIONS),
    *(f"{name}_min" for name in MIN_DIMENSIONS),
    *(f"{name}_late_minus_early" for name in TREND_DIMENSIONS),
    "dynamic_degree_std",
)
TARGETS = ("identity", "background", "motion", "overall")
V157_METHODS = (
    "ours_layer_interleaved10_reservoir4",
    "ours_layer_middle10_reservoir4",
    "ours_all_reservoir4_reference",
    "ours_all_recent8_reference",
)
PRIMARY = "ours_middle10_reservoir2_statemotionpair1"
FRESH = "ours_middle10_reservoir2_freshmotionpair1_reference"
OLD_MOTION = "ours_middle10_reservoir2_motionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
V160_METHOD_MAP = {
    "ours_middle10_reservoir2_freshmotionpair1": FRESH,
    OLD_MOTION: OLD_MOTION,
    RESERVOIR: RESERVOIR,
}
V160_TARGET_COLUMNS = {
    "identity": ("identity_continuity_-2_to_2",),
    "background": ("background_continuity_-2_to_2",),
    "motion": (
        "motion_naturalness_-2_to_2",
        "late_motion_stability_-2_to_2",
    ),
    "overall": ("overall_preference_-2_to_2",),
}
LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PATH_PATTERN = re.compile(
    r"(?:^|/)split_clip/(\d{6})-0/\1-0_(\d{3})\.mp4(?:/|$)"
)
FALLBACK_PATTERN = re.compile(r"(?:^|/)(\d{6})-0_(\d{3})\.mp4(?:/|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v157-parts-root",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v157_layer_gated_moviebench16"
            / "full8"
            / "metrics"
            / "vbench_long_parts"
        ),
    )
    parser.add_argument(
        "--v162-parts-root",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v162_automatic_calibration"
            / "vbench_parts"
        ),
    )
    parser.add_argument(
        "--v157-key",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "v157_layer_gated_moviebench16"
            / "metric_screened_review"
            / "v157_metric_screened_blind_key.json"
        ),
    )
    parser.add_argument(
        "--v157-review",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "v157_layer_gated_moviebench16"
            / "metric_screened_review"
            / "v157_metric_screened_review.csv"
        ),
    )
    parser.add_argument(
        "--v160-key",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "v160_fresh_motion_moviebench16"
            / "v160_wave1_blind_key.json"
        ),
    )
    parser.add_argument(
        "--v160-review",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "v160_fresh_motion_moviebench16"
            / "v160_wave1_review_sheet.csv"
        ),
    )
    parser.add_argument(
        "--v160-analysis",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v160_fresh_motion_moviebench16"
            / "full8"
            / "automated_screen"
            / "adaptive_review_analysis.json"
        ),
    )
    parser.add_argument(
        "--v160-wave2-review",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v160_fresh_motion_moviebench16"
            / "full8"
            / "adaptive_review"
            / "wave2"
            / "reviewer"
            / "v160_wave2_review_sheet.csv"
        ),
    )
    parser.add_argument(
        "--v161-screen",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "v161_state_matched_motion_moviebench16"
            / "automated_screen.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "v162_automatic_calibration"
            / "analysis"
            / "v162_metric_human_calibration.json"
        ),
    )
    parser.add_argument(
        "--history-preflight",
        action="store_true",
        help="Validate frozen v157/v160 labels and v157 clip metrics only.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {number}")
    return number


def candidate_record_lists(value: Any) -> Iterable[list[dict[str, Any]]]:
    """Yield VBench detail lists without mixing duplicated result views."""
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
    match = PATH_PATTERN.search(normalized) or FALLBACK_PATTERN.search(
        normalized
    )
    if match is None:
        raise ValueError(f"cannot recover prompt/clip from {video_path}")
    return int(match.group(1)), int(match.group(2))


def load_dimension(path: Path, dimension: str) -> dict[int, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or dimension not in payload:
        raise ValueError(f"invalid {dimension} result: {path}")
    expected_prompts = set(range(PROMPT_COUNT))
    expected_clips = set(range(CLIPS_PER_VIDEO))
    failures = []
    for candidate_index, records in enumerate(
        candidate_record_lists(payload[dimension])
    ):
        grouped: dict[int, dict[int, float]] = {}
        try:
            for record in records:
                prompt, clip = prompt_clip(str(record["video_path"]))
                value = finite(
                    record["video_results"],
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
                f"{PROMPT_COUNT * CLIPS_PER_VIDEO}"
            )
            continue
        return {
            prompt: [clips[index] for index in range(CLIPS_PER_VIDEO)]
            for prompt, clips in grouped.items()
        }
    detail = "; ".join(failures[:5]) or "no per-video detail list"
    raise ValueError(f"{dimension} has no complete clip view: {detail}")


def feature_row(series: dict[str, list[float]]) -> np.ndarray:
    values = []
    for dimension in MODEL_DIMENSIONS:
        values.append(statistics.mean(series[dimension]))
    for dimension in MIN_DIMENSIONS:
        values.append(min(series[dimension]))
    for dimension in TREND_DIMENSIONS:
        row = series[dimension]
        values.append(statistics.mean(row[-5:]) - statistics.mean(row[:5]))
    values.append(statistics.pstdev(series["dynamic_degree"]))
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FEATURE_NAMES),) or not np.isfinite(array).all():
        raise ValueError("invalid VBench feature vector")
    return array


def load_features(
    parts_root: Path,
    methods: tuple[str, ...],
) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for method in methods:
        dimensions = {
            dimension: load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
            )
            for dimension in MODEL_DIMENSIONS
        }
        for prompt in range(PROMPT_COUNT):
            result[(method, prompt)] = feature_row(
                {
                    dimension: dimensions[dimension][prompt]
                    for dimension in MODEL_DIMENSIONS
                }
            )
    return result


def feature_digest(features: dict[tuple[str, int], np.ndarray]) -> str:
    rows = [
        {
            "method": method,
            "prompt_index": prompt,
            "features": [float(value) for value in features[(method, prompt)]],
        }
        for method, prompt in sorted(features)
    ]
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_reviews(
    key_path: Path,
    review_path: Path,
    *,
    target_columns: dict[str, tuple[str, ...]],
    method_map: dict[str, str] | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key_rows = {str(row["video"]): row for row in key.get("rows", [])}
    review_rows = read_csv(review_path)
    if not key_rows or len(review_rows) != len(key_rows):
        raise ValueError(f"review/key row mismatch for {review_path}")
    result = {}
    for row in review_rows:
        video = str(row.get("video", ""))
        if video not in key_rows:
            raise ValueError(f"review video missing from blind key: {video}")
        hidden = key_rows[video]
        source_method = str(hidden["method"])
        method = (method_map or {}).get(source_method, source_method)
        prompt = int(hidden["prompt_index"])
        scores = {
            target: statistics.mean(
                finite(row[column], name=f"{video}:{column}")
                for column in columns
            )
            for target, columns in target_columns.items()
        }
        scores["severe"] = finite(
            row["severe_failure_0_or_1"],
            name=f"{video}:severe",
        )
        key_id = (method, prompt)
        if key_id in result:
            raise ValueError(f"duplicate human score for {key_id}")
        result[key_id] = scores
    return result


def pair_records(
    features: dict[tuple[str, int], np.ndarray],
    human: dict[tuple[str, int], dict[str, float]],
    target: str,
) -> list[dict[str, Any]]:
    prompts = sorted({prompt for _, prompt in human})
    records = []
    for prompt in prompts:
        methods = sorted(method for method, value in human if value == prompt)
        for left, right in itertools.combinations(methods, 2):
            left_key = (left, prompt)
            right_key = (right, prompt)
            if left_key not in features or right_key not in features:
                raise ValueError(f"missing features for {left_key} or {right_key}")
            records.append(
                {
                    "prompt": prompt,
                    "left": left,
                    "right": right,
                    "x": features[left_key] - features[right_key],
                    "y": human[left_key][target] - human[right_key][target],
                }
            )
    return records


def v160_analysis_pair_records(
    features: dict[tuple[str, int], np.ndarray],
    path: Path,
    target: str,
    prompt_order: list[int],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    combined = payload.get("combined") or {}
    prompt_set = [int(value) for value in combined.get("prompt_indices", [])]
    stored_order = combined.get("delta_prompt_order")
    prompts = (
        [int(value) for value in stored_order]
        if isinstance(stored_order, list)
        else [int(value) for value in prompt_order]
    )
    comparisons = combined.get("comparisons") or {}
    if (
        payload.get("experiment")
        != "v160_adaptive_blind_review_analysis"
        or payload.get("review_complete") is not True
        or int(combined.get("prompt_count", -1)) != 8
        or len(prompt_set) != 8
        or len(set(prompt_set)) != 8
        or len(prompts) != 8
        or set(prompts) != set(prompt_set)
        or set(comparisons) != {OLD_MOTION, RESERVOIR}
        or target not in V160_TARGET_COLUMNS
    ):
        raise ValueError("v160 combined review analysis violates the contract")

    deltas: dict[str, list[float]] = {}
    for reference in (OLD_MOTION, RESERVOIR):
        dimensions = []
        for column in V160_TARGET_COLUMNS[target]:
            raw = (comparisons[reference].get(column) or {}).get("deltas")
            if not isinstance(raw, list) or len(raw) != len(prompts):
                raise ValueError(
                    f"v160 {reference}:{column} delta coverage mismatch"
                )
            dimensions.append(
                [finite(value, name=f"v160:{reference}:{column}") for value in raw]
            )
        deltas[reference] = [
            statistics.mean(values)
            for values in zip(*dimensions)
        ]

    records = []
    for offset, prompt in enumerate(prompts):
        fresh_old = deltas[OLD_MOTION][offset]
        fresh_reservoir = deltas[RESERVOIR][offset]
        for left, right, value in (
            (FRESH, OLD_MOTION, fresh_old),
            (FRESH, RESERVOIR, fresh_reservoir),
            (OLD_MOTION, RESERVOIR, fresh_reservoir - fresh_old),
        ):
            left_key = (left, prompt)
            right_key = (right, prompt)
            if left_key not in features or right_key not in features:
                raise ValueError(
                    f"missing v160 transfer features for {left_key}/{right_key}"
                )
            records.append(
                {
                    "prompt": prompt,
                    "left": left,
                    "right": right,
                    "x": features[left_key] - features[right_key],
                    "y": float(value),
                }
            )
    return records


def review_prompt_order(*paths: Path) -> list[int]:
    order = []
    for path in paths:
        rows = read_csv(path)
        seen = set()
        for row in rows:
            prompt = int(row["prompt_index"])
            if prompt not in seen:
                seen.add(prompt)
                order.append(prompt)
        if len(rows) != 12 or len(seen) != 4:
            raise ValueError(f"v160 review prompt order is incomplete: {path}")
    if len(order) != 8 or len(set(order)) != 8:
        raise ValueError("v160 combined review prompt order is not unique")
    return order


def validate_v160_wave1_consistency(
    full_records: list[dict[str, Any]],
    wave1_records: list[dict[str, Any]],
) -> None:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, str, str], float]:
        return {
            (int(row["prompt"]), str(row["left"]), str(row["right"])): float(
                row["y"]
            )
            for row in rows
        }

    full = keyed(full_records)
    wave1 = keyed(wave1_records)
    missing = sorted(set(wave1) - set(full))
    conflicts = {
        key: (wave1[key], full[key])
        for key in wave1
        if key in full and not math.isclose(wave1[key], full[key], abs_tol=1e-9)
    }
    if missing or conflicts:
        raise ValueError(
            f"v160 Wave 1/combined review mismatch: missing={missing} "
            f"conflicts={conflicts}"
        )


def fit_ridge(records: list[dict[str, Any]], regularization: float) -> dict:
    if not records:
        raise ValueError("cannot fit an empty calibration set")
    matrix = np.stack([row["x"] for row in records])
    targets = np.asarray([row["y"] for row in records], dtype=np.float64)
    scale = np.sqrt(np.mean(np.square(matrix), axis=0))
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = matrix / scale
    gram = normalized.T @ normalized
    weights = np.linalg.solve(
        gram + float(regularization) * np.eye(gram.shape[0]),
        normalized.T @ targets,
    )
    return {
        "regularization": float(regularization),
        "scale": scale,
        "weights": weights,
    }


def predict(model: dict, feature_delta: np.ndarray) -> float:
    return float((feature_delta / model["scale"]) @ model["weights"])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = ranks(np.asarray(left, dtype=np.float64))
    right_rank = ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def prediction_metrics(
    records: list[dict[str, Any]],
    predictions: list[float],
) -> dict[str, Any]:
    if len(records) != len(predictions) or not records:
        raise ValueError("prediction metric inputs are incomplete")
    targets = [float(row["y"]) for row in records]
    non_ties = [index for index, value in enumerate(targets) if abs(value) > 1e-8]
    correct = sum(
        targets[index] * predictions[index] > 0.0 for index in non_ties
    )
    return {
        "pair_count": len(records),
        "non_tie_pair_count": len(non_ties),
        "directional_accuracy": (
            correct / len(non_ties) if non_ties else 0.0
        ),
        "spearman": spearman(targets, predictions),
        "mae": statistics.mean(
            abs(target - predicted)
            for target, predicted in zip(targets, predictions)
        ),
    }


def select_regularization(records: list[dict[str, Any]]) -> float:
    prompts = sorted({int(row["prompt"]) for row in records})
    if len(prompts) < 3:
        return 10.0
    scored = []
    for regularization in LAMBDA_GRID:
        validation_rows = []
        predictions = []
        for prompt in prompts:
            train = [row for row in records if row["prompt"] != prompt]
            held_out = [row for row in records if row["prompt"] == prompt]
            model = fit_ridge(train, regularization)
            validation_rows.extend(held_out)
            predictions.extend(predict(model, row["x"]) for row in held_out)
        metrics = prediction_metrics(validation_rows, predictions)
        scored.append(
            (
                metrics["directional_accuracy"],
                metrics["spearman"],
                -metrics["mae"],
                regularization,
            )
        )
    return float(max(scored)[-1])


def cross_validate(records: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    prompts = sorted({int(row["prompt"]) for row in records})
    held_out_rows = []
    predictions = []
    models = []
    selected = []
    for prompt in prompts:
        train = [row for row in records if row["prompt"] != prompt]
        held_out = [row for row in records if row["prompt"] == prompt]
        regularization = select_regularization(train)
        model = fit_ridge(train, regularization)
        models.append(model)
        selected.append(regularization)
        held_out_rows.extend(held_out)
        predictions.extend(predict(model, row["x"]) for row in held_out)
    metrics = prediction_metrics(held_out_rows, predictions)
    metrics["selected_regularization"] = selected
    return metrics, models


def bootstrap_ci(values: list[float], *, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = [
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(5000)
    ]
    means.sort()
    return [means[124], means[4874]]


def model_payload(model: dict) -> dict[str, Any]:
    return {
        "regularization": model["regularization"],
        "feature_weights": {
            name: float(weight / scale)
            for name, weight, scale in zip(
                FEATURE_NAMES,
                model["weights"],
                model["scale"],
            )
        },
    }


def primary_predictions(
    features: dict[tuple[str, int], np.ndarray],
    models: dict[str, dict],
    ensembles: dict[str, list[dict]],
) -> dict[str, Any]:
    result = {}
    for reference_index, reference in enumerate((FRESH, RESERVOIR, OLD_MOTION)):
        target_rows = {}
        for target in TARGETS:
            full_model = models[target]
            outer_models = ensembles[target]
            per_prompt = []
            for prompt in range(PROMPT_COUNT):
                delta = features[(PRIMARY, prompt)] - features[(reference, prompt)]
                ensemble_values = [predict(model, delta) for model in outer_models]
                per_prompt.append(
                    {
                        "prompt_index": prompt,
                        "predicted_delta": predict(full_model, delta),
                        "ensemble_mean": statistics.mean(ensemble_values),
                        "ensemble_std": statistics.pstdev(ensemble_values),
                    }
                )
            values = [row["predicted_delta"] for row in per_prompt]
            target_rows[target] = {
                "mean_predicted_delta": statistics.mean(values),
                "median_predicted_delta": statistics.median(values),
                "positive_prompts": sum(value > 0.0 for value in values),
                "bootstrap_mean_ci95": bootstrap_ci(
                    values,
                    seed=1622026 + 10 * reference_index + TARGETS.index(target),
                ),
                "mean_ensemble_std": statistics.mean(
                    row["ensemble_std"] for row in per_prompt
                ),
                "per_prompt": per_prompt,
            }
        result[reference] = target_rows
    return result


def screen_selection(screen: dict) -> dict[str, Any]:
    plan = screen.get("review_plan") or {}
    rows = plan.get("rows") or []
    flagged = sorted(
        {
            int(row["prompt_index"])
            for row in rows
            if row.get("automatic_flags")
        }
    )
    wave1 = plan.get("wave1") or []
    high_risk = next(
        (
            int(row["prompt_index"])
            for row in wave1
            if row.get("reason") == "highest_automatic_risk"
        ),
        flagged[0] if flagged else 0,
    )
    disagreement = next(
        (
            int(row["prompt_index"])
            for row in wave1
            if row.get("reason") == "largest_metric_disagreement"
        ),
        next(index for index in range(PROMPT_COUNT) if index != high_risk),
    )
    return {
        "automatic_safety_screen": bool(screen.get("automatic_safety_screen")),
        "flagged_prompt_indices": flagged,
        "sentinel_prompt_indices": [high_risk, disagreement],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# v162 Metric-Human Calibration",
        "",
        f"Calibration gate: **{report['calibration_gate']}**",
        f"Comparative auto gate: **{report['comparative_auto_gate']}**",
        f"Recommended review: **{report['review_recommendation']['mode']}**",
        "",
        "| Target | v157 LOPO accuracy | v157 rho | v160 transfer accuracy |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        development = report["v157_cross_validation"][target]
        transfer = report["v160_transfer"][target]
        lines.append(
            f"| {target} | {development['directional_accuracy']:.3f} | "
            f"{development['spearman']:.3f} | "
            f"{transfer['directional_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Predicted Primary Differences",
            "",
            "| Reference | Overall mean | 95% bootstrap CI | Positive prompts |",
            "|---|---:|---:|---:|",
        ]
    )
    for reference in (FRESH, RESERVOIR, OLD_MOTION):
        row = report["primary_predictions"][reference]["overall"]
        interval = row["bootstrap_mean_ci95"]
        lines.append(
            f"| {reference} | {row['mean_predicted_delta']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{row['positive_prompts']}/{PROMPT_COUNT} |"
        )
    lines.extend(
        [
            "",
            "Flagged prompts: "
            + ", ".join(
                str(index)
                for index in report["safety"]["flagged_prompt_indices"]
            ),
            "",
            "This calibration is for engineering triage. It does not replace "
            "a fixed human evaluation protocol for a paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    required = (
        args.v157_key,
        args.v157_review,
        args.v160_key,
        args.v160_review,
        args.v160_analysis,
        args.v160_wave2_review,
        args.v161_screen,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing calibration inputs: {missing}")

    v157_features = load_features(args.v157_parts_root, V157_METHODS)
    v157_human = load_reviews(
        args.v157_key,
        args.v157_review,
        target_columns={
            "identity": ("identity_continuity_-2_to_2",),
            "background": ("background_continuity_-2_to_2",),
            "motion": ("motion_quality_-2_to_2",),
            "overall": ("overall_preference_-2_to_2",),
        },
    )
    v160_human = load_reviews(
        args.v160_key,
        args.v160_review,
        target_columns=V160_TARGET_COLUMNS,
        method_map=V160_METHOD_MAP,
    )
    v160_prompt_order = review_prompt_order(
        args.v160_review,
        args.v160_wave2_review,
    )
    if args.history_preflight:
        expected_v157 = len(V157_METHODS) * PROMPT_COUNT
        if len(v157_features) != expected_v157 or len(v157_human) != expected_v157:
            raise ValueError("v157 historical calibration coverage is incomplete")
        if len(v160_human) != 12:
            raise ValueError("v160 transfer-review coverage must contain 12 videos")
        dummy = {
            (method, prompt): np.zeros(len(FEATURE_NAMES), dtype=np.float64)
            for method in (FRESH, OLD_MOTION, RESERVOIR)
            for prompt in range(PROMPT_COUNT)
        }
        for target in TARGETS:
            full_records = v160_analysis_pair_records(
                dummy,
                args.v160_analysis,
                target,
                v160_prompt_order,
            )
            wave1_records = pair_records(dummy, v160_human, target)
            validate_v160_wave1_consistency(full_records, wave1_records)
            if len(full_records) != 24:
                raise ValueError("v160 combined transfer requires 24 pairs")
        print(
            "[v162-history-preflight] "
            f"features={len(v157_features)} v157_reviews={len(v157_human)} "
            f"v160_reviews=24 v160_prompts=8 "
            f"feature_sha256={feature_digest(v157_features)} status=ok",
            flush=True,
        )
        return

    v162_features = load_features(args.v162_parts_root, V162_METHODS)

    development = {}
    transfer = {}
    final_models = {}
    ensembles = {}
    model_reports = {}
    for target in TARGETS:
        training_records = pair_records(
            v157_features,
            v157_human,
            target,
        )
        cv_metrics, outer_models = cross_validate(training_records)
        regularization = select_regularization(training_records)
        final_model = fit_ridge(training_records, regularization)
        transfer_records = v160_analysis_pair_records(
            v162_features,
            args.v160_analysis,
            target,
            v160_prompt_order,
        )
        validate_v160_wave1_consistency(
            transfer_records,
            pair_records(v162_features, v160_human, target),
        )
        transfer_predictions = [
            predict(final_model, row["x"]) for row in transfer_records
        ]
        development[target] = cv_metrics
        transfer[target] = prediction_metrics(
            transfer_records,
            transfer_predictions,
        )
        final_models[target] = final_model
        ensembles[target] = outer_models
        model_reports[target] = model_payload(final_model)

    calibration_checks = {
        "v157_overall_accuracy": (
            development["overall"]["directional_accuracy"] >= 0.60
        ),
        "v157_overall_spearman": development["overall"]["spearman"] >= 0.20,
        "v157_all_target_accuracy": min(
            development[target]["directional_accuracy"] for target in TARGETS
        )
        >= 0.55,
        "v160_overall_transfer": (
            transfer["overall"]["directional_accuracy"] >= 0.60
        ),
        "v160_motion_transfer": (
            transfer["motion"]["directional_accuracy"] >= 0.55
        ),
    }
    calibration_gate = all(calibration_checks.values())
    primary = primary_predictions(v162_features, final_models, ensembles)
    robustness_checks = {
        reference: (
            primary[reference]["overall"]["bootstrap_mean_ci95"][0] > 0.0
            and primary[reference]["overall"]["positive_prompts"] >= 10
        )
        for reference in (FRESH, RESERVOIR)
    }
    comparative_auto_gate = calibration_gate and all(
        robustness_checks.values()
    )
    screen = json.loads(args.v161_screen.read_text(encoding="utf-8"))
    safety = screen_selection(screen)
    mode = "safety_only" if comparative_auto_gate else "sentinel_blind"
    sentinel = safety["sentinel_prompt_indices"]
    safety_extras = [
        index for index in safety["flagged_prompt_indices"] if index not in sentinel
    ]
    manual_video_count = (
        len(safety["flagged_prompt_indices"])
        if mode == "safety_only"
        else len(sentinel) * 3 + len(safety_extras)
    )
    report = {
        "version": 1,
        "experiment": "v162_metric_human_calibration",
        "feature_names": list(FEATURE_NAMES),
        "v157_cross_validation": development,
        "v160_transfer": transfer,
        "models": model_reports,
        "calibration_checks": calibration_checks,
        "calibration_gate": calibration_gate,
        "primary_predictions": primary,
        "primary_robustness_checks": robustness_checks,
        "comparative_auto_gate": comparative_auto_gate,
        "safety": safety,
        "review_recommendation": {
            "mode": mode,
            "sentinel_prompt_indices": sentinel,
            "safety_extra_prompt_indices": safety_extras,
            "manual_video_count": manual_video_count,
            "blind_comparison_video_count": 0 if mode == "safety_only" else 6,
            "rule": (
                "use three primary-only safety checks when calibrated paired "
                "predictions are robust; otherwise use two prompts x three "
                "methods plus unmatched flagged-primary safety checks"
            ),
        },
        "inputs": {
            "v157_parts_root": str(args.v157_parts_root.resolve()),
            "v162_parts_root": str(args.v162_parts_root.resolve()),
            "v157_feature_sha256": feature_digest(v157_features),
            "v162_feature_sha256": feature_digest(v162_features),
            **{
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in (
                    ("v157_key", args.v157_key),
                    ("v157_review", args.v157_review),
                    ("v160_key", args.v160_key),
                    ("v160_review", args.v160_review),
                    ("v160_analysis", args.v160_analysis),
                    ("v160_wave2_review", args.v160_wave2_review),
                    ("v161_screen", args.v161_screen),
                )
            },
        },
        "claim_boundary": (
            "The learned metric calibration and adaptive sentinel selection "
            "are engineering triage only. They cannot replace a frozen human "
            "study used as paper evidence. Severe-failure safety remains a "
            "manual check because no severe-failure classifier is promoted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        "[v162-calibration] "
        f"calibration={calibration_gate} auto={comparative_auto_gate} "
        f"review={mode} videos={manual_video_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
