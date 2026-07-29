#!/usr/bin/env python3
"""Held-out threshold audit for the frozen v134 prompt-sensitivity profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_LAYERS = 30
EXPECTED_HEADS = 12
SCORE_FIELDS = {
    "raw_cphi": ("cphi_score",),
    "query_adjusted": ("cphi_score", "query_score"),
    "native_adjusted": ("cphi_score", "native_score"),
    "key_adjusted": ("cphi_score", "current_key_score"),
}


def _median(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else float("nan")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1)
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    left_rank = _rank([pair[0] for pair in pairs])
    right_rank = _rank([pair[1] for pair in pairs])
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_rank)
        * sum((value - right_mean) ** 2 for value in right_rank)
    )
    return numerator / denominator if denominator else float("nan")


def _score(row: dict[str, str], name: str) -> float:
    fields = SCORE_FIELDS[name]
    value = float(row[fields[0]])
    for field in fields[1:]:
        value -= float(row[field])
    return value


def _family(job_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"cf_(\d+)_([a-z_]+)", str(job_id))
    if not match:
        raise ValueError(f"unexpected v134 counterfactual job_id: {job_id}")
    return int(match.group(1)), match.group(2)


def _load_rows(path: Path, expected_jobs: int) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "job_id",
        "layer",
        "head",
        "cphi_score",
        "query_score",
        "native_score",
        "current_key_score",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"prompt job axes miss columns: {sorted(required - set(rows[0] if rows else {}))}"
        )
    jobs = {str(row["job_id"]) for row in rows}
    if len(jobs) != expected_jobs:
        raise ValueError(f"job count mismatch: {len(jobs)} != {expected_jobs}")
    expected_pairs = {
        (layer, head)
        for layer in range(EXPECTED_LAYERS)
        for head in range(EXPECTED_HEADS)
    }
    grouped = defaultdict(set)
    for row in rows:
        family, factor = _family(str(row["job_id"]))
        row["family_index"] = family
        row["factor"] = factor
        layer = int(row["layer"])
        head = int(row["head"])
        grouped[str(row["job_id"])].add((layer, head))
        for name in SCORE_FIELDS:
            row[name] = _score(row, name)
    bad = [job for job, pairs in grouped.items() if pairs != expected_pairs]
    if bad:
        raise ValueError(f"incomplete head grid for jobs: {bad[:5]}")
    return rows


def _aggregate(
    rows: list[dict], score_name: str, family_parity: int
) -> dict[tuple[int, int], float]:
    grouped = defaultdict(list)
    for row in rows:
        if int(row["family_index"]) % 2 != family_parity:
            continue
        grouped[(int(row["layer"]), int(row["head"]))].append(
            float(row[score_name])
        )
    expected = {
        (layer, head)
        for layer in range(EXPECTED_LAYERS)
        for head in range(EXPECTED_HEADS)
    }
    if set(grouped) != expected:
        raise ValueError("discovery/validation split has incomplete head coverage")
    return {key: _median(values) for key, values in grouped.items()}


def _otsu_threshold(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < 2 or ordered[0] == ordered[-1]:
        return ordered[0] if ordered else 0.0
    total_sum = sum(ordered)
    left_sum = 0.0
    best_score = -1.0
    best_threshold = ordered[0]
    for index in range(1, len(ordered)):
        left_sum += ordered[index - 1]
        if ordered[index - 1] == ordered[index]:
            continue
        left_count = index
        right_count = len(ordered) - index
        left_mean = left_sum / left_count
        right_mean = (total_sum - left_sum) / right_count
        score = left_count * right_count * (left_mean - right_mean) ** 2
        if score > best_score:
            best_score = score
            best_threshold = 0.5 * (ordered[index - 1] + ordered[index])
    return best_threshold


def _gmm_threshold(values: list[float]) -> tuple[float, dict]:
    values = [float(value) for value in values]
    mean0 = _quantile(values, 0.25)
    mean1 = _quantile(values, 0.75)
    variance = max(statistics.pvariance(values), 1e-6)
    variances = [variance, variance]
    weights = [0.5, 0.5]
    for _ in range(200):
        responsibilities = []
        for value in values:
            terms = []
            for weight, mean, component_variance in zip(
                weights, (mean0, mean1), variances
            ):
                density = math.exp(
                    -0.5 * (value - mean) ** 2 / component_variance
                ) / math.sqrt(2.0 * math.pi * component_variance)
                terms.append(max(1e-300, weight * density))
            scale = sum(terms)
            responsibilities.append([term / scale for term in terms])
        masses = [
            sum(row[component] for row in responsibilities)
            for component in range(2)
        ]
        new_means = [
            sum(
                row[component] * value
                for row, value in zip(responsibilities, values)
            )
            / max(masses[component], 1e-12)
            for component in range(2)
        ]
        new_variances = [
            max(
                sum(
                    row[component] * (value - new_means[component]) ** 2
                    for row, value in zip(responsibilities, values)
                )
                / max(masses[component], 1e-12),
                1e-8,
            )
            for component in range(2)
        ]
        new_weights = [mass / len(values) for mass in masses]
        delta = max(
            abs(mean0 - new_means[0]),
            abs(mean1 - new_means[1]),
            abs(variances[0] - new_variances[0]),
            abs(variances[1] - new_variances[1]),
        )
        mean0, mean1 = new_means
        variances = new_variances
        weights = new_weights
        if delta < 1e-10:
            break
    components = sorted(
        zip((mean0, mean1), variances, weights), key=lambda item: item[0]
    )
    (low_mean, low_variance, low_weight), (
        high_mean,
        high_variance,
        high_weight,
    ) = components
    candidates = [
        low_mean
        + (high_mean - low_mean) * index / 2000
        for index in range(1, 2000)
    ]

    def difference(value: float) -> float:
        low = math.log(max(low_weight, 1e-12)) - 0.5 * (
            math.log(low_variance)
            + (value - low_mean) ** 2 / low_variance
        )
        high = math.log(max(high_weight, 1e-12)) - 0.5 * (
            math.log(high_variance)
            + (value - high_mean) ** 2 / high_variance
        )
        return abs(low - high)

    threshold = min(candidates, key=difference)
    return threshold, {
        "means": [low_mean, high_mean],
        "variances": [low_variance, high_variance],
        "weights": [low_weight, high_weight],
    }


def _label_metrics(
    discovery: dict[tuple[int, int], float],
    validation: dict[tuple[int, int], float],
    threshold: float,
    *,
    scale: float,
) -> dict:
    keys = [key for key in sorted(discovery) if key[0] > 0]
    discovery_positive = {key for key in keys if discovery[key] > threshold}
    validation_positive = {key for key in keys if validation[key] > threshold}
    agreement = sum(
        (key in discovery_positive) == (key in validation_positive)
        for key in keys
    ) / len(keys)
    union = discovery_positive | validation_positive
    jaccard = (
        len(discovery_positive & validation_positive) / len(union)
        if union
        else 1.0
    )
    boundary = sum(
        abs(validation[key] - threshold) <= 0.1 * max(scale, 1e-12)
        for key in keys
    ) / len(keys)
    minority = min(
        len(validation_positive), len(keys) - len(validation_positive)
    ) / len(keys)
    return {
        "threshold": threshold,
        "discovery_positive": len(discovery_positive),
        "validation_positive": len(validation_positive),
        "active_head_count": len(keys),
        "label_agreement": agreement,
        "positive_jaccard": jaccard,
        "validation_minority_fraction": minority,
        "validation_boundary_fraction_0p1_iqr": boundary,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    rows: list[dict],
    *,
    output_dir: Path,
    expected_jobs: int,
) -> dict:
    del expected_jobs
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows = []
    sweep_rows = []
    head_rows = []
    score_reports = {}
    for score_name in SCORE_FIELDS:
        discovery = _aggregate(rows, score_name, family_parity=0)
        validation = _aggregate(rows, score_name, family_parity=1)
        active_values = [
            discovery[key] for key in sorted(discovery) if key[0] > 0
        ]
        iqr = _quantile(active_values, 0.75) - _quantile(
            active_values, 0.25
        )
        gmm_threshold, gmm = _gmm_threshold(active_values)
        thresholds = {
            "zero": 0.0,
            "otsu_discovery": _otsu_threshold(active_values),
            "gmm2_discovery": gmm_threshold,
        }
        method_metrics = {}
        for method, threshold in thresholds.items():
            metrics = _label_metrics(
                discovery, validation, threshold, scale=iqr
            )
            method_metrics[method] = metrics
            threshold_rows.append(
                {"score": score_name, "method": method, **metrics}
            )
        for percentile in range(0, 101, 5):
            threshold = _quantile(active_values, percentile / 100.0)
            sweep_rows.append(
                {
                    "score": score_name,
                    "discovery_percentile": percentile,
                    **_label_metrics(
                        discovery, validation, threshold, scale=iqr
                    ),
                }
            )
        keys = sorted(discovery)
        split_rho = _spearman(
            [discovery[key] for key in keys if key[0] > 0],
            [validation[key] for key in keys if key[0] > 0],
        )
        primary = method_metrics["zero"]
        primary_gate = (
            split_rho >= 0.60
            and primary["label_agreement"] >= 0.80
            and primary["validation_minority_fraction"] >= 0.05
            and primary["validation_boundary_fraction_0p1_iqr"] <= 0.20
        )
        score_reports[score_name] = {
            "discovery_validation_spearman": split_rho,
            "discovery_iqr": iqr,
            "gmm2": gmm,
            "thresholds": method_metrics,
            "zero_threshold_gate": primary_gate,
        }
        for key in keys:
            layer, head = key
            row = {
                "layer": layer,
                "head": head,
                "score": score_name,
                "discovery_score": discovery[key],
                "validation_score": validation[key],
            }
            for method, threshold in thresholds.items():
                row[f"{method}_discovery_label"] = int(
                    layer > 0 and discovery[key] > threshold
                )
                row[f"{method}_validation_label"] = int(
                    layer > 0 and validation[key] > threshold
                )
            head_rows.append(row)

    factor_rows = []
    for score_name in SCORE_FIELDS:
        score_groups = defaultdict(list)
        for row in rows:
            if int(row["layer"]) == 0:
                continue
            score_groups[
                (
                    str(row["factor"]),
                    int(row["layer"]),
                    int(row["head"]),
                )
            ].append(float(row[score_name]))
        for (factor, layer, head), values in sorted(score_groups.items()):
            factor_rows.append(
                {
                    "score": score_name,
                    "factor": factor,
                    "layer": layer,
                    "head": head,
                    "median": _median(values),
                    "positive_fraction": sum(value > 0 for value in values)
                    / len(values),
                    "samples": len(values),
                }
            )

    recommendation = (
        "query_adjusted_zero_threshold_candidate"
        if score_reports["query_adjusted"]["zero_threshold_gate"]
        else "no_thresholded_prompt_head_class_supported"
    )
    report = {
        "method": "v140_held_out_prompt_threshold_robustness",
        "input_rows": len(rows),
        "job_count": len({str(row["job_id"]) for row in rows}),
        "split": "even family index discovery; odd family index validation",
        "layer0_policy": "excluded from threshold fitting and forced invariant",
        "score_definitions": {
            "raw_cphi": "cphi_score",
            "query_adjusted": "cphi_score - query_score",
            "native_adjusted": "cphi_score - native_score",
            "key_adjusted": "cphi_score - current_key_score",
        },
        "score_reports": score_reports,
        "recommendation": recommendation,
    }
    _write_csv(output_dir / "threshold_methods.csv", threshold_rows)
    _write_csv(output_dir / "threshold_sweep.csv", sweep_rows)
    _write_csv(output_dir / "head_split_scores.csv", head_rows)
    _write_csv(output_dir / "head_factor_scores.csv", factor_rows)
    (output_dir / "threshold_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# v140 Prompt-Sensitivity Threshold Audit",
        "",
        f"- Recommendation: `{recommendation}`",
        "- Discovery/validation split: even/odd subject family.",
        "- Layer 0 is structural zero and is not used to fit thresholds.",
        "",
        "## Scores",
        "",
    ]
    for name, score_report in score_reports.items():
        zero = score_report["thresholds"]["zero"]
        summary.extend(
            [
                f"### {name}",
                "",
                (
                    "- Split Spearman / zero-label agreement: "
                    f"`{score_report['discovery_validation_spearman']:.4f}` / "
                    f"`{zero['label_agreement']:.4f}`"
                ),
                (
                    "- Validation positive / minority / boundary fractions: "
                    f"`{zero['validation_positive']}/{zero['active_head_count']}` / "
                    f"`{zero['validation_minority_fraction']:.4f}` / "
                    f"`{zero['validation_boundary_fraction_0p1_iqr']:.4f}`"
                ),
                (
                    "- Frozen zero-threshold gate: "
                    f"`{score_report['zero_threshold_gate']}`"
                ),
                "",
            ]
        )
    summary.extend(
        [
            "## Interpretation",
            "",
            "GMM and Otsu thresholds are fitted on discovery families only. "
            "They remain diagnostics unless their labels transfer to validation.",
            "The percentile sweep shows how strongly class membership depends "
            "on an arbitrary threshold; it must not be selected by PF overlap "
            "or downstream video metrics.",
            "",
        ]
    )
    (output_dir / "threshold_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-job-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int, default=128)
    args = parser.parse_args()
    rows = _load_rows(args.prompt_job_axes, args.expected_jobs)
    analyze(rows, output_dir=args.output_dir, expected_jobs=args.expected_jobs)


if __name__ == "__main__":
    main()
