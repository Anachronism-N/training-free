#!/usr/bin/env python3
"""Run a threshold-free/FDR audit of the packaged v145 head profiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


AXES = (
    "q_shift",
    "k_shift",
    "v_shift",
    "value_scale_shift",
    "policy_shift",
)
EXPECTED_FAMILIES = 16
EXPECTED_LAYERS = 30
EXPECTED_HEADS = 12
PERMUTATIONS = 1000
RANDOM_SEED = 1682026


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        result[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rank(np.asarray(left, dtype=float))
    right_rank = _rank(np.asarray(right, dtype=float))
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def bh_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    count = len(p_values)
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = count - reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def load_layer_residuals(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    variants = sorted({row["variant"] for row in rows})
    expected_rows = (
        EXPECTED_FAMILIES
        * len(variants)
        * EXPECTED_LAYERS
        * EXPECTED_HEADS
    )
    if len(rows) != expected_rows:
        raise ValueError(f"v145 row count mismatch: {len(rows)} != {expected_rows}")
    if variants != ["full_semantic", "identity", "paraphrase", "scene"]:
        raise ValueError(f"unexpected v145 counterfactual variants: {variants}")
    grouped = defaultdict(list)
    raw = {}
    for row in rows:
        family = int(row["family_index"])
        layer = int(row["layer"])
        head = int(row["head"])
        variant = row["variant"]
        for axis in AXES:
            for seed in (0, 1):
                key = (variant, axis, family, seed, layer, head)
                value = float(row[f"{axis}_seed{seed}"])
                raw[key] = value
                grouped[(variant, axis, family, seed, layer)].append(value)
    residuals = {}
    for key, value in raw.items():
        variant, axis, family, seed, layer, head = key
        center = _median(grouped[(variant, axis, family, seed, layer)])
        residuals[key] = value - center
    return variants, residuals


def _head_scores(
    residuals: dict,
    *,
    variant: str,
    axis: str,
    families: tuple[int, ...],
    seeds: tuple[int, ...],
) -> np.ndarray:
    result = []
    for layer in range(1, EXPECTED_LAYERS):
        for head in range(EXPECTED_HEADS):
            values = [
                residuals[(variant, axis, family, seed, layer, head)]
                for family in families
                for seed in seeds
            ]
            result.append(_median(values))
    return np.asarray(result, dtype=float)


def _permutation_pvalue(
    left: np.ndarray,
    right: np.ndarray,
    *,
    observed: float,
    rng: random.Random,
) -> float:
    exceed = 0
    matrix = right.reshape(EXPECTED_LAYERS - 1, EXPECTED_HEADS)
    for _ in range(PERMUTATIONS):
        permuted = np.empty_like(matrix)
        for layer in range(len(matrix)):
            order = list(range(EXPECTED_HEADS))
            rng.shuffle(order)
            permuted[layer] = matrix[layer, order]
        value = spearman(left, permuted.reshape(-1))
        if abs(value) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (PERMUTATIONS + 1)


def _threshold_rows(feature: str, discovery: np.ndarray, validation: np.ndarray):
    rows = []
    thresholds: list[tuple[str, float]] = [("zero", 0.0)]
    thresholds.extend(
        (f"p{percentile:02d}", float(np.percentile(discovery, percentile)))
        for percentile in range(10, 100, 10)
    )
    for name, threshold in thresholds:
        left = discovery > threshold
        right = validation > threshold
        union = np.logical_or(left, right).sum()
        rows.append(
            {
                "feature": feature,
                "threshold_name": name,
                "threshold": threshold,
                "discovery_positive_fraction": float(left.mean()),
                "validation_positive_fraction": float(right.mean()),
                "label_agreement": float((left == right).mean()),
                "positive_jaccard": (
                    float(np.logical_and(left, right).sum() / union)
                    if union
                    else 1.0
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path: Path, output_root: Path) -> dict:
    variants, residuals = load_layer_residuals(input_path)
    discovery_families = tuple(range(0, EXPECTED_FAMILIES, 2))
    validation_families = tuple(range(1, EXPECTED_FAMILIES, 2))
    all_families = tuple(range(EXPECTED_FAMILIES))
    rng = random.Random(RANDOM_SEED)
    feature_rows = []
    threshold_rows = []
    feature_vectors = {}
    for variant in variants:
        for axis in AXES:
            feature = f"{variant}.{axis}"
            discovery = _head_scores(
                residuals,
                variant=variant,
                axis=axis,
                families=discovery_families,
                seeds=(0, 1),
            )
            validation = _head_scores(
                residuals,
                variant=variant,
                axis=axis,
                families=validation_families,
                seeds=(0, 1),
            )
            seed0 = _head_scores(
                residuals,
                variant=variant,
                axis=axis,
                families=all_families,
                seeds=(0,),
            )
            seed1 = _head_scores(
                residuals,
                variant=variant,
                axis=axis,
                families=all_families,
                seeds=(1,),
            )
            family_rho = spearman(discovery, validation)
            seed_rho = spearman(seed0, seed1)
            p_value = _permutation_pvalue(
                discovery,
                validation,
                observed=family_rho,
                rng=rng,
            )
            feature_rows.append(
                {
                    "feature": feature,
                    "variant": variant,
                    "axis": axis,
                    "family_split_spearman": family_rho,
                    "seed_replicate_spearman": seed_rho,
                    "within_layer_permutation_p": p_value,
                }
            )
            threshold_rows.extend(_threshold_rows(feature, discovery, validation))
            feature_vectors[feature] = 0.5 * (discovery + validation)

    q_values = bh_adjust(
        [float(row["within_layer_permutation_p"]) for row in feature_rows]
    )
    for row, q_value in zip(feature_rows, q_values):
        row["bh_fdr_q"] = q_value
        row["threshold_free_screen_pass"] = int(
            q_value <= 0.05
            and float(row["family_split_spearman"]) >= 0.30
            and float(row["seed_replicate_spearman"]) >= 0.30
        )

    features = sorted(feature_vectors)
    matrix = np.vstack([feature_vectors[feature] for feature in features])
    correlation = np.corrcoef(matrix)
    eigenvalues = np.linalg.eigvalsh(correlation)
    positive = eigenvalues[eigenvalues > 1e-12]
    probabilities = positive / positive.sum()
    effective_rank = float(math.exp(-sum(p * math.log(p) for p in probabilities)))
    correlation_rows = [
        {
            "left_feature": left,
            "right_feature": right,
            "spearman": spearman(feature_vectors[left], feature_vectors[right]),
        }
        for left in features
        for right in features
    ]
    report = {
        "version": 1,
        "experiment": "head_profile_threshold_free_audit",
        "source": str(input_path),
        "feature_count": len(features),
        "threshold_free_screen_pass_count": sum(
            int(row["threshold_free_screen_pass"]) for row in feature_rows
        ),
        "effective_rank_of_feature_correlation": effective_rank,
        "permutations_per_feature": PERMUTATIONS,
        "multiple_testing": (
            f"Benjamini-Hochberg over all {len(features)} packaged "
            "counterfactual factor-axis tests"
        ),
        "claim_boundary": (
            "This audit measures observational reproducibility and feature "
            "redundancy after within-layer median residualization. Its zero "
            "split is a relative within-layer split, not the original semantic "
            "null. It does not establish a functional head class or "
            "trajectory-level cache utility."
        ),
        "features": feature_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "feature_fdr_audit.csv", feature_rows)
    _write_csv(output_root / "threshold_stability.csv", threshold_rows)
    _write_csv(output_root / "feature_correlations.csv", correlation_rows)
    (output_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Head Profile Threshold-Free Audit",
        "",
        f"- Features: `{len(features)}`",
        (
            "- Threshold-free FDR screen passes: "
            f"`{report['threshold_free_screen_pass_count']}`"
        ),
        f"- Correlation effective rank: `{effective_rank:.3f}`",
        f"- Within-layer permutations per feature: `{PERMUTATIONS}`",
        "",
        report["claim_boundary"],
        "",
    ]
    (output_root / "audit_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            root
            / "docs"
            / "results"
            / "v145_crossed_seed_head_profile"
            / "family_head_axes.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "docs" / "results" / "head_profile_threshold_free_audit",
    )
    args = parser.parse_args()
    report = analyze(args.input.resolve(), args.output_root.resolve())
    print(
        "[head-profile-audit] PASS "
        f"features={report['feature_count']} "
        f"fdr_pass={report['threshold_free_screen_pass_count']} "
        f"effective_rank={report['effective_rank_of_feature_correlation']:.3f}"
    )


if __name__ == "__main__":
    main()
