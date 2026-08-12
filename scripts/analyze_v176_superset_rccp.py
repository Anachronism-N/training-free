#!/usr/bin/env python3
"""Analyze v176 fair-teacher RCCP profiles without generation leakage."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import analyze_v173_cache_compatibility as base


DISCOVERY_SEED = 1762026
STABILITY_SEEDS = tuple(1763001 + index * 1009 for index in range(12))
DISCOVERY_COUNT = 64
VALIDATION_COUNT = 32
GENERATION_COUNT = 32
STABILITY_FREQUENCY = 0.75
MIN_DISCOVERY_PROMPTS = 32


def frozen_prompt_split(prompt_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    universe = sorted({int(value) for value in prompt_ids})
    if universe != list(range(base.PROMPTS)):
        raise ValueError("v176 requires complete prompt ids 0..127")
    order = [
        int(value)
        for value in np.random.default_rng(DISCOVERY_SEED).permutation(universe)
    ]
    discovery = sorted(order[:DISCOVERY_COUNT])
    validation = sorted(order[DISCOVERY_COUNT : DISCOVERY_COUNT + VALIDATION_COUNT])
    generation = sorted(order[-GENERATION_COUNT:])
    if set(discovery) & set(validation) or set(discovery) & set(generation):
        raise RuntimeError("v176 prompt split overlaps")
    if set(validation) & set(generation):
        raise RuntimeError("v176 prompt split overlaps")
    return discovery, validation, generation


def aggregate_profile(records: list[dict]) -> dict[str, np.ndarray | float]:
    """Aggregate the large record list once for all downstream diagnostics."""

    policy_count = len(base.POLICIES)
    frame_values = sorted({int(row["current_frame"]) for row in records})
    frame_midpoint = float(np.median(frame_values))
    error_sum = np.zeros(
        (base.PROMPTS, base.LAYERS, base.HEADS, policy_count),
        dtype=np.float64,
    )
    record_count = np.zeros((base.PROMPTS, base.LAYERS), dtype=np.int32)
    call_sum = np.zeros(
        (4, base.PROMPTS, base.LAYERS, base.HEADS, policy_count),
        dtype=np.float64,
    )
    call_count = np.zeros(
        (4, base.PROMPTS, base.LAYERS), dtype=np.int32
    )
    ar_sum = np.zeros(
        (2, base.PROMPTS, base.LAYERS, base.HEADS, policy_count),
        dtype=np.float64,
    )
    ar_count = np.zeros((2, base.PROMPTS, base.LAYERS), dtype=np.int32)
    budget_full_sum = np.zeros_like(error_sum)
    energy_sum = np.zeros(
        (base.PROMPTS, base.LAYERS, base.HEADS), dtype=np.float64
    )
    expected_budget = base.PROFILE_CONTRACTS["v176"]["expected_budget"]

    for record in records:
        prompt = int(record["prompt_id"])
        layer = int(record["layer"])
        call = int(record["call_index"])
        if call not in range(4):
            raise ValueError(f"v176 received unexpected call index {call}")
        ar_half = int(int(record["current_frame"]) > frame_midpoint)
        errors = np.stack(
            [
                np.log(
                    np.maximum(
                        np.asarray(
                            record["policies"][policy][
                                "residual_relative_mse"
                            ],
                            dtype=np.float64,
                        ),
                        1e-12,
                    )
                )
                for policy in base.POLICIES
            ],
            axis=-1,
        )
        if errors.shape != (base.HEADS, policy_count):
            raise ValueError("v176 residual error shape drift")
        error_sum[prompt, layer] += errors
        call_sum[call, prompt, layer] += errors
        ar_sum[ar_half, prompt, layer] += errors
        record_count[prompt, layer] += 1
        call_count[call, prompt, layer] += 1
        ar_count[ar_half, prompt, layer] += 1
        energy_sum[prompt, layer] += np.asarray(
            record["reference_residual_energy"], dtype=np.float64
        )
        for policy_index, policy in enumerate(base.POLICIES):
            observed = np.asarray(
                record["budgets"][policy][
                    "per_sequence_frame_equivalents"
                ],
                dtype=np.int64,
            )
            budget_full_sum[prompt, layer, :, policy_index] += (
                observed == expected_budget[policy]
            )

    if np.any(record_count <= 0) or np.any(call_count <= 0) or np.any(ar_count <= 0):
        raise ValueError("v176 aggregate contains an empty prompt/layer/context cell")
    errors = error_sum / record_count[:, :, None, None]
    call_errors = call_sum / call_count[:, :, :, None, None]
    ar_errors = ar_sum / ar_count[:, :, :, None, None]
    budget_full = budget_full_sum / record_count[:, :, None, None]
    energy = energy_sum / record_count[:, :, None]
    if not all(
        np.isfinite(value).all()
        for value in (errors, call_errors, ar_errors, budget_full, energy)
    ):
        raise ValueError("v176 aggregate contains non-finite values")
    return {
        "errors": errors,
        "call_errors": call_errors,
        "ar_errors": ar_errors,
        "budget_full": budget_full,
        "reference_residual_energy": energy,
        "frame_midpoint": frame_midpoint,
    }


def _oracle_policy(errors: dict[str, float]) -> tuple[str, float]:
    ranked = sorted(base.POLICIES, key=errors.__getitem__)
    return ranked[0], float(errors[ranked[1]] - errors[ranked[0]])


def _validation_stats(
    prompt_errors: np.ndarray,
    *,
    selected_policy: str,
    seed: int,
) -> dict:
    advantages = {}
    for alternative in base.POLICIES:
        if alternative == selected_policy:
            continue
        values = (
            prompt_errors[:, base.POLICIES.index(alternative)]
            - prompt_errors[:, base.POLICIES.index(selected_policy)]
        ).tolist()
        advantages[alternative] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "win_fraction": float(np.mean(np.asarray(values) > 0.0)),
            "bootstrap_ci95": base.bootstrap_ci(
                values,
                seed=seed + base.POLICIES.index(alternative),
                samples=2000,
            ),
            "sign_p": base.one_sided_sign_p(values),
        }
    return advantages


def _agreement(
    left: np.ndarray, right: np.ndarray
) -> float:
    if left.shape != right.shape:
        raise ValueError("policy agreement arrays have different shapes")
    return float(np.mean(left == right))


def _layer_residual_feature(value: float, layer_values: list[float]) -> float:
    return float(value - np.mean(layer_values))


def _load_v145_features(path: Path | None) -> tuple[dict, dict]:
    if path is None or not path.is_file():
        return {}, {"available": False, "path": None}
    by_head = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            layer = int(row["layer"])
            head = int(row["head"])
            for axis in ("q_shift", "k_shift", "v_shift", "policy_shift"):
                by_head[(layer, head)][f"v145_{variant}_{axis}"] = float(
                    row[f"discovery_{axis}_mean"]
                )
    expected = {
        (layer, head)
        for layer in range(base.LAYERS)
        for head in range(base.HEADS)
    }
    if set(by_head) != expected:
        raise ValueError("v145 feature table does not cover 30x12 heads")
    return dict(by_head), {
        "available": True,
        "path": str(path.resolve()),
        "sha256": base.sha256(path),
        "feature_count": len(next(iter(by_head.values()))),
    }


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    cursor = 0
    while cursor < array.size:
        end = cursor + 1
        while end < array.size and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    l_rank = _rankdata(left)
    r_rank = _rankdata(right)
    if np.std(l_rank) <= 1e-12 or np.std(r_rank) <= 1e-12:
        return None
    return float(np.corrcoef(l_rank, r_rank)[0, 1])


def _predictor_diagnostics(head_rows: list[dict], feature_rows: dict) -> dict:
    if not feature_rows:
        return {"available": False, "features": []}
    target_names = ("coverage_gain", "episode_gain")
    feature_names = sorted(next(iter(feature_rows.values())))
    diagnostics = []
    for feature in feature_names:
        raw = [feature_rows[(row["layer"], row["head"])][feature] for row in head_rows]
        residual = []
        for row, value in zip(head_rows, raw):
            layer_values = [
                feature_rows[(row["layer"], head)][feature]
                for head in range(base.HEADS)
            ]
            residual.append(_layer_residual_feature(value, layer_values))
        for target in target_names:
            target_values = [row[target] for row in head_rows]
            target_residual = []
            for row, value in zip(head_rows, target_values):
                layer_values = [
                    candidate[target]
                    for candidate in head_rows
                    if candidate["layer"] == row["layer"]
                ]
                target_residual.append(
                    _layer_residual_feature(value, layer_values)
                )
            diagnostics.append(
                {
                    "feature": feature,
                    "target": target,
                    "raw_spearman": _spearman(raw, target_values),
                    "within_layer_spearman": _spearman(residual, target_residual),
                }
            )
    return {"available": True, "features": diagnostics}


def _layer_diagnostics(head_rows: list[dict]) -> list[dict]:
    rows = []
    for layer in range(base.LAYERS):
        layer_rows = [row for row in head_rows if row["layer"] == layer]
        rows.append(
            {
                "layer": layer,
                "normalized_depth": (layer + 0.5) / base.LAYERS,
                "mean_coverage_gain": float(
                    np.mean([row["coverage_gain"] for row in layer_rows])
                ),
                "mean_episode_gain": float(
                    np.mean([row["episode_gain"] for row in layer_rows])
                ),
                "discovery_best_counts": dict(
                    sorted(Counter(row["discovery_best"] for row in layer_rows).items())
                ),
                "supported_nonlocal_count": int(
                    sum(row["supported_nonlocal"] for row in layer_rows)
                ),
            }
        )
    return rows


def _write_map(path: Path, labels: list[list[int]]) -> dict:
    return base.write_map(path, labels)


def analyze(
    profile_root: Path,
    output_dir: Path,
    *,
    v145_features: Path | None,
) -> dict:
    records, audit = base.load_records(
        profile_root, strict=True, contract="v176"
    )
    discovery, validation, generation = frozen_prompt_split(audit["prompt_ids"])
    aggregate = aggregate_profile(records)
    errors = aggregate["errors"]
    discovery_errors_by_prompt = errors[discovery]
    validation_errors_by_prompt = errors[validation]
    discovery_mean = discovery_errors_by_prompt.mean(axis=0)
    validation_budget = aggregate["budget_full"][discovery].mean(axis=0)
    discovery_energy = aggregate["reference_residual_energy"][discovery].mean(axis=0)

    stability_outcomes = np.empty(
        (len(STABILITY_SEEDS), base.LAYERS, base.HEADS), dtype=np.int8
    )
    for seed_index, seed in enumerate(STABILITY_SEEDS):
        permutation = [
            int(value)
            for value in np.random.default_rng(seed).permutation(discovery)
        ]
        subset = permutation[:MIN_DISCOVERY_PROMPTS]
        stability_outcomes[seed_index] = np.argmin(
            errors[subset].mean(axis=0), axis=-1
        )

    per_call = np.argmin(
        aggregate["call_errors"][:, discovery].mean(axis=1), axis=-1
    )
    call_first = np.argmin(
        aggregate["call_errors"][:2, discovery].mean(axis=(0, 1)), axis=-1
    )
    call_second = np.argmin(
        aggregate["call_errors"][2:, discovery].mean(axis=(0, 1)), axis=-1
    )
    ar_policy = np.argmin(
        aggregate["ar_errors"][:, discovery].mean(axis=1), axis=-1
    )
    ar_early, ar_late = ar_policy[0], ar_policy[1]
    frame_mid = float(aggregate["frame_midpoint"])

    feature_rows, feature_manifest = _load_v145_features(v145_features)
    head_rows = []
    maps = {
        "matched": [[base.LABELS["recent"]] * base.HEADS for _ in range(base.LAYERS)],
        "all_recent": [[base.LABELS["recent"]] * base.HEADS for _ in range(base.LAYERS)],
    }
    for layer in range(base.LAYERS):
        for head in range(base.HEADS):
            key = (layer, head)
            discovery_errors = {
                policy: float(discovery_mean[layer, head, policy_index])
                for policy_index, policy in enumerate(base.POLICIES)
            }
            best, margin = _oracle_policy(discovery_errors)
            comparisons = _validation_stats(
                validation_errors_by_prompt[:, layer, head, :],
                selected_policy=best,
                seed=1767001 + layer * 101 + head * 7,
            )
            best_index = base.POLICIES.index(best)
            outcomes_index = stability_outcomes[:, layer, head]
            outcomes = [base.POLICIES[int(value)] for value in outcomes_index]
            frequency = float(np.mean(outcomes_index == best_index))
            competing_nonlocal = {
                int(value)
                for value in outcomes_index
                if int(value) not in {0, best_index}
            }
            stable_policy = (
                frequency >= STABILITY_FREQUENCY
                and not competing_nonlocal
            )
            validation_gate = all(
                row["mean"] >= math.log(1.02)
                and row["bootstrap_ci95"][0] > 0.0
                and row["win_fraction"] >= 0.60
                for row in comparisons.values()
            )
            context_gate = (
                np.all(per_call[:, layer, head] == best_index)
                and ar_early[layer, head] == ar_late[layer, head] == best_index
            )
            budget_fraction = float(
                validation_budget[layer, head, best_index]
            )
            budget_gate = budget_fraction >= 0.80
            energy = float(discovery_energy[layer, head])
            layer_energy_median = float(np.median(discovery_energy[layer]))
            salience_gate = energy > 1e-12
            supported = bool(
                best != "recent"
                and margin >= 0.01
                and stable_policy
                and validation_gate
                and context_gate
                and budget_gate
                and salience_gate
            )
            if supported:
                maps["matched"][layer][head] = base.LABELS[best]
            row = {
                "layer": layer,
                "head": head,
                "discovery_best": best,
                "discovery_log_errors": discovery_errors,
                "discovery_margin": margin,
                "coverage_gain": discovery_errors["recent"] - discovery_errors["coverage"],
                "episode_gain": discovery_errors["recent"] - discovery_errors["episode"],
                "stability_outcomes": dict(sorted(Counter(outcomes).items())),
                "stability_frequency": frequency,
                "competing_nonlocal_policies": [
                    base.POLICIES[value] for value in sorted(competing_nonlocal)
                ],
                "validation_comparisons": comparisons,
                "call_pair": [
                    base.POLICIES[int(call_first[layer, head])],
                    base.POLICIES[int(call_second[layer, head])],
                ],
                "per_call_policy": {
                    str(call): base.POLICIES[int(per_call[call, layer, head])]
                    for call in range(4)
                },
                "ar_pair": [
                    base.POLICIES[int(ar_early[layer, head])],
                    base.POLICIES[int(ar_late[layer, head])],
                ],
                "budget_and_salience": {
                    "full_budget_fraction": budget_fraction,
                    "reference_residual_energy": energy,
                    "layer_median_reference_residual_energy": layer_energy_median,
                    "relative_to_layer_median": (
                        energy / max(layer_energy_median, 1e-12)
                    ),
                    "salient": salience_gate,
                },
                "gates": {
                    "margin": margin >= 0.01,
                    "same_nonlocal_9_of_12_without_competitor": stable_policy,
                    "heldout_validation": validation_gate,
                    "call_and_ar_consistency": context_gate,
                    "full_budget_ge_0p80": budget_gate,
                    "nonzero_reference_residual_energy": salience_gate,
                },
                "supported_nonlocal": supported,
                "assigned_policy": best if supported else "recent",
            }
            if feature_rows:
                row["legacy_features"] = feature_rows[key]
            head_rows.append(row)

    # Correct the two validation comparisons per head together. Bootstrap
    # controls effect uncertainty; BH controls the 720 simultaneous sign
    # tests used to turn a continuous score into a sparse membership map.
    pair_tests = []
    for row in head_rows:
        for alternative, comparison in row["validation_comparisons"].items():
            pair_tests.append(
                {
                    "row": row,
                    "alternative": alternative,
                    "p_value": comparison["sign_p"],
                }
            )
    base.benjamini_hochberg(pair_tests)
    for test in pair_tests:
        test["row"]["validation_comparisons"][test["alternative"]][
            "bh_q"
        ] = test["q_value"]
    for row in head_rows:
        fdr_gate = all(
            comparison["bh_q"] <= 0.10
            for comparison in row["validation_comparisons"].values()
        )
        row["gates"]["global_bh_q_le_0p10"] = fdr_gate
        if row["supported_nonlocal"] and not fdr_gate:
            row["supported_nonlocal"] = False
            row["assigned_policy"] = "recent"
            maps["matched"][row["layer"]][row["head"]] = base.LABELS["recent"]

    maps["all_coverage"] = [
        [base.LABELS["coverage"]] * base.HEADS
        for _ in range(base.LAYERS)
    ]
    maps["all_episode"] = [
        [base.LABELS["episode"]] * base.HEADS
        for _ in range(base.LAYERS)
    ]
    # Near-miss controls keep exact per-layer policy counts but route the
    # nonlocal operators to the strongest rejected heads. They are stricter
    # than random maps and isolate membership from operator exposure.
    for replica in range(4):
        control = [
            [base.LABELS["recent"]] * base.HEADS
            for _ in range(base.LAYERS)
        ]
        for layer in range(base.LAYERS):
            for policy in ("coverage", "episode"):
                label = base.LABELS[policy]
                count = maps["matched"][layer].count(label)
                if count == 0:
                    continue
                candidates = [
                    row
                    for row in head_rows
                    if row["layer"] == layer
                    and not row["supported_nonlocal"]
                    and control[layer][row["head"]] == base.LABELS["recent"]
                ]
                score_name = f"{policy}_gain"
                candidates.sort(
                    key=lambda row: (
                        row[score_name],
                        float(
                            np.random.default_rng(
                                1769001 + replica * 1009 + layer * 31 + row["head"]
                            ).uniform(-1e-9, 1e-9)
                        ),
                    ),
                    reverse=True,
                )
                pool = candidates[: max(count, min(len(candidates), count * 4))]
                if len(pool) < count:
                    raise ValueError(
                        f"L{layer}: insufficient rejected heads for {policy} control"
                    )
                offset = replica % len(pool)
                chosen = [pool[(offset + index) % len(pool)] for index in range(count)]
                if len({row["head"] for row in chosen}) != count:
                    raise ValueError("v176 hard-negative control contains duplicate heads")
                for row in chosen:
                    control[layer][row["head"]] = label
        maps[f"hard_negative_{replica}"] = control

    stable_counts = Counter(row["assigned_policy"] for row in head_rows)
    nonlocal_count = sum(
        row["supported_nonlocal"] for row in head_rows
    )
    discovery_sizes = (8, 16, 24, 32, 48, 64)
    sample_efficiency = []
    full_policy = np.argmin(discovery_mean, axis=-1)
    for count in discovery_sizes:
        agreements = []
        nonlocal_jaccards = []
        coverage_recalls = []
        episode_recalls = []
        for seed in STABILITY_SEEDS:
            subset = [
                int(value)
                for value in np.random.default_rng(seed).permutation(discovery)[:count]
            ]
            predicted = np.argmin(errors[subset].mean(axis=0), axis=-1)
            agreements.append(_agreement(predicted, full_policy))
            full_nonlocal = set(zip(*np.where(full_policy != 0)))
            predicted_nonlocal = set(zip(*np.where(predicted != 0)))
            union = full_nonlocal | predicted_nonlocal
            nonlocal_jaccards.append(
                1.0 if not union else len(full_nonlocal & predicted_nonlocal) / len(union)
            )
            for policy_index, values in (
                (1, coverage_recalls),
                (2, episode_recalls),
            ):
                target = full_policy == policy_index
                values.append(
                    None
                    if not np.any(target)
                    else float(np.mean(predicted[target] == policy_index))
                )
        sample_efficiency.append(
            {
                "prompt_count": count,
                "mean_policy_agreement": float(np.mean(agreements)),
                "min_policy_agreement": float(np.min(agreements)),
                "mean_nonlocal_jaccard": float(np.mean(nonlocal_jaccards)),
                "min_nonlocal_jaccard": float(np.min(nonlocal_jaccards)),
                "mean_coverage_recall": (
                    None
                    if all(value is None for value in coverage_recalls)
                    else float(np.mean([value for value in coverage_recalls if value is not None]))
                ),
                "mean_episode_recall": (
                    None
                    if all(value is None for value in episode_recalls)
                    else float(np.mean([value for value in episode_recalls if value is not None]))
                ),
            }
        )

    threshold_sensitivity = []
    for margin_threshold in (0.0, 0.005, 0.01, 0.02, 0.04):
        counts = Counter()
        members = set()
        for row in head_rows:
            if (
                row["discovery_best"] != "recent"
                and row["discovery_margin"] >= margin_threshold
            ):
                counts[row["discovery_best"]] += 1
                members.add((row["layer"], row["head"], row["discovery_best"]))
        primary_members = {
            (row["layer"], row["head"], row["discovery_best"])
            for row in head_rows
            if row["discovery_best"] != "recent"
            and row["discovery_margin"] >= 0.01
        }
        union = members | primary_members
        threshold_sensitivity.append(
            {
                "margin_threshold": margin_threshold,
                "nonlocal_candidate_counts": dict(sorted(counts.items())),
                "nonlocal_candidate_count": len(members),
                "jaccard_to_primary_0p01": (
                    1.0 if not union else len(members & primary_members) / len(union)
                ),
            }
        )

    maps_dir = output_dir / "maps"
    map_manifest = {
        name: _write_map(maps_dir / f"{name}.csv", rows)
        for name, rows in maps.items()
    }
    scores_path = output_dir / "head_scores.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "layer", "head", "discovery_best", "discovery_margin",
            "coverage_gain", "episode_gain", "stability_frequency",
            "supported_nonlocal", "assigned_policy",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in head_rows:
            writer.writerow({field: row[field] for field in fields})

    payload = {
        "version": 1,
        "experiment": "v176_superset_rccp",
        "profile_audit": audit,
        "teacher_contract": {
            "candidate_physical_superset_required": True,
            "local_teacher_is_not_generation_utility": True,
            "union_max_ffe": 17,
        },
        "prompt_split": {
            "seed": DISCOVERY_SEED,
            "discovery_prompt_ids": discovery,
            "validation_prompt_ids": validation,
            "generation_prompt_ids": generation,
            "generation_prompts_used_for_membership": False,
        },
        "context_agreement": {
            "call_halves": _agreement(call_first, call_second),
            "ar_halves": _agreement(ar_early, ar_late),
            "ar_midpoint": frame_mid,
        },
        "sample_efficiency": sample_efficiency,
        "threshold_sensitivity": threshold_sensitivity,
        "legacy_feature_manifest": feature_manifest,
        "legacy_feature_predictors": _predictor_diagnostics(
            head_rows, feature_rows
        ),
        "layer_diagnostics": _layer_diagnostics(head_rows),
        "assigned_policy_counts": dict(sorted(stable_counts.items())),
        "supported_nonlocal_head_count": nonlocal_count,
        "head_rows": head_rows,
        "maps": map_manifest,
        "head_scores_csv": {
            "path": str(scores_path.resolve()),
            "sha256": base.sha256(scores_path),
        },
        "generation_ready": bool(nonlocal_count > 0),
        "claim_boundary": (
            "v176 membership is a stable local cache-compatibility hypothesis; "
            "it becomes a generation method only if matched routing beats "
            "layer/count-matched hard negatives on untouched prompts."
        ),
    }
    output_path = output_dir / "analysis.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# v176 Superset RCCP Analysis",
                "",
                f"- Complete profile: {audit['complete_profile']}",
                f"- Supported nonlocal heads: {nonlocal_count}",
                f"- Assigned policies: {dict(sorted(stable_counts.items()))}",
                f"- Call-half agreement: {_agreement(call_first, call_second):.4f}",
                f"- AR-half agreement: {_agreement(ar_early, ar_late):.4f}",
                f"- Generation ready: {payload['generation_ready']}",
                "",
                payload["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        "[v176-analysis] "
        f"prompts=128 nonlocal={nonlocal_count} "
        f"generation_ready={payload['generation_ready']} output={output_path}"
    )
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=root / "runs" / "v176_superset_rccp" / "profiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "v176_superset_rccp" / "analysis",
    )
    parser.add_argument(
        "--v145-features",
        type=Path,
        default=(
            root
            / "docs"
            / "results"
            / "v145_crossed_seed_head_profile"
            / "head_factor_reproducibility.csv"
        ),
    )
    args = parser.parse_args()
    analyze(
        args.profile_root,
        args.output_dir,
        v145_features=args.v145_features,
    )


if __name__ == "__main__":
    main()
