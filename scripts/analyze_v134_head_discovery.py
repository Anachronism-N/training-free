#!/usr/bin/env python3
"""Analyze v134 counterfactual and temporal head-profile artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def _median(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else float("nan")


def _mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(values)) if values else float("nan")


def _percentile(values, fraction: float) -> float:
    ordered = sorted(
        float(value) for value in values if math.isfinite(float(value))
    )
    if not ordered:
        return float("nan")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1 - alpha) + ordered[upper] * alpha


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return float("nan")
    left_rank = _rank([pair[0] for pair in pairs])
    right_rank = _rank([pair[1] for pair in pairs])
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator > 0 else float("nan")


def _relative_per_head(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            "paired signatures must have identical [heads, features] shape"
        )
    left = left.float()
    right = right.float()
    numerator = (left - right).norm(dim=-1)
    denominator = 0.5 * (
        left.norm(dim=-1) + right.norm(dim=-1)
    )
    return (
        numerator / denominator.clamp_min(1e-6)
    ).cpu().tolist()


def _record_key(record: dict) -> tuple:
    return (
        str(record["mode"]),
        int(record["current_frame"]),
        int(record["nominal_timestep"]),
        int(record["layer"]),
    )


def load_profiles(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.pt"))
    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) != 2:
            raise ValueError(f"unsupported profile version in {path}")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def collect_counterfactual_observations(
    profiles: list[dict],
) -> tuple[list[dict], list[dict]]:
    observations = []
    missing = []
    signature_names = (
        "residual_signature",
        "native_signature",
        "query_signature",
        "current_key_signature",
    )
    for profile in profiles:
        job = profile["job"]
        records = defaultdict(dict)
        for record in profile["records"]:
            records[_record_key(record)][str(record["branch"])] = record
        for key, branches in records.items():
            if "base" not in branches:
                missing.append(
                    {"job_id": job["job_id"], "key": key, "missing": ["base"]}
                )
                continue
            if not {"semantic", "null"}.issubset(branches):
                absent = sorted({"semantic", "null"} - set(branches))
                missing.append(
                    {"job_id": job["job_id"], "key": key, "missing": absent}
                )
                continue
            base = branches["base"]
            semantic = branches["semantic"]
            null = branches["null"]
            distances = {}
            for name in signature_names:
                distances[f"semantic_{name}"] = _relative_per_head(
                    semantic[name], base[name]
                )
                distances[f"null_{name}"] = _relative_per_head(
                    null[name], base[name]
                )
            num_heads = len(distances["semantic_residual_signature"])
            for head in range(num_heads):
                semantic_interaction = distances[
                    "semantic_residual_signature"
                ][head]
                null_interaction = distances[
                    "null_residual_signature"
                ][head]
                row = {
                    "dataset_index": int(job["dataset_index"]),
                    "job_id": str(job["job_id"]),
                    "family_id": str(job.get("family_id", job["job_id"])),
                    "factor": str(job.get("factor", "unknown")),
                    "mode": str(base["mode"]),
                    "current_frame": int(base["current_frame"]),
                    "nominal_timestep": int(base["nominal_timestep"]),
                    "layer": int(base["layer"]),
                    "head": int(head),
                    "semantic_interaction": semantic_interaction,
                    "null_interaction": null_interaction,
                    "interaction_log_ratio": math.log(
                        (semantic_interaction + 1e-4)
                        / (null_interaction + 1e-4)
                    ),
                }
                for name in signature_names[1:]:
                    short = name.removesuffix("_signature")
                    semantic_response = distances[f"semantic_{name}"][head]
                    null_response = distances[f"null_{name}"][head]
                    row[f"semantic_{short}_response"] = semantic_response
                    row[f"null_{short}_response"] = null_response
                    row[f"{short}_log_ratio"] = math.log(
                        (semantic_response + 1e-4)
                        / (null_response + 1e-4)
                    )
                observations.append(row)
    return observations, missing


def collect_temporal_observations(profiles: list[dict]) -> list[dict]:
    observations = []
    for profile in profiles:
        job = profile["job"]
        for record in profile["records"]:
            if record["branch"] != "base":
                continue
            probs = record["temporal_probs"].float()
            logits = record["temporal_logits"].float()
            frame_ids = record["history_frame_ids"].float()
            ages = float(record["current_frame"]) - frame_ids
            entropy_denominator = math.log(max(2, probs.shape[-1]))
            for head in range(probs.shape[0]):
                head_probs = probs[head]
                head_probs = head_probs / head_probs.sum().clamp_min(1e-8)
                entropy = -(
                    head_probs
                    * head_probs.clamp_min(1e-8).log()
                ).sum() / entropy_denominator
                peak_index = int(head_probs.argmax().item())
                observations.append(
                    {
                        "dataset_index": int(job["dataset_index"]),
                        "job_id": str(job["job_id"]),
                        "kind": str(job["kind"]),
                        "factor": str(job.get("factor", "natural")),
                        "mode": str(record["mode"]),
                        "current_frame": int(record["current_frame"]),
                        "nominal_timestep": int(
                            record["nominal_timestep"]
                        ),
                        "layer": int(record["layer"]),
                        "head": int(head),
                        "expected_age": float(
                            (head_probs * ages).sum().item()
                        ),
                        "recent4_mass": float(
                            head_probs[ages <= 4].sum().item()
                        ),
                        "old12_mass": float(
                            head_probs[ages > 12].sum().item()
                        ),
                        "temporal_entropy": float(entropy.item()),
                        "peak_age": float(ages[peak_index].item()),
                        "positive_logit_fraction": float(
                            (logits[head] > 0).float().mean().item()
                        ),
                    }
                )
    return observations


def collect_family_base_consistency(profiles: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for profile in profiles:
        job = profile["job"]
        family_id = str(job.get("family_id", job["job_id"]))
        for record in profile["records"]:
            if record["branch"] == "base":
                grouped[(family_id, _record_key(record))].append(
                    (str(job["job_id"]), record)
                )
    observations = []
    for (family_id, key), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[0])
        if len(rows) < 2:
            continue
        reference = rows[0][1]
        for _, record in rows[1:]:
            residual = _relative_per_head(
                record["residual_signature"],
                reference["residual_signature"],
            )
            native = _relative_per_head(
                record["native_signature"],
                reference["native_signature"],
            )
            for head in range(len(residual)):
                observations.append(
                    {
                        "family_id": family_id,
                        "mode": key[0],
                        "current_frame": key[1],
                        "nominal_timestep": key[2],
                        "layer": key[3],
                        "head": head,
                        "residual_base_drift": residual[head],
                        "native_base_drift": native[head],
                    }
                )
    return observations


def _group_median(rows: list[dict], keys: tuple[str, ...], fields: tuple[str, ...]):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[name] for name in keys)
        for field in fields:
            grouped[key][field].append(row[field])
    output = []
    for key, values in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        for field in fields:
            row[field] = _median(values[field])
            row[f"{field}_samples"] = len(values[field])
        output.append(row)
    return output


def _group_cluster_median(
    rows: list[dict],
    keys: tuple[str, ...],
    *,
    cluster_key: str,
    fields: tuple[str, ...],
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        key = tuple(row[name] for name in keys)
        cluster = str(row[cluster_key])
        for field in fields:
            grouped[key][cluster][field].append(row[field])
    output = []
    for key, clusters in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["cluster_count"] = len(clusters)
        for field in fields:
            cluster_values = [
                _median(values[field]) for values in clusters.values()
            ]
            row[field] = _median(cluster_values)
            row[f"{field}_cluster_q25"] = _percentile(
                cluster_values, 0.25
            )
            row[f"{field}_cluster_q75"] = _percentile(
                cluster_values, 0.75
            )
            if field.endswith("log_ratio"):
                row[f"{field}_positive_cluster_fraction"] = _mean(
                    [float(value > 0) for value in cluster_values]
                )
        output.append(row)
    return output


def _bootstrap_head_scores(
    observations: list[dict],
    *,
    rounds: int,
    seed: int,
) -> dict[tuple[int, int], dict]:
    by_head_family = defaultdict(lambda: defaultdict(list))
    by_head_jobs = defaultdict(set)
    for row in observations:
        key = (int(row["layer"]), int(row["head"]))
        by_head_family[key][str(row["family_id"])].append(
            float(row["interaction_log_ratio"])
        )
        by_head_jobs[key].add(str(row["job_id"]))
    rng = random.Random(seed)
    output = {}
    for key, families in sorted(by_head_family.items()):
        family_values = {
            family_id: _median(values)
            for family_id, values in families.items()
        }
        family_ids = sorted(family_values)
        samples = []
        for _ in range(rounds):
            selected = [
                rng.choice(family_ids) for _ in family_ids
            ]
            samples.append(
                _median(
                    [family_values[family_id] for family_id in selected]
                )
            )
        output[key] = {
            "job_count": len(by_head_jobs[key]),
            "bootstrap_family_count": len(family_ids),
            "bootstrap_low": _percentile(samples, 0.025),
            "bootstrap_high": _percentile(samples, 0.975),
            "bootstrap_p_conditional": _mean(
                [float(value > 0) for value in samples]
            ),
        }
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _head_axis_summary(
    rows: list[dict],
    keys: tuple[str, ...],
    *,
    score_field: str = "interaction_log_ratio",
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        scores = [float(row[score_field]) for row in values]
        output.append(
            {
                **{
                    name: value
                    for name, value in zip(keys, key)
                },
                "head_count": len(values),
                "score_median": _median(scores),
                "score_q25": _percentile(scores, 0.25),
                "score_q75": _percentile(scores, 0.75),
                "prompt_conditional_head_fraction": _mean(
                    [float(score > 0) for score in scores]
                ),
            }
        )
    return output


def _load_legacy_map(spec: str) -> tuple[str, list[list[int]]]:
    if "=" not in spec:
        raise ValueError("--legacy-map must be NAME=PATH")
    name, path_text = spec.split("=", 1)
    with Path(path_text).open("r", encoding="utf-8", newline="") as handle:
        matrix = [
            [int(value) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(matrix) != 30 or any(len(row) != 12 for row in matrix):
        raise ValueError(f"legacy map {name} must be 30x12")
    return name, matrix


def analyze(
    observational_profiles: list[dict],
    counterfactual_profiles: list[dict],
    *,
    output_dir: Path,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    expected_count: int,
    legacy_specs: list[str],
) -> dict:
    interactions, missing_pairs = collect_counterfactual_observations(
        counterfactual_profiles
    )
    temporal = collect_temporal_observations(
        observational_profiles + counterfactual_profiles
    )
    base_consistency = collect_family_base_consistency(
        counterfactual_profiles
    )
    if not interactions:
        raise ValueError("no complete counterfactual observations found")

    interaction_fields = (
        "semantic_interaction",
        "null_interaction",
        "interaction_log_ratio",
        "semantic_native_response",
        "null_native_response",
        "semantic_query_response",
        "null_query_response",
        "query_log_ratio",
        "semantic_current_key_response",
        "null_current_key_response",
        "current_key_log_ratio",
        "native_log_ratio",
    )
    per_head = _group_median(
        interactions,
        ("layer", "head"),
        interaction_fields,
    )
    bootstrap = _bootstrap_head_scores(
        interactions,
        rounds=bootstrap_rounds,
        seed=bootstrap_seed,
    )
    temporal_head = _group_median(
        temporal,
        ("layer", "head"),
        (
            "expected_age",
            "recent4_mass",
            "old12_mass",
            "temporal_entropy",
            "peak_age",
            "positive_logit_fraction",
        ),
    )
    temporal_by_key = {
        (int(row["layer"]), int(row["head"])): row
        for row in temporal_head
    }
    for row in per_head:
        key = (int(row["layer"]), int(row["head"]))
        row.update(bootstrap[key])
        probability = float(row["bootstrap_p_conditional"])
        score = float(row["interaction_log_ratio"])
        row["label"] = (
            "prompt_conditional" if score > 0 else "prompt_invariant"
        )
        row["label_code"] = 1 if score > 0 else 0
        row["bootstrap_confidence"] = max(probability, 1 - probability)
        row["reliable_at_80pct"] = int(
            row["bootstrap_confidence"] >= 0.80
        )
        for field, value in temporal_by_key.get(key, {}).items():
            if field not in {"layer", "head"}:
                row[field] = value

    factor_rows = _group_cluster_median(
        interactions,
        ("factor", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "semantic_interaction",
            "null_interaction",
            "interaction_log_ratio",
            "native_log_ratio",
            "query_log_ratio",
            "current_key_log_ratio",
        ),
    )
    job_rows = _group_median(
        interactions,
        (
            "dataset_index",
            "job_id",
            "family_id",
            "factor",
            "layer",
            "head",
        ),
        (
            "semantic_interaction",
            "null_interaction",
            "interaction_log_ratio",
            "native_log_ratio",
            "query_log_ratio",
            "current_key_log_ratio",
        ),
    )
    timestep_rows = _group_cluster_median(
        interactions,
        ("mode", "nominal_timestep", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "semantic_interaction",
            "null_interaction",
            "interaction_log_ratio",
            "native_log_ratio",
            "query_log_ratio",
            "current_key_log_ratio",
        ),
    )
    ar_rows = _group_cluster_median(
        interactions,
        ("mode", "current_frame", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "semantic_interaction",
            "null_interaction",
            "interaction_log_ratio",
            "native_log_ratio",
            "query_log_ratio",
            "current_key_log_ratio",
        ),
    )
    temporal_timestep_rows = _group_median(
        temporal,
        ("kind", "mode", "nominal_timestep", "layer", "head"),
        (
            "expected_age",
            "recent4_mass",
            "old12_mass",
            "temporal_entropy",
            "positive_logit_fraction",
        ),
    )
    temporal_ar_rows = _group_median(
        temporal,
        ("kind", "mode", "current_frame", "layer", "head"),
        (
            "expected_age",
            "recent4_mass",
            "old12_mass",
            "temporal_entropy",
            "positive_logit_fraction",
        ),
    )
    base_consistency_rows = _group_median(
        base_consistency,
        ("family_id", "layer", "head"),
        ("residual_base_drift", "native_base_drift"),
    )
    layer_summary_rows = _head_axis_summary(per_head, ("layer",))
    factor_layer_rows = _head_axis_summary(
        factor_rows, ("factor", "layer")
    )
    timestep_layer_rows = _head_axis_summary(
        timestep_rows, ("mode", "nominal_timestep", "layer")
    )
    ar_layer_rows = _head_axis_summary(
        ar_rows, ("mode", "current_frame", "layer")
    )

    family_order = {
        family_id: index
        for index, family_id in enumerate(
            sorted({str(row["family_id"]) for row in interactions})
        )
    }
    even_scores = _group_median(
        [
            row
            for row in interactions
            if family_order[str(row["family_id"])] % 2 == 0
        ],
        ("layer", "head"),
        ("interaction_log_ratio",),
    )
    odd_scores = _group_median(
        [
            row
            for row in interactions
            if family_order[str(row["family_id"])] % 2 == 1
        ],
        ("layer", "head"),
        ("interaction_log_ratio",),
    )
    even_map = {
        (row["layer"], row["head"]): row["interaction_log_ratio"]
        for row in even_scores
    }
    odd_map = {
        (row["layer"], row["head"]): row["interaction_log_ratio"]
        for row in odd_scores
    }
    head_keys = [(layer, head) for layer in range(30) for head in range(12)]
    split_spearman = _spearman(
        [even_map.get(key, float("nan")) for key in head_keys],
        [odd_map.get(key, float("nan")) for key in head_keys],
    )
    split_label_agreement = _mean(
        [
            float((even_map[key] > 0) == (odd_map[key] > 0))
            for key in head_keys
            if key in even_map and key in odd_map
        ]
    )

    label_counts = {
        "prompt_conditional": sum(
            row["label"] == "prompt_conditional" for row in per_head
        ),
        "prompt_invariant": sum(
            row["label"] == "prompt_invariant" for row in per_head
        ),
    }
    reliable_fraction = _mean(
        [row["reliable_at_80pct"] for row in per_head]
    )
    score_vector = [row["interaction_log_ratio"] for row in per_head]
    correlations = {}
    for field in (
        "expected_age",
        "recent4_mass",
        "old12_mass",
        "temporal_entropy",
        "positive_logit_fraction",
    ):
        correlations[field] = _spearman(
            score_vector, [row.get(field, float("nan")) for row in per_head]
        )
    representation_correlations = {
        field: _spearman(
            score_vector, [row.get(field, float("nan")) for row in per_head]
        )
        for field in (
            "native_log_ratio",
            "query_log_ratio",
            "current_key_log_ratio",
        )
    }

    threshold_rows = []
    for threshold_index in range(-10, 11):
        threshold = threshold_index * 0.05
        conditional = sum(
            row["interaction_log_ratio"] > threshold for row in per_head
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "prompt_conditional": conditional,
                "prompt_invariant": len(per_head) - conditional,
                "conditional_fraction": conditional / len(per_head),
            }
        )

    posthoc = {}
    for spec in legacy_specs:
        name, matrix = _load_legacy_map(spec)
        table = defaultdict(int)
        for row in per_head:
            legacy = matrix[int(row["layer"])][int(row["head"])]
            table[(int(legacy), str(row["label"]))] += 1
        posthoc[name] = {
            f"legacy_{legacy}__{label}": count
            for (legacy, label), count in sorted(table.items())
        }

    semantic_global = _median(
        [row["semantic_interaction"] for row in interactions]
    )
    null_global = _median(
        [row["null_interaction"] for row in interactions]
    )
    minority_fraction = min(label_counts.values()) / max(1, len(per_head))
    gates = {
        "observational_profile_count": {
            "observed": len(observational_profiles),
            "required": expected_count,
            "passed": len(observational_profiles) == expected_count,
        },
        "counterfactual_profile_count": {
            "observed": len(counterfactual_profiles),
            "required": expected_count,
            "passed": len(counterfactual_profiles) == expected_count,
        },
        "complete_counterfactual_pairs": {
            "missing": len(missing_pairs),
            "required": 0,
            "passed": not missing_pairs,
        },
        "semantic_exceeds_paraphrase": {
            "semantic_median": semantic_global,
            "paraphrase_median": null_global,
            "passed": semantic_global > null_global,
        },
        "split_half_rank_reproducibility": {
            "observed": split_spearman,
            "required": 0.30,
            "passed": split_spearman >= 0.30,
        },
        "bootstrap_reliable_fraction": {
            "observed": reliable_fraction,
            "required": 0.70,
            "passed": reliable_fraction >= 0.70,
        },
        "nondegenerate_binary_partition": {
            "observed": minority_fraction,
            "required": 0.10,
            "passed": minority_fraction >= 0.10,
        },
    }
    accepted = all(row["passed"] for row in gates.values())
    report = {
        "version": 1,
        "method": "counterfactual_prompt_history_interaction",
        "threshold": {
            "value": 0.0,
            "meaning": "semantic intervention exceeds paraphrase variation",
            "learned_or_count_matched": False,
        },
        "profile_counts": {
            "observational": len(observational_profiles),
            "counterfactual": len(counterfactual_profiles),
        },
        "observation_counts": {
            "counterfactual_head_records": len(interactions),
            "temporal_head_records": len(temporal),
        },
        "label_counts": label_counts,
        "global_interaction": {
            "semantic_median": semantic_global,
            "paraphrase_median": null_global,
            "median_log_ratio": _median(
                [row["interaction_log_ratio"] for row in interactions]
            ),
        },
        "reproducibility": {
            "split_half_spearman": split_spearman,
            "split_half_label_agreement": split_label_agreement,
            "bootstrap_reliable_fraction": reliable_fraction,
            "matched_base_residual_drift_median": _median(
                [row["residual_base_drift"] for row in base_consistency]
            ),
            "matched_base_native_drift_median": _median(
                [row["native_base_drift"] for row in base_consistency]
            ),
        },
        "prompt_temporal_spearman": correlations,
        "prompt_representation_spearman": representation_correlations,
        "posthoc_legacy_cross_tabs": posthoc,
        "acceptance_gates": {"accepted": accepted, "checks": gates},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "head_scores.csv", per_head)
    _write_csv(output_dir / "head_job_scores.csv", job_rows)
    _write_csv(output_dir / "head_factor_scores.csv", factor_rows)
    _write_csv(output_dir / "head_timestep_scores.csv", timestep_rows)
    _write_csv(output_dir / "head_ar_scores.csv", ar_rows)
    _write_csv(
        output_dir / "temporal_timestep_scores.csv",
        temporal_timestep_rows,
    )
    _write_csv(output_dir / "temporal_ar_scores.csv", temporal_ar_rows)
    _write_csv(
        output_dir / "family_base_consistency.csv",
        base_consistency_rows,
    )
    _write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    _write_csv(output_dir / "layer_summary.csv", layer_summary_rows)
    _write_csv(
        output_dir / "factor_layer_summary.csv", factor_layer_rows
    )
    _write_csv(
        output_dir / "timestep_layer_summary.csv",
        timestep_layer_rows,
    )
    _write_csv(output_dir / "ar_layer_summary.csv", ar_layer_rows)
    with (output_dir / "head_map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        score_lookup = {
            (int(row["layer"]), int(row["head"])): int(row["label_code"])
            for row in per_head
        }
        for layer in range(30):
            writer.writerow(
                [score_lookup[(layer, head)] for head in range(12)]
            )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_debug.json").write_text(
        json.dumps(
            {
                "missing_counterfactual_records": missing_pairs,
                "counterfactual_profile_paths": [
                    profile["_path"] for profile in counterfactual_profiles
                ],
                "observational_profile_paths": [
                    profile["_path"] for profile in observational_profiles
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# v134 Head Discovery Analysis",
        "",
        f"- Acceptance gates: **{'PASS' if accepted else 'FAIL'}**",
        (
            "- Profiles: "
            f"{len(observational_profiles)} observational, "
            f"{len(counterfactual_profiles)} counterfactual"
        ),
        (
            "- Zero-threshold partition: "
            f"{label_counts['prompt_conditional']} prompt-conditional, "
            f"{label_counts['prompt_invariant']} prompt-invariant"
        ),
        (
            "- Semantic/paraphrase median interaction: "
            f"{semantic_global:.6f} / {null_global:.6f}"
        ),
        (
            "- Split-half Spearman / label agreement: "
            f"{split_spearman:.4f} / {split_label_agreement:.4f}"
        ),
        f"- Bootstrap-reliable head fraction: {reliable_fraction:.4f}",
        (
            "- Matched-base residual/native drift: "
            f"{report['reproducibility']['matched_base_residual_drift_median']:.6f} / "
            f"{report['reproducibility']['matched_base_native_drift_median']:.6f}"
        ),
        "",
        "## Prompt-Temporal Relations",
        "",
    ]
    for field, value in correlations.items():
        summary_lines.append(f"- CPHI vs `{field}` Spearman: {value:.4f}")
    summary_lines.extend(["", "## Representation Relations", ""])
    for field, value in representation_correlations.items():
        summary_lines.append(f"- CPHI vs `{field}` Spearman: {value:.4f}")
    summary_lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            (
                "The static binary map is admissible only when the acceptance "
                "gates pass. If the global map fails but timestep/factor tables "
                "are reproducible, use a continuous timestep-conditioned gate "
                "instead of forcing a binary partition."
            ),
            "",
        ]
    )
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observational-dir", required=True, type=Path)
    parser.add_argument("--counterfactual-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--legacy-map",
        action="append",
        default=[],
        help="Optional post-hoc comparison in NAME=/path/to/30x12.csv form.",
    )
    parser.add_argument("--strict-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        load_profiles(args.observational_dir),
        load_profiles(args.counterfactual_dir),
        output_dir=args.output_dir,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
        expected_count=args.expected_count,
        legacy_specs=args.legacy_map,
    )
    print(
        "[v134-analysis] "
        f"labels={report['label_counts']} "
        f"accepted={report['acceptance_gates']['accepted']} "
        f"output={args.output_dir}"
    )
    if args.strict_gates and not report["acceptance_gates"]["accepted"]:
        failed = [
            name
            for name, row in report["acceptance_gates"]["checks"].items()
            if not row["passed"]
        ]
        raise SystemExit("v134 acceptance gates failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
