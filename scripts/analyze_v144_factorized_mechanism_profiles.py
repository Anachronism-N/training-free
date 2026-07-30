#!/usr/bin/env python3
"""Analyze matched v144 Q/K/V, policy, and spatial-topology profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
FAMILIES = 16
VARIANTS = (
    "base",
    "seed_control",
    "paraphrase",
    "identity",
    "scene",
    "action",
    "camera",
    "full_semantic",
)
SEMANTIC_FACTORS = ("identity", "scene", "action", "camera")
SEED_CORRECTED_VARIANTS = (
    *SEMANTIC_FACTORS,
    "paraphrase",
    "full_semantic",
)
TOPOLOGY_FIELDS = (
    "normalized_entropy",
    "diagonal_mass",
    "expected_displacement",
    "directional_coherence",
    "top1_displacement",
)
EPSILON = 1e-8
DOMINANT_MIN_Z = 0.50
DOMINANT_MIN_MARGIN = 0.25


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_key(record: dict) -> tuple[str, int, int, int]:
    return (
        str(record["mode"]),
        int(record["current_frame"]),
        int(record["nominal_timestep"]),
        int(record["layer"]),
    )


def _load_profiles(directory: Path, expected_count: int) -> list[dict]:
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v144 profiles, found {len(paths)}"
        )
    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != 7:
            raise RuntimeError(f"{path} is not a v144 mechanism profile")
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        if str(job.get("kind")) != "factorized_head_mechanism":
            raise RuntimeError(f"{path} has the wrong job kind")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete calls")
        if not metadata.get("descriptor_export"):
            raise RuntimeError(f"{path} lacks Q/K/V descriptors")
        if not metadata.get("spatial_topology_metrics"):
            raise RuntimeError(f"{path} lacks spatial topology metrics")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def _aligned_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[0] != HEADS:
        raise ValueError(
            f"descriptor shapes do not align: {left.shape} vs {right.shape}"
        )
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1).flatten(1).mean(dim=-1)


def _descriptor_similarity(
    query: torch.Tensor, history_key: torch.Tensor
) -> torch.Tensor:
    if (
        query.ndim != 3
        or history_key.ndim != 4
        or query.shape[0] != HEADS
        or query.shape[0] != history_key.shape[0]
        or query.shape[1] != history_key.shape[2]
        or query.shape[2] != history_key.shape[3]
    ):
        raise ValueError("projected Q/K descriptor shapes do not align")
    query = torch.nn.functional.normalize(query.float(), dim=-1)
    history_key = torch.nn.functional.normalize(
        history_key.float(), dim=-1
    )
    frame_similarity = torch.einsum(
        "hsp,hfsp->hfs", query, history_key
    ).mean(dim=-1)
    return frame_similarity.max(dim=-1).values


def _policy_vector(record: dict) -> torch.Tensor:
    metrics = record.get("causal_policy_metrics") or {}
    names = sorted(metrics)
    if not names:
        return torch.full((HEADS, 0), float("nan"))
    values = torch.stack(
        [
            metrics[name]["projected_relative_error"].float()
            for name in names
        ],
        dim=-1,
    )
    if values.shape[0] != HEADS:
        raise ValueError("causal policy vector has the wrong head count")
    return values / values.sum(dim=-1, keepdim=True).clamp_min(EPSILON)


def _policy_shift(base: dict, variant: dict) -> torch.Tensor:
    left, right = _policy_vector(base), _policy_vector(variant)
    if left.shape != right.shape:
        raise ValueError("base/variant policy candidates differ")
    if left.shape[-1] == 0:
        return torch.full((HEADS,), float("nan"))
    return (left - right).abs().sum(dim=-1)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left_rank, right_rank = _rankdata(left[finite]), _rankdata(right[finite])
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _layer_residual(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(LAYERS, HEADS)
    return (matrix - np.median(matrix, axis=1, keepdims=True)).reshape(-1)


def _layer_eta_squared(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    grand = float(array.mean())
    total = float(np.square(array - grand).sum())
    if total <= 1e-12:
        return 0.0
    means = array.reshape(LAYERS, HEADS).mean(axis=1)
    return float(HEADS * np.square(means - grand).sum() / total)


def _profile_index(profiles: list[dict]) -> tuple[dict, list[dict]]:
    indexed = {}
    audits = []
    for profile in profiles:
        job = profile["job"]
        family, variant = int(job["family_index"]), str(job["variant"])
        key = (family, variant)
        if key in indexed:
            raise RuntimeError(f"duplicate v144 profile: {key}")
        if not 0 <= family < FAMILIES or variant not in VARIANTS:
            raise RuntimeError(f"invalid v144 profile coordinate: {key}")
        records = {}
        for record in profile["records"]:
            if str(record["branch"]) != "base":
                raise RuntimeError(f"{key} contains a non-base branch")
            state = _state_key(record)
            if state in records:
                raise RuntimeError(f"{key} duplicates state {state}")
            for field in (
                "query_projection",
                "history_key_projection",
                "history_value_projection",
                "history_value_rms",
                "spatial_topology_metrics",
            ):
                if field not in record:
                    raise RuntimeError(f"{key}/{state} lacks {field}")
            records[state] = record
        layers_by_state = Counter(state[:3] for state in records)
        if not layers_by_state or set(layers_by_state.values()) != {LAYERS}:
            raise RuntimeError(f"{key} has incomplete layer states")
        indexed[key] = {"profile": profile, "records": records}
        audits.append(
            {
                "dataset_index": int(job["dataset_index"]),
                "family_index": family,
                "variant": variant,
                "seed": int(profile["metadata"]["seed"]),
                "state_count": len(layers_by_state),
                "record_count": len(records),
                "path": profile["_path"],
            }
        )
    expected = {
        (family, variant)
        for family in range(FAMILIES)
        for variant in VARIANTS
    }
    if set(indexed) != expected:
        raise RuntimeError(
            f"v144 profile grid is incomplete: missing={sorted(expected-set(indexed))}"
        )
    return indexed, audits


def _state_observations(indexed: dict) -> list[dict]:
    rows = []
    for family in range(FAMILIES):
        base = indexed[(family, "base")]
        base_job = base["profile"]["job"]
        for variant in VARIANTS[1:]:
            donor = indexed[(family, variant)]
            donor_job = donor["profile"]["job"]
            if set(base["records"]) != set(donor["records"]):
                raise RuntimeError(
                    f"family={family} variant={variant} state grid differs"
                )
            for state in sorted(base["records"]):
                base_record = base["records"][state]
                donor_record = donor["records"][state]
                query_shift = 1.0 - _aligned_cosine(
                    base_record["query_projection"],
                    donor_record["query_projection"],
                )
                key_shift = 1.0 - _aligned_cosine(
                    base_record["history_key_projection"],
                    donor_record["history_key_projection"],
                )
                value_shift = 1.0 - _aligned_cosine(
                    base_record["history_value_projection"],
                    donor_record["history_value_projection"],
                )
                base_rms = base_record["history_value_rms"].float()
                donor_rms = donor_record["history_value_rms"].float()
                value_scale_shift = (
                    (donor_rms + EPSILON).log()
                    - (base_rms + EPSILON).log()
                ).abs().flatten(1).mean(dim=-1)
                own_compatibility = _descriptor_similarity(
                    base_record["query_projection"],
                    base_record["history_key_projection"],
                )
                donor_compatibility = _descriptor_similarity(
                    base_record["query_projection"],
                    donor_record["history_key_projection"],
                )
                compatibility_loss = (
                    own_compatibility - donor_compatibility
                )
                policy_shift = _policy_shift(base_record, donor_record)
                topology_delta = {}
                for field in TOPOLOGY_FIELDS:
                    base_value = base_record["spatial_topology_metrics"][
                        field
                    ].float()
                    donor_value = donor_record["spatial_topology_metrics"][
                        field
                    ].float()
                    topology_delta[field] = (donor_value - base_value).abs()
                mode, frame, timestep, layer = state
                for head in range(HEADS):
                    row = {
                        "family_index": family,
                        "variant": variant,
                        "same_seed_as_base": int(
                            bool(donor_job["same_seed_as_base"])
                        ),
                        "token_jaccard_to_base": float(
                            donor_job["token_jaccard_to_base"]
                        ),
                        "normalized_token_edit_distance": float(
                            donor_job["normalized_token_edit_distance"]
                        ),
                        "base_seed": int(base_job["seed"]),
                        "variant_seed": int(donor_job["seed"]),
                        "mode": mode,
                        "current_frame": frame,
                        "nominal_timestep": timestep,
                        "layer": layer,
                        "head": head,
                        "query_shift": float(query_shift[head]),
                        "key_shift": float(key_shift[head]),
                        "value_shift": float(value_shift[head]),
                        "value_scale_shift": float(value_scale_shift[head]),
                        "own_history_compatibility": float(
                            own_compatibility[head]
                        ),
                        "donor_history_compatibility": float(
                            donor_compatibility[head]
                        ),
                        "compatibility_loss": float(
                            compatibility_loss[head]
                        ),
                        "policy_shift": float(policy_shift[head]),
                    }
                    for field in TOPOLOGY_FIELDS:
                        row[f"topology_{field}_shift"] = float(
                            topology_delta[field][head]
                        )
                        row[f"base_topology_{field}"] = float(
                            base_record["spatial_topology_metrics"][field][
                                head
                            ]
                        )
                    rows.append(row)
    return rows


MEASURES = (
    "query_shift",
    "key_shift",
    "value_shift",
    "value_scale_shift",
    "compatibility_loss",
    "policy_shift",
    *(f"topology_{field}_shift" for field in TOPOLOGY_FIELDS),
)


def _family_head_axes(observations: list[dict]) -> list[dict]:
    grouped = {}
    for row in observations:
        key = (
            int(row["family_index"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["variant"]),
        )
        grouped.setdefault(key, []).append(row)
    rows = []
    for family in range(FAMILIES):
        for layer in range(LAYERS):
            for head in range(HEADS):
                row = {
                    "family_index": family,
                    "layer": layer,
                    "head": head,
                }
                for variant in VARIANTS[1:]:
                    values = grouped[(family, layer, head, variant)]
                    for measure in MEASURES:
                        row[f"{variant}.{measure}"] = float(
                            np.median(
                                [float(value[measure]) for value in values]
                            )
                        )
                for factor in (*SEMANTIC_FACTORS, "paraphrase", "full_semantic"):
                    for measure in MEASURES:
                        row[f"{factor}.{measure}_excess_seed"] = (
                            row[f"{factor}.{measure}"]
                            - row[f"seed_control.{measure}"]
                        )
                rows.append(row)
    return rows


def _context_observations_with_seed_excess(
    observations: list[dict],
) -> list[dict]:
    """Add family-paired semantic-minus-seed rows before context aggregation."""
    indexed = {}
    for row in observations:
        key = (
            int(row["family_index"]),
            str(row["variant"]),
            str(row["mode"]),
            int(row["current_frame"]),
            int(row["nominal_timestep"]),
            int(row["layer"]),
            int(row["head"]),
        )
        if key in indexed:
            raise RuntimeError(f"duplicate v144 context observation: {key}")
        indexed[key] = row

    augmented = list(observations)
    for row in observations:
        variant = str(row["variant"])
        if variant not in SEED_CORRECTED_VARIANTS:
            continue
        seed_key = (
            int(row["family_index"]),
            "seed_control",
            str(row["mode"]),
            int(row["current_frame"]),
            int(row["nominal_timestep"]),
            int(row["layer"]),
            int(row["head"]),
        )
        seed_row = indexed.get(seed_key)
        if seed_row is None:
            raise RuntimeError(
                "missing family/state-matched seed control for "
                f"{seed_key}"
            )
        corrected = dict(row)
        corrected["variant"] = f"{variant}_excess_seed"
        for measure in MEASURES:
            corrected[measure] = (
                float(row[measure]) - float(seed_row[measure])
            )
        augmented.append(corrected)
    return augmented


def _context_head_axes(
    observations: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = {}
    contexts = set()
    for row in _context_observations_with_seed_excess(observations):
        context = (
            str(row["variant"]),
            str(row["mode"]),
            int(row["current_frame"]),
            int(row["nominal_timestep"]),
        )
        contexts.add(context)
        key = (
            *context,
            int(row["layer"]),
            int(row["head"]),
            int(row["family_index"]) % 2,
        )
        grouped.setdefault(key, []).append(row)
    context_rows = []
    for variant, mode, frame, timestep in sorted(contexts):
        for layer in range(LAYERS):
            for head in range(HEADS):
                row = {
                    "variant": variant,
                    "mode": mode,
                    "current_frame": frame,
                    "nominal_timestep": timestep,
                    "layer": layer,
                    "head": head,
                }
                for split_name, parity in (
                    ("discovery", 0),
                    ("validation", 1),
                ):
                    values = grouped[
                        (
                            variant,
                            mode,
                            frame,
                            timestep,
                            layer,
                            head,
                            parity,
                        )
                    ]
                    for measure in MEASURES:
                        row[f"{split_name}_{measure}"] = float(
                            np.median(
                                [float(value[measure]) for value in values]
                            )
                        )
                context_rows.append(row)

    audits = []
    vectors = {}
    for variant, mode, frame, timestep in sorted(contexts):
        selected = [
            row
            for row in context_rows
            if (
                row["variant"],
                row["mode"],
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            == (variant, mode, frame, timestep)
        ]
        if len(selected) != TOTAL_HEADS:
            raise RuntimeError(
                "v144 context/head table is incomplete: "
                f"{variant}/{mode}/{frame}/{timestep}"
            )
        for measure in MEASURES:
            discovery = np.asarray(
                [float(row[f"discovery_{measure}"]) for row in selected]
            )
            validation = np.asarray(
                [float(row[f"validation_{measure}"]) for row in selected]
            )
            residual_discovery = _layer_residual(discovery)
            residual_validation = _layer_residual(validation)
            vectors[
                (variant, measure, mode, frame, timestep)
            ] = residual_discovery
            audits.append(
                {
                    "variant": variant,
                    "measure": measure,
                    "mode": mode,
                    "current_frame": frame,
                    "nominal_timestep": timestep,
                    "raw_split_spearman": _spearman(
                        discovery, validation
                    ),
                    "layer_residual_split_spearman": _spearman(
                        residual_discovery, residual_validation
                    ),
                    "discovery_layer_eta_squared": _layer_eta_squared(
                        discovery
                    ),
                    "validation_layer_eta_squared": _layer_eta_squared(
                        validation
                    ),
                }
            )

    stability = []
    for variant in sorted(
        {current_variant for current_variant, _, _, _ in contexts}
    ):
        variant_contexts = sorted(
            {
                (mode, frame, timestep)
                for current_variant, mode, frame, timestep in contexts
                if current_variant == variant
            }
        )
        for measure in MEASURES:
            correlations = []
            for left, right in itertools.combinations(variant_contexts, 2):
                correlations.append(
                    _spearman(
                        vectors[(variant, measure, *left)],
                        vectors[(variant, measure, *right)],
                    )
                )
            relevant_audits = [
                row
                for row in audits
                if row["variant"] == variant and row["measure"] == measure
            ]
            stability.append(
                {
                    "variant": variant,
                    "measure": measure,
                    "context_count": len(variant_contexts),
                    "median_layer_residual_split_spearman": float(
                        np.median(
                            [
                                row["layer_residual_split_spearman"]
                                for row in relevant_audits
                            ]
                        )
                    ),
                    "minimum_layer_residual_split_spearman": float(
                        min(
                            row["layer_residual_split_spearman"]
                            for row in relevant_audits
                        )
                    ),
                    "median_layer_residual_cross_context_spearman": float(
                        np.median(correlations)
                    ),
                    "minimum_layer_residual_cross_context_spearman": float(
                        min(correlations)
                    ),
                    "state_interpretation": (
                        "context_stable_candidate"
                        if (
                            np.median(correlations) >= 0.30
                            and np.median(
                                [
                                    row["layer_residual_split_spearman"]
                                    for row in relevant_audits
                                ]
                            )
                            >= 0.50
                        )
                        else "state_conditioned_or_unstable"
                    ),
                }
            )
    return context_rows, audits, stability


def _head_axes(
    family_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    value_fields = [
        field
        for field in family_rows[0]
        if field not in {"family_index", "layer", "head"}
    ]
    by_coordinate = {
        (int(row["family_index"]), int(row["layer"]), int(row["head"])): row
        for row in family_rows
    }
    head_rows = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            row = {"layer": layer, "head": head}
            for field in value_fields:
                discovery_values = [
                    float(by_coordinate[(family, layer, head)][field])
                    for family in range(FAMILIES)
                    if family % 2 == 0
                ]
                validation_values = [
                    float(by_coordinate[(family, layer, head)][field])
                    for family in range(FAMILIES)
                    if family % 2 == 1
                ]
                row[f"discovery_{field}"] = float(
                    np.median(discovery_values)
                )
                row[f"validation_{field}"] = float(
                    np.median(validation_values)
                )
            head_rows.append(row)

    audits = []
    for field in value_fields:
        discovery = np.asarray(
            [float(row[f"discovery_{field}"]) for row in head_rows]
        )
        validation = np.asarray(
            [float(row[f"validation_{field}"]) for row in head_rows]
        )
        residual_discovery = _layer_residual(discovery)
        residual_validation = _layer_residual(validation)
        audits.append(
            {
                "feature": field,
                "raw_split_spearman": _spearman(discovery, validation),
                "layer_residual_split_spearman": _spearman(
                    residual_discovery, residual_validation
                ),
                "discovery_layer_eta_squared": _layer_eta_squared(discovery),
                "validation_layer_eta_squared": _layer_eta_squared(validation),
                "raw_discovery_iqr": float(
                    np.quantile(discovery, 0.75)
                    - np.quantile(discovery, 0.25)
                ),
                "layer_residual_discovery_iqr": float(
                    np.quantile(residual_discovery, 0.75)
                    - np.quantile(residual_discovery, 0.25)
                ),
            }
        )

    factor_scores = {}
    factor_scaling = {}
    for factor in SEMANTIC_FACTORS:
        field = f"{factor}.compatibility_loss_excess_seed"
        discovery = _layer_residual(
            np.asarray(
                [float(row[f"discovery_{field}"]) for row in head_rows]
            )
        )
        validation = _layer_residual(
            np.asarray(
                [float(row[f"validation_{field}"]) for row in head_rows]
            )
        )
        center = float(np.median(discovery))
        scale = float(
            np.quantile(discovery, 0.75)
            - np.quantile(discovery, 0.25)
        )
        if scale <= 1e-8:
            scale = float(discovery.std())
        if scale <= 1e-8:
            scale = 1.0
        factor_scores[factor] = (
            (discovery - center) / scale,
            (validation - center) / scale,
        )
        factor_scaling[factor] = {
            "discovery_layer_residual_median": center,
            "discovery_layer_residual_scale": scale,
        }

    def dominant_label(scores: dict[str, float]) -> tuple[str, float, float]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_factor, top_score = ordered[0]
        margin = top_score - ordered[1][1]
        if top_score < DOMINANT_MIN_Z or margin < DOMINANT_MIN_MARGIN:
            return "unresolved", top_score, margin
        return top_factor, top_score, margin

    discovery_labels, validation_labels = [], []
    for flat_head, row in enumerate(head_rows):
        discovery_values = {
            factor: float(factor_scores[factor][0][flat_head])
            for factor in SEMANTIC_FACTORS
        }
        validation_values = {
            factor: float(factor_scores[factor][1][flat_head])
            for factor in SEMANTIC_FACTORS
        }
        discovery_label, discovery_score, discovery_margin = dominant_label(
            discovery_values
        )
        validation_label, validation_score, validation_margin = dominant_label(
            validation_values
        )
        discovery_labels.append(discovery_label)
        validation_labels.append(validation_label)
        for factor in SEMANTIC_FACTORS:
            row[f"discovery_{factor}_standardized_selectivity"] = (
                discovery_values[factor]
            )
            row[f"validation_{factor}_standardized_selectivity"] = (
                validation_values[factor]
            )
        row["discovery_dominant_semantic_score"] = discovery_score
        row["validation_dominant_semantic_score"] = validation_score
        row["discovery_dominant_semantic_margin"] = discovery_margin
        row["validation_dominant_semantic_margin"] = validation_margin
    for row, discovery, validation in zip(
        head_rows, discovery_labels, validation_labels
    ):
        row["discovery_dominant_semantic_factor"] = discovery
        row["validation_dominant_semantic_factor"] = validation
        row["dominant_semantic_factor_split_agree"] = int(
            discovery == validation
        )
    counts = Counter(discovery_labels)
    label_space = (*SEMANTIC_FACTORS, "unresolved")
    dominant_report = {
        "split_agreement": float(
            np.mean(np.asarray(discovery_labels) == np.asarray(validation_labels))
        ),
        "discovery_counts": {
            label: int(counts.get(label, 0)) for label in label_space
        },
        "minimum_resolved_class_fraction": min(
            counts.get(label, 0) for label in SEMANTIC_FACTORS
        )
        / TOTAL_HEADS,
        "unresolved_fraction": counts.get("unresolved", 0) / TOTAL_HEADS,
        "minimum_standardized_score": DOMINANT_MIN_Z,
        "minimum_standardized_margin": DOMINANT_MIN_MARGIN,
        "factor_scaling": factor_scaling,
        "functional_claim_admissible": False,
        "reason": (
            "Factors are layer-residualized and robust-standardized using "
            "discovery families, but descriptor dominance remains "
            "observational; factor-selective head interventions are required"
        ),
    }
    return head_rows, audits, dominant_report


def analyze(profile_dir: Path, output_dir: Path, expected_count: int) -> dict:
    indexed, profile_audit = _profile_index(
        _load_profiles(profile_dir, expected_count)
    )
    observations = _state_observations(indexed)
    family_axes = _family_head_axes(observations)
    context_axes, context_audit, context_stability = _context_head_axes(
        observations
    )
    head_axes, feature_audit, dominant = _head_axes(family_axes)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "factor_state_head_observations.csv.gz", observations
    )
    _write_csv(output_dir / "family_head_axes.csv", family_axes)
    _write_csv(output_dir / "context_head_axes.csv", context_axes)
    _write_csv(output_dir / "context_feature_audit.csv", context_audit)
    _write_csv(
        output_dir / "context_feature_stability.csv", context_stability
    )
    _write_csv(output_dir / "head_factor_axes.csv", head_axes)
    _write_csv(output_dir / "feature_stability_audit.csv", feature_audit)
    _write_csv(output_dir / "profile_contract_audit.csv", profile_audit)
    stable_residual = [
        row["feature"]
        for row in feature_audit
        if float(row["layer_residual_split_spearman"]) >= 0.3
        and float(row["layer_residual_discovery_iqr"]) > 1e-8
    ]
    report = {
        "version": 1,
        "profile_count": len(indexed),
        "family_count": FAMILIES,
        "variants": list(VARIANTS),
        "state_observation_count": len(observations),
        "family_head_count": len(family_axes),
        "context_head_count": len(context_axes),
        "head_count": len(head_axes),
        "split_stable_layer_residual_features": stable_residual,
        "split_stable_layer_residual_feature_count": len(stable_residual),
        "dominant_semantic_factor": dominant,
        "context_stable_feature_count": sum(
            row["state_interpretation"] == "context_stable_candidate"
            for row in context_stability
        ),
        "interpretation": {
            "seed_control": (
                "same text with a different random trajectory; estimates "
                "trajectory noise rather than prompt semantics"
            ),
            "semantic_excess_seed": (
                "semantic factor response minus the different-seed response; "
                "a diagnostic difference-of-differences, not a causal effect"
            ),
            "spatial_topology": (
                "recent cross-frame attention correspondence topology; no "
                "optical-flow labels are used, so it is not a motion score"
            ),
        },
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# v144 Factorized Head-mechanism Analysis",
        "",
        f"- Profiles: `{report['profile_count']}`",
        f"- State/head observations: `{len(observations)}`",
        (
            "- Split-stable layer-residual features: "
            f"`{len(stable_residual)}`"
        ),
        (
            "- Dominant semantic-factor split agreement: "
            f"`{dominant['split_agreement']:.4f}`"
        ),
        "",
        "The primary comparison is semantic-factor response relative to the "
        "same-prompt different-seed control. Q, K, V, policy, and recent "
        "spatial-topology changes are kept separate. No functional head name "
        "is assigned until a head-selective generation intervention passes.",
    ]
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.profile_dir, args.output_dir, args.expected_count),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
