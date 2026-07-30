#!/usr/bin/env python3
"""Analyze crossed-seed prompt-factor head profiles without static role names."""

from __future__ import annotations

import argparse
import csv
import gzip
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
REPLICATES = (0, 1)
VARIANTS = (
    "base",
    "paraphrase",
    "identity",
    "scene",
    "full_semantic",
)
FACTOR_VARIANTS = VARIANTS[1:]
DESCRIPTORS = {
    "q": "query_projection",
    "k": "history_key_projection",
    "v": "history_value_projection",
}
EFFECT_AXES = (
    "q_shift",
    "k_shift",
    "v_shift",
    "value_scale_shift",
    "policy_shift",
)
EPSILON = 1e-8
MIN_FAMILY_SPLIT_RHO = 0.30
MIN_SEED_REPLICATE_RHO = 0.30
MIN_DIRECTION_COSINE = 0.05
MIN_DIRECTION_MARGIN = 0.02


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    left_rank = _rankdata(left[finite])
    right_rank = _rankdata(right[finite])
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _layer_residual(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (TOTAL_HEADS,) or not np.isfinite(array).all():
        raise ValueError("head vector must contain 360 finite values")
    matrix = array.reshape(LAYERS, HEADS)
    return (matrix - np.median(matrix, axis=1, keepdims=True)).reshape(-1)


def _aligned_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[0] != HEADS:
        raise ValueError(
            f"descriptor shapes do not align: {left.shape} vs {right.shape}"
        )
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1).flatten(1).mean(dim=-1)


def _mean_delta(
    base: torch.Tensor, variant: torch.Tensor
) -> torch.Tensor:
    if base.shape != variant.shape or base.shape[0] != HEADS:
        raise ValueError("base/variant descriptor shapes do not align")
    delta = variant.float() - base.float()
    return delta.reshape(HEADS, -1, delta.shape[-1]).mean(dim=1)


def _vector_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[0] != HEADS:
        raise ValueError("head-vector shapes do not align")
    left_norm = left.square().sum(dim=-1).sqrt()
    right_norm = right.square().sum(dim=-1).sqrt()
    valid = (left_norm > EPSILON) & (right_norm > EPSILON)
    cosine = (left * right).sum(dim=-1) / (
        left_norm * right_norm
    ).clamp_min(EPSILON)
    return torch.where(
        valid,
        cosine.clamp(-1, 1),
        torch.full_like(cosine, float("nan")),
    )


def _delta_direction_cosine(
    base_left: torch.Tensor,
    variant_left: torch.Tensor,
    base_right: torch.Tensor,
    variant_right: torch.Tensor,
) -> torch.Tensor:
    return _vector_cosine(
        _mean_delta(base_left, variant_left),
        _mean_delta(base_right, variant_right),
    )


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
        raise ValueError("causal-policy vector has the wrong head count")
    return values / values.sum(dim=-1, keepdim=True).clamp_min(EPSILON)


def _policy_shift(base: dict, variant: dict) -> torch.Tensor:
    left, right = _policy_vector(base), _policy_vector(variant)
    if left.shape != right.shape:
        raise ValueError("base/variant policy candidates differ")
    if left.shape[-1] == 0:
        return torch.full((HEADS,), float("nan"))
    return (left - right).abs().sum(dim=-1)


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
            f"expected {expected_count} v145 profiles, found {len(paths)}"
        )
    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        if int(payload.get("version", -1)) != 7:
            raise RuntimeError(f"{path} is not a mechanism profile")
        if str(job.get("kind")) != "crossed_seed_head_mechanism":
            raise RuntimeError(f"{path} has the wrong job kind")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete calls")
        if not metadata.get("descriptor_export"):
            raise RuntimeError(f"{path} lacks Q/K/V descriptors")
        if not metadata.get("causal_policy_metrics"):
            raise RuntimeError(f"{path} lacks causal-policy metrics")
        if metadata.get("spatial_topology_metrics"):
            raise RuntimeError(f"{path} unexpectedly enabled topology metrics")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def _profile_index(profiles: list[dict]) -> tuple[dict, list[dict]]:
    indexed = {}
    audits = []
    for profile in profiles:
        job = profile["job"]
        key = (
            int(job["family_index"]),
            int(job["seed_replicate"]),
            str(job["variant"]),
        )
        if key in indexed:
            raise RuntimeError(f"duplicate v145 profile: {key}")
        if (
            not 0 <= key[0] < FAMILIES
            or key[1] not in REPLICATES
            or key[2] not in VARIANTS
        ):
            raise RuntimeError(f"invalid v145 profile coordinate: {key}")
        declared_seed = int(job["seed"])
        reference_seed = int(job["reference_seed"])
        actual_seed = int(metadata["seed"])
        if declared_seed != reference_seed or actual_seed != declared_seed:
            raise RuntimeError(
                f"{key} violates paired-seed contract: "
                f"actual={actual_seed} declared={declared_seed} "
                f"reference={reference_seed}"
            )
        records = {}
        for record in profile["records"]:
            if str(record["branch"]) != "base":
                raise RuntimeError(f"{key} contains a non-base branch")
            state = _state_key(record)
            if state in records:
                raise RuntimeError(f"{key} duplicates state {state}")
            for field in (
                *DESCRIPTORS.values(),
                "history_value_rms",
                "causal_policy_metrics",
            ):
                if field not in record:
                    raise RuntimeError(f"{key}/{state} lacks {field}")
            records[state] = record
        states = Counter(state[:3] for state in records)
        if len(states) != 6 or set(states.values()) != {LAYERS}:
            raise RuntimeError(f"{key} has an invalid state/layer grid")
        indexed[key] = {"profile": profile, "records": records}
        audits.append(
            {
                "dataset_index": int(job["dataset_index"]),
                "family_index": key[0],
                "seed_replicate": key[1],
                "variant": key[2],
                "seed": actual_seed,
                "declared_seed": declared_seed,
                "reference_seed": reference_seed,
                "state_count": len(states),
                "record_count": len(records),
                "path": profile["_path"],
            }
        )
    expected = {
        (family, replicate, variant)
        for family in range(FAMILIES)
        for replicate in REPLICATES
        for variant in VARIANTS
    }
    if set(indexed) != expected:
        raise RuntimeError(
            "v145 profile grid is incomplete: "
            f"missing={sorted(expected - set(indexed))}"
        )
    for family in range(FAMILIES):
        seeds = {}
        for replicate in REPLICATES:
            values = {
                int(
                    indexed[(family, replicate, variant)]["profile"][
                        "metadata"
                    ]["seed"]
                )
                for variant in VARIANTS
            }
            if len(values) != 1:
                raise RuntimeError(
                    f"family={family} replicate={replicate} "
                    f"uses unpaired seeds: {sorted(values)}"
                )
            seeds[replicate] = next(iter(values))
        if seeds[0] == seeds[1]:
            raise RuntimeError(
                f"family={family} repeats the same seed across replicates"
            )
    return indexed, audits


def _effect_shifts(base: dict, variant: dict) -> dict[str, torch.Tensor]:
    base_rms = base["history_value_rms"].float()
    variant_rms = variant["history_value_rms"].float()
    return {
        "q_shift": 1.0
        - _aligned_cosine(
            base["query_projection"], variant["query_projection"]
        ),
        "k_shift": 1.0
        - _aligned_cosine(
            base["history_key_projection"],
            variant["history_key_projection"],
        ),
        "v_shift": 1.0
        - _aligned_cosine(
            base["history_value_projection"],
            variant["history_value_projection"],
        ),
        "value_scale_shift": (
            (variant_rms + EPSILON).log()
            - (base_rms + EPSILON).log()
        )
        .abs()
        .flatten(1)
        .mean(dim=-1),
        "policy_shift": _policy_shift(base, variant),
    }


def _state_observations(indexed: dict) -> list[dict]:
    rows = []
    for family in range(FAMILIES):
        reference_states = set(
            indexed[(family, 0, "base")]["records"]
        )
        for replicate in REPLICATES:
            for variant in VARIANTS:
                if set(indexed[(family, replicate, variant)]["records"]) != (
                    reference_states
                ):
                    raise RuntimeError(
                        f"state grid differs for {family}/{replicate}/{variant}"
                    )
        for variant in FACTOR_VARIANTS:
            other_variants = [
                value for value in FACTOR_VARIANTS if value != variant
            ]
            for state in sorted(reference_states):
                base = {
                    replicate: indexed[(family, replicate, "base")][
                        "records"
                    ][state]
                    for replicate in REPLICATES
                }
                current = {
                    replicate: indexed[(family, replicate, variant)][
                        "records"
                    ][state]
                    for replicate in REPLICATES
                }
                shifts = {
                    replicate: _effect_shifts(
                        base[replicate], current[replicate]
                    )
                    for replicate in REPLICATES
                }
                direction = {}
                cross_direction = {}
                for short_name, field in DESCRIPTORS.items():
                    direction[short_name] = _delta_direction_cosine(
                        base[0][field],
                        current[0][field],
                        base[1][field],
                        current[1][field],
                    )
                    negatives = []
                    for other in other_variants:
                        other_zero = indexed[(family, 0, other)][
                            "records"
                        ][state]
                        other_one = indexed[(family, 1, other)][
                            "records"
                        ][state]
                        negatives.extend(
                            (
                                _delta_direction_cosine(
                                    base[0][field],
                                    current[0][field],
                                    base[1][field],
                                    other_one[field],
                                ),
                                _delta_direction_cosine(
                                    base[1][field],
                                    current[1][field],
                                    base[0][field],
                                    other_zero[field],
                                ),
                            )
                        )
                    cross_direction[short_name] = torch.nanmedian(
                        torch.stack(negatives, dim=0), dim=0
                    ).values
                mode, frame, timestep, layer = state
                for head in range(HEADS):
                    row = {
                        "family_index": family,
                        "family_split": (
                            "discovery"
                            if family % 2 == 0
                            else "validation"
                        ),
                        "variant": variant,
                        "mode": mode,
                        "current_frame": frame,
                        "nominal_timestep": timestep,
                        "layer": layer,
                        "head": head,
                    }
                    for axis in EFFECT_AXES:
                        left = float(shifts[0][axis][head])
                        right = float(shifts[1][axis][head])
                        row[f"{axis}_seed0"] = left
                        row[f"{axis}_seed1"] = right
                        row[f"{axis}_mean"] = 0.5 * (left + right)
                        row[f"{axis}_seed_gap"] = abs(left - right)
                    for short_name in DESCRIPTORS:
                        same = float(direction[short_name][head])
                        cross = float(cross_direction[short_name][head])
                        row[f"{short_name}_delta_seed_cosine"] = same
                        row[
                            f"{short_name}_delta_cross_factor_cosine"
                        ] = cross
                        row[
                            f"{short_name}_delta_specificity_margin"
                        ] = same - cross
                    rows.append(row)
    return rows


def _aggregate_family_heads(observations: list[dict]) -> list[dict]:
    metadata = {
        "family_index",
        "family_split",
        "variant",
        "mode",
        "current_frame",
        "nominal_timestep",
        "layer",
        "head",
    }
    fields = [field for field in observations[0] if field not in metadata]
    grouped = {}
    for row in observations:
        key = (
            int(row["family_index"]),
            str(row["variant"]),
            int(row["layer"]),
            int(row["head"]),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for family in range(FAMILIES):
        for variant in FACTOR_VARIANTS:
            for layer in range(LAYERS):
                for head in range(HEADS):
                    values = grouped[(family, variant, layer, head)]
                    row = {
                        "family_index": family,
                        "family_split": (
                            "discovery"
                            if family % 2 == 0
                            else "validation"
                        ),
                        "variant": variant,
                        "layer": layer,
                        "head": head,
                    }
                    for field in fields:
                        row[field] = float(
                            np.nanmedian(
                                [float(value[field]) for value in values]
                            )
                        )
                    output.append(row)
    return output


def _aggregate_heads(family_rows: list[dict]) -> list[dict]:
    metadata = {
        "family_index",
        "family_split",
        "variant",
        "layer",
        "head",
    }
    fields = [field for field in family_rows[0] if field not in metadata]
    indexed = {}
    for row in family_rows:
        key = (
            str(row["variant"]),
            int(row["layer"]),
            int(row["head"]),
        )
        indexed.setdefault(key, []).append(row)
    output = []
    for variant in FACTOR_VARIANTS:
        for layer in range(LAYERS):
            for head in range(HEADS):
                values = indexed[(variant, layer, head)]
                row = {"variant": variant, "layer": layer, "head": head}
                for split in ("discovery", "validation", "all"):
                    selected = (
                        values
                        if split == "all"
                        else [
                            value for value in values
                            if value["family_split"] == split
                        ]
                    )
                    for field in fields:
                        row[f"{split}_{field}"] = float(
                            np.nanmedian(
                                [float(value[field]) for value in selected]
                            )
                        )
                output.append(row)
    return output


def _feature_audit(head_rows: list[dict]) -> list[dict]:
    output = []
    for variant in FACTOR_VARIANTS:
        selected = [
            row for row in head_rows if row["variant"] == variant
        ]
        if len(selected) != TOTAL_HEADS:
            raise RuntimeError(f"incomplete v145 head table for {variant}")
        for axis in EFFECT_AXES:
            discovery = np.asarray(
                [float(row[f"discovery_{axis}_mean"]) for row in selected]
            )
            validation = np.asarray(
                [float(row[f"validation_{axis}_mean"]) for row in selected]
            )
            seed0 = np.asarray(
                [float(row[f"all_{axis}_seed0"]) for row in selected]
            )
            seed1 = np.asarray(
                [float(row[f"all_{axis}_seed1"]) for row in selected]
            )
            family_rho = _spearman(
                _layer_residual(discovery),
                _layer_residual(validation),
            )
            seed_rho = _spearman(
                _layer_residual(seed0), _layer_residual(seed1)
            )
            short_name = axis.removesuffix("_shift")
            if short_name in DESCRIPTORS:
                direction = np.asarray(
                    [
                        float(
                            row[
                                f"all_{short_name}_delta_seed_cosine"
                            ]
                        )
                        for row in selected
                    ]
                )
                specificity = np.asarray(
                    [
                        float(
                            row[
                                f"all_{short_name}_delta_specificity_margin"
                            ]
                        )
                        for row in selected
                    ]
                )
                median_direction = float(np.nanmedian(direction))
                median_specificity = float(np.nanmedian(specificity))
                direction_gate = (
                    median_direction >= MIN_DIRECTION_COSINE
                    and median_specificity >= MIN_DIRECTION_MARGIN
                )
            else:
                median_direction = float("nan")
                median_specificity = float("nan")
                direction_gate = True
            output.append(
                {
                    "variant": variant,
                    "axis": axis,
                    "raw_family_split_spearman": _spearman(
                        discovery, validation
                    ),
                    "layer_residual_family_split_spearman": family_rho,
                    "raw_seed_replicate_spearman": _spearman(seed0, seed1),
                    "layer_residual_seed_replicate_spearman": seed_rho,
                    "median_seed_delta_direction_cosine": median_direction,
                    "median_cross_factor_specificity_margin": (
                        median_specificity
                    ),
                    "discovery_layer_residual_iqr": float(
                        np.quantile(_layer_residual(discovery), 0.75)
                        - np.quantile(
                            _layer_residual(discovery), 0.25
                        )
                    ),
                    "reproducible_factor_axis_candidate": int(
                        family_rho >= MIN_FAMILY_SPLIT_RHO
                        and seed_rho >= MIN_SEED_REPLICATE_RHO
                        and direction_gate
                    ),
                }
            )
    return output


def _context_audit(observations: list[dict]) -> list[dict]:
    contexts = sorted(
        {
            (
                str(row["mode"]),
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in observations
        }
    )
    output = []
    for variant in FACTOR_VARIANTS:
        for mode, frame, timestep in contexts:
            selected = [
                row
                for row in observations
                if row["variant"] == variant
                and row["mode"] == mode
                and int(row["current_frame"]) == frame
                and int(row["nominal_timestep"]) == timestep
            ]
            by_head = {}
            for row in selected:
                key = (int(row["layer"]), int(row["head"]))
                by_head.setdefault(key, []).append(row)
            if (
                len(by_head) != TOTAL_HEADS
                or any(len(values) != FAMILIES for values in by_head.values())
            ):
                raise RuntimeError(
                    "incomplete v145 context: "
                    f"{variant}/{mode}/{frame}/{timestep}"
                )
            for short_name in DESCRIPTORS:
                axis = f"{short_name}_shift"
                discovery, validation, seed0, seed1 = [], [], [], []
                direction, specificity = [], []
                for layer in range(LAYERS):
                    for head in range(HEADS):
                        values = by_head[(layer, head)]
                        discovery.append(
                            np.median(
                                [
                                    float(value[f"{axis}_mean"])
                                    for value in values
                                    if value["family_split"] == "discovery"
                                ]
                            )
                        )
                        validation.append(
                            np.median(
                                [
                                    float(value[f"{axis}_mean"])
                                    for value in values
                                    if value["family_split"] == "validation"
                                ]
                            )
                        )
                        seed0.append(
                            np.median(
                                [
                                    float(value[f"{axis}_seed0"])
                                    for value in values
                                ]
                            )
                        )
                        seed1.append(
                            np.median(
                                [
                                    float(value[f"{axis}_seed1"])
                                    for value in values
                                ]
                            )
                        )
                        direction.append(
                            np.nanmedian(
                                [
                                    float(
                                        value[
                                            f"{short_name}_delta_seed_cosine"
                                        ]
                                    )
                                    for value in values
                                ]
                            )
                        )
                        specificity.append(
                            np.nanmedian(
                                [
                                    float(
                                        value[
                                            f"{short_name}_delta_specificity_margin"
                                        ]
                                    )
                                    for value in values
                                ]
                            )
                        )
                discovery = np.asarray(discovery)
                validation = np.asarray(validation)
                seed0 = np.asarray(seed0)
                seed1 = np.asarray(seed1)
                family_rho = _spearman(
                    _layer_residual(discovery),
                    _layer_residual(validation),
                )
                seed_rho = _spearman(
                    _layer_residual(seed0), _layer_residual(seed1)
                )
                median_direction = float(np.nanmedian(direction))
                median_specificity = float(np.nanmedian(specificity))
                output.append(
                    {
                        "variant": variant,
                        "axis": axis,
                        "mode": mode,
                        "current_frame": frame,
                        "nominal_timestep": timestep,
                        "layer_residual_family_split_spearman": family_rho,
                        "layer_residual_seed_replicate_spearman": seed_rho,
                        "median_seed_delta_direction_cosine": (
                            median_direction
                        ),
                        "median_cross_factor_specificity_margin": (
                            median_specificity
                        ),
                        "reproducible_context_candidate": int(
                            family_rho >= MIN_FAMILY_SPLIT_RHO
                            and seed_rho >= MIN_SEED_REPLICATE_RHO
                            and median_direction >= MIN_DIRECTION_COSINE
                            and median_specificity >= MIN_DIRECTION_MARGIN
                        ),
                    }
                )
    return output


def analyze(
    profile_dir: Path, output_dir: Path, expected_count: int
) -> dict:
    indexed, profile_audit = _profile_index(
        _load_profiles(profile_dir, expected_count)
    )
    observations = _state_observations(indexed)
    family_rows = _aggregate_family_heads(observations)
    head_rows = _aggregate_heads(family_rows)
    feature_rows = _feature_audit(head_rows)
    context_rows = _context_audit(observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "state_head_observations.csv.gz", observations
    )
    _write_csv(output_dir / "family_head_axes.csv", family_rows)
    _write_csv(output_dir / "head_factor_reproducibility.csv", head_rows)
    _write_csv(output_dir / "feature_reproducibility_audit.csv", feature_rows)
    _write_csv(output_dir / "context_reproducibility_audit.csv", context_rows)
    _write_csv(output_dir / "profile_contract_audit.csv", profile_audit)
    accepted = [
        f"{row['variant']}.{row['axis']}"
        for row in feature_rows
        if int(row["reproducible_factor_axis_candidate"])
    ]
    report = {
        "version": 1,
        "profile_count": len(indexed),
        "family_count": FAMILIES,
        "seed_replicates": list(REPLICATES),
        "variants": list(VARIANTS),
        "state_head_observation_count": len(observations),
        "reproducible_factor_axis_candidates": accepted,
        "reproducible_factor_axis_candidate_count": len(accepted),
        "reproducible_context_candidate_count": sum(
            int(row["reproducible_context_candidate"])
            for row in context_rows
        ),
        "thresholds": {
            "minimum_family_split_spearman": MIN_FAMILY_SPLIT_RHO,
            "minimum_seed_replicate_spearman": MIN_SEED_REPLICATE_RHO,
            "minimum_delta_direction_cosine": MIN_DIRECTION_COSINE,
            "minimum_cross_factor_specificity_margin": MIN_DIRECTION_MARGIN,
        },
        "claim_boundary": (
            "Passing identifies a reproducible observational factor axis, "
            "not a functional head class. Head-selective interventions and "
            "count-matched random controls remain required."
        ),
        "static_factor_taxonomy_admissible": False,
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# v145 Crossed-seed Head Profiling",
        "",
        f"- Profiles: `{report['profile_count']}`",
        (
            "- Reproducible factor-axis candidates: "
            f"`{len(accepted)}`"
        ),
        (
            "- Reproducible state-specific candidates: "
            f"`{report['reproducible_context_candidate_count']}`"
        ),
        "",
        "A candidate must reproduce across held-out prompt families and "
        "independent seeds. Q/K/V delta direction must also be more similar "
        "for the same factor than for other factors. These are observational "
        "screening gates, not functional role labels.",
    ]
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=160)
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
