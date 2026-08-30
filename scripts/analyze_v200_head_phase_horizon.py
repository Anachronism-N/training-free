#!/usr/bin/env python3
"""Audit AR-horizon structure in v189 Head x Denoising-Phase profiles.

This analysis is deliberately generation-free.  It asks whether conditioning
Coverage membership on the autoregressive horizon predicts held-out shadow
readout utility beyond a static Head x Denoising-Phase selector with the same
number of Coverage exposures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from prepare_v189_structured_head_phase_profile import (
    CALLS,
    HEADS,
    LAYERS,
    OPERATORS,
    PROMPT_COUNT,
    sha256,
    verify,
)

EXPERIMENT = "v200_head_phase_ar_horizon_audit"
SOURCE_EXPERIMENT = "v189_structured_head_phase_profile"
PROFILE_VERSION = 4
PROFILE_METHOD = "structured_head_phase_cache_compatibility"
POLICIES = ("recent", "coverage")
EXPECTED_CURRENT_FRAMES = tuple(range(12, 120, 9))
EXPECTED_POSITIONS = len(EXPECTED_CURRENT_FRAMES)
SPARSITY_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PRIMARY_FRACTION = 0.10
DEFAULT_BOOTSTRAP_SAMPLES = 5_000
DEFAULT_PERMUTATIONS = 5_000


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.pt"))
    if not paths:
        paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise ValueError(f"no v189 profile shards below {root}")
    return paths


def _validate_record_contract(record: dict, *, path: Path, index: int) -> None:
    policies = record.get("policies") or {}
    budgets = record.get("budgets") or {}
    if (
        record.get("profile_contract") != "v189"
        or str(record.get("cache_update_mode")) != "noisy"
        or str(record.get("cfg_branch")) != "cond"
        or int(record.get("heads", -1)) != HEADS
        or set(policies) != set(POLICIES)
        or set(budgets) != {"recent", "coverage", "union"}
    ):
        raise ValueError(f"{path}:{index}: v189 record contract drift")
    union = budgets["union"]
    if (
        union.get("superset_verification_contract") != "v189"
        or union.get("candidate_representation_subset_verified") is not True
        or int(union.get("candidate_representation_subset_checks", -1))
        != len(POLICIES) * HEADS
        or int(union.get("candidate_representation_subset_failures", -1)) != 0
        or int(union.get("max_frame_equivalents", -1)) > 13
        or int(budgets["recent"].get("max_frame_equivalents", -1)) > 9
        or int(budgets["coverage"].get("max_frame_equivalents", -1)) > 9
    ):
        raise ValueError(f"{path}:{index}: representation-complete Union drift")


def load_operator_tensor(
    root: Path, operator: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Load one operator as [prompt, call, layer, head, AR-position]."""

    import torch

    shape = (PROMPT_COUNT, CALLS, LAYERS, HEADS, EXPECTED_POSITIONS)
    gain = np.full(shape, np.nan, dtype=np.float64)
    energy = np.full(shape, np.nan, dtype=np.float64)
    full_budget = np.full(shape, np.nan, dtype=np.float64)
    seen = np.zeros((PROMPT_COUNT, CALLS, LAYERS, EXPECTED_POSITIONS), dtype=np.uint8)
    frame_to_position = {
        frame: position for position, frame in enumerate(EXPECTED_CURRENT_FRAMES)
    }
    source_kinds: Counter[str] = Counter()
    shards = []

    for path in _profile_paths(root):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        if (
            payload.get("version") != PROFILE_VERSION
            or payload.get("contract") != "v189"
            or payload.get("method") != PROFILE_METHOD
            or tuple(payload.get("policies") or ()) != POLICIES
            or metadata.get("profile_contract") != "v189"
            or metadata.get("coverage_operator") != operator
        ):
            raise ValueError(f"{path}: v189 profile artifact drift")
        records = payload.get("records") or []
        if not records:
            raise ValueError(f"{path}: empty v189 profile shard")
        shard_prompts = set()
        for record_index, record in enumerate(records):
            _validate_record_contract(record, path=path, index=record_index)
            prompt = int(record["prompt_id"])
            call = int(record["call_index"])
            layer = int(record["layer"])
            current_frame = int(record["current_frame"])
            if (
                not 0 <= prompt < PROMPT_COUNT
                or not 0 <= call < CALLS
                or not 0 <= layer < LAYERS
                or current_frame not in frame_to_position
            ):
                raise ValueError(f"{path}:{record_index}: invalid profile location")
            position = frame_to_position[current_frame]
            if seen[prompt, call, layer, position]:
                raise ValueError(
                    "duplicate v189 horizon location: "
                    f"{prompt}/{call}/{layer}/{current_frame}"
                )

            policy_rows = record["policies"]
            recent_error = np.asarray(
                policy_rows["recent"]["residual_relative_mse"], dtype=np.float64
            )
            coverage_error = np.asarray(
                policy_rows["coverage"]["residual_relative_mse"], dtype=np.float64
            )
            reference_energy = np.asarray(
                record["reference_residual_energy"], dtype=np.float64
            )
            coverage_budget = np.asarray(
                record["budgets"]["coverage"]["per_sequence_frame_equivalents"],
                dtype=np.float64,
            )
            if (
                recent_error.shape != (HEADS,)
                or coverage_error.shape != (HEADS,)
                or reference_energy.shape != (HEADS,)
                or coverage_budget.shape != (HEADS,)
                or not np.isfinite(recent_error).all()
                or not np.isfinite(coverage_error).all()
                or not np.isfinite(reference_energy).all()
                or np.any(reference_energy < 0.0)
                or np.any(coverage_budget > 9.0)
            ):
                raise ValueError(f"{path}:{record_index}: invalid profile vectors")
            gain[prompt, call, layer, :, position] = np.log(
                np.maximum(recent_error, 1e-12)
            ) - np.log(np.maximum(coverage_error, 1e-12))
            energy[prompt, call, layer, :, position] = reference_energy
            full_budget[prompt, call, layer, :, position] = coverage_budget >= 9.0
            seen[prompt, call, layer, position] = 1
            shard_prompts.add(prompt)
            source_kinds.update(
                str(value)
                for value in (
                    record["budgets"]["coverage"].get("selected_source_codebook") or []
                )
            )
        shards.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "record_count": len(records),
                "prompt_ids": sorted(shard_prompts),
            }
        )

    if not np.all(seen == 1):
        observed = Counter(int(value) for value in seen.sum(axis=-1).reshape(-1))
        raise ValueError(
            "v189 horizon profile is incomplete; expected "
            f"{EXPECTED_POSITIONS} positions, observed={dict(sorted(observed.items()))}"
        )
    if any(np.isnan(value).any() for value in (gain, energy, full_budget)):
        raise ValueError("v189 horizon tensor contains unfilled entries")
    report = {
        "operator": operator,
        "shape": list(shape),
        "current_frames": list(EXPECTED_CURRENT_FRAMES),
        "shard_count": len(shards),
        "record_count": int(seen.sum()),
        "source_codebook_counts": dict(sorted(source_kinds.items())),
        "shards": shards,
    }
    return {"gain": gain, "energy": energy, "full_budget": full_budget}, report


def _rankdata(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        end = start + 1
        while end < flat.size and flat[order[end]] == flat[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks.reshape(np.asarray(values).shape)


def _correlation(
    left: np.ndarray, right: np.ndarray, *, rank: bool = False
) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("correlation inputs must share a nontrivial shape")
    if rank:
        left = _rankdata(left)
        right = _rankdata(right)
    x = left.reshape(-1)
    y = right.reshape(-1)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prompt_slopes(gain: np.ndarray) -> np.ndarray:
    """Fit a centered linear horizon slope for every prompt and cell."""

    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 5 or gain.shape[-1] < 3:
        raise ValueError("horizon gain must have shape [P,C,L,H,T]")
    positions = np.linspace(-1.0, 1.0, gain.shape[-1], dtype=np.float64)
    denominator = float(np.square(positions).sum())
    return np.sum(gain * positions, axis=-1) / denominator


def bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or samples <= 0:
        raise ValueError("bootstrap requires non-empty values and positive samples")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(int(samples), values.size))].mean(
        axis=1
    )
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def one_sided_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    wins = int(np.count_nonzero(values > 0.0))
    losses = int(np.count_nonzero(values < 0.0))
    total = wins + losses
    if total == 0:
        return 1.0
    return float(sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total)


def _top_mask(scores: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < count < flat.size:
        raise ValueError("top-mask count must leave a non-empty complement")
    selected = np.zeros(flat.size, dtype=np.bool_)
    indices = np.argpartition(flat, flat.size - count)[-count:]
    selected[indices] = True
    return selected.reshape(np.asarray(scores).shape)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def selector_test(
    gain: np.ndarray,
    discovery: list[int],
    validation: list[int],
    *,
    fraction: float,
    seed: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> dict:
    """Compare horizon-conditioned and static selectors at equal exposure."""

    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 5 or not discovery or not validation:
        raise ValueError("selector test requires [P,C,L,H,T] and non-empty splits")
    if set(discovery) & set(validation):
        raise ValueError("discovery and validation prompts must be disjoint")
    cells = int(np.prod(gain.shape[1:-1]))
    positions = int(gain.shape[-1])
    count = max(1, min(cells - 1, round(cells * float(fraction))))
    discovery_mean = gain[discovery].mean(axis=0)
    validation_gain = gain[validation]

    static_scores = discovery_mean.mean(axis=-1)
    static_cell_mask = _top_mask(static_scores, count)
    static_mask = np.repeat(static_cell_mask[..., None], positions, axis=-1)
    horizon_mask = np.zeros_like(discovery_mean, dtype=np.bool_)
    for position in range(positions):
        horizon_mask[..., position] = _top_mask(discovery_mean[..., position], count)
    expected_exposures = count * positions
    if (
        int(static_mask.sum()) != expected_exposures
        or int(horizon_mask.sum()) != expected_exposures
    ):
        raise RuntimeError("static and horizon selector exposure budgets differ")

    static_values = validation_gain[:, static_mask].mean(axis=1)
    horizon_values = validation_gain[:, horizon_mask].mean(axis=1)
    deltas = horizon_values - static_values

    validation_mean = validation_gain.mean(axis=0)
    pair_utility = np.empty((positions, positions), dtype=np.float64)
    for source_position in range(positions):
        source_mask = horizon_mask[..., source_position]
        for target_position in range(positions):
            pair_utility[source_position, target_position] = float(
                validation_mean[..., target_position][source_mask].mean()
            )
    observed_horizon_utility = float(horizon_values.mean())
    rng = np.random.default_rng(seed + 91)
    null = np.empty(int(permutations), dtype=np.float64)
    target = np.arange(positions)
    for draw in range(int(permutations)):
        assignment = rng.permutation(positions)
        null[draw] = float(pair_utility[assignment, target].mean())
    permutation_p = float(
        (1 + np.count_nonzero(null >= observed_horizon_utility)) / (len(null) + 1)
    )
    adjacent_overlap = [
        _jaccard(horizon_mask[..., left], horizon_mask[..., left + 1])
        for left in range(positions - 1)
    ]
    return {
        "fraction": float(fraction),
        "cells_per_position": count,
        "position_count": positions,
        "coverage_exposures": expected_exposures,
        "equal_exposure_verified": True,
        "static_validation_utility": float(static_values.mean()),
        "horizon_validation_utility": observed_horizon_utility,
        "paired_delta_mean": float(deltas.mean()),
        "paired_delta_median": float(np.median(deltas)),
        "paired_delta_ci95": bootstrap_ci(deltas, seed=seed, samples=bootstrap_samples),
        "paired_win_fraction": float(np.mean(deltas > 0.0)),
        "paired_sign_p": one_sided_sign_p(deltas),
        "time_assignment_permutation_p": permutation_p,
        "time_assignment_null_mean": float(null.mean()),
        "time_assignment_null_std": float(null.std()),
        "static_horizon_membership_jaccard": _jaccard(static_mask, horizon_mask),
        "adjacent_horizon_membership_jaccard_mean": float(np.mean(adjacent_overlap)),
        "unique_horizon_cells": int(horizon_mask.any(axis=-1).sum()),
    }


def analyze_operator_tensor(
    tensors: dict[str, np.ndarray],
    *,
    discovery: list[int],
    validation: list[int],
    operator_index: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> dict:
    gain = np.asarray(tensors["gain"], dtype=np.float64)
    energy = np.asarray(tensors["energy"], dtype=np.float64)
    full_budget = np.asarray(tensors["full_budget"], dtype=np.float64)
    if gain.shape != energy.shape or gain.shape != full_budget.shape:
        raise ValueError("v200 gain/energy/budget tensors must share a shape")
    if not np.isfinite(gain).all() or not np.isfinite(energy).all():
        raise ValueError("v200 tensors contain non-finite values")

    discovery_mean = gain[discovery].mean(axis=0)
    validation_mean = gain[validation].mean(axis=0)
    discovery_interaction = discovery_mean - discovery_mean.mean(axis=-1, keepdims=True)
    validation_interaction = validation_mean - validation_mean.mean(
        axis=-1, keepdims=True
    )
    slopes = prompt_slopes(gain)
    discovery_slope = slopes[discovery].mean(axis=0)
    validation_slope = slopes[validation].mean(axis=0)
    global_prompt_slopes = slopes.mean(axis=(1, 2, 3))

    curves = {
        "discovery_mean_gain": gain[discovery].mean(axis=(0, 1, 2, 3)).tolist(),
        "validation_mean_gain": gain[validation].mean(axis=(0, 1, 2, 3)).tolist(),
        "discovery_full_budget_fraction": full_budget[discovery]
        .mean(axis=(0, 1, 2, 3))
        .tolist(),
        "validation_full_budget_fraction": full_budget[validation]
        .mean(axis=(0, 1, 2, 3))
        .tolist(),
        "discovery_reference_energy": energy[discovery]
        .mean(axis=(0, 1, 2, 3))
        .tolist(),
        "validation_reference_energy": energy[validation]
        .mean(axis=(0, 1, 2, 3))
        .tolist(),
    }
    selector_rows = [
        selector_test(
            gain,
            discovery,
            validation,
            fraction=fraction,
            seed=2000000 + operator_index * 10000 + index * 101,
            bootstrap_samples=bootstrap_samples,
            permutations=permutations,
        )
        for index, fraction in enumerate(SPARSITY_FRACTIONS)
    ]
    primary = next(row for row in selector_rows if row["fraction"] == PRIMARY_FRACTION)
    ordered_fractions = sorted(SPARSITY_FRACTIONS)
    primary_index = ordered_fractions.index(PRIMARY_FRACTION)
    adjacent_fractions = set()
    if primary_index > 0:
        adjacent_fractions.add(ordered_fractions[primary_index - 1])
    if primary_index + 1 < len(ordered_fractions):
        adjacent_fractions.add(ordered_fractions[primary_index + 1])
    adjacent_supported = sum(
        row["paired_delta_ci95"][0] >= 0.0 and row["paired_delta_mean"] > 0.0
        for row in selector_rows
        if row["fraction"] in adjacent_fractions
    )
    gate = bool(
        primary["paired_delta_ci95"][0] > 0.0
        and primary["paired_win_fraction"] >= 0.55
        and primary["time_assignment_permutation_p"] <= 0.05
        and adjacent_supported >= 1
    )
    return {
        "shape": list(gain.shape),
        "continuous_reproducibility": {
            "gain_spearman_all_cell_positions": _correlation(
                discovery_mean, validation_mean, rank=True
            ),
            "horizon_interaction_spearman": _correlation(
                discovery_interaction, validation_interaction, rank=True
            ),
            "cell_slope_pearson": _correlation(discovery_slope, validation_slope),
            "cell_slope_spearman": _correlation(
                discovery_slope, validation_slope, rank=True
            ),
        },
        "global_horizon_slope": {
            "discovery_mean": float(global_prompt_slopes[discovery].mean()),
            "validation_mean": float(global_prompt_slopes[validation].mean()),
            "validation_ci95": bootstrap_ci(
                global_prompt_slopes[validation],
                seed=2008000 + operator_index,
                samples=bootstrap_samples,
            ),
            "validation_win_fraction": float(
                np.mean(global_prompt_slopes[validation] > 0.0)
            ),
            "monotonic_increase_is_required": False,
        },
        "horizon_curves": curves,
        "selector_tests": selector_rows,
        "primary_fraction": PRIMARY_FRACTION,
        "adjacent_fractions": sorted(adjacent_fractions),
        "adjacent_supported_count": int(adjacent_supported),
        "horizon_conditioning_gate": gate,
        "gate_contract": {
            "primary_paired_ci_lower_strictly_positive": True,
            "primary_validation_win_fraction_min": 0.55,
            "time_assignment_permutation_p_max": 0.05,
            "minimum_supported_adjacent_sparsities": 1,
            "global_positive_slope_required": False,
        },
        "prompt_slopes": slopes,
        "discovery_cell_slope": discovery_slope,
        "validation_cell_slope": validation_slope,
        "discovery_cell_mean_gain": discovery_mean.mean(axis=-1),
        "validation_cell_mean_gain": validation_mean.mean(axis=-1),
    }


def _validate_sources(
    manifest_path: Path,
    analysis_path: Path,
    audit_path: Path,
) -> tuple[dict, dict, dict]:
    manifest = verify(manifest_path)
    analysis = _load_json(analysis_path)
    audit = _load_json(audit_path)
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or analysis.get("experiment") != SOURCE_EXPERIMENT
        or analysis.get("input_manifest_sha256") != sha256(manifest_path)
        or analysis.get("manual_review_required") is not False
        or set(analysis.get("operators") or {}) != set(OPERATORS)
        or audit.get("experiment") != "v189_structured_head_phase_profile_audit"
        or audit.get("ok") is not True
        or set(audit.get("operators") or {}) != set(OPERATORS)
    ):
        raise ValueError("v200 received mismatched or incomplete v189 sources")
    split = manifest.get("prompt_split") or {}
    discovery = list(split.get("discovery") or [])
    validation = list(split.get("validation") or [])
    holdout = list(split.get("generation_holdout") or [])
    if (
        len(discovery) != 64
        or len(validation) != 32
        or len(holdout) != 32
        or set(discovery) & set(validation)
        or set(discovery) & set(holdout)
        or set(validation) & set(holdout)
        or set(discovery + validation + holdout) != set(range(PROMPT_COUNT))
    ):
        raise ValueError("v189 split contract drift")
    return manifest, analysis, audit


def _validate_loaded_audit(loaded: dict, audited: dict, operator: str) -> None:
    audited_shards = {
        Path(row["path"]).name: row["sha256"] for row in audited.get("shards") or []
    }
    loaded_shards = {
        Path(row["path"]).name: row["sha256"] for row in loaded.get("shards") or []
    }
    if (
        loaded.get("operator") != operator
        or int(loaded.get("record_count", -1))
        != PROMPT_COUNT * CALLS * LAYERS * EXPECTED_POSITIONS
        or loaded_shards != audited_shards
    ):
        raise ValueError(f"v200 loaded profile disagrees with v189 audit: {operator}")


def analyze(
    manifest_path: Path,
    analysis_path: Path,
    audit_path: Path,
    profile_root: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> dict:
    manifest, source_analysis, source_audit = _validate_sources(
        manifest_path, analysis_path, audit_path
    )
    split = manifest["prompt_split"]
    discovery = list(split["discovery"])
    validation = list(split["validation"])
    output_dir.mkdir(parents=True, exist_ok=True)
    operator_reports = {}
    cell_rows = []
    curve_rows = []
    selector_rows = []

    for operator_index, operator in enumerate(OPERATORS):
        tensors, loaded_audit = load_operator_tensor(profile_root / operator, operator)
        _validate_loaded_audit(
            loaded_audit, source_audit["operators"][operator], operator
        )
        computed = analyze_operator_tensor(
            tensors,
            discovery=discovery,
            validation=validation,
            operator_index=operator_index,
            bootstrap_samples=bootstrap_samples,
            permutations=permutations,
        )
        for position, current_frame in enumerate(EXPECTED_CURRENT_FRAMES):
            curve_rows.append(
                {
                    "operator": operator,
                    "position_index": position,
                    "current_frame": current_frame,
                    **{
                        key: computed["horizon_curves"][key][position]
                        for key in computed["horizon_curves"]
                    },
                }
            )
        for call in range(CALLS):
            for layer in range(LAYERS):
                for head in range(HEADS):
                    cell_rows.append(
                        {
                            "operator": operator,
                            "call_index": call,
                            "layer": layer,
                            "head": head,
                            "discovery_mean_gain": computed["discovery_cell_mean_gain"][
                                call, layer, head
                            ],
                            "validation_mean_gain": computed[
                                "validation_cell_mean_gain"
                            ][call, layer, head],
                            "discovery_horizon_slope": computed["discovery_cell_slope"][
                                call, layer, head
                            ],
                            "validation_horizon_slope": computed[
                                "validation_cell_slope"
                            ][call, layer, head],
                        }
                    )
        for row in computed["selector_tests"]:
            selector_rows.append({"operator": operator, **row})
        operator_reports[operator] = {
            key: value
            for key, value in computed.items()
            if key
            not in {
                "prompt_slopes",
                "discovery_cell_slope",
                "validation_cell_slope",
                "discovery_cell_mean_gain",
                "validation_cell_mean_gain",
            }
        }
        operator_reports[operator]["profile_audit"] = loaded_audit
        operator_reports[operator]["v189_generation_candidate"] = operator in (
            source_analysis.get("generation_candidates") or []
        )

    candidates = [
        operator
        for operator in OPERATORS
        if operator_reports[operator]["horizon_conditioning_gate"]
    ]
    if candidates:
        recommendation = "advance_head_phase_horizon_to_runtime_design"
    elif source_analysis.get("generation_candidates"):
        recommendation = "retain_v189_head_phase_without_ar_horizon"
    else:
        recommendation = "no_reproducible_classifier_structure_do_not_generate"
    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "current_frames": list(EXPECTED_CURRENT_FRAMES),
        "current_frame_role": (
            "Ordered AR-horizon readout locations; they are not denoising timesteps."
        ),
        "split": {
            "discovery": discovery,
            "validation": validation,
            "generation_holdout": list(split["generation_holdout"]),
            "generation_holdout_used": False,
        },
        "operators": operator_reports,
        "generation_candidates": candidates,
        "recommendation": recommendation,
        "manual_review_required": False,
        "changes_v189_frozen_map": False,
        "new_video_generation_required": False,
        "claim_boundary": (
            "v200 is a cross-fit shadow-readout audit. It can authorize a new "
            "runtime design, but it cannot establish generated-video quality or "
            "replace a causal equal-exposure generation screen."
        ),
        "source": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "v189_analysis": str(analysis_path.resolve()),
            "v189_analysis_sha256": sha256(analysis_path),
            "v189_profile_audit": str(audit_path.resolve()),
            "v189_profile_audit_sha256": sha256(audit_path),
        },
    }

    json_path = output_dir / "analysis.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NpEncoder) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "horizon_curves.csv", curve_rows)
    _write_csv(output_dir / "cell_horizon_slopes.csv", cell_rows)
    _write_csv(output_dir / "selector_tests.csv", _flatten_selector_rows(selector_rows))
    (output_dir / "analysis.md").write_text(render(report), encoding="utf-8")
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_selector_rows(rows: list[dict]) -> list[dict]:
    flattened = []
    for row in rows:
        output = {
            key: value for key, value in row.items() if key != "paired_delta_ci95"
        }
        output["paired_delta_ci_lower"] = row["paired_delta_ci95"][0]
        output["paired_delta_ci_upper"] = row["paired_delta_ci95"][1]
        flattened.append(output)
    return flattened


def render(report: dict) -> str:
    lines = [
        "# v200 Head x Denoising Phase x AR-Horizon Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Horizon candidates: `{report['generation_candidates']}`",
        "- New videos required: `False`",
        "- Manual review required: `False`",
        "- The frozen v189 map is unchanged.",
        "",
        "| Operator | Horizon gate | Interaction rho | Primary delta | CI95 | Win | Time-perm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operator in OPERATORS:
        row = report["operators"][operator]
        primary = next(
            item
            for item in row["selector_tests"]
            if item["fraction"] == PRIMARY_FRACTION
        )
        correlation = row["continuous_reproducibility"]["horizon_interaction_spearman"]
        correlation_text = "n/a" if correlation is None else f"{correlation:.4f}"
        lines.append(
            f"| {operator} | {row['horizon_conditioning_gate']} | "
            f"{correlation_text} | {primary['paired_delta_mean']:.6f} | "
            f"[{primary['paired_delta_ci95'][0]:.6f}, "
            f"{primary['paired_delta_ci95'][1]:.6f}] | "
            f"{primary['paired_win_fraction']:.3f} | "
            f"{primary['time_assignment_permutation_p']:.4g} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v189-analysis", type=Path, required=True)
    parser.add_argument("--v189-profile-audit", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    args = parser.parse_args()
    if args.bootstrap_samples < 500 or args.permutations < 500:
        raise ValueError("v200 requires at least 500 bootstrap/permutation draws")
    report = analyze(
        args.manifest.resolve(),
        args.v189_analysis.resolve(),
        args.v189_profile_audit.resolve(),
        args.profile_root.resolve(),
        args.output_dir.resolve(),
        bootstrap_samples=args.bootstrap_samples,
        permutations=args.permutations,
    )
    print(
        "[v200-horizon] "
        f"recommendation={report['recommendation']} "
        f"candidates={','.join(report['generation_candidates']) or 'none'} "
        "manual_review=false"
    )


if __name__ == "__main__":
    main()
