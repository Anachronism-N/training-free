#!/usr/bin/env python3
"""Analyze structured cache compatibility for every head and denoising call."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

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


PROFILE_VERSION = 4
PROFILE_METHOD = "structured_head_phase_cache_compatibility"
POLICIES = ("recent", "coverage")
EXPECTED_RECORDS_PER_CELL = 12
DISCOVERY_GAIN = 0.02
VALIDATION_GAIN = 0.0
VALIDATION_CI_LOWER = -0.01
VALIDATION_WIN = 0.60
PHASE_CONTRAST = 0.01
PHASE_VALIDATION_WIN = 0.55
MIN_BUDGET_FRACTION = 0.80
MIN_RELATIVE_ENERGY = 0.10
TOPK_PER_CALL = 12


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 1000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(samples, values.size))].mean(
        axis=1
    )
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def one_sided_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def bh_qvalues(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def profile_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.pt"))
    if not paths:
        paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise ValueError(f"no v189 profile shards below {root}")
    return paths


def aggregate_operator(root: Path, operator: str) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    gain_sum = np.zeros((PROMPT_COUNT, CALLS, LAYERS, HEADS), dtype=np.float64)
    energy_sum = np.zeros_like(gain_sum)
    budget_sum = np.zeros_like(gain_sum)
    counts = np.zeros((PROMPT_COUNT, CALLS, LAYERS), dtype=np.int16)
    locations: set[tuple[int, int, int, int, str, str]] = set()
    shard_rows = []
    source_kinds = Counter()
    for path in profile_paths(root):
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
            raise ValueError(f"{path}: v189 artifact contract drift")
        records = payload.get("records") or []
        if not records:
            raise ValueError(f"{path}: empty v189 profile shard")
        prompt_ids = set()
        for record_index, record in enumerate(records):
            prompt = int(record["prompt_id"])
            call = int(record["call_index"])
            layer = int(record["layer"])
            if not (
                0 <= prompt < PROMPT_COUNT
                and 0 <= call < CALLS
                and 0 <= layer < LAYERS
                and int(record.get("heads", -1)) == HEADS
                and record.get("profile_contract") == "v189"
            ):
                raise ValueError(f"{path}:{record_index}: location contract drift")
            location = (
                prompt,
                layer,
                int(record["current_frame"]),
                call,
                str(record["cache_update_mode"]),
                str(record["cfg_branch"]),
            )
            if location in locations:
                raise ValueError(f"duplicate v189 location: {location}")
            locations.add(location)
            if location[-2:] != ("noisy", "cond"):
                raise ValueError(f"{path}:{record_index}: branch/mode drift")
            policies = record.get("policies") or {}
            budgets = record.get("budgets") or {}
            if set(policies) != set(POLICIES) or set(budgets) != {
                "recent",
                "coverage",
                "union",
            }:
                raise ValueError(f"{path}:{record_index}: policy set drift")
            errors = {}
            for policy in POLICIES:
                values = np.asarray(
                    policies[policy]["residual_relative_mse"], dtype=np.float64
                )
                if values.shape != (HEADS,) or not np.isfinite(values).all():
                    raise ValueError(f"{path}:{record_index}: residual shape drift")
                errors[policy] = np.log(np.maximum(values, 1e-12))
            reference_energy = np.asarray(
                record["reference_residual_energy"], dtype=np.float64
            )
            coverage_budget = np.asarray(
                budgets["coverage"]["per_sequence_frame_equivalents"],
                dtype=np.float64,
            )
            if (
                reference_energy.shape != (HEADS,)
                or coverage_budget.shape != (HEADS,)
                or not np.isfinite(reference_energy).all()
                or np.any(reference_energy < 0)
                or np.any(coverage_budget > 9)
                or int(budgets["recent"]["max_frame_equivalents"]) > 9
                or int(budgets["coverage"]["max_frame_equivalents"]) > 9
                or int(budgets["union"]["max_frame_equivalents"]) > 13
            ):
                raise ValueError(f"{path}:{record_index}: budget/energy drift")
            union = budgets["union"]
            if (
                union.get("superset_verification_contract") != "v189"
                or union.get("candidate_representation_subset_verified") is not True
                or int(union.get("candidate_representation_subset_checks", -1))
                != len(POLICIES) * HEADS
                or int(union.get("candidate_representation_subset_failures", -1))
                != 0
            ):
                raise ValueError(f"{path}:{record_index}: Union is not a strict superset")
            gain_sum[prompt, call, layer] += errors["recent"] - errors["coverage"]
            energy_sum[prompt, call, layer] += reference_energy
            budget_sum[prompt, call, layer] += coverage_budget >= 9
            counts[prompt, call, layer] += 1
            prompt_ids.add(prompt)
            codebook = budgets["coverage"].get("selected_source_codebook") or []
            source_kinds.update(str(value) for value in codebook)
        shard_rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "record_count": len(records),
                "prompt_ids": sorted(prompt_ids),
            }
        )
    if not np.all(counts == EXPECTED_RECORDS_PER_CELL):
        observed = Counter(int(value) for value in counts.reshape(-1))
        raise ValueError(
            "v189 profile is incomplete; expected 12 records per "
            f"prompt/call/layer, observed={dict(sorted(observed.items()))}"
        )
    aggregate = {
        "gain": gain_sum / counts[:, :, :, None],
        "energy": energy_sum / counts[:, :, :, None],
        "full_budget": budget_sum / counts[:, :, :, None],
    }
    audit = {
        "operator": operator,
        "shard_count": len(shard_rows),
        "record_count": int(np.sum(counts)),
        "prompt_count": PROMPT_COUNT,
        "records_per_prompt_call_layer": EXPECTED_RECORDS_PER_CELL,
        "candidate_representation_subset_checks_per_record": len(POLICIES) * HEADS,
        "source_codebook_counts": dict(sorted(source_kinds.items())),
        "shards": shard_rows,
    }
    return aggregate, audit


def _cell_rows(
    aggregate: dict[str, np.ndarray],
    *,
    operator: str,
    discovery: list[int],
    validation: list[int],
) -> list[dict]:
    gain = aggregate["gain"]
    energy = aggregate["energy"]
    full_budget = aggregate["full_budget"]
    rows = []
    p_values = []
    for call in range(CALLS):
        discovery_layer_energy = energy[discovery, call].mean(axis=0)
        layer_medians = np.median(discovery_layer_energy, axis=1)
        for layer in range(LAYERS):
            for head in range(HEADS):
                discovery_values = gain[discovery, call, layer, head]
                validation_values = gain[validation, call, layer, head]
                other_calls = [value for value in range(CALLS) if value != call]
                discovery_phase = discovery_values - gain[
                    np.ix_(discovery, other_calls, [layer], [head])
                ].reshape(len(discovery), len(other_calls)).mean(axis=1)
                validation_phase = validation_values - gain[
                    np.ix_(validation, other_calls, [layer], [head])
                ].reshape(len(validation), len(other_calls)).mean(axis=1)
                layer_median = max(float(layer_medians[layer]), 1e-12)
                relative_energy = float(
                    discovery_layer_energy[layer, head] / layer_median
                )
                validation_ci = bootstrap_ci(
                    validation_values,
                    seed=1890000 + call * 10000 + layer * 100 + head,
                )
                phase_ci = bootstrap_ci(
                    validation_phase,
                    seed=1895000 + call * 10000 + layer * 100 + head,
                )
                compatible = _compatibility_gate(
                    discovery_values,
                    validation_values,
                    validation_ci=validation_ci,
                    full_budget_values=full_budget[
                        discovery, call, layer, head
                    ],
                    relative_energy=relative_energy,
                )
                phase_selective = bool(
                    compatible
                    and float(np.mean(discovery_phase)) >= PHASE_CONTRAST
                    and float(np.mean(validation_phase)) > 0
                    and float(np.mean(validation_phase > 0)) >= PHASE_VALIDATION_WIN
                )
                sign_p = one_sided_sign_p(validation_values)
                p_values.append(sign_p)
                rows.append(
                    {
                        "operator": operator,
                        "call_index": call,
                        "layer": layer,
                        "head": head,
                        "discovery_gain": float(np.mean(discovery_values)),
                        "validation_gain": float(np.mean(validation_values)),
                        "validation_ci95": validation_ci,
                        "validation_win_fraction": float(
                            np.mean(validation_values > 0)
                        ),
                        "validation_sign_p": sign_p,
                        "phase_contrast_discovery": float(np.mean(discovery_phase)),
                        "phase_contrast_validation": float(np.mean(validation_phase)),
                        "phase_contrast_validation_ci95": phase_ci,
                        "phase_contrast_validation_win_fraction": float(
                            np.mean(validation_phase > 0)
                        ),
                        "full_budget_fraction": float(
                            np.mean(full_budget[discovery, call, layer, head])
                        ),
                        "relative_reference_energy": relative_energy,
                        "compatible": compatible,
                        "phase_selective": phase_selective,
                    }
                )
    for row, q_value in zip(rows, bh_qvalues(p_values)):
        row["validation_bh_q"] = float(q_value)
    return rows


def _mask(rows: list[dict], field: str) -> list[list[list[bool]]]:
    result = [
        [[False for _ in range(HEADS)] for _ in range(LAYERS)]
        for _ in range(CALLS)
    ]
    for row in rows:
        result[int(row["call_index"])][int(row["layer"])][int(row["head"])] = bool(
            row[field]
        )
    return result


def _topk_mask(rows: list[dict]) -> list[list[list[bool]]]:
    result = [
        [[False for _ in range(HEADS)] for _ in range(LAYERS)]
        for _ in range(CALLS)
    ]
    for call in range(CALLS):
        eligible = [
            row
            for row in rows
            if int(row["call_index"]) == call
            and float(row["discovery_gain"]) > 0
            and float(row["full_budget_fraction"]) >= MIN_BUDGET_FRACTION
            and float(row["relative_reference_energy"]) >= MIN_RELATIVE_ENERGY
        ]
        ranked = sorted(
            eligible,
            key=lambda row: (
                float(row["discovery_gain"]),
                float(row["phase_contrast_discovery"]),
                -int(row["layer"]),
                -int(row["head"]),
            ),
            reverse=True,
        )[:TOPK_PER_CALL]
        for row in ranked:
            result[call][int(row["layer"])][int(row["head"])] = True
    return result


def _compatibility_gate(
    discovery_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    validation_ci: list[float],
    full_budget_values: np.ndarray,
    relative_energy: float,
) -> bool:
    return bool(
        float(np.mean(discovery_values)) >= DISCOVERY_GAIN
        and float(np.mean(validation_values)) >= VALIDATION_GAIN
        and validation_ci[0] >= VALIDATION_CI_LOWER
        and float(np.mean(validation_values > 0)) >= VALIDATION_WIN
        and float(np.mean(full_budget_values)) >= MIN_BUDGET_FRACTION
        and relative_energy >= MIN_RELATIVE_ENERGY
    )


def _factorized_masks(
    aggregate: dict[str, np.ndarray],
    *,
    operator: str,
    discovery: list[int],
    validation: list[int],
) -> tuple[dict[str, list[list[list[bool]]]], list[dict]]:
    """Fit call-invariant head and head-invariant call/layer controls."""

    gain = aggregate["gain"]
    energy = aggregate["energy"]
    full_budget = aggregate["full_budget"]
    head_mask = [[False for _ in range(HEADS)] for _ in range(LAYERS)]
    phase_layer_mask = [
        [False for _ in range(LAYERS)] for _ in range(CALLS)
    ]
    rows = []

    head_energy = energy[discovery].mean(axis=(0, 1))
    head_layer_medians = np.median(head_energy, axis=1)
    for layer in range(LAYERS):
        for head in range(HEADS):
            discovery_values = gain[discovery, :, layer, head].mean(axis=1)
            validation_values = gain[validation, :, layer, head].mean(axis=1)
            validation_ci = bootstrap_ci(
                validation_values,
                seed=1897000 + layer * 100 + head,
            )
            relative_energy = float(
                head_energy[layer, head]
                / max(float(head_layer_medians[layer]), 1e-12)
            )
            budget_values = full_budget[discovery, :, layer, head].mean(axis=1)
            selected = _compatibility_gate(
                discovery_values,
                validation_values,
                validation_ci=validation_ci,
                full_budget_values=budget_values,
                relative_energy=relative_energy,
            )
            head_mask[layer][head] = selected
            rows.append(
                {
                    "operator": operator,
                    "factor": "head_only",
                    "call_index": -1,
                    "layer": layer,
                    "head": head,
                    "discovery_gain": float(np.mean(discovery_values)),
                    "validation_gain": float(np.mean(validation_values)),
                    "validation_ci_lower": validation_ci[0],
                    "validation_ci_upper": validation_ci[1],
                    "validation_win_fraction": float(
                        np.mean(validation_values > 0)
                    ),
                    "full_budget_fraction": float(np.mean(budget_values)),
                    "relative_reference_energy": relative_energy,
                    "selected": selected,
                }
            )

    for call in range(CALLS):
        for layer in range(LAYERS):
            discovery_values = gain[discovery, call, layer].mean(axis=1)
            validation_values = gain[validation, call, layer].mean(axis=1)
            validation_ci = bootstrap_ci(
                validation_values,
                seed=1898000 + call * 10000 + layer * 100,
            )
            per_head_energy = energy[discovery, call, layer].mean(axis=0)
            relative_energy = float(
                np.mean(per_head_energy)
                / max(float(np.median(per_head_energy)), 1e-12)
            )
            budget_values = full_budget[discovery, call, layer].mean(axis=1)
            selected = _compatibility_gate(
                discovery_values,
                validation_values,
                validation_ci=validation_ci,
                full_budget_values=budget_values,
                relative_energy=relative_energy,
            )
            phase_layer_mask[call][layer] = selected
            rows.append(
                {
                    "operator": operator,
                    "factor": "phase_layer_only",
                    "call_index": call,
                    "layer": layer,
                    "head": -1,
                    "discovery_gain": float(np.mean(discovery_values)),
                    "validation_gain": float(np.mean(validation_values)),
                    "validation_ci_lower": validation_ci[0],
                    "validation_ci_upper": validation_ci[1],
                    "validation_win_fraction": float(
                        np.mean(validation_values > 0)
                    ),
                    "full_budget_fraction": float(np.mean(budget_values)),
                    "relative_reference_energy": relative_energy,
                    "selected": selected,
                }
            )

    head_masks = [
        [[bool(value) for value in layer] for layer in head_mask]
        for _ in range(CALLS)
    ]
    phase_layer_masks = [
        [
            [bool(phase_layer_mask[call][layer]) for _ in range(HEADS)]
            for layer in range(LAYERS)
        ]
        for call in range(CALLS)
    ]
    return {
        "head_only_compatible": head_masks,
        "phase_layer_only_compatible": phase_layer_masks,
    }, rows


def _write_map(
    path: Path,
    *,
    operator: str,
    classification: str,
    masks: list[list[list[bool]]],
    source_audit: dict,
    holdout: list[int],
) -> dict:
    counts = [
        int(sum(value for layer in call_rows for value in layer))
        for call_rows in masks
    ]
    payload = {
        "version": 1,
        "experiment": "v189_structured_head_phase_profile",
        "classification": classification,
        "coverage_operator": operator,
        "call_count": CALLS,
        "layer_count": LAYERS,
        "head_count": HEADS,
        "coverage_masks": masks,
        "coverage_count_by_call": counts,
        "generation_holdout_prompt_ids": holdout,
        "source_profile_shards": [
            {"sha256": row["sha256"], "record_count": row["record_count"]}
            for row in source_audit["shards"]
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["map_id"] = f"v189-{operator}-{classification}-{hashlib.sha256(canonical).hexdigest()[:12]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _threshold_sensitivity(rows: list[dict]) -> list[dict]:
    output = []
    for gain_threshold in (0.0, 0.01, 0.02, 0.05):
        for win_threshold in (0.55, 0.60, 0.65):
            selected = [
                row
                for row in rows
                if float(row["discovery_gain"]) >= gain_threshold
                and float(row["validation_gain"]) >= 0
                and float(row["validation_win_fraction"]) >= win_threshold
                and float(row["full_budget_fraction"]) >= MIN_BUDGET_FRACTION
                and float(row["relative_reference_energy"]) >= MIN_RELATIVE_ENERGY
            ]
            output.append(
                {
                    "discovery_gain_threshold": gain_threshold,
                    "validation_win_threshold": win_threshold,
                    "cell_count": len(selected),
                    "count_by_call": [
                        sum(int(row["call_index"]) == call for row in selected)
                        for call in range(CALLS)
                    ],
                }
            )
    return output


def analyze(manifest_path: Path, profile_root: Path, output_dir: Path) -> dict:
    manifest = verify(manifest_path)
    split = manifest["prompt_split"]
    discovery = [int(value) for value in split["discovery"]]
    validation = [int(value) for value in split["validation"]]
    holdout = [int(value) for value in split["generation_holdout"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_factor_rows = []
    audits = {}
    operators = {}
    map_payloads = {}
    for operator in OPERATORS:
        aggregate, audit = aggregate_operator(profile_root / operator, operator)
        rows = _cell_rows(
            aggregate,
            operator=operator,
            discovery=discovery,
            validation=validation,
        )
        all_rows.extend(rows)
        audits[operator] = audit
        compatible = _mask(rows, "compatible")
        selective = _mask(rows, "phase_selective")
        topk = _topk_mask(rows)
        factor_masks, factor_rows = _factorized_masks(
            aggregate,
            operator=operator,
            discovery=discovery,
            validation=validation,
        )
        all_factor_rows.extend(factor_rows)
        operator_maps = {}
        for name, masks in (
            ("compatible", compatible),
            ("phase_selective", selective),
            ("top12_discovery", topk),
            *factor_masks.items(),
        ):
            operator_maps[name] = _write_map(
                output_dir / "maps" / f"{operator}_{name}.json",
                operator=operator,
                classification=name,
                masks=masks,
                source_audit=audit,
                holdout=holdout,
            )
        map_payloads[operator] = operator_maps
        compatible_rows = [row for row in rows if row["compatible"]]
        selective_rows = [row for row in rows if row["phase_selective"]]
        operators[operator] = {
            "compatible_cell_count": len(compatible_rows),
            "phase_selective_cell_count": len(selective_rows),
            "head_only_compatible_cell_count": int(
                sum(
                    value
                    for call_rows in factor_masks["head_only_compatible"]
                    for layer in call_rows
                    for value in layer
                )
            ),
            "phase_layer_only_compatible_cell_count": int(
                sum(
                    value
                    for call_rows in factor_masks["phase_layer_only_compatible"]
                    for layer in call_rows
                    for value in layer
                )
            ),
            "compatible_count_by_call": operator_maps["compatible"][
                "coverage_count_by_call"
            ],
            "phase_selective_count_by_call": operator_maps["phase_selective"][
                "coverage_count_by_call"
            ],
            "compatible_validation_gain_mean": (
                float(np.mean([row["validation_gain"] for row in compatible_rows]))
                if compatible_rows
                else None
            ),
            "phase_selective_validation_gain_mean": (
                float(np.mean([row["validation_gain"] for row in selective_rows]))
                if selective_rows
                else None
            ),
            "heads_selected_any_call": len(
                {
                    (int(row["layer"]), int(row["head"]))
                    for row in compatible_rows
                }
            ),
            "heads_selected_all_calls": sum(
                all(
                    compatible[call][layer][head]
                    for call in range(CALLS)
                )
                for layer in range(LAYERS)
                for head in range(HEADS)
            ),
            "threshold_sensitivity": _threshold_sensitivity(rows),
            "maps": {
                name: {
                    "path": str(
                        (output_dir / "maps" / f"{operator}_{name}.json").resolve()
                    ),
                    "sha256": sha256(
                        output_dir / "maps" / f"{operator}_{name}.json"
                    ),
                    "map_id": payload["map_id"],
                    "coverage_count_by_call": payload["coverage_count_by_call"],
                }
                for name, payload in operator_maps.items()
            },
        }

    overlap = []
    for call in range(CALLS):
        left = {
            (layer, head)
            for layer in range(LAYERS)
            for head in range(HEADS)
            if map_payloads["landmark"]["compatible"]["coverage_masks"][call][
                layer
            ][head]
        }
        right = {
            (layer, head)
            for layer in range(LAYERS)
            for head in range(HEADS)
            if map_payloads["retrieval"]["compatible"]["coverage_masks"][call][
                layer
            ][head]
        }
        union = left | right
        overlap.append(
            {
                "call_index": call,
                "landmark_count": len(left),
                "retrieval_count": len(right),
                "intersection": len(left & right),
                "jaccard": float(len(left & right) / len(union)) if union else 1.0,
            }
        )
    generation_candidates = [
        operator
        for operator in OPERATORS
        if operators[operator]["compatible_cell_count"] >= 4
        and sum(
            value > 0 for value in operators[operator]["compatible_count_by_call"]
        )
        >= 2
    ]
    report = {
        "version": 1,
        "experiment": "v189_structured_head_phase_profile",
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "claim_boundary": (
            "The report supports shadow-readout compatibility only. Generated "
            "video causality requires a separate screen on prompts excluded "
            "from map fitting and operator selection."
        ),
        "split": split,
        "thresholds": {
            "discovery_gain": DISCOVERY_GAIN,
            "validation_gain": VALIDATION_GAIN,
            "validation_ci_lower": VALIDATION_CI_LOWER,
            "validation_win_fraction": VALIDATION_WIN,
            "phase_contrast": PHASE_CONTRAST,
            "phase_validation_win_fraction": PHASE_VALIDATION_WIN,
            "minimum_full_budget_fraction": MIN_BUDGET_FRACTION,
            "minimum_relative_reference_energy": MIN_RELATIVE_ENERGY,
            "topk_per_call_sensitivity": TOPK_PER_CALL,
        },
        "operators": operators,
        "operator_overlap_by_call": overlap,
        "profile_audits": audits,
        "generation_candidates": generation_candidates,
        "recommendation": (
            "advance_head_phase_maps_to_causal_screen"
            if generation_candidates
            else "do_not_generate_from_v189_classifier"
        ),
        "manual_review_required": False,
    }
    json_path = output_dir / "analysis.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NpEncoder) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "operator",
        "call_index",
        "layer",
        "head",
        "discovery_gain",
        "validation_gain",
        "validation_ci_lower",
        "validation_ci_upper",
        "validation_win_fraction",
        "validation_sign_p",
        "validation_bh_q",
        "phase_contrast_discovery",
        "phase_contrast_validation",
        "phase_contrast_validation_ci_lower",
        "phase_contrast_validation_ci_upper",
        "phase_contrast_validation_win_fraction",
        "full_budget_fraction",
        "relative_reference_energy",
        "compatible",
        "phase_selective",
    ]
    with (output_dir / "cell_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            flat.update(
                {
                    "validation_ci_lower": row["validation_ci95"][0],
                    "validation_ci_upper": row["validation_ci95"][1],
                    "phase_contrast_validation_ci_lower": row[
                        "phase_contrast_validation_ci95"
                    ][0],
                    "phase_contrast_validation_ci_upper": row[
                        "phase_contrast_validation_ci95"
                    ][1],
                }
            )
            writer.writerow(flat)
    factor_fieldnames = [
        "operator",
        "factor",
        "call_index",
        "layer",
        "head",
        "discovery_gain",
        "validation_gain",
        "validation_ci_lower",
        "validation_ci_upper",
        "validation_win_fraction",
        "full_budget_fraction",
        "relative_reference_energy",
        "selected",
    ]
    with (output_dir / "factor_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=factor_fieldnames)
        writer.writeheader()
        writer.writerows(all_factor_rows)
    lines = [
        "# v189 Structured Head x Phase Profile",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Generation candidates: `{generation_candidates}`",
        "- Holdout prompts were not used for classification.",
        "- Classification is per denoising call; cross-call consistency is not a gate.",
        "",
        "| Operator | Joint cells | Head-only cells | Phase/layer-only cells | Phase-selective cells | Per-call joint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for operator in OPERATORS:
        row = operators[operator]
        lines.append(
            f"| {operator} | {row['compatible_cell_count']} | "
            f"{row['head_only_compatible_cell_count']} | "
            f"{row['phase_layer_only_compatible_cell_count']} | "
            f"{row['phase_selective_cell_count']} | "
            f"{row['compatible_count_by_call']} |"
        )
    lines.extend(
        [
            "",
            "The `compatible` map is the primary generation candidate. "
            "`phase_selective` and `top12_discovery` are preregistered mechanism/threshold controls, not alternative claims selected after video review.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.profile_root, args.output_dir)
    print(
        "[v189-analysis] "
        f"recommendation={report['recommendation']} "
        f"candidates={report['generation_candidates']}"
    )


if __name__ == "__main__":
    main()
