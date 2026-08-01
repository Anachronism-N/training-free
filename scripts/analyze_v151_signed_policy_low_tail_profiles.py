#!/usr/bin/env python3
"""Analyze the independent v151 signed-policy and low-tail causal suite."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_v149_calibrated_causal_profiles import (
        EPSILON,
        HEADS,
        LAYERS,
        PROFILE_VERSION,
        REPLICATES,
        _canonical_digest,
        _downstream_rows,
        _lookup,
        _paired_log_effects,
        _probe_summary,
        _summarize_effects,
        _write_csv,
    )
except ModuleNotFoundError:
    from analyze_v149_calibrated_causal_profiles import (
        EPSILON,
        HEADS,
        LAYERS,
        PROFILE_VERSION,
        REPLICATES,
        _canonical_digest,
        _downstream_rows,
        _lookup,
        _paired_log_effects,
        _probe_summary,
        _summarize_effects,
        _write_csv,
    )


SUITE = "v151_signed_policy_low_tail_core"
EXPECTED_PROFILES = 64
EXPECTED_PROMPTS = 32
EXPECTED_PROBES = 32
PER_LAYER_COUNT = 4
RANDOM_CONTROL_COUNT = 8
CONTEXTS = (
    "noisy_t1000",
    "noisy_t750",
    "noisy_t500",
    "noisy_t250",
)
CHANNEL_METRICS = {
    "susceptibility": "mean_raw_projected_relative_rms",
    "leverage": "x0_relative_rms",
}
MIN_RANDOM_POSITIVE_MAPS = 6
MIN_LOG_EFFECT = math.log(1.05)
MIN_POSITIVE_FRACTION = 0.65
MIN_SEED_REPLICATE_SPEARMAN = 0.30
MAX_CALIBRATION_RELATIVE_ERROR = 0.02
MIN_ACCEPTABLE_CALIBRATION_SCALE = 0.005
MAX_ACCEPTABLE_CALIBRATION_SCALE = 50.0


def _load_plan(path: Path) -> tuple[dict, str, int]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(plan.get("version", -1)) != 1
        or plan.get("suite") != SUITE
        or int(plan.get("layers", -1)) != LAYERS
        or int(plan.get("heads", -1)) != HEADS
    ):
        raise RuntimeError("v151 probe plan has an invalid model contract")
    probes = plan.get("probes") or []
    names = {str(probe.get("name")) for probe in probes}
    if len(probes) != EXPECTED_PROBES or len(names) != EXPECTED_PROBES:
        raise RuntimeError("v151 probe plan must contain 32 unique probes")
    if int(plan.get("random_control_count", -1)) != RANDOM_CONTROL_COUNT:
        raise RuntimeError("v151 random-control count changed")
    contexts = plan.get("contexts") or []
    if [int(row["nominal_timestep"]) for row in contexts] != [
        1000,
        750,
        500,
        250,
    ]:
        raise RuntimeError("v151 timestep grid changed")
    refinement_steps = {
        int(probe.get("calibration", {}).get("refinement_steps", -1))
        for probe in probes
    }
    if (
        len(refinement_steps) != 1
        or not 0 < next(iter(refinement_steps)) <= 8
    ):
        raise RuntimeError("v151 refinement-step contract changed")
    expected_refinement_steps = refinement_steps.pop()
    for probe in probes:
        calibration = probe.get("calibration") or {}
        if (
            calibration.get("mode") != "projected_relative_rms"
            or float(calibration.get("target", -1)) != 0.02
            or float(calibration.get("min_scale", -1)) != 0.001
            or float(calibration.get("max_scale", -1)) != 50.0
        ):
            raise RuntimeError("v151 calibration contract changed")
        head_map = probe.get("head_map") or {}
        if set(head_map) != {str(layer) for layer in range(LAYERS)}:
            raise RuntimeError(f"v151 {probe['name']} has an incomplete map")
        if any(
            len({int(head) for head in heads}) != PER_LAYER_COUNT
            for heads in head_map.values()
        ):
            raise RuntimeError(f"v151 {probe['name']} has an invalid map")
        intervention = probe.get("intervention")
        if intervention == "uniform" and probe.get("policy_args") != {
            "left": "uniform8",
            "right": "recent8",
        }:
            raise RuntimeError("v151 uniform contrast changed")
        if intervention == "boundary" and probe.get("policy_args") != {
            "left": "boundary8",
            "right": "recent8",
        }:
            raise RuntimeError("v151 boundary contrast changed")
    families = plan.get("families") or {}
    if set(families) != {"scalar", "signed"}:
        raise RuntimeError("v151 comparison families are incomplete")
    for family in families.values():
        referenced = {
            probe
            for cells in family["probes"].values()
            for probe in cells.values()
        } | set(family["random_uniform_probes"])
        if not referenced <= names:
            raise RuntimeError("v151 comparison references an unknown probe")
        if len(family["random_uniform_probes"]) != RANDOM_CONTROL_COUNT:
            raise RuntimeError("v151 comparison has the wrong random controls")
    return plan, _canonical_digest(plan), expected_refinement_steps


def _load_profiles(
    directory: Path,
    *,
    plan: dict,
    plan_sha256: str,
    expected_count: int,
    expected_refinement_steps: int,
) -> tuple[list[dict], list[dict]]:
    import torch

    if expected_count != EXPECTED_PROFILES:
        raise RuntimeError("v151 is frozen at 64 profiles")
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v151 profiles, found {len(paths)}"
        )
    expected_downstream = len(CONTEXTS) * (len(plan["probes"]) + 1)
    expected_records = len(CONTEXTS) * LAYERS
    profiles = []
    audits = []
    seen = set()
    prompt_seeds = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
        if key in seen:
            raise RuntimeError(f"duplicate v151 profile coordinate: {key}")
        seen.add(key)
        if (
            int(payload.get("version", -1)) != PROFILE_VERSION
            or str(job.get("kind")) != SUITE
            or not (
                int(metadata["seed"])
                == int(job["seed"])
                == int(job["reference_seed"])
            )
        ):
            raise RuntimeError(f"{path} violates the v151 profile contract")
        records = payload.get("records") or []
        downstream = payload.get("downstream_probe_records") or []
        if (
            metadata.get("incomplete_calls")
            or int(metadata.get("captured_calls", -1)) != len(CONTEXTS)
            or len(records) != expected_records
            or int(metadata.get("record_count", -1)) != expected_records
            or len(downstream) != expected_downstream
            or int(payload.get("downstream_probe_expected_count", -1))
            != expected_downstream
        ):
            raise RuntimeError(f"{path} has an invalid v151 capture grid")
        refined_layers = [
            layer_metadata
            for row in downstream
            if str(row.get("probe_name")) != "native_replay"
            for layer_metadata in (row.get("layer_metadata") or {}).values()
        ]
        if (
            len(refined_layers) != len(CONTEXTS) * EXPECTED_PROBES * LAYERS
            or any(
                int(layer.get("calibration_refinement_steps", -1))
                != expected_refinement_steps
                for layer in refined_layers
            )
        ):
            raise RuntimeError(
                f"{path} did not execute the frozen v151 refinement path"
            )
        state_layers = Counter(
            (
                str(row["mode"]),
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in records
        )
        if len(state_layers) != len(CONTEXTS) or set(
            state_layers.values()
        ) != {LAYERS}:
            raise RuntimeError(f"{path} has an invalid state/layer grid")
        plan_metadata = metadata.get("downstream_probe_plan") or {}
        if str(plan_metadata.get("sha256")) != plan_sha256:
            raise RuntimeError(f"{path} uses a different v151 probe plan")
        probe_grid = Counter(
            (
                f"noisy_t{int(row['nominal_timestep'])}",
                str(row["probe_name"]),
            )
            for row in downstream
        )
        if (
            {context for context, _ in probe_grid} != set(CONTEXTS)
            or len(probe_grid) != expected_downstream
            or set(probe_grid.values()) != {1}
        ):
            raise RuntimeError(f"{path} has an invalid downstream grid")
        prompt_seeds[key] = int(metadata["seed"])
        profiles.append(payload)
        audits.append(
            {
                "dataset_index": int(job["dataset_index"]),
                "prompt_slot": key[0],
                "source_prompt_index": int(job["source_prompt_index"]),
                "seed_replicate": key[1],
                "seed": int(metadata["seed"]),
                "captured_calls": int(metadata["captured_calls"]),
                "record_count": len(records),
                "downstream_record_count": len(downstream),
                "refined_layer_count": len(refined_layers),
                "refinement_bound_hit_count": sum(
                    bool(layer["calibration_refinement_bound_hit"])
                    for layer in refined_layers
                ),
                "path": str(path),
            }
        )
    expected_grid = {
        (prompt, replicate)
        for prompt in range(EXPECTED_PROMPTS)
        for replicate in REPLICATES
    }
    if seen != expected_grid:
        raise RuntimeError("v151 prompt/seed grid is incomplete")
    for prompt in range(EXPECTED_PROMPTS):
        if prompt_seeds[(prompt, 0)] == prompt_seeds[(prompt, 1)]:
            raise RuntimeError(f"v151 prompt {prompt} repeats its seed")
    return profiles, audits


def _enrich_rows(rows: list[dict], plan: dict) -> list[dict]:
    probes = {probe["name"]: probe for probe in plan["probes"]}
    enriched = []
    for row in rows:
        if row["probe_name"] == "native_replay":
            enriched.append(
                {
                    **row,
                    "target": 0.0,
                    "rank_group": "native",
                    "control_family": "native",
                    "intervention": "native",
                }
            )
            continue
        probe = probes.get(row["probe_name"])
        if probe is None:
            raise RuntimeError(f"unknown v151 probe {row['probe_name']}")
        if (
            row["policy"] != probe["policy"]
            or row["group"] != probe["group"]
            or row["selected_head_count"] != LAYERS * PER_LAYER_COUNT
        ):
            raise RuntimeError(
                f"runtime metadata changed for {row['probe_name']}"
            )
        enriched.append(
            {
                **row,
                "target": float(probe["calibration"]["target"]),
                "rank_group": str(probe["rank_group"]),
                "control_family": str(probe["control_family"]),
                "intervention": str(probe["intervention"]),
            }
        )
    return enriched


def _comparison_integrity(
    probes: list[str], context: str, integrity: dict[tuple[str, str], bool]
) -> bool:
    return all(integrity.get((probe, context), False) for probe in probes)


def _qualifies(row: dict) -> bool:
    return bool(
        row["context"] != "pooled"
        and row["median_effect"] >= MIN_LOG_EFFECT
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= MIN_POSITIVE_FRACTION
        )
        and row["seed_replicate_spearman"]
        >= MIN_SEED_REPLICATE_SPEARMAN
    )


def _probe_integrity(
    rows: list[dict], *, expected_units: int
) -> tuple[list[dict], dict[tuple[str, str], bool], dict]:
    """Validate calibration over v151's frozen four-context grid."""

    native = [row for row in rows if row["probe_name"] == "native_replay"]
    if len(native) != expected_units * len(CONTEXTS):
        raise RuntimeError("v151 native replay grid is incomplete")
    replay_max = max(
        max(row["flow_relative_rms"], row["x0_relative_rms"])
        for row in native
    )
    grouped = defaultdict(list)
    for row in rows:
        if row["probe_name"] != "native_replay":
            grouped[(row["probe_name"], row["context"])].append(row)
    expected_cells = EXPECTED_PROBES * len(CONTEXTS)
    if len(grouped) != expected_cells:
        raise RuntimeError(
            f"v151 expected {expected_cells} probe/context cells, "
            f"found {len(grouped)}"
        )

    summaries = []
    lookup = {}
    for (probe, context), values in sorted(grouped.items()):
        if context not in CONTEXTS or len(values) != expected_units:
            raise RuntimeError(
                f"{probe}/{context} has {len(values)} units, expected "
                f"{expected_units}"
            )
        clipped = sum(row["calibration_clipped_count"] for row in values)
        degenerate = sum(
            row["calibration_degenerate_count"] for row in values
        )
        max_error = max(
            row["calibration_relative_error_max"] for row in values
        )
        min_scale = min(row["calibration_scale_min"] for row in values)
        max_scale = max(row["calibration_scale_max"] for row in values)
        target_min = min(row["calibration_target_min"] for row in values)
        target_max = max(row["calibration_target_max"] for row in values)
        target = float(values[0]["target"])
        target_consistent = (
            abs(target_min - target) <= 1e-8
            and abs(target_max - target) <= 1e-8
        )
        contrast_valid = all(row["policy_contrast_valid"] for row in values)
        shift_valid = (
            all(row["min_shifted_old_frames"] > 1 for row in values)
            if values[0]["policy"] in {"key_shift", "value_shift"}
            else True
        )
        passed = bool(
            clipped == 0
            and degenerate == 0
            and max_error <= MAX_CALIBRATION_RELATIVE_ERROR
            and min_scale >= MIN_ACCEPTABLE_CALIBRATION_SCALE
            and max_scale <= MAX_ACCEPTABLE_CALIBRATION_SCALE
            and target_consistent
            and contrast_valid
            and shift_valid
        )
        summary = {
            "probe_name": probe,
            "policy": values[0]["policy"],
            "rank_group": values[0]["rank_group"],
            "target": target,
            "context": context,
            "unit_count": len(values),
            "calibrated_layer_count": sum(
                row["calibrated_layer_count"] for row in values
            ),
            "calibration_clipped_count": clipped,
            "calibration_degenerate_count": degenerate,
            "calibration_max_relative_error": max_error,
            "calibration_scale_min": min_scale,
            "calibration_scale_max": max_scale,
            "target_consistent": int(target_consistent),
            "policy_or_shift_valid": int(contrast_valid and shift_valid),
            "integrity_pass": int(passed),
        }
        summaries.append(summary)
        lookup[(probe, context)] = passed

    return summaries, lookup, {
        "native_replay_max_relative_rms": replay_max,
        "native_replay_pass": replay_max <= 1e-4,
        "probe_context_count": len(summaries),
        "probe_context_integrity_pass_count": sum(
            row["integrity_pass"] for row in summaries
        ),
        "probe_context_integrity_pass_rate": float(
            np.mean([row["integrity_pass"] for row in summaries])
        ),
        "calibration_clipped_layer_count": sum(
            row["calibration_clipped_count"] for row in summaries
        ),
        "calibration_degenerate_layer_count": sum(
            row["calibration_degenerate_count"] for row in summaries
        ),
        "calibration_max_relative_error": max(
            row["calibration_max_relative_error"] for row in summaries
        ),
        "calibration_scale_min": min(
            row["calibration_scale_min"] for row in summaries
        ),
        "calibration_scale_max": max(
            row["calibration_scale_max"] for row in summaries
        ),
    }


def _summarize_pair(
    lookup: dict,
    *,
    left: str,
    right: str,
    channel: str,
    family: str,
    intervention: str,
    comparison: str,
    seed: int,
) -> list[dict]:
    return _summarize_effects(
        _paired_log_effects(lookup, left_probe=left, right_probe=right),
        label=(f"{channel}:{family}:{intervention}:{comparison}"),
        metric=CHANNEL_METRICS[channel],
        metadata={
            "channel": channel,
            "family": family,
            "intervention": intervention,
            "comparison_group": comparison,
        },
        bootstrap_seed=seed,
    )


def _ensemble_effects(
    lookup: dict,
    *,
    left_probe: str | None,
    right_probe: str | None,
    random_probes: list[str],
) -> dict[tuple[int, int, str], float]:
    reference = left_probe or right_probe
    coordinates = {key[:3] for key in lookup if key[3] == reference}
    effects = {}
    for coordinate in coordinates:
        random_log = float(
            np.mean(
                [
                    math.log(max(lookup[(*coordinate, probe)], EPSILON))
                    for probe in random_probes
                ]
            )
        )
        if left_probe is not None:
            effects[coordinate] = (
                math.log(max(lookup[(*coordinate, left_probe)], EPSILON))
                - random_log
            )
        else:
            effects[coordinate] = random_log - math.log(
                max(lookup[(*coordinate, right_probe)], EPSILON)
            )
    return effects


def _context_rows(rows: list[dict]) -> dict[str, dict]:
    return {str(row["context"]): row for row in rows if row["context"] != "pooled"}


def _family_analysis(
    rows: list[dict],
    plan: dict,
    integrity: dict[tuple[str, str], bool],
) -> tuple[dict[str, list[dict]], dict]:
    outputs = {
        "group_comparisons": [],
        "random_map_comparisons": [],
        "intervention_specificity": [],
        "contrast_diagnostics": [],
    }
    result = {}
    seed = 15100
    for family_name, hypothesis in plan["families"].items():
        random_probes = list(hypothesis["random_uniform_probes"])
        family_result = {
            "susceptibility_contexts": [],
            "leverage_contexts": [],
            "specificity_contexts": [],
            "confirmed_contexts": [],
            "random_positive_counts": {},
        }
        summaries_by_channel = {}
        individual_random = {}
        for channel, metric in CHANNEL_METRICS.items():
            lookup = _lookup(rows, metric)
            uniform = hypothesis["probes"]["uniform"]
            comparisons = {
                "high>low": (uniform["high"], uniform["low"]),
                "middle>low": (uniform["middle"], uniform["low"]),
                "high>middle": (uniform["high"], uniform["middle"]),
            }
            channel_summaries = {}
            for label, (left, right) in comparisons.items():
                summary = _summarize_pair(
                    lookup,
                    left=left,
                    right=right,
                    channel=channel,
                    family=family_name,
                    intervention="uniform",
                    comparison=label,
                    seed=seed,
                )
                seed += 101
                outputs["group_comparisons"].extend(summary)
                channel_summaries[label] = _context_rows(summary)

            high_random = _summarize_effects(
                _ensemble_effects(
                    lookup,
                    left_probe=uniform["high"],
                    right_probe=None,
                    random_probes=random_probes,
                ),
                label=f"{channel}:{family_name}:uniform:high>random_ensemble",
                metric=metric,
                metadata={
                    "channel": channel,
                    "family": family_name,
                    "intervention": "uniform",
                    "comparison_group": "high>random_ensemble",
                },
                bootstrap_seed=seed,
            )
            seed += 101
            random_low = _summarize_effects(
                _ensemble_effects(
                    lookup,
                    left_probe=None,
                    right_probe=uniform["low"],
                    random_probes=random_probes,
                ),
                label=f"{channel}:{family_name}:uniform:random_ensemble>low",
                metric=metric,
                metadata={
                    "channel": channel,
                    "family": family_name,
                    "intervention": "uniform",
                    "comparison_group": "random_ensemble>low",
                },
                bootstrap_seed=seed,
            )
            seed += 101
            outputs["group_comparisons"].extend(high_random)
            outputs["group_comparisons"].extend(random_low)
            channel_summaries["high>random_ensemble"] = _context_rows(
                high_random
            )
            channel_summaries["random_ensemble>low"] = _context_rows(
                random_low
            )
            summaries_by_channel[channel] = channel_summaries

            individual_random[channel] = {"high": [], "low": []}
            for index, random_probe in enumerate(random_probes):
                for side, left, right in (
                    ("high", uniform["high"], random_probe),
                    ("low", random_probe, uniform["low"]),
                ):
                    summary = _summarize_pair(
                        lookup,
                        left=left,
                        right=right,
                        channel=channel,
                        family=family_name,
                        intervention="uniform",
                        comparison=(
                            f"high>random{index}"
                            if side == "high"
                            else f"random{index}>low"
                        ),
                        seed=seed,
                    )
                    seed += 101
                    outputs["random_map_comparisons"].extend(summary)
                    individual_random[channel][side].append(
                        _context_rows(summary)
                    )

        uniform_probes = hypothesis["probes"]["uniform"]
        scalar_random_integrity = [
            uniform_probes["high"],
            uniform_probes["middle"],
            uniform_probes["low"],
            *random_probes,
        ]
        for context in CONTEXTS:
            susceptibility = summaries_by_channel["susceptibility"]
            leverage = summaries_by_channel["leverage"]
            high_positive = sum(
                item[context]["median_effect"] > 0
                for item in individual_random["susceptibility"]["high"]
            )
            low_positive = sum(
                item[context]["median_effect"] > 0
                for item in individual_random["susceptibility"]["low"]
            )
            family_result["random_positive_counts"][context] = {
                "high_over_random": high_positive,
                "random_over_low": low_positive,
            }
            if family_name == "scalar":
                susceptibility_pass = bool(
                    _comparison_integrity(
                        scalar_random_integrity, context, integrity
                    )
                    and _qualifies(susceptibility["high>low"][context])
                    and _qualifies(susceptibility["middle>low"][context])
                    and _qualifies(
                        susceptibility["random_ensemble>low"][context]
                    )
                    and low_positive >= MIN_RANDOM_POSITIVE_MAPS
                )
                leverage_pass = bool(
                    _comparison_integrity(
                        list(uniform_probes.values()), context, integrity
                    )
                    and _qualifies(leverage["high>low"][context])
                    and _qualifies(leverage["middle>low"][context])
                )
            else:
                source_pass = bool(
                    plan["source"].get("signed_source_screen_pass", False)
                )
                susceptibility_pass = bool(
                    source_pass
                    and _comparison_integrity(
                        scalar_random_integrity, context, integrity
                    )
                    and _qualifies(susceptibility["high>low"][context])
                    and _qualifies(susceptibility["high>middle"][context])
                    and _qualifies(
                        susceptibility["high>random_ensemble"][context]
                    )
                    and high_positive >= MIN_RANDOM_POSITIVE_MAPS
                )
                leverage_pass = bool(
                    source_pass
                    and _comparison_integrity(
                        scalar_random_integrity, context, integrity
                    )
                    and _qualifies(leverage["high>low"][context])
                    and _qualifies(leverage["high>middle"][context])
                    and _qualifies(leverage["high>random_ensemble"][context])
                )
            if susceptibility_pass:
                family_result["susceptibility_contexts"].append(context)
            if leverage_pass:
                family_result["leverage_contexts"].append(context)

        pair_roles = ("middle", "low") if family_name == "scalar" else (
            "high",
            "low",
        )
        for channel, metric in CHANNEL_METRICS.items():
            lookup = _lookup(rows, metric)
            pair_effects = {}
            pair_probes = {}
            for intervention in ("uniform", "boundary", "key_shift", "value_shift"):
                cells = hypothesis["probes"][intervention]
                left = cells[pair_roles[0]]
                right = cells[pair_roles[1]]
                pair_effects[intervention] = _paired_log_effects(
                    lookup, left_probe=left, right_probe=right
                )
                pair_probes[intervention] = (left, right)
            specificity_summaries = {}
            for control in ("key_shift", "value_shift"):
                difference = {
                    key: pair_effects["uniform"][key]
                    - pair_effects[control][key]
                    for key in pair_effects["uniform"]
                }
                summary = _summarize_effects(
                    difference,
                    label=(
                        f"{channel}:{family_name}:uniform>{control}:"
                        f"{pair_roles[0]}>{pair_roles[1]}"
                    ),
                    metric="pair_log_effect_difference",
                    metadata={
                        "channel": channel,
                        "family": family_name,
                        "intervention": "uniform",
                        "control": control,
                        "comparison_group": (
                            f"{pair_roles[0]}>{pair_roles[1]}"
                        ),
                    },
                    bootstrap_seed=seed,
                    effect_definition="difference_of_log_ratios",
                )
                seed += 101
                outputs["intervention_specificity"].extend(summary)
                specificity_summaries[control] = _context_rows(summary)
            boundary_difference = {
                key: pair_effects["uniform"][key]
                - pair_effects["boundary"][key]
                for key in pair_effects["uniform"]
            }
            boundary_summary = _summarize_effects(
                boundary_difference,
                label=(
                    f"{channel}:{family_name}:uniform>boundary:"
                    f"{pair_roles[0]}>{pair_roles[1]}"
                ),
                metric="pair_log_effect_difference",
                metadata={
                    "channel": channel,
                    "family": family_name,
                    "intervention": "uniform",
                    "control": "boundary",
                    "comparison_group": f"{pair_roles[0]}>{pair_roles[1]}",
                },
                bootstrap_seed=seed,
                effect_definition="difference_of_log_ratios",
            )
            seed += 101
            outputs["contrast_diagnostics"].extend(boundary_summary)
            if channel == "leverage":
                all_specificity_probes = [
                    probe
                    for intervention in ("uniform", "key_shift", "value_shift")
                    for probe in pair_probes[intervention]
                ]
                family_result["specificity_contexts"] = [
                    context
                    for context in CONTEXTS
                    if _comparison_integrity(
                        all_specificity_probes, context, integrity
                    )
                    and all(
                        _qualifies(specificity_summaries[control][context])
                        for control in ("key_shift", "value_shift")
                    )
                ]

        family_result["confirmed_contexts"] = sorted(
            set(family_result["susceptibility_contexts"])
            & set(family_result["leverage_contexts"])
            & set(family_result["specificity_contexts"])
        )
        result[family_name] = family_result
    return outputs, result


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# v151 Signed Policy / Low-Tail Confirmation",
        "",
        "## Integrity",
        "",
        f"- Profiles: `{report['profile_count']}`",
        f"- Prompts: `{report['prompt_count']}`",
        (
            "- Native replay maximum relative RMS: "
            f"`{report['native_replay_max_relative_rms']:.6g}`"
        ),
        (
            "- Probe/context calibration pass rate: "
            f"`{report['probe_context_integrity_pass_count']}/"
            f"{report['probe_context_count']}`"
        ),
        f"- Intact contexts: `{report['intact_contexts']}`",
        f"- Invalid contexts: `{report['invalid_contexts']}`",
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(report["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "The scalar and signed branches are separate hypotheses. A passing "
        "one-step gate does not establish improved long-video generation.",
        "Contexts that fail calibration integrity remain in diagnostic CSVs "
        "but cannot satisfy any confirmation gate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    profile_dir: Path,
    *,
    probe_plan_path: Path,
    output_dir: Path,
    expected_count: int = EXPECTED_PROFILES,
) -> dict:
    plan, plan_sha256, refinement_steps = _load_plan(probe_plan_path)
    profiles, audits = _load_profiles(
        profile_dir,
        plan=plan,
        plan_sha256=plan_sha256,
        expected_count=expected_count,
        expected_refinement_steps=refinement_steps,
    )
    rows = _enrich_rows(_downstream_rows(profiles), plan)
    integrity_rows, integrity_lookup, integrity = _probe_integrity(
        rows, expected_units=expected_count
    )
    context_integrity = {
        context: all(
            integrity_lookup.get((probe["name"], context), False)
            for probe in plan["probes"]
        )
        for context in CONTEXTS
    }
    outputs, families = _family_analysis(rows, plan, integrity_lookup)
    if not integrity["native_replay_pass"]:
        for family in families.values():
            family["confirmed_contexts"] = []
    gates = {
        "g0_native_replay": integrity["native_replay_pass"],
        "g1_scalar_low_tail_susceptibility": bool(
            families["scalar"]["susceptibility_contexts"]
        ),
        "g2_scalar_policy_leverage": bool(
            families["scalar"]["leverage_contexts"]
        ),
        "g3_scalar_intervention_specificity": bool(
            families["scalar"]["specificity_contexts"]
        ),
        "g4_scalar_candidate_confirmed": bool(
            families["scalar"]["confirmed_contexts"]
        ),
        "g5_signed_source_screen": bool(
            plan["source"].get("signed_source_screen_pass", False)
        ),
        "g6_signed_group_effect": {
            "susceptibility": bool(
                families["signed"]["susceptibility_contexts"]
            ),
            "leverage": bool(families["signed"]["leverage_contexts"]),
        },
        "g7_signed_intervention_specificity": bool(
            families["signed"]["specificity_contexts"]
        ),
        "g8_signed_candidate_confirmed": bool(
            families["signed"]["confirmed_contexts"]
        ),
    }
    report = {
        "version": 1,
        "suite": SUITE,
        "profile_count": len(profiles),
        "prompt_count": EXPECTED_PROMPTS,
        "seed_replicates": list(REPLICATES),
        "contexts": list(CONTEXTS),
        "probe_plan": str(probe_plan_path),
        "probe_plan_sha256": plan_sha256,
        "downstream_observation_count": len(rows),
        **integrity,
        "gates": gates,
        "families": families,
        "context_integrity": context_integrity,
        "intact_contexts": sorted(
            context for context, passed in context_integrity.items() if passed
        ),
        "invalid_contexts": sorted(
            context for context, passed in context_integrity.items() if not passed
        ),
        "calibration_refinement_bound_hit_count": sum(
            int(row["refinement_bound_hit_count"]) for row in audits
        ),
        "minimum_random_positive_maps": MIN_RANDOM_POSITIVE_MAPS,
        "calibration_refinement_steps": refinement_steps,
        "thresholds": {
            "minimum_median_effect_log_ratio": MIN_LOG_EFFECT,
            "minimum_median_effect_ratio": math.exp(MIN_LOG_EFFECT),
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "minimum_seed_replicate_spearman": (
                MIN_SEED_REPLICATE_SPEARMAN
            ),
            "maximum_calibration_relative_error": (
                MAX_CALIBRATION_RELATIVE_ERROR
            ),
            "minimum_accepted_calibration_scale": (
                MIN_ACCEPTABLE_CALIBRATION_SCALE
            ),
            "maximum_accepted_calibration_scale": (
                MAX_ACCEPTABLE_CALIBRATION_SCALE
            ),
        },
        "source": plan["source"],
        "claim_boundary": plan["claim_boundary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_audit.csv", audits)
    _write_csv(output_dir / "downstream_observations.csv.gz", rows)
    _write_csv(output_dir / "probe_effect_summary.csv", _probe_summary(rows))
    _write_csv(output_dir / "probe_integrity.csv", integrity_rows)
    for name, values in outputs.items():
        _write_csv(output_dir / f"{name}.csv", values)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_PROFILES)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.profile_dir,
                probe_plan_path=args.probe_plan,
                output_dir=args.output_dir,
                expected_count=args.expected_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
