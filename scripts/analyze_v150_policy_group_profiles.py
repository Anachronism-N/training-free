#!/usr/bin/env python3
"""Analyze v150 policy-group confirmation profiles."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_v149_calibrated_causal_profiles import (
        CONTEXTS,
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
        CONTEXTS,
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


SUITES = {"v150_policy_group_core", "v150_policy_group_strength"}
EXPECTED_PROBES = 33
RANDOM_CONTROL_COUNT = 8
PER_LAYER_COUNT = 4
PRIMARY_POLICY = "policy_contrast"
MIN_LOG_EFFECT = math.log(1.05)
MIN_TARGET_RESPONSE = math.log(1.20)
MIN_RANDOM_POSITIVE_MAPS = 6
MAX_CALIBRATION_RELATIVE_ERROR = 0.02
MIN_ACCEPTABLE_CALIBRATION_SCALE = 0.005
MAX_ACCEPTABLE_CALIBRATION_SCALE = 50.0
CHANNEL_METRICS = {
    "susceptibility": "mean_raw_projected_relative_rms",
    "leverage": "x0_relative_rms",
}


def _expected_prompt_count(plan: dict) -> int:
    return 32 if plan["suite"].endswith("core") else 16


def _expected_profile_count(plan: dict) -> int:
    return 2 * _expected_prompt_count(plan)


def _target_key(target: float) -> str:
    return f"{float(target):.6g}"


def _head_map_signature(head_map: dict) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted(int(head) for head in head_map[str(layer)]))
        for layer in range(LAYERS)
    )


def _validate_map_contract(probes: list[dict]) -> None:
    maps = {}
    for probe in probes:
        group = str(probe.get("rank_group"))
        head_map = probe.get("head_map") or {}
        if set(head_map) != {str(layer) for layer in range(LAYERS)}:
            raise RuntimeError(f"{group} does not cover all model layers")
        if any(
            len(set(int(head) for head in head_map[str(layer)]))
            != PER_LAYER_COUNT
            for layer in range(LAYERS)
        ):
            raise RuntimeError(f"{group} is not four-head-per-layer")
        if any(
            not 0 <= int(head) < HEADS
            for layer in range(LAYERS)
            for head in head_map[str(layer)]
        ):
            raise RuntimeError(f"{group} contains an invalid head index")
        signature = _head_map_signature(head_map)
        if group in maps and maps[group] != signature:
            raise RuntimeError(f"{group} changes across v150 probes")
        maps[group] = signature
    expected_groups = {
        "top4",
        "bottom4",
        "middle4",
        *{f"random{index}" for index in range(RANDOM_CONTROL_COUNT)},
    }
    if set(maps) != expected_groups:
        raise RuntimeError("v150 group map set is incomplete")
    if len(set(maps.values())) != len(maps):
        raise RuntimeError("v150 contains duplicate global group maps")
    for layer in range(LAYERS):
        fixed = [
            set(maps[group][layer])
            for group in ("top4", "bottom4", "middle4")
        ]
        if any(fixed[i] & fixed[j] for i in range(3) for j in range(i)):
            raise RuntimeError(f"layer {layer} fixed rank groups overlap")
        if set().union(*fixed) != set(range(HEADS)):
            raise RuntimeError(f"layer {layer} fixed groups do not partition")
        random_sets = [
            set(maps[f"random{index}"][layer])
            for index in range(RANDOM_CONTROL_COUNT)
        ]
        if len({frozenset(values) for values in random_sets}) != len(
            random_sets
        ):
            raise RuntimeError(f"layer {layer} repeats a random map")
        if any(values in fixed for values in random_sets):
            raise RuntimeError(f"layer {layer} random map equals a rank group")
        usage = Counter(head for values in random_sets for head in values)
        if set(usage) != set(range(HEADS)) or max(usage.values()) - min(
            usage.values()
        ) > 1:
            raise RuntimeError(f"layer {layer} random usage is unbalanced")


def _load_plan(path: Path) -> tuple[dict, str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(plan.get("version", -1)) != 1
        or int(plan.get("layers", -1)) != LAYERS
        or int(plan.get("heads", -1)) != HEADS
        or plan.get("suite") not in SUITES
    ):
        raise RuntimeError("v150 probe plan has an invalid model contract")
    probes = plan.get("probes") or []
    names = [str(probe.get("name")) for probe in probes]
    if (
        len(probes) != EXPECTED_PROBES
        or len(names) != len(set(names))
        or int(plan.get("random_control_count", -1))
        != RANDOM_CONTROL_COUNT
    ):
        raise RuntimeError("v150 probe plan has an invalid probe count")
    _validate_map_contract(probes)
    comparisons = plan.get("comparisons") or []
    if len(comparisons) != 3:
        raise RuntimeError("v150 requires exactly three comparison cells")
    plan_targets = {float(value) for value in plan["calibration_targets"]}
    expected_targets = (
        {0.02} if plan["suite"].endswith("core") else {0.01, 0.02, 0.05}
    )
    if plan_targets != expected_targets:
        raise RuntimeError("v150 calibration target grid changed")
    probe_names = set(names)
    for probe in probes:
        calibration = probe.get("calibration") or {}
        if (
            calibration.get("mode") != "projected_relative_rms"
            or float(calibration.get("min_scale", -1)) != 0.001
            or float(calibration.get("max_scale", -1)) != 50.0
            or float(calibration.get("target", -1)) not in expected_targets
        ):
            raise RuntimeError("v150 calibration contract changed")
        if probe["policy"] == PRIMARY_POLICY and probe.get(
            "policy_args"
        ) != {"left": "uniform8", "right": "recent8"}:
            raise RuntimeError("v150 policy contrast changed")
    for comparison in comparisons:
        referenced = {
            comparison["top_probe"],
            comparison["bottom_probe"],
            comparison["middle_probe"],
            *comparison["random_probes"],
        }
        if (
            len(comparison["random_probes"]) != RANDOM_CONTROL_COUNT
            or not referenced <= probe_names
        ):
            raise RuntimeError("v150 comparison references invalid probes")
    return plan, _canonical_digest(plan)


def _load_profiles(
    directory: Path,
    *,
    plan: dict,
    plan_sha256: str,
    expected_count: int,
) -> tuple[list[dict], list[dict]]:
    import torch

    if expected_count != _expected_profile_count(plan):
        raise RuntimeError("v150 expected profile count disagrees with suite")
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v150 profiles, found {len(paths)}"
        )
    prompt_count = _expected_prompt_count(plan)
    expected_downstream = 2 * (len(plan["probes"]) + 1)
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
            raise RuntimeError(f"duplicate v150 profile coordinate: {key}")
        seen.add(key)
        if (
            int(payload.get("version", -1)) != PROFILE_VERSION
            or str(job.get("kind")) != plan["suite"]
            or not (
                int(metadata["seed"])
                == int(job["seed"])
                == int(job["reference_seed"])
            )
        ):
            raise RuntimeError(f"{path} violates the v150 profile contract")
        records = payload.get("records") or []
        downstream = payload.get("downstream_probe_records") or []
        if (
            metadata.get("incomplete_calls")
            or int(metadata.get("captured_calls", -1)) != 2
            or int(metadata.get("record_count", -1)) != 2 * LAYERS
            or len(records) != 2 * LAYERS
            or len(downstream) != expected_downstream
            or int(payload.get("downstream_probe_expected_count", -1))
            != expected_downstream
        ):
            raise RuntimeError(f"{path} has an invalid capture grid")
        state_layers = Counter(
            (
                str(row["mode"]),
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in records
        )
        if len(state_layers) != 2 or set(state_layers.values()) != {LAYERS}:
            raise RuntimeError(f"{path} has an invalid state/layer grid")
        plan_meta = metadata.get("downstream_probe_plan") or {}
        if str(plan_meta.get("sha256")) != plan_sha256:
            raise RuntimeError(f"{path} uses a different v150 probe plan")
        probe_grid = Counter(
            (
                "clean"
                if str(row["mode"]) == "clean"
                else f"noisy_t{int(row['nominal_timestep'])}",
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
                "path": str(path),
            }
        )
    expected_grid = {
        (prompt, replicate)
        for prompt in range(prompt_count)
        for replicate in REPLICATES
    }
    if seen != expected_grid:
        raise RuntimeError("v150 prompt/seed grid is incomplete")
    for prompt in range(prompt_count):
        if prompt_seeds[(prompt, 0)] == prompt_seeds[(prompt, 1)]:
            raise RuntimeError(f"v150 prompt {prompt} repeats its seed")
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
                }
            )
            continue
        probe = probes.get(row["probe_name"])
        if probe is None:
            raise RuntimeError(f"unknown v150 probe {row['probe_name']}")
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
            }
        )
    return enriched


def _probe_integrity(
    rows: list[dict], *, expected_units: int
) -> tuple[list[dict], dict[tuple[str, str], bool], dict]:
    native = [row for row in rows if row["probe_name"] == "native_replay"]
    replay_max = max(
        max(row["flow_relative_rms"], row["x0_relative_rms"])
        for row in native
    )
    if len(native) != expected_units * len(CONTEXTS):
        raise RuntimeError("v150 native replay grid is incomplete")
    grouped = defaultdict(list)
    for row in rows:
        if row["probe_name"] != "native_replay":
            grouped[(row["probe_name"], row["context"])].append(row)
    summaries = []
    lookup = {}
    for (probe, context), values in sorted(grouped.items()):
        if len(values) != expected_units:
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
        shift_valid = all(
            row["min_shifted_old_frames"] > 1 for row in values
        ) if values[0]["policy"] in {"key_shift", "value_shift"} else True
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


def _random_ensemble_effects(
    lookup: dict[tuple, float],
    *,
    top_probe: str,
    random_probes: list[str],
) -> dict[tuple[int, int, str], float]:
    if len(random_probes) != RANDOM_CONTROL_COUNT:
        raise RuntimeError("v150 requires eight random controls")
    coordinates = {key[:3] for key in lookup if key[3] == top_probe}
    effects = {}
    for coordinate in coordinates:
        top = math.log(max(lookup[(*coordinate, top_probe)], EPSILON))
        random_log = float(
            np.mean(
                [
                    math.log(max(lookup[(*coordinate, probe)], EPSILON))
                    for probe in random_probes
                ]
            )
        )
        effects[coordinate] = top - random_log
    return effects


def _ensemble_metric(
    lookup: dict[tuple, float], random_probes: list[str]
) -> dict[tuple[int, int, str], float]:
    coordinates = {key[:3] for key in lookup if key[3] == random_probes[0]}
    return {
        coordinate: math.exp(
            float(
                np.mean(
                    [
                        math.log(
                            max(lookup[(*coordinate, probe)], EPSILON)
                        )
                        for probe in random_probes
                    ]
                )
            )
        )
        for coordinate in coordinates
    }


def _summarize_pair(
    effects: dict[tuple[int, int, str], float],
    *,
    channel: str,
    policy: str,
    target: float,
    control: str,
    seed: int,
) -> list[dict]:
    return _summarize_effects(
        effects,
        label=(
            f"{channel}:{policy}:top4>{control}:"
            f"target={_target_key(target)}"
        ),
        metric=CHANNEL_METRICS[channel],
        metadata={
            "channel": channel,
            "axis": "policy",
            "policy": policy,
            "target": float(target),
            "control": control,
        },
        bootstrap_seed=seed,
    )


def _qualifies(row: dict, *, minimum: float = MIN_LOG_EFFECT) -> bool:
    return bool(
        row["context"] != "pooled"
        and row["median_effect"] >= minimum
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= 0.65
        )
        and row["seed_replicate_spearman"] >= 0.30
    )


def _context_rows(rows: list[dict]) -> dict[str, dict]:
    return {row["context"]: row for row in rows if row["context"] != "pooled"}


def _comparison_integrity(
    probe_names: list[str],
    context: str,
    integrity: dict[tuple[str, str], bool],
) -> bool:
    return all(integrity.get((probe, context), False) for probe in probe_names)


def _group_analysis(
    rows: list[dict],
    plan: dict,
    integrity: dict[tuple[str, str], bool],
) -> tuple[list[dict], list[dict], dict, dict]:
    comparisons = []
    random_rows = []
    group_contexts = {
        channel: defaultdict(dict) for channel in CHANNEL_METRICS
    }
    random_positive_counts = {
        channel: defaultdict(dict) for channel in CHANNEL_METRICS
    }
    seed = 15000
    for channel, metric in CHANNEL_METRICS.items():
        metric_lookup = _lookup(rows, metric)
        for hypothesis in plan["comparisons"]:
            policy = hypothesis["policy"]
            target = float(hypothesis["target"])
            top = hypothesis["top_probe"]
            control_effects = {
                "bottom4": _paired_log_effects(
                    metric_lookup,
                    left_probe=top,
                    right_probe=hypothesis["bottom_probe"],
                ),
                "middle4": _paired_log_effects(
                    metric_lookup,
                    left_probe=top,
                    right_probe=hypothesis["middle_probe"],
                ),
                "random_ensemble": _random_ensemble_effects(
                    metric_lookup,
                    top_probe=top,
                    random_probes=hypothesis["random_probes"],
                ),
            }
            summary_by_control = {}
            for control, effects in control_effects.items():
                summary = _summarize_pair(
                    effects,
                    channel=channel,
                    policy=policy,
                    target=target,
                    control=control,
                    seed=seed,
                )
                seed += 101
                comparisons.extend(summary)
                summary_by_control[control] = _context_rows(summary)

            individual = []
            for index, random_probe in enumerate(hypothesis["random_probes"]):
                effects = _paired_log_effects(
                    metric_lookup,
                    left_probe=top,
                    right_probe=random_probe,
                )
                summary = _summarize_pair(
                    effects,
                    channel=channel,
                    policy=policy,
                    target=target,
                    control=f"random{index}",
                    seed=seed,
                )
                seed += 101
                random_rows.extend(summary)
                individual.append(_context_rows(summary))

            probe_names = [
                top,
                hypothesis["bottom_probe"],
                hypothesis["middle_probe"],
                *hypothesis["random_probes"],
            ]
            target_key = _target_key(target)
            passing_contexts = []
            counts = {}
            for context in CONTEXTS:
                positive_count = sum(
                    item[context]["median_effect"] > 0 for item in individual
                )
                counts[context] = positive_count
                if (
                    _comparison_integrity(probe_names, context, integrity)
                    and all(
                        _qualifies(summary_by_control[control][context])
                        for control in (
                            "bottom4",
                            "middle4",
                            "random_ensemble",
                        )
                    )
                    and positive_count >= MIN_RANDOM_POSITIVE_MAPS
                ):
                    passing_contexts.append(context)
            group_contexts[channel][policy][target_key] = passing_contexts
            random_positive_counts[channel][policy][target_key] = counts
    normalized_contexts = {
        channel: {
            policy: dict(targets) for policy, targets in policies.items()
        }
        for channel, policies in group_contexts.items()
    }
    normalized_counts = {
        channel: {
            policy: dict(targets) for policy, targets in policies.items()
        }
        for channel, policies in random_positive_counts.items()
    }
    return comparisons, random_rows, normalized_contexts, normalized_counts


def _core_specificity(
    rows: list[dict],
    plan: dict,
    integrity: dict[tuple[str, str], bool],
) -> tuple[list[dict], dict]:
    hypotheses = {item["policy"]: item for item in plan["comparisons"]}
    if set(hypotheses) != {"key_shift", "value_shift", PRIMARY_POLICY}:
        raise RuntimeError("v150 core intervention grid is incomplete")
    result_rows = []
    contexts_by_channel = {}
    seed = 16000
    for channel, metric in CHANNEL_METRICS.items():
        metric_lookup = _lookup(rows, metric)
        effects = {
            policy: _paired_log_effects(
                metric_lookup,
                left_probe=hypothesis["top_probe"],
                right_probe=hypothesis["bottom_probe"],
            )
            for policy, hypothesis in hypotheses.items()
        }
        summaries = {}
        for off_policy in ("key_shift", "value_shift"):
            difference = {
                key: effects[PRIMARY_POLICY][key] - effects[off_policy][key]
                for key in effects[PRIMARY_POLICY]
            }
            summary = _summarize_effects(
                difference,
                label=(
                    f"{channel}:policy_contrast>{off_policy}:"
                    f"target={_target_key(0.02)}"
                ),
                metric="top_bottom_log_effect",
                metadata={
                    "channel": channel,
                    "axis": "policy",
                    "policy": PRIMARY_POLICY,
                    "target": 0.02,
                    "control": off_policy,
                },
                bootstrap_seed=seed,
                effect_definition="difference_of_log_ratios",
            )
            seed += 101
            result_rows.extend(summary)
            summaries[off_policy] = _context_rows(summary)
        all_probes = [
            hypotheses[policy][probe]
            for policy in hypotheses
            for probe in ("top_probe", "bottom_probe")
        ]
        contexts_by_channel[channel] = [
            context
            for context in CONTEXTS
            if _comparison_integrity(all_probes, context, integrity)
            and all(
                _qualifies(summaries[off_policy][context])
                for off_policy in ("key_shift", "value_shift")
            )
        ]
    return result_rows, contexts_by_channel


def _target_response_analysis(
    rows: list[dict],
    plan: dict,
    integrity: dict[tuple[str, str], bool],
) -> tuple[list[dict], dict]:
    by_target = {
        float(item["target"]): item for item in plan["comparisons"]
    }
    if set(by_target) != {0.01, 0.02, 0.05}:
        raise RuntimeError("v150 strength target grid is incomplete")
    metric_lookup = _lookup(rows, "x0_relative_rms")
    low = by_target[0.01]
    high = by_target[0.05]
    result_rows = []
    summaries = {}
    seed = 17000
    direct_groups = {
        "top4": "top_probe",
        "bottom4": "bottom_probe",
        "middle4": "middle_probe",
    }
    for group, field in direct_groups.items():
        effects = _paired_log_effects(
            metric_lookup,
            left_probe=high[field],
            right_probe=low[field],
        )
        summary = _summarize_effects(
            effects,
            label=f"leverage:{group}:target0.05>0.01",
            metric="x0_relative_rms",
            metadata={
                "channel": "leverage",
                "axis": "policy",
                "policy": PRIMARY_POLICY,
                "target": 0.05,
                "control": "target0.01",
                "rank_group": group,
            },
            bootstrap_seed=seed,
            effect_definition="target_log_ratio",
        )
        seed += 101
        result_rows.extend(summary)
        summaries[group] = _context_rows(summary)
    high_random = _ensemble_metric(metric_lookup, high["random_probes"])
    low_random = _ensemble_metric(metric_lookup, low["random_probes"])
    random_effects = {
        key: math.log(max(high_random[key], EPSILON) / max(low_random[key], EPSILON))
        for key in high_random
    }
    random_summary = _summarize_effects(
        random_effects,
        label="leverage:random_ensemble:target0.05>0.01",
        metric="x0_relative_rms",
        metadata={
            "channel": "leverage",
            "axis": "policy",
            "policy": PRIMARY_POLICY,
            "target": 0.05,
            "control": "target0.01",
            "rank_group": "random_ensemble",
        },
        bootstrap_seed=seed,
        effect_definition="target_log_ratio",
    )
    result_rows.extend(random_summary)
    summaries["random_ensemble"] = _context_rows(random_summary)
    all_probes = [
        low["top_probe"],
        low["bottom_probe"],
        low["middle_probe"],
        *low["random_probes"],
        high["top_probe"],
        high["bottom_probe"],
        high["middle_probe"],
        *high["random_probes"],
    ]
    contexts = [
        context
        for context in CONTEXTS
        if _comparison_integrity(all_probes, context, integrity)
        and all(
            _qualifies(summaries[group][context], minimum=MIN_TARGET_RESPONSE)
            for group in ("top4", "bottom4", "random_ensemble")
        )
    ]
    return result_rows, {"leverage": contexts}


def _analyze(
    rows: list[dict],
    plan: dict,
    integrity_lookup: dict[tuple[str, str], bool],
) -> tuple[dict[str, list[dict]], dict]:
    group_rows, random_rows, group_contexts, random_counts = _group_analysis(
        rows, plan, integrity_lookup
    )
    outputs = {
        "group_comparisons": group_rows,
        "random_map_comparisons": random_rows,
    }
    if plan["suite"].endswith("core"):
        specificity_rows, specificity_contexts = _core_specificity(
            rows, plan, integrity_lookup
        )
        outputs["intervention_specificity"] = specificity_rows
        primary = {
            channel: group_contexts[channel][PRIMARY_POLICY][
                _target_key(0.02)
            ]
            for channel in CHANNEL_METRICS
        }
        confirmed = {
            channel: sorted(
                set(primary[channel]) & set(specificity_contexts[channel])
            )
            for channel in CHANNEL_METRICS
        }
        analysis = {
            "g1_count_matched_group_effect": {
                channel: bool(contexts)
                for channel, contexts in primary.items()
            },
            "g2_intervention_specificity": {
                channel: bool(contexts)
                for channel, contexts in specificity_contexts.items()
            },
            "g3_policy_group_confirmed": {
                channel: bool(contexts)
                for channel, contexts in confirmed.items()
            },
            "qualifying_group_contexts": group_contexts,
            "qualifying_primary_contexts": primary,
            "qualifying_specificity_contexts": specificity_contexts,
            "confirmed_contexts": confirmed,
            "random_positive_map_counts": random_counts,
        }
    else:
        target_rows, target_contexts = _target_response_analysis(
            rows, plan, integrity_lookup
        )
        outputs["target_response"] = target_rows
        robust = {}
        for channel in CHANNEL_METRICS:
            robust[channel] = [
                context
                for context in CONTEXTS
                if sum(
                    context
                    in group_contexts[channel][PRIMARY_POLICY][
                        _target_key(target)
                    ]
                    for target in (0.01, 0.02, 0.05)
                )
                >= 2
            ]
        confirmed = {
            "leverage": sorted(
                set(robust["leverage"]) & set(target_contexts["leverage"])
            )
        }
        analysis = {
            "g1_group_effect_at_multiple_targets": {
                channel: bool(contexts)
                for channel, contexts in robust.items()
            },
            "g2_target_response_sanity": {
                "leverage": bool(target_contexts["leverage"])
            },
            "g3_strength_robust_policy_group": {
                "leverage": bool(confirmed["leverage"])
            },
            "qualifying_group_contexts": group_contexts,
            "robust_target_contexts": robust,
            "target_response_contexts": target_contexts,
            "confirmed_contexts": confirmed,
            "random_positive_map_counts": random_counts,
        }
    return outputs, analysis


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# v150 Policy-Group Confirmation Results",
        "",
        "## Integrity",
        "",
        f"- Suite: `{report['suite']}`",
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
        (
            "- Calibration clipped / degenerate layers: "
            f"`{report['calibration_clipped_layer_count']} / "
            f"{report['calibration_degenerate_layer_count']}`"
        ),
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(report["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation Boundary",
        "",
        (
            "A policy group is confirmed only when top4 beats bottom4, "
            "middle4, and the eight-map random ensemble in one intact "
            "context. Core confirmation additionally requires policy "
            "contrast to exceed both K- and V-shift separations."
        ),
        "",
        (
            "These are one-step frame-117 causal measurements. They do not "
            "establish trajectory-level video quality."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    profile_dir: Path,
    *,
    probe_plan_path: Path,
    output_dir: Path,
    expected_count: int,
) -> dict:
    plan, plan_sha256 = _load_plan(probe_plan_path)
    profiles, audits = _load_profiles(
        profile_dir,
        plan=plan,
        plan_sha256=plan_sha256,
        expected_count=expected_count,
    )
    rows = _enrich_rows(_downstream_rows(profiles), plan)
    integrity_rows, integrity_lookup, integrity = _probe_integrity(
        rows, expected_units=expected_count
    )
    outputs, analysis = _analyze(rows, plan, integrity_lookup)
    gates = {
        "g0_native_replay": integrity["native_replay_pass"],
        **{
            key: value for key, value in analysis.items() if key.startswith("g")
        },
    }
    report = {
        "suite": plan["suite"],
        "profile_count": len(profiles),
        "prompt_count": _expected_prompt_count(plan),
        "seed_replicates": list(REPLICATES),
        "probe_plan": str(probe_plan_path),
        "probe_plan_sha256": plan_sha256,
        "downstream_observation_count": len(rows),
        **integrity,
        "gates": gates,
        **{key: value for key, value in analysis.items() if not key.startswith("g")},
        "minimum_qualifying_median_log_effect": MIN_LOG_EFFECT,
        "minimum_target_response_log_effect": MIN_TARGET_RESPONSE,
        "minimum_random_positive_maps": MIN_RANDOM_POSITIVE_MAPS,
        "maximum_calibration_relative_error": MAX_CALIBRATION_RELATIVE_ERROR,
        "acceptable_calibration_scale_range": [
            MIN_ACCEPTABLE_CALIBRATION_SCALE,
            MAX_ACCEPTABLE_CALIBRATION_SCALE,
        ],
        "source": plan["source"],
        "claim_boundary": (
            "Passing v150 supports a count-matched collective policy-group "
            "effect at a measured denoising context, not a binary taxonomy "
            "or improved long-video generation."
        ),
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
    parser.add_argument("--expected-count", type=int, required=True)
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
