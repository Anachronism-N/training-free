#!/usr/bin/env python3
"""Analyze v149 calibrated susceptibility/leverage head profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LAYERS = 30
HEADS = 12
REPLICATES = (0, 1)
PROFILE_VERSION = 8
CONTEXTS = ("noisy_t1000", "noisy_t500")
EPSILON = 1e-10
MIN_LOG_EFFECT = math.log(1.05)
MAX_CALIBRATION_RELATIVE_ERROR = 0.02
MIN_ACCEPTABLE_CALIBRATION_SCALE = 0.02
MAX_ACCEPTABLE_CALIBRATION_SCALE = 50.0
SUITES = {"v149_calibrated_core", "v149_calibrated_dose"}
CHANNEL_METRICS = {
    "susceptibility": "mean_raw_projected_relative_rms",
    "leverage": "x0_relative_rms",
}


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
        while (
            end < values.size
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x_rank = _rankdata(x[finite])
    y_rank = _rankdata(y[finite])
    if x_rank.std() <= 1e-12 or y_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _bootstrap_ci(
    values: dict[int, float],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    array = np.asarray(
        [values[key] for key in sorted(values)], dtype=np.float64
    )
    if array.size < 2:
        value = float(array.mean()) if array.size else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    boot = array[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(
        np.quantile(boot, 0.975)
    )


def _context_key(record: dict) -> str:
    if str(record["mode"]) == "clean":
        return "clean"
    return f"noisy_t{int(record['nominal_timestep'])}"


def _canonical_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_plan(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("version", -1)) != 1
        or int(payload.get("layers", -1)) != LAYERS
        or int(payload.get("heads", -1)) != HEADS
        or payload.get("suite") not in SUITES
    ):
        raise RuntimeError("v149 probe plan has an invalid model contract")
    probes = payload.get("probes") or []
    names = [str(probe.get("name")) for probe in probes]
    expected = 30 if payload["suite"].endswith("core") else 24
    if (
        len(probes) != expected
        or len(names) != len(set(names))
        or any(probe.get("calibration") is None for probe in probes)
    ):
        raise RuntimeError("v149 probe plan has an invalid probe contract")
    return payload, _canonical_digest(payload)


def _expected_prompt_count(plan: dict) -> int:
    return 32 if plan["suite"].endswith("core") else 16


def _load_profiles(
    directory: Path,
    *,
    plan: dict,
    plan_sha256: str,
    expected_count: int,
) -> tuple[list[dict], list[dict]]:
    import torch

    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v149 profiles, found {len(paths)}"
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
            raise RuntimeError(f"duplicate v149 profile coordinate: {key}")
        seen.add(key)
        if (
            int(payload.get("version", -1)) != PROFILE_VERSION
            or str(job.get("kind")) != plan["suite"]
        ):
            raise RuntimeError(f"{path} is not a {plan['suite']} profile")
        if not (
            int(metadata["seed"])
            == int(job["seed"])
            == int(job["reference_seed"])
        ):
            raise RuntimeError(f"{path} violates the seed contract")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete captures")
        records = payload.get("records") or []
        downstream = payload.get("downstream_probe_records") or []
        if (
            int(metadata.get("captured_calls", -1)) != 2
            or int(metadata.get("record_count", -1)) != 2 * LAYERS
            or len(records) != 2 * LAYERS
            or len(downstream) != expected_downstream
            or int(payload.get("downstream_probe_expected_count", -1))
            != expected_downstream
        ):
            raise RuntimeError(f"{path} has an invalid capture grid")
        plan_meta = metadata.get("downstream_probe_plan") or {}
        if str(plan_meta.get("sha256")) != plan_sha256:
            raise RuntimeError(f"{path} uses a different probe plan")
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
        probe_grid = Counter(
            (_context_key(row), str(row["probe_name"]))
            for row in downstream
        )
        if (
            set(context for context, _ in probe_grid) != set(CONTEXTS)
            or len(probe_grid) != expected_downstream
            or set(probe_grid.values()) != {1}
        ):
            raise RuntimeError(f"{path} has an invalid probe grid")
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
        raise RuntimeError(
            f"incomplete v149 prompt/seed grid: "
            f"missing={sorted(expected_grid - seen)}"
        )
    for prompt in range(prompt_count):
        if prompt_seeds[(prompt, 0)] == prompt_seeds[(prompt, 1)]:
            raise RuntimeError(f"prompt {prompt} repeats its seed")
    return profiles, audits


def _tensor_values(values: list[object]) -> list[float]:
    return [float(value) for value in values]


def _contrast_valid(metadata: dict) -> bool:
    contrast = metadata.get("policy_contrast")
    indices = metadata.get("frame_indices")
    if not isinstance(contrast, dict) or not isinstance(indices, dict):
        return False
    left = str(contrast.get("left") or "")
    right = str(contrast.get("right") or "")
    if left != "uniform8" or right != "recent8":
        return False
    if left not in indices or right not in indices:
        return False
    left_values = np.asarray(indices[left]).reshape(-1)
    right_values = np.asarray(indices[right]).reshape(-1)
    return bool(
        left_values.size == right_values.size == 8
        and not np.array_equal(left_values, right_values)
    )


def _downstream_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    for payload in profiles:
        job = payload["job"]
        for record in payload["downstream_probe_records"]:
            layer_metadata = record.get("layer_metadata") or {}
            is_native = str(record["probe_name"]) == "native_replay"
            calibrated = list(layer_metadata.values())
            if not is_native:
                required = {
                    "raw_replacement_relative_rms",
                    "replacement_relative_rms",
                    "raw_projected_replacement_relative_rms",
                    "projected_replacement_relative_rms",
                    "calibration_target",
                    "calibration_scale",
                    "calibration_clipped",
                    "calibration_degenerate",
                    "calibration_relative_error",
                }
                if len(calibrated) != LAYERS or any(
                    not required.issubset(metadata)
                    for metadata in calibrated
                ):
                    raise RuntimeError(
                        f"{job['job_id']}/{record['probe_name']} lacks "
                        "complete layer calibration metadata"
                    )
            raw_local = _tensor_values(
                [
                    value["raw_replacement_relative_rms"]
                    for value in calibrated
                ]
            )
            applied_local = _tensor_values(
                [value["replacement_relative_rms"] for value in calibrated]
            )
            raw_projected = _tensor_values(
                [
                    value["raw_projected_replacement_relative_rms"]
                    for value in calibrated
                ]
            )
            applied_projected = _tensor_values(
                [
                    value["projected_replacement_relative_rms"]
                    for value in calibrated
                ]
            )
            scales = _tensor_values(
                [value["calibration_scale"] for value in calibrated]
            )
            errors = _tensor_values(
                [value["calibration_relative_error"] for value in calibrated]
            )
            targets = _tensor_values(
                [value["calibration_target"] for value in calibrated]
            )
            shifted = [
                int(value.get("shifted_old_frames", 0))
                for value in calibrated
                if "shifted_old_frames" in value
            ]
            contrast_rows = [
                _contrast_valid(value)
                for value in calibrated
                if record["policy"] == "policy_contrast"
            ]
            rows.append(
                {
                    "prompt_slot": int(job["prompt_slot"]),
                    "source_prompt_index": int(
                        job["source_prompt_index"]
                    ),
                    "seed_replicate": int(job["seed_replicate"]),
                    "seed": int(job["seed"]),
                    "context": _context_key(record),
                    "mode": str(record["mode"]),
                    "nominal_timestep": int(record["nominal_timestep"]),
                    "probe_name": str(record["probe_name"]),
                    "policy": str(record["policy"]),
                    "group": str(record["group"]),
                    "selected_head_count": int(
                        record["selected_head_count"]
                    ),
                    "flow_relative_rms": float(
                        record["flow_metrics"]["relative_rms"]
                    ),
                    "x0_relative_rms": float(
                        record["x0_metrics"]["relative_rms"]
                    ),
                    "mean_raw_local_relative_rms": (
                        float(np.mean(raw_local)) if raw_local else 0.0
                    ),
                    "mean_applied_local_relative_rms": (
                        float(np.mean(applied_local))
                        if applied_local
                        else 0.0
                    ),
                    "mean_raw_projected_relative_rms": (
                        float(np.mean(raw_projected))
                        if raw_projected
                        else 0.0
                    ),
                    "mean_projected_relative_rms": (
                        float(np.mean(applied_projected))
                        if applied_projected
                        else 0.0
                    ),
                    "calibration_scale_mean": (
                        float(np.mean(scales)) if scales else 1.0
                    ),
                    "calibration_scale_min": (
                        min(scales) if scales else 1.0
                    ),
                    "calibration_scale_max": (
                        max(scales) if scales else 1.0
                    ),
                    "calibration_relative_error_max": (
                        max(errors) if errors else 0.0
                    ),
                    "calibration_target_min": (
                        min(targets) if targets else 0.0
                    ),
                    "calibration_target_max": (
                        max(targets) if targets else 0.0
                    ),
                    "calibration_clipped_count": sum(
                        bool(value["calibration_clipped"])
                        for value in calibrated
                    ),
                    "calibration_degenerate_count": sum(
                        bool(value["calibration_degenerate"])
                        for value in calibrated
                    ),
                    "calibrated_layer_count": len(calibrated),
                    "min_shifted_old_frames": (
                        min(shifted) if shifted else 0
                    ),
                    "policy_contrast_valid": (
                        int(all(contrast_rows))
                        if contrast_rows
                        else int(record["policy"] != "policy_contrast")
                    ),
                }
            )
    return rows


def _lookup(rows: list[dict], metric: str) -> dict[tuple, float]:
    result = {}
    for row in rows:
        key = (
            int(row["prompt_slot"]),
            int(row["seed_replicate"]),
            str(row["context"]),
            str(row["probe_name"]),
        )
        if key in result:
            raise RuntimeError(f"duplicate downstream observation: {key}")
        value = float(row[metric])
        if row["probe_name"] != "native_replay" and value <= 0:
            raise RuntimeError(f"{metric} is non-positive for {key}")
        result[key] = value
    return result


def _paired_log_effects(
    lookup: dict[tuple, float],
    *,
    left_probe: str,
    right_probe: str,
) -> dict[tuple[int, int, str], float]:
    effects = {}
    coordinates = {
        key[:3] for key in lookup if key[3] == left_probe
    }
    for coordinate in coordinates:
        left = lookup[(*coordinate, left_probe)]
        right = lookup[(*coordinate, right_probe)]
        effects[coordinate] = math.log(
            max(left, EPSILON) / max(right, EPSILON)
        )
    return effects


def _random_ensemble_effects(
    lookup: dict[tuple, float],
    *,
    top_probe: str,
    random_probes: list[str],
) -> dict[tuple[int, int, str], float]:
    if len(random_probes) != 2:
        raise RuntimeError("v149 requires exactly two random controls")
    effects = {}
    coordinates = {key[:3] for key in lookup if key[3] == top_probe}
    for coordinate in coordinates:
        top = math.log(max(lookup[(*coordinate, top_probe)], EPSILON))
        random_log = np.mean(
            [
                math.log(max(lookup[(*coordinate, probe)], EPSILON))
                for probe in random_probes
            ]
        )
        effects[coordinate] = top - float(random_log)
    return effects


def _summarize_effects(
    effects: dict[tuple[int, int, str], float],
    *,
    label: str,
    metric: str,
    metadata: dict,
    bootstrap_seed: int,
    effect_definition: str = "paired_log_ratio",
) -> list[dict]:
    rows = []
    contexts = sorted({key[2] for key in effects})
    prompt_ids = sorted({key[0] for key in effects})
    for context in [*contexts, "pooled"]:
        selected = {
            key: value
            for key, value in effects.items()
            if context == "pooled" or key[2] == context
        }
        prompt_values = {
            prompt: float(
                np.mean(
                    [
                        value
                        for (item_prompt, _, _), value in selected.items()
                        if item_prompt == prompt
                    ]
                )
            )
            for prompt in prompt_ids
        }
        low, high = _bootstrap_ci(
            prompt_values,
            seed=bootstrap_seed + len(rows) * 17,
        )
        values = np.asarray(list(selected.values()), dtype=np.float64)
        replicate_values = {
            replicate: [
                float(
                    np.mean(
                        [
                            value
                            for (
                                item_prompt,
                                item_replicate,
                                _,
                            ), value in selected.items()
                            if item_prompt == prompt
                            and item_replicate == replicate
                        ]
                    )
                )
                for prompt in prompt_ids
            ]
            for replicate in REPLICATES
        }
        rows.append(
            {
                "comparison": label,
                **metadata,
                "metric": metric,
                "effect_definition": effect_definition,
                "context": context,
                "unit_count": len(values),
                "prompt_count": len(prompt_values),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "positive_fraction": float((values > 0).mean()),
                "prompt_bootstrap_mean_ci_low": low,
                "prompt_bootstrap_mean_ci_high": high,
                "seed_replicate_spearman": _spearman(
                    replicate_values[0], replicate_values[1]
                ),
            }
        )
    return rows


def _qualifies(row: dict) -> bool:
    return bool(
        row["context"] != "pooled"
        and row["median_effect"] >= MIN_LOG_EFFECT
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= 0.65
        )
        and row["seed_replicate_spearman"] >= 0.30
    )


def _probe_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["probe_name"], row["context"])].append(row)
    summaries = []
    for (probe, context), values in sorted(grouped.items()):
        summaries.append(
            {
                "probe_name": probe,
                "context": context,
                "unit_count": len(values),
                "selected_head_count": values[0]["selected_head_count"],
                "mean_x0_relative_rms": float(
                    np.mean([row["x0_relative_rms"] for row in values])
                ),
                "mean_raw_projected_relative_rms": float(
                    np.mean(
                        [
                            row["mean_raw_projected_relative_rms"]
                            for row in values
                        ]
                    )
                ),
                "mean_projected_relative_rms": float(
                    np.mean(
                        [
                            row["mean_projected_relative_rms"]
                            for row in values
                        ]
                    )
                ),
                "mean_calibration_scale": float(
                    np.mean(
                        [row["calibration_scale_mean"] for row in values]
                    )
                ),
                "max_calibration_relative_error": max(
                    row["calibration_relative_error_max"]
                    for row in values
                ),
                "calibration_clipped_count": sum(
                    row["calibration_clipped_count"] for row in values
                ),
                "calibration_degenerate_count": sum(
                    row["calibration_degenerate_count"] for row in values
                ),
            }
        )
    return summaries


def _integrity(rows: list[dict]) -> dict:
    native = [row for row in rows if row["probe_name"] == "native_replay"]
    interventions = [
        row for row in rows if row["probe_name"] != "native_replay"
    ]
    replay_max = max(
        max(row["flow_relative_rms"], row["x0_relative_rms"])
        for row in native
    )
    clipped = sum(
        row["calibration_clipped_count"] for row in interventions
    )
    degenerate = sum(
        row["calibration_degenerate_count"] for row in interventions
    )
    target_error = max(
        row["calibration_relative_error_max"] for row in interventions
    )
    scale_min = min(row["calibration_scale_min"] for row in interventions)
    scale_max = max(row["calibration_scale_max"] for row in interventions)
    target_min = min(
        row["calibration_target_min"] for row in interventions
    )
    target_max = max(
        row["calibration_target_max"] for row in interventions
    )
    shifted = [
        row["min_shifted_old_frames"]
        for row in interventions
        if row["policy"] in {"key_shift", "value_shift"}
    ]
    contrast_valid = all(
        bool(row["policy_contrast_valid"])
        for row in interventions
        if row["policy"] == "policy_contrast"
    )
    return {
        "native_replay_max_relative_rms": replay_max,
        "native_replay_pass": replay_max <= 1e-4,
        "calibration_clipped_layer_count": clipped,
        "calibration_degenerate_layer_count": degenerate,
        "calibration_max_relative_error": target_error,
        "calibration_scale_min": scale_min,
        "calibration_scale_max": scale_max,
        "calibration_target_min": target_min,
        "calibration_target_max": target_max,
        "calibration_pass": bool(
            clipped == 0
            and degenerate == 0
            and target_error <= MAX_CALIBRATION_RELATIVE_ERROR
            and abs(target_max - target_min) <= 1e-8
            and scale_min >= MIN_ACCEPTABLE_CALIBRATION_SCALE
            and scale_max <= MAX_ACCEPTABLE_CALIBRATION_SCALE
        ),
        "shift_interventions_non_degenerate": bool(
            shifted and min(shifted) > 1
        ),
        "policy_contrast_valid": contrast_valid,
    }


def _core_analysis(rows: list[dict], plan: dict) -> tuple[list, list, dict]:
    comparisons = []
    specificity = []
    effects_by_channel = {}
    seed = 14900
    for channel, metric in CHANNEL_METRICS.items():
        lookup = _lookup(rows, metric)
        for hypothesis in plan["hypotheses"]:
            axis = hypothesis["axis"]
            policy = hypothesis["policy"]
            effects = _paired_log_effects(
                lookup,
                left_probe=hypothesis["top_probe"],
                right_probe=hypothesis["bottom_probe"],
            )
            effects_by_channel[
                (channel, axis, policy, "bottom")
            ] = effects
            comparisons.extend(
                _summarize_effects(
                    effects,
                    label=f"{channel}:{axis}:{policy}:top>bottom",
                    metric=metric,
                    metadata={
                        "channel": channel,
                        "axis": axis,
                        "policy": policy,
                        "control": "bottom",
                        "matched_intervention": int(
                            hypothesis["matched"]
                        ),
                    },
                    bootstrap_seed=seed,
                )
            )
            seed += 101
            if hypothesis["random_probes"]:
                random_effects = _random_ensemble_effects(
                    lookup,
                    top_probe=hypothesis["top_probe"],
                    random_probes=hypothesis["random_probes"],
                )
                effects_by_channel[
                    (channel, axis, policy, "random_ensemble")
                ] = random_effects
                comparisons.extend(
                    _summarize_effects(
                        random_effects,
                        label=(
                            f"{channel}:{axis}:{policy}:"
                            "top>random_ensemble"
                        ),
                        metric=metric,
                        metadata={
                            "channel": channel,
                            "axis": axis,
                            "policy": policy,
                            "control": "random_ensemble",
                            "matched_intervention": 1,
                        },
                        bootstrap_seed=seed,
                    )
                )
                seed += 101
        for hypothesis in plan["pf_matched_hypotheses"]:
            effects = _paired_log_effects(
                lookup,
                left_probe=hypothesis["top_probe"],
                right_probe=hypothesis["bottom_probe"],
            )
            comparisons.extend(
                _summarize_effects(
                    effects,
                    label=(
                        f"{channel}:{hypothesis['axis']}:"
                        f"{hypothesis['policy']}:pfmatched_top>bottom"
                    ),
                    metric=metric,
                    metadata={
                        "channel": channel,
                        "axis": hypothesis["axis"],
                        "policy": hypothesis["policy"],
                        "control": "pf_label_matched",
                        "matched_intervention": 1,
                    },
                    bootstrap_seed=seed,
                )
            )
            seed += 101
        for axis in ("k", "v", "policy"):
            diagonal = next(
                item["policy"]
                for item in plan["hypotheses"]
                if item["axis"] == axis and item["matched"]
            )
            diagonal_effects = effects_by_channel[
                (channel, axis, diagonal, "bottom")
            ]
            for off_policy in (
                policy
                for policy in ("key_shift", "value_shift", "policy_contrast")
                if policy != diagonal
            ):
                off_effects = effects_by_channel[
                    (channel, axis, off_policy, "bottom")
                ]
                difference = {
                    key: diagonal_effects[key] - off_effects[key]
                    for key in diagonal_effects
                }
                specificity.extend(
                    _summarize_effects(
                        difference,
                        label=(
                            f"{channel}:{axis}:matched>{off_policy}"
                        ),
                        metric="top_bottom_log_effect",
                        metadata={
                            "channel": channel,
                            "axis": axis,
                            "policy": diagonal,
                            "control": off_policy,
                            "matched_intervention": 1,
                        },
                        bootstrap_seed=seed,
                        effect_definition="difference_of_log_ratios",
                    )
                )
                seed += 101

    matched_contexts = {}
    pf_contexts = {}
    specificity_contexts = {}
    for channel in CHANNEL_METRICS:
        matched_contexts[channel] = {}
        pf_contexts[channel] = {}
        specificity_contexts[channel] = {}
        for axis in ("k", "v", "policy"):
            policy = next(
                item["policy"]
                for item in plan["hypotheses"]
                if item["axis"] == axis and item["matched"]
            )
            bottom = {
                row["context"]
                for row in comparisons
                if row["channel"] == channel
                and row["axis"] == axis
                and row["policy"] == policy
                and row["control"] == "bottom"
                and _qualifies(row)
            }
            random = {
                row["context"]
                for row in comparisons
                if row["channel"] == channel
                and row["axis"] == axis
                and row["policy"] == policy
                and row["control"] == "random_ensemble"
                and _qualifies(row)
            }
            matched_contexts[channel][axis] = sorted(bottom & random)
            pf_contexts[channel][axis] = sorted(
                {
                    row["context"]
                    for row in comparisons
                    if row["channel"] == channel
                    and row["axis"] == axis
                    and row["control"] == "pf_label_matched"
                    and _qualifies(row)
                }
            )
            passing = []
            for off_policy in (
                item
                for item in ("key_shift", "value_shift", "policy_contrast")
                if item != policy
            ):
                passing.append(
                    {
                        row["context"]
                        for row in specificity
                        if row["channel"] == channel
                        and row["axis"] == axis
                        and row["control"] == off_policy
                        and _qualifies(row)
                    }
                )
            specificity_contexts[channel][axis] = sorted(
                set.intersection(*passing)
            )
    return comparisons, specificity, {
        "g1_matched_axis_effect": {
            channel: {
                axis: bool(contexts)
                for axis, contexts in values.items()
            }
            for channel, values in matched_contexts.items()
        },
        "g2_pf_independent_effect": {
            channel: {
                axis: bool(contexts)
                for axis, contexts in values.items()
            }
            for channel, values in pf_contexts.items()
        },
        "g3_intervention_specificity": {
            channel: {
                axis: bool(contexts)
                for axis, contexts in values.items()
            }
            for channel, values in specificity_contexts.items()
        },
        "qualifying_matched_contexts": matched_contexts,
        "qualifying_pf_contexts": pf_contexts,
        "qualifying_specificity_contexts": specificity_contexts,
    }


def _dose_analysis(rows: list[dict], plan: dict) -> tuple[list, dict]:
    comparisons = []
    qualifying = {}
    seed = 15900
    for channel, metric in CHANNEL_METRICS.items():
        lookup = _lookup(rows, metric)
        qualifying[channel] = {}
        for hypothesis in plan["dose_hypotheses"]:
            axis = hypothesis["axis"]
            for pair in hypothesis["pairs"]:
                effects = _paired_log_effects(
                    lookup,
                    left_probe=pair["top_probe"],
                    right_probe=pair["bottom_probe"],
                )
                comparisons.extend(
                    _summarize_effects(
                        effects,
                        label=(
                            f"{channel}:{axis}:{hypothesis['policy']}:"
                            f"top>bottom:dose{pair['dose']}"
                        ),
                        metric=metric,
                        metadata={
                            "channel": channel,
                            "axis": axis,
                            "policy": hypothesis["policy"],
                            "control": "bottom",
                            "matched_intervention": 1,
                            "dose": int(pair["dose"]),
                        },
                        bootstrap_seed=seed,
                    )
                )
                seed += 101
            qualifying[channel][axis] = sorted(
                {
                    int(row["dose"])
                    for row in comparisons
                    if row["channel"] == channel
                    and row["axis"] == axis
                    and _qualifies(row)
                }
            )
    return comparisons, {
        "g1_positive_separation_at_multiple_doses": {
            channel: {
                axis: len(doses) >= 2
                for axis, doses in values.items()
            }
            for channel, values in qualifying.items()
        },
        "qualifying_doses": qualifying,
    }


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# v149 Calibrated Causal Profiling Results",
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
            "- Calibration maximum relative target error: "
            f"`{report['calibration_max_relative_error']:.6g}`"
        ),
        (
            "- Calibration clipped layers: "
            f"`{report['calibration_clipped_layer_count']}`"
        ),
        (
            "- Calibration degenerate layers: "
            f"`{report['calibration_degenerate_layer_count']}`"
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
            "`susceptibility` is the raw projected local replacement before "
            "calibration. `leverage` is the final x0 effect after every layer "
            "has been calibrated to the same projected relative RMS."
        ),
        "",
        (
            "These are one-step downstream causal measurements at frame 117, "
            "not trajectory-level video quality or a validated cache method."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    *,
    profile_dir: Path,
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
    rows = _downstream_rows(profiles)
    integrity = _integrity(rows)
    if plan["suite"].endswith("core"):
        comparisons, specificity, analysis = _core_analysis(rows, plan)
    else:
        comparisons, analysis = _dose_analysis(rows, plan)
        specificity = []
    integrity_pass = bool(
        integrity["native_replay_pass"]
        and integrity["calibration_pass"]
        and integrity["shift_interventions_non_degenerate"]
        and integrity["policy_contrast_valid"]
    )
    report = {
        "suite": plan["suite"],
        "profile_count": len(profiles),
        "prompt_count": _expected_prompt_count(plan),
        "seed_replicates": list(REPLICATES),
        "probe_plan": str(probe_plan_path),
        "probe_plan_sha256": plan_sha256,
        "downstream_observation_count": len(rows),
        **integrity,
        "gates": {
            "g0_integrity_and_calibration": integrity_pass,
            **{
                key: value
                for key, value in analysis.items()
                if key.startswith("g")
            },
        },
        **{
            key: value
            for key, value in analysis.items()
            if not key.startswith("g")
        },
        "minimum_qualifying_median_log_effect": MIN_LOG_EFFECT,
        "maximum_calibration_relative_error": (
            MAX_CALIBRATION_RELATIVE_ERROR
        ),
        "acceptable_calibration_scale_range": [
            MIN_ACCEPTABLE_CALIBRATION_SCALE,
            MAX_ACCEPTABLE_CALIBRATION_SCALE,
        ],
        "source": plan["source"],
        "claim_boundary": (
            "Passing susceptibility identifies a reproducible local response. "
            "Passing calibrated leverage identifies downstream amplification "
            "at equal projected perturbation strength. Neither establishes "
            "trajectory-level video improvement."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_audit.csv", audits)
    _write_csv(output_dir / "downstream_observations.csv.gz", rows)
    _write_csv(output_dir / "probe_effect_summary.csv", _probe_summary(rows))
    _write_csv(output_dir / "channel_comparisons.csv", comparisons)
    if specificity:
        _write_csv(output_dir / "channel_specificity.csv", specificity)
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
                profile_dir=args.profile_dir,
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
