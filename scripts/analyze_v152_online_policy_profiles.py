#!/usr/bin/env python3
"""Analyze v152 frozen-native dynamic head-policy probes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


LAYERS = 30
HEADS = 12
SEED_REPLICATES = (0, 1)
EPSILON = 1e-12
MIN_EFFECT = math.log(1.03)
MIN_RANDOM_EFFECT = math.log(1.01)
MIN_POSITIVE_FRACTION = 0.65
MIN_SEED_SPEARMAN = 0.30
MIN_ALIGNMENT_SPEARMAN = 0.30
MIN_ALIGNMENT_JACCARD = 0.30


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


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _context_key(row: dict) -> str:
    return (
        f"{row['mode']}_f{int(row['current_frame'])}_"
        f"t{int(row['nominal_timestep'])}"
    )


def _expected_contexts(plan: dict) -> list[str]:
    return [_context_key(row) for row in plan["contexts"]]


def _bootstrap_mean_ci(
    effects: dict[tuple[int, int], float], *, seed: int, samples: int = 2000
) -> tuple[float, float]:
    prompt_slots = sorted({key[0] for key in effects})
    by_prompt = {
        prompt: np.asarray(
            [
                effects[(prompt, replicate)]
                for replicate in SEED_REPLICATES
            ],
            dtype=np.float64,
        )
        for prompt in prompt_slots
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(prompt_slots, size=len(prompt_slots), replace=True)
        values[index] = np.mean([by_prompt[int(prompt)].mean() for prompt in sampled])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _summarize_effects(
    rows: list[dict],
    *,
    value_field: str,
    label: str,
    group: str,
    context: str,
    seed: int,
) -> dict:
    effects = {
        (int(row["prompt_slot"]), int(row["seed_replicate"])): float(
            row[value_field]
        )
        for row in rows
    }
    prompt_slots = sorted({key[0] for key in effects})
    expected = {
        (prompt, replicate)
        for prompt in prompt_slots
        for replicate in SEED_REPLICATES
    }
    if set(effects) != expected:
        raise RuntimeError(f"{label}/{context} has an incomplete seed grid")
    values = np.asarray(list(effects.values()), dtype=np.float64)
    left = np.asarray([effects[(prompt, 0)] for prompt in prompt_slots])
    right = np.asarray([effects[(prompt, 1)] for prompt in prompt_slots])
    low, high = _bootstrap_mean_ci(effects, seed=seed)
    return {
        "comparison": label,
        "group": group,
        "context": context,
        "metric": value_field,
        "unit_count": len(values),
        "prompt_count": len(prompt_slots),
        "mean_effect": float(values.mean()),
        "median_effect": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "prompt_bootstrap_mean_ci_low": low,
        "prompt_bootstrap_mean_ci_high": high,
        "seed_replicate_spearman": _spearman(left, right),
    }


def _qualifies(row: dict, *, minimum_effect: float = MIN_EFFECT) -> bool:
    return bool(
        float(row["median_effect"]) >= minimum_effect
        and (
            float(row["prompt_bootstrap_mean_ci_low"]) > 0
            or float(row["positive_fraction"]) >= MIN_POSITIVE_FRACTION
        )
        and float(row["seed_replicate_spearman"]) >= MIN_SEED_SPEARMAN
    )


def _load_profiles(
    profile_dir: Path,
    *,
    plan: dict,
    expected_count: int,
) -> tuple[list[dict], list[dict], float]:
    paths = sorted(profile_dir.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v152 profiles, found {len(paths)}"
        )
    probe_by_name = {str(probe["name"]): probe for probe in plan["probes"]}
    expected_names = {"native_replay", *probe_by_name}
    contexts = set(_expected_contexts(plan))
    profiles = []
    audits = []
    replay_max = 0.0
    coordinates = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        if int(payload.get("version", -1)) != 8:
            raise RuntimeError(f"{path} is not a downstream causal profile")
        if str(job.get("kind")) != str(plan["suite"]):
            raise RuntimeError(f"{path} has the wrong job kind")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete base calls")
        coordinate = (int(job["prompt_slot"]), int(job["seed_replicate"]))
        if coordinate in coordinates:
            raise RuntimeError(f"duplicate v152 coordinate {coordinate}")
        coordinates.add(coordinate)
        if int(metadata["seed"]) != int(job["seed"]):
            raise RuntimeError(f"{path} violates the seed contract")
        rows = payload.get("downstream_probe_records") or []
        expected_rows = len(expected_names) * len(contexts)
        if len(rows) != expected_rows or int(
            payload.get("downstream_probe_expected_count", -1)
        ) != expected_rows:
            raise RuntimeError(f"{path} has incomplete downstream probes")
        seen = set()
        for row in rows:
            context = _context_key(row)
            name = str(row["probe_name"])
            key = (context, name)
            if context not in contexts or name not in expected_names or key in seen:
                raise RuntimeError(f"{path} has invalid probe coordinate {key}")
            seen.add(key)
            if name == "native_replay":
                replay_max = max(
                    replay_max,
                    float(row["flow_metrics"]["relative_rms"]),
                    float(row["x0_metrics"]["relative_rms"]),
                )
            else:
                expected_probe = probe_by_name[name]
                if str(row["policy"]) != str(expected_probe["policy"]):
                    raise RuntimeError(f"{path}/{key} has the wrong policy")
                if int(row["selected_head_count"]) != LAYERS * 4:
                    raise RuntimeError(f"{path}/{key} selects the wrong head count")
                layer_metadata = row.get("layer_metadata") or {}
                if {int(layer) for layer in layer_metadata} != set(range(LAYERS)):
                    raise RuntimeError(f"{path}/{key} has incomplete layers")
                for layer in layer_metadata.values():
                    indices = torch.as_tensor(layer["frame_indices"]).long()
                    if indices.numel() != 8 or torch.unique(indices).numel() != 8:
                        raise RuntimeError(f"{path}/{key} violates the 8-frame budget")
                    history_frames = int(layer["history_frames"])
                    values = sorted(int(value) for value in indices.tolist())
                    if values[0] < 0 or values[-1] >= history_frames:
                        raise RuntimeError(f"{path}/{key} has invalid frame ids")
                    if row["policy"] == "recent8" and values != list(
                        range(history_frames - 8, history_frames)
                    ):
                        raise RuntimeError(f"{path}/{key} is not recent8")
                    if row["policy"] == "uniform8":
                        if values[-4:] != list(
                            range(history_frames - 4, history_frames)
                        ) or values == list(range(history_frames - 8, history_frames)):
                            raise RuntimeError(f"{path}/{key} is not uniform8")
                    if "calibration_target" in layer:
                        raise RuntimeError(f"{path}/{key} unexpectedly uses calibration")
        if seen != {(context, name) for context in contexts for name in expected_names}:
            raise RuntimeError(f"{path} has an incomplete context/probe grid")
        profiles.append(payload)
        audits.append(
            {
                "dataset_index": int(job["dataset_index"]),
                "prompt_slot": coordinate[0],
                "source_prompt_index": int(job["source_prompt_index"]),
                "seed_replicate": coordinate[1],
                "seed": int(job["seed"]),
                "captured_calls": int(metadata["captured_calls"]),
                "record_count": int(metadata["record_count"]),
                "downstream_record_count": len(rows),
                "path": str(path),
            }
        )
    if replay_max > 1e-4:
        raise RuntimeError(f"native replay failed: {replay_max:.6g}")
    prompt_count = expected_count // len(SEED_REPLICATES)
    expected_coordinates = {
        (prompt, replicate)
        for prompt in range(prompt_count)
        for replicate in SEED_REPLICATES
    }
    if coordinates != expected_coordinates:
        raise RuntimeError("v152 prompt/seed grid is incomplete")
    return profiles, audits, replay_max


def _index_probe_rows(profile: dict) -> dict[tuple[str, str], dict]:
    return {
        (_context_key(row), str(row["probe_name"])): row
        for row in profile["downstream_probe_records"]
    }


def _same_selector(left: dict, right: dict) -> bool:
    if left["selected_heads"] != right["selected_heads"]:
        return False
    return bool(
        torch.equal(torch.as_tensor(left["scores"]), torch.as_tensor(right["scores"]))
    )


def _extract(
    profiles: list[dict], plan: dict
) -> tuple[list[dict], dict, list[dict]]:
    pair_rows = []
    selector_index = {}
    probe_rows = []
    contexts = _expected_contexts(plan)
    for profile in profiles:
        job = profile["job"]
        indexed = _index_probe_rows(profile)
        base = {
            "prompt_slot": int(job["prompt_slot"]),
            "source_prompt_index": int(job["source_prompt_index"]),
            "seed_replicate": int(job["seed_replicate"]),
            "seed": int(job["seed"]),
        }
        for context in contexts:
            for group, pair in plan["pair_probes"].items():
                uniform = indexed[(context, pair["uniform8"])]
                recent = indexed[(context, pair["recent8"])]
                expected_policy = plan["groups"][group]["expected_policy"]
                x0_uniform = float(uniform["x0_metrics"]["relative_rms"])
                x0_recent = float(recent["x0_metrics"]["relative_rms"])
                flow_uniform = float(uniform["flow_metrics"]["relative_rms"])
                flow_recent = float(recent["flow_metrics"]["relative_rms"])
                x0_advantage = math.log(
                    (x0_recent + EPSILON) / (x0_uniform + EPSILON)
                )
                flow_advantage = math.log(
                    (flow_recent + EPSILON) / (flow_uniform + EPSILON)
                )
                preferred_sign = (
                    1.0
                    if expected_policy == "uniform8"
                    else -1.0 if expected_policy == "recent8" else float("nan")
                )
                pair_rows.append(
                    {
                        **base,
                        "context": context,
                        "group": group,
                        "group_kind": plan["groups"][group]["kind"],
                        "expected_policy": expected_policy or "diagnostic",
                        "x0_uniform_relative_rms": x0_uniform,
                        "x0_recent_relative_rms": x0_recent,
                        "flow_uniform_relative_rms": flow_uniform,
                        "flow_recent_relative_rms": flow_recent,
                        "uniform_advantage_x0": x0_advantage,
                        "uniform_advantage_flow": flow_advantage,
                        "preferred_advantage_x0": x0_advantage * preferred_sign,
                        "preferred_advantage_flow": flow_advantage * preferred_sign,
                    }
                )
                for layer in range(LAYERS):
                    uniform_heads = [
                        int(value)
                        for value in uniform["layer_metadata"][layer][
                            "selected_heads"
                        ]
                    ]
                    recent_heads = [
                        int(value)
                        for value in recent["layer_metadata"][layer][
                            "selected_heads"
                        ]
                    ]
                    if uniform_heads != recent_heads:
                        raise RuntimeError(
                            f"head map changed across policies: {base}/{context}/{group}/{layer}"
                        )
                    if plan["groups"][group]["kind"] == "static_control":
                        expected_heads = sorted(
                            int(value)
                            for value in next(
                                probe
                                for probe in plan["probes"]
                                if probe["name"] == pair["uniform8"]
                            )["head_map"][str(layer)]
                        )
                        if uniform_heads != expected_heads:
                            raise RuntimeError(
                                f"static map differs: {base}/{context}/{group}/{layer}"
                            )
                for policy, row in (("uniform8", uniform), ("recent8", recent)):
                    probe_rows.append(
                        {
                            **base,
                            "context": context,
                            "group": group,
                            "policy": policy,
                            "probe_name": str(row["probe_name"]),
                            "x0_relative_rms": float(
                                row["x0_metrics"]["relative_rms"]
                            ),
                            "flow_relative_rms": float(
                                row["flow_metrics"]["relative_rms"]
                            ),
                        }
                    )
                if plan["groups"][group]["kind"] != "dynamic":
                    continue
                for layer in range(LAYERS):
                    left = uniform["layer_metadata"][layer]["head_selector"]
                    right = recent["layer_metadata"][layer]["head_selector"]
                    if not _same_selector(left, right):
                        raise RuntimeError(
                            f"selector changed across policies: {base}/{context}/{group}/{layer}"
                        )
                    if len(left["selected_heads"]) != 4:
                        raise RuntimeError("dynamic selector did not select four heads")
                    scores = [float(value) for value in torch.as_tensor(left["scores"])]
                    if str(left["direction"]) == "high":
                        ranked = sorted(
                            range(HEADS), key=lambda head: (-scores[head], head)
                        )
                    else:
                        ranked = sorted(
                            range(HEADS), key=lambda head: (scores[head], -head)
                        )
                    if sorted(ranked[:4]) != sorted(
                        int(value) for value in left["selected_heads"]
                    ):
                        raise RuntimeError("dynamic selector ids do not match scores")
                    key = (
                        base["prompt_slot"],
                        base["seed_replicate"],
                        context,
                        layer,
                        group,
                    )
                    selector_index[key] = {
                        "selected_heads": [int(value) for value in left["selected_heads"]],
                        "scores": np.asarray(
                            torch.as_tensor(left["scores"]).float().tolist(),
                            dtype=np.float64,
                        ),
                        "type": str(left["type"]),
                        "direction": str(left["direction"]),
                    }
    return pair_rows, selector_index, probe_rows


def _pair_summaries(pair_rows: list[dict], plan: dict) -> list[dict]:
    summaries = []
    seed = 15200
    contexts = _expected_contexts(plan)
    for group in plan["groups"]:
        for context in contexts:
            subset = [
                row
                for row in pair_rows
                if row["group"] == group and row["context"] == context
            ]
            summaries.append(
                _summarize_effects(
                    subset,
                    value_field="uniform_advantage_x0",
                    label=f"{group}:uniform_over_recent:x0",
                    group=group,
                    context=context,
                    seed=seed,
                )
            )
            seed += 101
            if plan["groups"][group]["expected_policy"] is not None:
                summaries.append(
                    _summarize_effects(
                        subset,
                        value_field="preferred_advantage_x0",
                        label=f"{group}:preferred_policy:x0",
                        group=group,
                        context=context,
                        seed=seed,
                    )
                )
                seed += 101
    return summaries


def _random_control_effects(
    pair_rows: list[dict], plan: dict
) -> tuple[list[dict], list[dict]]:
    random_groups = [
        group
        for group, metadata in plan["groups"].items()
        if metadata["kind"] == "static_control" and group.startswith("random")
    ]
    if len(random_groups) < 4:
        raise RuntimeError("v152 requires at least four random control maps")
    indexed = {
        (
            int(row["prompt_slot"]),
            int(row["seed_replicate"]),
            str(row["context"]),
            str(row["group"]),
        ): row
        for row in pair_rows
    }
    effects = []
    for group, metadata in plan["groups"].items():
        expected = metadata["expected_policy"]
        if expected is None:
            continue
        sign = 1.0 if expected == "uniform8" else -1.0
        for prompt in sorted({int(row["prompt_slot"]) for row in pair_rows}):
            for replicate in SEED_REPLICATES:
                for context in _expected_contexts(plan):
                    value = indexed[(prompt, replicate, context, group)]
                    random_advantage = np.mean(
                        [
                            sign
                            * float(
                                indexed[
                                    (prompt, replicate, context, random_group)
                                ]["uniform_advantage_x0"]
                            )
                            for random_group in random_groups
                        ]
                    )
                    effects.append(
                        {
                            "prompt_slot": prompt,
                            "seed_replicate": replicate,
                            "context": context,
                            "group": group,
                            "preferred_advantage_x0": float(
                                value["preferred_advantage_x0"]
                            ),
                            "random_direction_matched_advantage_x0": float(
                                random_advantage
                            ),
                            "preferred_minus_random": float(
                                value["preferred_advantage_x0"] - random_advantage
                            ),
                        }
                    )
    summaries = []
    seed = 25200
    for group in [
        value
        for value, metadata in plan["groups"].items()
        if metadata["expected_policy"] is not None
    ]:
        for context in _expected_contexts(plan):
            subset = [
                row
                for row in effects
                if row["group"] == group and row["context"] == context
            ]
            summaries.append(
                _summarize_effects(
                    subset,
                    value_field="preferred_minus_random",
                    label=f"{group}:preferred_minus_random",
                    group=group,
                    context=context,
                    seed=seed,
                )
            )
            seed += 101
    return effects, summaries


def _selector_diagnostics(
    selector_index: dict, plan: dict
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    snapshots = []
    for key, value in sorted(selector_index.items()):
        prompt, replicate, context, layer, group = key
        snapshots.append(
            {
                "prompt_slot": prompt,
                "seed_replicate": replicate,
                "context": context,
                "layer": layer,
                "group": group,
                "selector_type": value["type"],
                "direction": value["direction"],
                "selected_heads": json.dumps(value["selected_heads"]),
                "scores": json.dumps([float(item) for item in value["scores"]]),
                "score_min": float(value["scores"].min()),
                "score_median": float(np.median(value["scores"])),
                "score_max": float(value["scores"].max()),
            }
        )

    alignment = []
    prompt_slots = sorted({key[0] for key in selector_index})
    contexts = _expected_contexts(plan)
    for prompt in prompt_slots:
        for replicate in SEED_REPLICATES:
            for context in contexts:
                for layer in range(LAYERS):
                    oracle_high = selector_index[
                        (prompt, replicate, context, layer, "oracle_uniform4")
                    ]
                    oracle_low = selector_index[
                        (prompt, replicate, context, layer, "oracle_recent4")
                    ]
                    qk_high = selector_index[
                        (prompt, replicate, context, layer, "qk_uniform4")
                    ]
                    qk_low = selector_index[
                        (prompt, replicate, context, layer, "qk_recent4")
                    ]
                    mass_high = selector_index[
                        (prompt, replicate, context, layer, "mass_uniform4")
                    ]
                    mass_low = selector_index[
                        (prompt, replicate, context, layer, "mass_recent4")
                    ]
                    for high, low in (
                        (oracle_high, oracle_low),
                        (qk_high, qk_low),
                        (mass_high, mass_low),
                    ):
                        if not np.array_equal(high["scores"], low["scores"]):
                            raise RuntimeError(
                                "high/low directions did not share frozen scores"
                            )
                        if set(high["selected_heads"]) & set(
                            low["selected_heads"]
                        ):
                            raise RuntimeError(
                                "high/low dynamic selector tails overlap"
                            )
                    alignment.append(
                        {
                            "prompt_slot": prompt,
                            "seed_replicate": replicate,
                            "context": context,
                            "layer": layer,
                            "oracle_qk_score_spearman": _spearman(
                                oracle_high["scores"], qk_high["scores"]
                            ),
                            "oracle_mass_score_spearman": _spearman(
                                oracle_high["scores"], mass_high["scores"]
                            ),
                            "oracle_qk_uniform_jaccard": _jaccard(
                                oracle_high["selected_heads"],
                                qk_high["selected_heads"],
                            ),
                            "oracle_qk_recent_jaccard": _jaccard(
                                oracle_low["selected_heads"],
                                qk_low["selected_heads"],
                            ),
                            "oracle_mass_uniform_jaccard": _jaccard(
                                oracle_high["selected_heads"],
                                mass_high["selected_heads"],
                            ),
                            "oracle_mass_recent_jaccard": _jaccard(
                                oracle_low["selected_heads"],
                                mass_low["selected_heads"],
                            ),
                        }
                    )
    alignment_summary = []
    for context in contexts:
        subset = [row for row in alignment if row["context"] == context]
        alignment_summary.append(
            {
                "context": context,
                **{
                    field: float(np.median([float(row[field]) for row in subset]))
                    for field in (
                        "oracle_qk_score_spearman",
                        "oracle_mass_score_spearman",
                        "oracle_qk_uniform_jaccard",
                        "oracle_qk_recent_jaccard",
                        "oracle_mass_uniform_jaccard",
                        "oracle_mass_recent_jaccard",
                    )
                },
            }
        )

    recurrence = []
    groups = list(_dynamic_groups_from_plan(plan))
    for prompt in prompt_slots:
        for context in contexts:
            for layer in range(LAYERS):
                for group in groups:
                    left = selector_index[(prompt, 0, context, layer, group)]
                    right = selector_index[(prompt, 1, context, layer, group)]
                    recurrence.append(
                        {
                            "axis": "seed",
                            "prompt_slot": prompt,
                            "context_left": context,
                            "context_right": context,
                            "layer": layer,
                            "group": group,
                            "jaccard": _jaccard(
                                left["selected_heads"], right["selected_heads"]
                            ),
                        }
                    )
    for prompt in prompt_slots:
        for replicate in SEED_REPLICATES:
            for left_context, right_context in zip(contexts[:-1], contexts[1:]):
                for layer in range(LAYERS):
                    for group in groups:
                        left = selector_index[
                            (prompt, replicate, left_context, layer, group)
                        ]
                        right = selector_index[
                            (prompt, replicate, right_context, layer, group)
                        ]
                        recurrence.append(
                            {
                                "axis": "adjacent_timestep",
                                "prompt_slot": prompt,
                                "context_left": left_context,
                                "context_right": right_context,
                                "layer": layer,
                                "group": group,
                                "jaccard": _jaccard(
                                    left["selected_heads"],
                                    right["selected_heads"],
                                ),
                            }
                        )
    recurrence_summary = []
    for axis in ("seed", "adjacent_timestep"):
        for group in groups:
            values = [
                float(row["jaccard"])
                for row in recurrence
                if row["axis"] == axis and row["group"] == group
            ]
            recurrence_summary.append(
                {
                    "axis": axis,
                    "group": group,
                    "unit_count": len(values),
                    "mean_jaccard": float(np.mean(values)),
                    "median_jaccard": float(np.median(values)),
                    "p10_jaccard": float(np.quantile(values, 0.10)),
                    "p90_jaccard": float(np.quantile(values, 0.90)),
                }
            )
    return snapshots, alignment, alignment_summary, recurrence, recurrence_summary


def _dynamic_groups_from_plan(plan: dict):
    return (
        group
        for group, metadata in plan["groups"].items()
        if metadata["kind"] == "dynamic"
    )


def _build_report(
    *,
    plan: dict,
    profile_count: int,
    replay_max: float,
    pair_summaries: list[dict],
    random_summaries: list[dict],
    alignment_summary: list[dict],
    recurrence_summary: list[dict],
) -> dict:
    pair_lookup = {
        (row["group"], row["context"], row["metric"]): row
        for row in pair_summaries
    }
    random_lookup = {
        (row["group"], row["context"]): row for row in random_summaries
    }
    alignment_lookup = {row["context"]: row for row in alignment_summary}
    oracle_contexts = []
    qk_contexts = []
    mass_contexts = []
    qk_random_contexts = []
    aligned_contexts = []
    for context in _expected_contexts(plan):
        if all(
            _qualifies(pair_lookup[(group, context, "preferred_advantage_x0")])
            for group in ("oracle_uniform4", "oracle_recent4")
        ):
            oracle_contexts.append(context)
        if all(
            _qualifies(pair_lookup[(group, context, "preferred_advantage_x0")])
            for group in ("qk_uniform4", "qk_recent4")
        ):
            qk_contexts.append(context)
        if all(
            _qualifies(pair_lookup[(group, context, "preferred_advantage_x0")])
            for group in ("mass_uniform4", "mass_recent4")
        ):
            mass_contexts.append(context)
        if all(
            _qualifies(
                random_lookup[(group, context)], minimum_effect=MIN_RANDOM_EFFECT
            )
            for group in ("qk_uniform4", "qk_recent4")
        ):
            qk_random_contexts.append(context)
        alignment = alignment_lookup[context]
        if (
            float(alignment["oracle_qk_score_spearman"])
            >= MIN_ALIGNMENT_SPEARMAN
            and float(alignment["oracle_qk_uniform_jaccard"])
            >= MIN_ALIGNMENT_JACCARD
            and float(alignment["oracle_qk_recent_jaccard"])
            >= MIN_ALIGNMENT_JACCARD
        ):
            aligned_contexts.append(context)
    candidate_contexts = sorted(
        set(oracle_contexts)
        & set(qk_contexts)
        & set(qk_random_contexts)
        & set(aligned_contexts)
    )
    recurrence_lookup = {
        (row["axis"], row["group"]): row for row in recurrence_summary
    }
    return {
        "version": 1,
        "suite": str(plan["suite"]),
        "profile_count": profile_count,
        "prompt_count": profile_count // 2,
        "native_replay_max_relative_rms": replay_max,
        "contexts": _expected_contexts(plan),
        "gates": {
            "g0_native_replay_and_contract": replay_max <= 1e-4,
            "g1_oracle_policy_choice": bool(oracle_contexts),
            "g2_qk_policy_choice": bool(qk_contexts),
            "g3_qk_beats_count_matched_random": bool(qk_random_contexts),
            "g4_qk_matches_oracle": bool(aligned_contexts),
            "g5_online_qk_candidate_confirmed": bool(candidate_contexts),
            "g6_old_mass_baseline": bool(mass_contexts),
        },
        "qualifying_contexts": {
            "oracle_policy_choice": oracle_contexts,
            "qk_policy_choice": qk_contexts,
            "qk_beats_random": qk_random_contexts,
            "qk_oracle_alignment": aligned_contexts,
            "online_qk_candidate": candidate_contexts,
            "old_mass_policy_choice": mass_contexts,
        },
        "selection_recurrence": {
            group: {
                "seed_median_jaccard": float(
                    recurrence_lookup[("seed", group)]["median_jaccard"]
                ),
                "adjacent_timestep_median_jaccard": float(
                    recurrence_lookup[("adjacent_timestep", group)][
                        "median_jaccard"
                    ]
                ),
            }
            for group in ("qk_uniform4", "qk_recent4")
        },
        "thresholds": {
            "minimum_preferred_policy_effect_log_ratio": MIN_EFFECT,
            "minimum_random_margin_log_ratio": MIN_RANDOM_EFFECT,
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "minimum_seed_replicate_spearman": MIN_SEED_SPEARMAN,
            "minimum_oracle_score_spearman": MIN_ALIGNMENT_SPEARMAN,
            "minimum_oracle_topk_jaccard": MIN_ALIGNMENT_JACCARD,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def _write_report(path: Path, report: dict) -> None:
    lines = [
        "# v152 Online State-Conditioned Policy Profiling",
        "",
        f"- Profiles: `{report['profile_count']}`",
        f"- Prompts: `{report['prompt_count']}`",
        (
            "- Native replay maximum relative RMS: "
            f"`{report['native_replay_max_relative_rms']:.6g}`"
        ),
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(report["gates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Qualifying Contexts",
        "",
        "```json",
        json.dumps(report["qualifying_contexts"], indent=2, sort_keys=True),
        "```",
        "",
        "The oracle uses native full-history policy errors and is not deployable. "
        "Only a passing QK branch is eligible for trajectory-level routing. "
        "Its shared candidate-bank cost must still be measured.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    profile_dir: Path,
    probe_plan: Path,
    output_dir: Path,
    expected_count: int,
) -> dict:
    plan = json.loads(probe_plan.read_text(encoding="utf-8"))
    if int(plan.get("version", -1)) != 2 or str(plan.get("suite")) != "v152_online_policy_core":
        raise RuntimeError("analyzer received a non-v152 plan")
    profiles, audits, replay_max = _load_profiles(
        profile_dir, plan=plan, expected_count=expected_count
    )
    pair_rows, selector_index, probe_rows = _extract(profiles, plan)
    pair_summaries = _pair_summaries(pair_rows, plan)
    random_effects, random_summaries = _random_control_effects(pair_rows, plan)
    (
        snapshots,
        alignment,
        alignment_summary,
        recurrence,
        recurrence_summary,
    ) = _selector_diagnostics(selector_index, plan)
    report = _build_report(
        plan=plan,
        profile_count=len(profiles),
        replay_max=replay_max,
        pair_summaries=pair_summaries,
        random_summaries=random_summaries,
        alignment_summary=alignment_summary,
        recurrence_summary=recurrence_summary,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_audit.csv", audits)
    _write_csv(output_dir / "probe_observations.csv.gz", probe_rows)
    _write_csv(output_dir / "policy_pair_effects.csv.gz", pair_rows)
    _write_csv(output_dir / "policy_pair_summary.csv", pair_summaries)
    _write_csv(output_dir / "random_control_effects.csv.gz", random_effects)
    _write_csv(output_dir / "random_control_summary.csv", random_summaries)
    _write_csv(output_dir / "selector_snapshots.csv.gz", snapshots)
    _write_csv(output_dir / "selector_alignment.csv.gz", alignment)
    _write_csv(output_dir / "selector_alignment_summary.csv", alignment_summary)
    _write_csv(output_dir / "selector_recurrence.csv.gz", recurrence)
    _write_csv(output_dir / "selector_recurrence_summary.csv", recurrence_summary)
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
    parser.add_argument("--expected-count", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                profile_dir=args.profile_dir,
                probe_plan=args.probe_plan,
                output_dir=args.output_dir,
                expected_count=args.expected_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
