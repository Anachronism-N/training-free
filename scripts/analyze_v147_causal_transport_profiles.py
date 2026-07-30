#!/usr/bin/env python3
"""Analyze v147 downstream-causal and QK-V transport profiles."""

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
import torch


LAYERS = 30
HEADS = 12
PROMPTS = 32
REPLICATES = (0, 1)
PROFILE_VERSION = 8
EPSILON = 1e-10
MOTION_METRICS = (
    "raw_value_coordinate_error",
    "refined_value_coordinate_error",
    "semantic_refinement_gain",
    "raw_value_top1_match",
    "refined_value_top1_match",
    "raw_value_direction_cosine",
    "refined_value_direction_cosine",
    "qk_displacement",
    "value_displacement",
    "normalized_qk_entropy",
)


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
    keys = sorted(values)
    array = np.asarray([values[key] for key in keys], dtype=np.float64)
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
    mode = str(record["mode"])
    timestep = int(record["nominal_timestep"])
    return "clean" if mode == "clean" else f"noisy_t{timestep}"


def _load_plan(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("version", -1)) != 1
        or int(payload.get("layers", -1)) != LAYERS
        or int(payload.get("heads", -1)) != HEADS
    ):
        raise RuntimeError("v147 probe plan has an invalid model contract")
    probes = payload.get("probes") or []
    names = [str(probe.get("name")) for probe in probes]
    required = {
        "top_recent4",
        "bottom_recent4",
        "random_recent4",
        "all_recent4",
        "top_uniform8",
        "random_uniform8",
        "top_q_retrieval8",
        "bottom_q_retrieval8",
        "random_q_retrieval8",
        "top_value_shift",
        "bottom_value_shift",
        "random_value_shift",
    }
    if not required.issubset(names) or len(set(names)) != len(names):
        raise RuntimeError(
            "v147 probe plan lacks a core probe or repeats a name"
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload, digest


def _load_profiles(
    directory: Path,
    *,
    expected_count: int,
    plan_sha256: str,
    probe_count: int,
) -> tuple[list[dict], list[dict]]:
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v147 profiles, found {len(paths)}"
        )
    profiles = []
    audits = []
    seen = set()
    prompt_seeds: dict[tuple[int, int], int] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
        if key in seen:
            raise RuntimeError(f"duplicate v147 profile coordinate: {key}")
        seen.add(key)
        if (
            int(payload.get("version", -1)) != PROFILE_VERSION
            or str(job.get("kind")) != "causal_transport_profile"
        ):
            raise RuntimeError(f"{path} is not a v147 causal profile")
        if int(metadata["seed"]) != int(job["seed"]):
            raise RuntimeError(f"{path} violates the runtime seed contract")
        if int(job["reference_seed"]) != int(job["seed"]):
            raise RuntimeError(f"{path} violates paired-reference seeding")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete captures")
        expected_records = 3 * LAYERS
        if (
            int(metadata.get("captured_calls", -1)) != 3
            or int(metadata.get("record_count", -1)) != expected_records
        ):
            raise RuntimeError(f"{path} has an invalid 3x30 capture grid")
        plan_meta = metadata.get("downstream_probe_plan") or {}
        if str(plan_meta.get("sha256")) != plan_sha256:
            raise RuntimeError(f"{path} uses a different probe plan")
        records = payload.get("records") or []
        downstream = payload.get("downstream_probe_records") or []
        expected_downstream = 3 * (probe_count + 1)
        if (
            len(records) != expected_records
            or len(downstream) != expected_downstream
        ):
            raise RuntimeError(f"{path} has an invalid record count")
        if (
            int(payload.get("downstream_probe_expected_count", -1))
            != expected_downstream
        ):
            raise RuntimeError(f"{path} declares the wrong probe count")
        state_layers = Counter(
            (
                str(row["mode"]),
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in records
        )
        if len(state_layers) != 3 or set(state_layers.values()) != {LAYERS}:
            raise RuntimeError(f"{path} has an invalid motion state grid")
        for row in records:
            metrics = row.get("motion_correspondence_metrics") or {}
            for name in MOTION_METRICS:
                value = metrics.get(name)
                if not isinstance(value, torch.Tensor) or value.numel() != HEADS:
                    raise RuntimeError(
                        f"{path} lacks 12-head motion metric {name}"
                    )
        probe_grid = Counter(
            (_context_key(row), str(row["probe_name"]))
            for row in downstream
        )
        if (
            len(probe_grid) != expected_downstream
            or set(probe_grid.values()) != {1}
        ):
            raise RuntimeError(f"{path} has a duplicate/incomplete probe grid")
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
                "record_count": int(metadata["record_count"]),
                "downstream_record_count": len(downstream),
                "path": str(path),
            }
        )
    expected = {
        (prompt, replicate)
        for prompt in range(PROMPTS)
        for replicate in REPLICATES
    }
    if seen != expected:
        raise RuntimeError(
            f"incomplete v147 prompt/seed grid: missing={sorted(expected-seen)}"
        )
    for prompt in range(PROMPTS):
        if prompt_seeds[(prompt, 0)] == prompt_seeds[(prompt, 1)]:
            raise RuntimeError(f"prompt {prompt} repeats its seed")
    return profiles, audits


def _downstream_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    for payload in profiles:
        job = payload["job"]
        for record in payload["downstream_probe_records"]:
            layer_metadata = record.get("layer_metadata") or {}
            replacement = [
                float(value["replacement_relative_rms"])
                for value in layer_metadata.values()
                if "replacement_relative_rms" in value
            ]
            shifted = [
                int(value.get("shifted_old_frames", 0))
                for value in layer_metadata.values()
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
                    "nominal_timestep": int(
                        record["nominal_timestep"]
                    ),
                    "probe_name": str(record["probe_name"]),
                    "policy": str(record["policy"]),
                    "group": str(record["group"]),
                    "selected_head_count": int(
                        record["selected_head_count"]
                    ),
                    "flow_relative_rms": float(
                        record["flow_metrics"]["relative_rms"]
                    ),
                    "flow_cosine": float(
                        record["flow_metrics"]["cosine"]
                    ),
                    "flow_max_abs_delta": float(
                        record["flow_metrics"]["max_abs_delta"]
                    ),
                    "x0_relative_rms": float(
                        record["x0_metrics"]["relative_rms"]
                    ),
                    "x0_cosine": float(record["x0_metrics"]["cosine"]),
                    "x0_max_abs_delta": float(
                        record["x0_metrics"]["max_abs_delta"]
                    ),
                    "mean_local_replacement_relative_rms": (
                        float(np.mean(replacement))
                        if replacement
                        else 0.0
                    ),
                    "min_shifted_old_frames": min(shifted) if shifted else 0,
                }
            )
    return rows


def _comparison_summary(
    rows: list[dict],
    *,
    left_probe: str,
    right_probe: str,
    metric: str,
    direction: str,
    label: str,
    bootstrap_seed: int,
) -> dict:
    lookup = {
        (
            int(row["prompt_slot"]),
            int(row["seed_replicate"]),
            str(row["context"]),
            str(row["probe_name"]),
        ): float(row[metric])
        for row in rows
    }
    contexts = sorted({str(row["context"]) for row in rows})
    unit_effects: dict[tuple[int, int, str], float] = {}
    for prompt in range(PROMPTS):
        for replicate in REPLICATES:
            for context in contexts:
                left = lookup[(prompt, replicate, context, left_probe)]
                right = lookup[(prompt, replicate, context, right_probe)]
                if direction == "log_ratio":
                    effect = math.log((left + EPSILON) / (right + EPSILON))
                elif direction == "right_minus_left_relative":
                    effect = (right - left) / (right + EPSILON)
                else:
                    raise ValueError(f"unknown comparison direction {direction}")
                unit_effects[(prompt, replicate, context)] = effect

    summaries = []
    for context in [*contexts, "pooled"]:
        selected = {
            key: value
            for key, value in unit_effects.items()
            if context == "pooled" or key[2] == context
        }
        prompt_values = {
            prompt: float(
                np.mean(
                    [
                        value
                        for (current_prompt, _, _), value in selected.items()
                        if current_prompt == prompt
                    ]
                )
            )
            for prompt in range(PROMPTS)
        }
        replicate_values = {
            replicate: [
                float(
                    np.mean(
                        [
                            value
                            for (
                                current_prompt,
                                current_replicate,
                                _,
                            ), value in selected.items()
                            if current_prompt == prompt
                            and current_replicate == replicate
                        ]
                    )
                )
                for prompt in range(PROMPTS)
            ]
            for replicate in REPLICATES
        }
        low, high = _bootstrap_ci(
            prompt_values, seed=bootstrap_seed + len(summaries)
        )
        values = np.asarray(list(selected.values()), dtype=np.float64)
        summaries.append(
            {
                "comparison": label,
                "left_probe": left_probe,
                "right_probe": right_probe,
                "metric": metric,
                "effect_definition": direction,
                "context": context,
                "unit_count": int(values.size),
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
    return {"rows": summaries, "unit_effects": unit_effects}


def _downstream_comparisons(rows: list[dict]) -> list[dict]:
    comparisons = []
    definitions = []
    for output in ("x0", "flow"):
        metric = f"{output}_relative_rms"
        for policy in ("recent4", "q_retrieval8", "value_shift"):
            for control in ("random", "bottom"):
                definitions.append(
                    (
                        f"{policy}:top>{control}:{output}",
                        f"top_{policy}",
                        f"{control}_{policy}",
                        metric,
                        "log_ratio",
                    )
                )
        definitions.extend(
            [
                (
                    f"retrieval_rescue:top:{output}",
                    "top_q_retrieval8",
                    "top_recent4",
                    metric,
                    "right_minus_left_relative",
                ),
                (
                    f"uniform_rescue:top:{output}",
                    "top_uniform8",
                    "top_recent4",
                    metric,
                    "right_minus_left_relative",
                ),
                (
                    f"retrieval_vs_uniform:top:{output}",
                    "top_q_retrieval8",
                    "top_uniform8",
                    metric,
                    "right_minus_left_relative",
                ),
                (
                    f"retrieval_rescue:random:{output}",
                    "random_q_retrieval8",
                    "random_recent4",
                    metric,
                    "right_minus_left_relative",
                ),
                (
                    f"retrieval_vs_uniform:random:{output}",
                    "random_q_retrieval8",
                    "random_uniform8",
                    metric,
                    "right_minus_left_relative",
                ),
            ]
        )
    for index, (
        label,
        left,
        right,
        metric,
        direction,
    ) in enumerate(definitions):
        result = _comparison_summary(
            rows,
            left_probe=left,
            right_probe=right,
            metric=metric,
            direction=direction,
            label=label,
            bootstrap_seed=14700 + index * 17,
        )
        comparisons.extend(result["rows"])
    comparisons.extend(_retrieval_selectivity(rows))
    return comparisons


def _retrieval_selectivity(rows: list[dict]) -> list[dict]:
    lookup = {
        (
            int(row["prompt_slot"]),
            int(row["seed_replicate"]),
            str(row["context"]),
            str(row["probe_name"]),
        ): float(row["x0_relative_rms"])
        for row in rows
    }
    contexts = sorted({str(row["context"]) for row in rows})
    unit_effects = {}
    for prompt in range(PROMPTS):
        for replicate in REPLICATES:
            for context in contexts:
                prefix = (prompt, replicate, context)
                top_uniform = lookup[(*prefix, "top_uniform8")]
                top_retrieval = lookup[(*prefix, "top_q_retrieval8")]
                random_uniform = lookup[(*prefix, "random_uniform8")]
                random_retrieval = lookup[
                    (*prefix, "random_q_retrieval8")
                ]
                top_gain = (top_uniform - top_retrieval) / (
                    top_uniform + EPSILON
                )
                random_gain = (random_uniform - random_retrieval) / (
                    random_uniform + EPSILON
                )
                unit_effects[(prompt, replicate, context)] = (
                    top_gain - random_gain
                )
    summaries = []
    for context in [*contexts, "pooled"]:
        selected = {
            key: value
            for key, value in unit_effects.items()
            if context == "pooled" or key[2] == context
        }
        prompt_values = {
            prompt: float(
                np.mean(
                    [
                        value
                        for (current_prompt, _, _), value in selected.items()
                        if current_prompt == prompt
                    ]
                )
            )
            for prompt in range(PROMPTS)
        }
        low, high = _bootstrap_ci(
            prompt_values, seed=15000 + len(summaries) * 23
        )
        values = np.asarray(list(selected.values()), dtype=np.float64)
        summaries.append(
            {
                "comparison": "retrieval_selectivity:top_minus_random:x0",
                "left_probe": "top_q_retrieval8_vs_top_uniform8",
                "right_probe": (
                    "random_q_retrieval8_vs_random_uniform8"
                ),
                "metric": "x0_relative_rms",
                "effect_definition": "top_gain_minus_random_gain",
                "context": context,
                "unit_count": int(values.size),
                "prompt_count": len(prompt_values),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "positive_fraction": float((values > 0).mean()),
                "prompt_bootstrap_mean_ci_low": low,
                "prompt_bootstrap_mean_ci_high": high,
                "seed_replicate_spearman": _spearman(
                    [
                        float(
                            np.mean(
                                [
                                    value
                                    for (
                                        current_prompt,
                                        replicate,
                                        _,
                                    ), value in selected.items()
                                    if current_prompt == prompt
                                    and replicate == 0
                                ]
                            )
                        )
                        for prompt in range(PROMPTS)
                    ],
                    [
                        float(
                            np.mean(
                                [
                                    value
                                    for (
                                        current_prompt,
                                        replicate,
                                        _,
                                    ), value in selected.items()
                                    if current_prompt == prompt
                                    and replicate == 1
                                ]
                            )
                        )
                        for prompt in range(PROMPTS)
                    ],
                ),
            }
        )
    return summaries


def _absolute_probe_summary(
    rows: list[dict], probe_names: tuple[str, ...]
) -> list[dict]:
    selected = [
        row for row in rows if str(row["probe_name"]) in probe_names
    ]
    summaries = []
    contexts = sorted({str(row["context"]) for row in selected})
    for probe_name in probe_names:
        for metric in ("x0_relative_rms", "flow_relative_rms"):
            for context in contexts:
                current = [
                    row
                    for row in selected
                    if row["probe_name"] == probe_name
                    and row["context"] == context
                ]
                lookup = {
                    (int(row["prompt_slot"]), int(row["seed_replicate"])): (
                        float(row[metric])
                    )
                    for row in current
                }
                if len(lookup) != PROMPTS * len(REPLICATES):
                    raise RuntimeError(
                        f"incomplete layer-band probe {probe_name}/{context}"
                    )
                prompt_values = {
                    prompt: float(
                        np.mean(
                            [
                                lookup[(prompt, replicate)]
                                for replicate in REPLICATES
                            ]
                        )
                    )
                    for prompt in range(PROMPTS)
                }
                low, high = _bootstrap_ci(
                    prompt_values,
                    seed=14900 + len(summaries) * 19,
                )
                values = np.asarray(list(lookup.values()))
                summaries.append(
                    {
                        "probe_name": probe_name,
                        "metric": metric,
                        "context": context,
                        "unit_count": int(values.size),
                        "mean_effect": float(values.mean()),
                        "median_effect": float(np.median(values)),
                        "prompt_bootstrap_mean_ci_low": low,
                        "prompt_bootstrap_mean_ci_high": high,
                        "seed_replicate_spearman": _spearman(
                            [
                                lookup[(prompt, 0)]
                                for prompt in range(PROMPTS)
                            ],
                            [
                                lookup[(prompt, 1)]
                                for prompt in range(PROMPTS)
                            ],
                        ),
                    }
                )
    return summaries


def _motion_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    for payload in profiles:
        job = payload["job"]
        for record in payload["records"]:
            metrics = record["motion_correspondence_metrics"]
            for head in range(HEADS):
                row = {
                    "prompt_slot": int(job["prompt_slot"]),
                    "source_prompt_index": int(
                        job["source_prompt_index"]
                    ),
                    "seed_replicate": int(job["seed_replicate"]),
                    "seed": int(job["seed"]),
                    "context": _context_key(record),
                    "layer": int(record["layer"]),
                    "head": head,
                    "sample_count": int(metrics["sample_count"]),
                    "topk": int(metrics["topk"]),
                }
                for name in MOTION_METRICS:
                    row[name] = float(metrics[name][head])
                rows.append(row)
    return rows


def _plan_head_sets(plan: dict) -> dict[str, dict[int, set[int]]]:
    groups = {}
    for probe in plan["probes"]:
        group = str(probe["group"])
        if group not in {"top", "bottom", "random"} or group in groups:
            continue
        groups[group] = {
            int(layer): {int(head) for head in heads}
            for layer, heads in probe["head_map"].items()
        }
    if set(groups) != {"top", "bottom", "random"}:
        raise RuntimeError("probe plan lacks top/bottom/random head maps")
    return groups


def _motion_group_summary(
    rows: list[dict], plan: dict
) -> list[dict]:
    groups = _plan_head_sets(plan)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["prompt_slot"]),
                int(row["seed_replicate"]),
                str(row["context"]),
            )
        ].append(row)
    unit_values = []
    for unit, current_rows in sorted(grouped.items()):
        for group, head_map in groups.items():
            selected = [
                row
                for row in current_rows
                if int(row["head"]) in head_map[int(row["layer"])]
            ]
            expected = LAYERS * len(next(iter(head_map.values())))
            if len(selected) != expected:
                raise RuntimeError(
                    f"motion group {group} has {len(selected)} heads, "
                    f"expected {expected}"
                )
            for metric in MOTION_METRICS:
                unit_values.append(
                    {
                        "prompt_slot": unit[0],
                        "seed_replicate": unit[1],
                        "context": unit[2],
                        "group": group,
                        "metric": metric,
                        "mean": float(
                            np.mean([row[metric] for row in selected])
                        ),
                    }
                )
    lookup = {
        (
            row["prompt_slot"],
            row["seed_replicate"],
            row["context"],
            row["group"],
            row["metric"],
        ): row["mean"]
        for row in unit_values
    }
    summaries = []
    contexts = sorted({row["context"] for row in unit_values})
    for metric in MOTION_METRICS:
        for control in ("bottom", "random"):
            for context in [*contexts, "pooled"]:
                effects = {}
                for prompt in range(PROMPTS):
                    for replicate in REPLICATES:
                        selected_contexts = (
                            contexts if context == "pooled" else [context]
                        )
                        effects[(prompt, replicate)] = float(
                            np.mean(
                                [
                                    lookup[
                                        (
                                            prompt,
                                            replicate,
                                            current,
                                            "top",
                                            metric,
                                        )
                                    ]
                                    - lookup[
                                        (
                                            prompt,
                                            replicate,
                                            current,
                                            control,
                                            metric,
                                        )
                                    ]
                                    for current in selected_contexts
                                ]
                            )
                        )
                prompt_values = {
                    prompt: float(
                        np.mean(
                            [
                                effects[(prompt, replicate)]
                                for replicate in REPLICATES
                            ]
                        )
                    )
                    for prompt in range(PROMPTS)
                }
                low, high = _bootstrap_ci(
                    prompt_values,
                    seed=14800 + len(summaries) * 13,
                )
                values = np.asarray(list(effects.values()))
                summaries.append(
                    {
                        "comparison": f"top_minus_{control}",
                        "metric": metric,
                        "context": context,
                        "unit_count": int(values.size),
                        "mean_difference": float(values.mean()),
                        "median_difference": float(np.median(values)),
                        "positive_fraction": float((values > 0).mean()),
                        "prompt_bootstrap_mean_ci_low": low,
                        "prompt_bootstrap_mean_ci_high": high,
                        "seed_replicate_spearman": _spearman(
                            [
                                effects[(prompt, 0)]
                                for prompt in range(PROMPTS)
                            ],
                            [
                                effects[(prompt, 1)]
                                for prompt in range(PROMPTS)
                            ],
                        ),
                    }
                )
    return summaries


def _gate_report(
    downstream_rows: list[dict],
    comparisons: list[dict],
) -> dict:
    native = [
        row
        for row in downstream_rows
        if row["probe_name"] == "native_replay"
    ]
    replay_max = max(
        max(row["flow_relative_rms"], row["x0_relative_rms"])
        for row in native
    )
    causal = [
        row
        for row in comparisons
        if row["context"] != "pooled"
        and row["comparison"].startswith(
            ("recent4:top>", "q_retrieval8:top>", "value_shift:top>")
        )
        and row["metric"] == "x0_relative_rms"
        and row["median_effect"] > 0
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= 0.65
        )
        and row["seed_replicate_spearman"] >= 0.30
    ]
    rescue = [
        row
        for row in comparisons
        if row["context"] != "pooled"
        and row["comparison"] == "retrieval_rescue:top:x0"
        and row["median_effect"] >= 0.05
        and row["positive_fraction"] >= 0.60
        and row["seed_replicate_spearman"] >= 0.30
    ]
    matched_retrieval = [
        row
        for row in comparisons
        if row["context"] != "pooled"
        and row["comparison"] == "retrieval_vs_uniform:top:x0"
        and row["median_effect"] >= 0.02
        and row["positive_fraction"] >= 0.55
        and row["seed_replicate_spearman"] >= 0.20
    ]
    retrieval_contexts = {
        row["context"] for row in rescue
    } & {row["context"] for row in matched_retrieval}
    selective_retrieval = [
        row
        for row in comparisons
        if row["context"] != "pooled"
        and row["comparison"]
        == "retrieval_selectivity:top_minus_random:x0"
        and row["median_effect"] > 0
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= 0.65
        )
        and row["seed_replicate_spearman"] >= 0.30
    ]
    shifted = [
        row["min_shifted_old_frames"]
        for row in downstream_rows
        if row["policy"] == "value_shift"
    ]
    gates = {
        "g0_native_replay_parity": replay_max <= 1e-4,
        "g1_ranked_heads_have_reproducible_downstream_effect": bool(causal),
        "g2_q_retrieval_rescues_ranked_heads": bool(
            retrieval_contexts
        ),
        "g3_value_shift_is_non_degenerate": bool(shifted)
        and min(shifted) > 1,
        "g4_q_retrieval_is_head_selective": bool(selective_retrieval),
    }
    return {
        "gates": gates,
        "all_required_integrity_gates_pass": gates[
            "g0_native_replay_parity"
        ]
        and gates["g3_value_shift_is_non_degenerate"],
        "mechanism_supported": gates[
            "g1_ranked_heads_have_reproducible_downstream_effect"
        ],
        "retrieval_design_supported": gates[
            "g2_q_retrieval_rescues_ranked_heads"
        ],
        "retrieval_head_selectivity_supported": gates[
            "g4_q_retrieval_is_head_selective"
        ],
        "native_replay_max_relative_rms": replay_max,
        "qualifying_causal_comparisons": causal,
        "qualifying_retrieval_comparisons": rescue,
        "qualifying_matched_budget_retrieval_comparisons": (
            matched_retrieval
        ),
        "qualifying_retrieval_contexts": sorted(retrieval_contexts),
        "qualifying_retrieval_selectivity_comparisons": (
            selective_retrieval
        ),
        "claim_boundary": (
            "A v145 ranking is a functional head mechanism only when G1 "
            "passes. QK-V correspondence remains descriptive unless it "
            "aligns with a passing downstream intervention. Retrieval is "
            "a method candidate only when G2 also passes."
        ),
    }


def analyze(
    *,
    profile_dir: Path,
    probe_plan_path: Path,
    output_dir: Path,
    expected_count: int = 64,
) -> dict:
    plan, plan_sha256 = _load_plan(probe_plan_path)
    profiles, audits = _load_profiles(
        profile_dir,
        expected_count=expected_count,
        plan_sha256=plan_sha256,
        probe_count=len(plan["probes"]),
    )
    downstream = _downstream_rows(profiles)
    comparisons = _downstream_comparisons(downstream)
    band_names = tuple(
        name
        for name in (
            "top_recent4_early",
            "top_recent4_middle",
            "top_recent4_late",
        )
        if any(
            str(probe["name"]) == name for probe in plan["probes"]
        )
    )
    layer_bands = (
        _absolute_probe_summary(downstream, band_names)
        if band_names
        else []
    )
    motion = _motion_rows(profiles)
    motion_summary = _motion_group_summary(motion, plan)
    report = _gate_report(downstream, comparisons)
    report.update(
        {
            "profile_count": len(profiles),
            "prompt_count": len(
                {row["prompt_slot"] for row in audits}
            ),
            "seed_replicates": sorted(
                {row["seed_replicate"] for row in audits}
            ),
            "downstream_observation_count": len(downstream),
            "motion_head_observation_count": len(motion),
            "probe_plan": str(probe_plan_path),
            "probe_plan_sha256": plan_sha256,
            "source_axis": plan["source"]["selected_axis"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_audit.csv", audits)
    _write_csv(
        output_dir / "downstream_observations.csv.gz", downstream
    )
    _write_csv(output_dir / "downstream_comparisons.csv", comparisons)
    _write_csv(output_dir / "layer_band_effects.csv", layer_bands)
    _write_csv(output_dir / "qkv_head_observations.csv.gz", motion)
    _write_csv(output_dir / "qkv_group_comparisons.csv", motion_summary)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate_lines = "\n".join(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in report["gates"].items()
    )
    source = report["source_axis"]
    markdown = f"""# v147 Causal Transport Profiling Results

## Integrity and mechanism gates

{gate_lines}

- Native replay maximum relative RMS: `{report['native_replay_max_relative_rms']:.6g}`
- Source ranking: `{source['variant']} / {source['axis']}`
- Profiles: `{report['profile_count']}`; prompts: `{report['prompt_count']}`; seeds per prompt: `2`

## Interpretation

`G1` is required before assigning a functional role to the v145 ranking.
`G2` is required before turning Q-retrieval into a proposed cache design.
QK-V correspondence is a transport-alignment diagnostic, not optical flow
and not an independently validated motion-head label.

## Artifacts

- `profile_audit.csv`
- `downstream_observations.csv.gz`
- `downstream_comparisons.csv`
- `layer_band_effects.csv`
- `qkv_head_observations.csv.gz`
- `qkv_group_comparisons.csv`
- `report.json`
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=64)
    args = parser.parse_args()
    report = analyze(
        profile_dir=args.profile_dir,
        probe_plan_path=args.probe_plan,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
