#!/usr/bin/env python3
"""Analyze v143 natural-region and A-B output-causal head profiles."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from lifecycle_kv.head_taxonomy import (
    dummy_forcing_labels,
    forcing_kv_labels,
    head_forcing_labels,
)


def _load_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V141 = _load_script("v141_analysis", "analyze_v141_full_prompt_switch_profiles.py")
V142 = _load_script("v142_analysis", "analyze_v142_output_causal_profiles.py")

LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
BRANCHES = {"base", "exact_a", "exact_b", "paraphrase_a", "paraphrase_b"}
AB_SWITCH = 57
EPSILON = 1e-6
AB_CONTEXT_AXES = (
    "prompt_history_excess",
    "policy_prompt_score",
    "stale_a_mass",
    "persistent_content",
    "persistent_positioned",
    "persistent_output",
)


def _median(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _profile_files(directory: Path, expected: int) -> list[Path]:
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} profiles in {directory}, found {len(paths)}"
        )
    return paths


def _load(path: Path, expected_kind: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != 6:
        raise RuntimeError(f"{path} is not a version-6 profile")
    metadata = payload.get("metadata") or {}
    if metadata.get("incomplete_calls"):
        raise RuntimeError(f"{path} contains incomplete calls")
    if not metadata.get("region_attention_metrics"):
        raise RuntimeError(f"{path} does not contain region attention metrics")
    if (
        metadata.get("region_attention_method")
        != "sampled_token_softmax_cartesian"
    ):
        raise RuntimeError(f"{path} uses an unsupported region mass estimator")
    if str(payload.get("job", {}).get("kind")) != expected_kind:
        raise RuntimeError(f"{path} has the wrong job kind")
    return payload


def _head_slice(layer: int) -> slice:
    return slice(layer * HEADS, (layer + 1) * HEADS)


def _region(record: dict, field: str) -> np.ndarray:
    metrics = record.get("region_attention_metrics") or {}
    if field not in metrics:
        raise KeyError(f"missing region attention field: {field}")
    return V142._as_numpy(metrics[field])


def _majority(strings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(strings, axis=0)
    labels = []
    fractions = []
    for head in range(stacked.shape[1]):
        counts = Counter(stacked[:, head].tolist())
        label, count = sorted(
            counts.items(), key=lambda item: (-item[1], str(item[0]))
        )[0]
        labels.append(str(label))
        fractions.append(count / stacked.shape[0])
    return np.asarray(labels), np.asarray(fractions, dtype=np.float64)


def _natural_profile(path: Path) -> tuple[dict, dict, dict]:
    payload = _load(path, "multiaxis_region_natural")
    job = payload["job"]
    state_values: dict[tuple, dict[str, np.ndarray]] = {}
    for record in payload["records"]:
        if str(record["branch"]) != "base":
            raise RuntimeError(f"{path} natural profile contains shadow branches")
        layer = int(record["layer"])
        head_slice = _head_slice(layer)
        key = (
            str(record["mode"]),
            int(record["current_frame"]),
            int(record["nominal_timestep"]),
        )
        state = state_values.setdefault(
            key,
            {
                name: np.full(TOTAL_HEADS, np.nan)
                for name in (
                    "current_mass",
                    "oldest1_mass",
                    "global_sink1_mass",
                    "middle_mass",
                    "recent4_mass",
                    "last4_mass",
                    "recent4_non_oldest_ratio",
                    "history_positive_rate",
                    "history_mean_logit",
                )
            },
        )
        for name in (
            "current_mass",
            "oldest1_mass",
            "global_sink1_mass",
            "middle_mass",
            "recent4_mass",
            "last4_mass",
            "recent4_non_oldest_ratio",
        ):
            state[name][head_slice] = _region(record, name)
        region = record["region_attention_metrics"]
        frame_ids = V142._as_numpy(region["frame_ids"]).astype(np.int64)
        logits = V142._as_numpy(region["frame_logits"])
        history_mask = frame_ids < int(record["current_frame"])
        if not history_mask.any():
            raise RuntimeError(f"{path} has a region state without history")
        history_logits = logits[:, history_mask]
        state["history_positive_rate"][head_slice] = (
            history_logits > 0
        ).mean(axis=-1)
        state["history_mean_logit"][head_slice] = history_logits.mean(axis=-1)
        if int(record["current_frame"]) == 9:
            if not bool(region["global_sink_available"]):
                raise RuntimeError(f"{path} frame-9 calibration lacks frame 0")
    if not state_values:
        raise RuntimeError(f"{path} has no natural region states")
    calibration_states = [
        state
        for key, state in state_values.items()
        if int(key[1]) == 9
    ]
    if len(calibration_states) != 3:
        raise RuntimeError(
            f"{path} expected three frame-9 calibration states, "
            f"found {len(calibration_states)}"
        )
    calibration = {
        name: np.median(
            np.stack([state[name] for state in calibration_states], axis=0),
            axis=0,
        )
        for name in (
            "current_mass",
            "global_sink1_mass",
            "last4_mass",
            "oldest1_mass",
        )
    }
    calibration["non_oldest_mass"] = 1.0 - calibration["oldest1_mass"]
    if any(not np.isfinite(value).all() for value in calibration.values()):
        raise RuntimeError(f"{path} has incomplete frame-9 calibration")
    prompt_axes = {
        name: np.median(
            np.stack([state[name] for state in state_values.values()], axis=0),
            axis=0,
        )
        for name in next(iter(state_values.values()))
    }
    labels = {
        "forcing_kv": forcing_kv_labels(
            calibration["last4_mass"],
            calibration["non_oldest_mass"],
        ),
        "head_forcing": head_forcing_labels(
            calibration["global_sink1_mass"],
            calibration["current_mass"],
        ),
        "dummy_forcing": dummy_forcing_labels(calibration["current_mass"]),
    }
    audit = {
        "dataset_index": int(job["dataset_index"]),
        "state_count": len(state_values),
        "record_count": len(payload["records"]),
        "calibration_frame": 9,
        "calibration_state_count": len(calibration_states),
    }
    return job, prompt_axes, labels, audit


def analyze_natural(
    paths: list[Path],
) -> tuple[dict, list[dict], list[dict]]:
    jobs, axes, labels, audits = [], [], [], []
    for path in paths:
        job, prompt_axes, prompt_labels, audit = _natural_profile(path)
        jobs.append(job)
        axes.append(prompt_axes)
        labels.append(prompt_labels)
        audits.append(audit)
    order = np.argsort([int(job["dataset_index"]) for job in jobs])
    jobs = [jobs[index] for index in order]
    axes = [axes[index] for index in order]
    labels = [labels[index] for index in order]
    if [int(job["dataset_index"]) for job in jobs] != list(range(128)):
        raise RuntimeError("natural dataset indices are incomplete")
    indices = np.arange(128)
    discovery = indices % 2 == 0
    validation = ~discovery
    feature_names = sorted(axes[0])
    stacked = {
        name: np.stack([row[name] for row in axes], axis=0)
        for name in feature_names
    }
    label_outputs = {}
    for name in sorted(labels[0]):
        values = [row[name] for row in labels]
        full_label, full_fraction = _majority(values)
        discovery_label, _ = _majority(
            [value for value, keep in zip(values, discovery) if keep]
        )
        validation_label, _ = _majority(
            [value for value, keep in zip(values, validation) if keep]
        )
        label_outputs[name] = {
            "full": full_label,
            "fraction": full_fraction,
            "discovery": discovery_label,
            "validation": validation_label,
        }
    policy_report, policy_rows, _, _ = V142.analyze_natural(paths)
    policy_map = {
        (int(row["layer"]), int(row["head"])): row for row in policy_rows
    }
    head_rows = []
    for flat_head in range(TOTAL_HEADS):
        layer, head = divmod(flat_head, HEADS)
        row = {"layer": layer, "head": head}
        for name in feature_names:
            row[f"discovery_{name}"] = float(
                np.median(stacked[name][discovery, flat_head])
            )
            row[f"validation_{name}"] = float(
                np.median(stacked[name][validation, flat_head])
            )
        for name, output in label_outputs.items():
            row[f"{name}_label"] = str(output["full"][flat_head])
            row[f"{name}_vote_fraction"] = float(
                output["fraction"][flat_head]
            )
            row[f"{name}_split_agree"] = int(
                output["discovery"][flat_head]
                == output["validation"][flat_head]
            )
        row.update(
            {
                key: value
                for key, value in policy_map[(layer, head)].items()
                if key not in {"layer", "head"}
            }
        )
        head_rows.append(row)
    report = {
        "profile_count": len(paths),
        "states_per_profile": sorted(
            {int(row["state_count"]) for row in audits}
        ),
        "calibration": (
            "median over noisy-1000, noisy-500, and clean at frame 9; "
            "global frame 0 is still inside the native window"
        ),
        "published_taxonomy_fidelity": {
            "forcing_kv": (
                "official open-source threshold formula, remeasured on v143"
            ),
            "head_forcing": (
                "paper-formula reimplementation; v143 calibration protocol "
                "differs from the paper's 20-prompt random-AR protocol"
            ),
            "dummy_forcing": (
                "paper top-25%-current diagnostic; not the full dynamic "
                "head-programming runtime"
            ),
            "pyramid_forcing": (
                "official published label map is used only as a post-hoc "
                "reference in clustering"
            ),
        },
        "long_context_scope": (
            "native 21-frame Self-Forcing window, not persistent full history"
        ),
        "label_split_agreement": {
            name: float(
                np.mean(output["discovery"] == output["validation"])
            )
            for name, output in label_outputs.items()
        },
        "label_counts": {
            name: dict(Counter(output["full"].tolist()))
            for name, output in label_outputs.items()
        },
        "output_causal_policy": policy_report,
    }
    return report, head_rows, audits


def _signature_score(
    branches: dict[str, dict],
    field: str,
    *,
    active_a: bool,
) -> np.ndarray:
    matched = "exact_a" if active_a else "exact_b"
    opposite = "exact_b" if active_a else "exact_a"
    paraphrase = "paraphrase_a" if active_a else "paraphrase_b"
    base = branches["base"][field]
    switch = V141._relative(branches[opposite][field], base).numpy()
    control = V141._relative(branches[paraphrase][field], base).numpy()
    parity = V141._relative(branches[matched][field], base).numpy()
    return np.stack(
        (
            np.log((switch + EPSILON) / (control + EPSILON)),
            parity,
        ),
        axis=-1,
    )


def _profile_ab(path: Path) -> tuple[dict, dict, dict, dict]:
    payload = _load(path, "multiaxis_full_prompt_ab")
    job = payload["job"]
    groups = V142._branch_groups(payload)
    values = defaultdict(lambda: [[] for _ in range(TOTAL_HEADS)])
    contexts = {}
    matching_parity = []
    reconstruction_max, reconstruction_rms = [], []
    for key, branches in groups.items():
        if set(branches) != BRANCHES:
            raise RuntimeError(f"{path} state {key} has incomplete branches")
        _, frame, _, layer = key
        active_a = frame < AB_SWITCH
        signature = _signature_score(
            branches, "residual_signature", active_a=active_a
        )
        query = _signature_score(
            branches, "query_signature", active_a=active_a
        )
        prompt_history_excess = signature[:, 0] - query[:, 0]
        vectors = {
            branch: V142._normalized_policy_vector(record)
            for branch, record in branches.items()
        }
        exact_distance = V142._l1(vectors["exact_a"], vectors["exact_b"])
        paraphrase_distance = 0.5 * (
            V142._l1(vectors["exact_a"], vectors["paraphrase_a"])
            + V142._l1(vectors["exact_b"], vectors["paraphrase_b"])
        )
        policy_score = np.log(
            (exact_distance + EPSILON)
            / (paraphrase_distance + EPSILON)
        )
        head_slice = _head_slice(layer)
        base = branches["base"]
        base_probs = V141._normalize_probabilities(
            base["temporal_probs"]
        ).numpy()
        frame_ids = V142._as_numpy(base["history_frame_ids"]).astype(np.int64)
        stale_a_mass = base_probs[:, frame_ids < AB_SWITCH].sum(axis=-1)
        stale_a_visible = bool((frame_ids < AB_SWITCH).any())
        context_key = (
            str(base["mode"]),
            int(frame),
            int(base["nominal_timestep"]),
        )
        context = contexts.setdefault(
            context_key,
            {
                "axes": {
                    name: np.full(TOTAL_HEADS, np.nan)
                    for name in AB_CONTEXT_AXES
                },
                "stale_a_visible": stale_a_visible,
            },
        )
        if bool(context["stale_a_visible"]) != stale_a_visible:
            raise RuntimeError(
                f"{path} has inconsistent history visibility at {context_key}"
            )
        context_axes = context["axes"]
        context_axes["prompt_history_excess"][head_slice] = (
            prompt_history_excess
        )
        context_axes["policy_prompt_score"][head_slice] = policy_score
        context_axes["stale_a_mass"][head_slice] = stale_a_mass
        for local, flat_head in enumerate(
            range(head_slice.start, head_slice.stop)
        ):
            values["prompt_history_excess"][flat_head].append(
                prompt_history_excess[local]
            )
            values["policy_prompt_score"][flat_head].append(policy_score[local])
            values["stale_a_mass"][flat_head].append(stale_a_mass[local])
            split = "A" if active_a else "B"
            values[f"prompt_history_excess_{split}"][flat_head].append(
                prompt_history_excess[local]
            )
            values[f"policy_prompt_score_{split}"][flat_head].append(
                policy_score[local]
            )
            values[f"stale_a_mass_{split}"][flat_head].append(
                stale_a_mass[local]
            )
            if not active_a and stale_a_visible:
                values["stale_a_mass_B_visible"][flat_head].append(
                    stale_a_mass[local]
                )
        matched = "exact_a" if active_a else "exact_b"
        matching_parity.extend(
            np.abs(
                V142._raw_policy_vector(base)
                - V142._raw_policy_vector(branches[matched])
            ).reshape(-1)
        )
        metadata = base["causal_policy_metadata"]
        reconstruction_max.append(
            metadata["native_reconstruction_relative_max"]
        )
        reconstruction_rms.append(
            metadata["native_reconstruction_relative_rms"]
        )
        if "persistent_probe_metrics" not in base:
            continue
        for name, metric, lower_is_better in (
            ("persistent_content", "content_top1_cosine", False),
            (
                "persistent_positioned",
                "positioned_top1_cosine",
                False,
            ),
            (
                "persistent_output",
                "output_projected_relative_error",
                True,
            ),
        ):
            selectivity, noise = V142._persistent_selectivity(
                branches, metric, lower_is_better=lower_is_better
            )
            context_axes[name][head_slice] = selectivity
            for local, flat_head in enumerate(
                range(head_slice.start, head_slice.stop)
            ):
                values[name][flat_head].append(selectivity[local])
                values[f"{name}_noise"][flat_head].append(noise[local])
                split = "A" if active_a else "B"
                values[f"{name}_{split}"][flat_head].append(
                    selectivity[local]
                )

    result = {
        name: np.asarray([_median(items) for items in rows])
        for name, rows in values.items()
    }
    for key, context in contexts.items():
        incomplete = [
            name
            for name, value in context["axes"].items()
            if not np.isfinite(value).all()
        ]
        if incomplete:
            raise RuntimeError(
                f"{path} has incomplete AB context {key}: {incomplete}"
            )
    audit = {
        "dataset_index": int(job["dataset_index"]),
        "family_index": int(job["family_index"]),
        "switch_type": str(job["switch_type"]),
        "group_count": len(groups),
        "matching_parity_p99": float(np.quantile(matching_parity, 0.99)),
        "reconstruction_max": max(reconstruction_max),
        "reconstruction_rms": max(reconstruction_rms),
    }
    return job, result, audit, contexts


def _family_matrix(jobs: list[dict], values: np.ndarray) -> np.ndarray:
    rows = []
    for family in range(16):
        indices = [
            index
            for index, job in enumerate(jobs)
            if int(job["family_index"]) == family
        ]
        if len(indices) != 2:
            raise RuntimeError(f"family {family} lacks two switch types")
        rows.append(np.median(values[indices], axis=0))
    return np.stack(rows, axis=0)


def _top_fraction_jaccard(
    left: np.ndarray,
    right: np.ndarray,
    *,
    fraction: float = 0.25,
) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        return float("nan")
    count = max(1, int(math.ceil(indices.size * fraction)))
    left_top = set(indices[np.argsort(left[indices], kind="mergesort")[-count:]])
    right_top = set(
        indices[np.argsort(right[indices], kind="mergesort")[-count:]]
    )
    return float(len(left_top & right_top) / len(left_top | right_top))


def _ab_context_analysis(
    jobs: list[dict],
    contexts: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    entries = []
    context_rows = []
    for switch_type in sorted({str(job["switch_type"]) for job in jobs}):
        all_indices = [
            index
            for index, job in enumerate(jobs)
            if str(job["switch_type"]) == switch_type
        ]
        if len(all_indices) != 16:
            raise RuntimeError(
                f"switch_type={switch_type} has {len(all_indices)} profiles"
            )
        state_keys = set(contexts[all_indices[0]])
        for index in all_indices[1:]:
            if set(contexts[index]) != state_keys:
                raise RuntimeError(
                    f"AB context states disagree for switch_type={switch_type}"
                )
        split_indices = {
            "all": all_indices,
            "discovery": [
                index
                for index in all_indices
                if int(jobs[index]["family_index"]) % 2 == 0
            ],
            "validation": [
                index
                for index in all_indices
                if int(jobs[index]["family_index"]) % 2 == 1
            ],
        }
        if any(
            len(indices) != expected
            for split, indices in split_indices.items()
            for expected in (16 if split == "all" else 8,)
        ):
            raise RuntimeError(
                f"switch_type={switch_type} has invalid prompt splits"
            )
        for prompt_split, indices in split_indices.items():
            for mode, frame, timestep in sorted(
                state_keys, key=lambda item: (item[1], item[0], item[2])
            ):
                axes = {
                    axis: np.median(
                        np.stack(
                            [
                                contexts[index][
                                    (mode, frame, timestep)
                                ]["axes"][axis]
                                for index in indices
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )
                    for axis in AB_CONTEXT_AXES
                }
                visibility = {
                    bool(
                        contexts[index][(mode, frame, timestep)][
                            "stale_a_visible"
                        ]
                    )
                    for index in indices
                }
                if len(visibility) != 1:
                    raise RuntimeError(
                        "history visibility disagrees across prompt families: "
                        f"{switch_type}/{mode}/{frame}/{timestep}"
                    )
                entry = {
                    "prompt_split": prompt_split,
                    "switch_type": switch_type,
                    "mode": str(mode),
                    "current_frame": int(frame),
                    "nominal_timestep": int(timestep),
                    "episode": "A" if int(frame) < AB_SWITCH else "B",
                    "stale_a_visible": bool(next(iter(visibility))),
                    "axes": axes,
                }
                if prompt_split == "all":
                    entries.append(entry)
                for flat_head in range(TOTAL_HEADS):
                    row = {
                        key: value
                        for key, value in entry.items()
                        if key != "axes"
                    }
                    row["boundary_offset"] = int(frame) - AB_SWITCH
                    row["layer"] = flat_head // HEADS
                    row["head"] = flat_head % HEADS
                    row["stale_a_visible"] = int(row["stale_a_visible"])
                    for axis, values in axes.items():
                        row[axis] = float(values[flat_head])
                    context_rows.append(row)

    groupers = {
        "switch_type": lambda row: row["switch_type"],
        "episode": lambda row: row["episode"],
        "frame": lambda row: f"frame_{row['current_frame']:03d}",
        "denoise_state": lambda row: (
            f"{row['mode']}:t{row['nominal_timestep']}"
        ),
        "switch_type_episode": lambda row: (
            f"{row['switch_type']}:{row['episode']}"
        ),
    }
    stability_rows = []
    summary = {}
    for grouping, grouper in groupers.items():
        for axis in AB_CONTEXT_AXES:
            grouped = defaultdict(list)
            for entry in entries:
                if axis == "stale_a_mass" and not (
                    entry["episode"] == "B" and entry["stale_a_visible"]
                ):
                    continue
                grouped[grouper(entry)].append(entry)
            vectors = {
                label: np.median(
                    np.stack([entry["axes"][axis] for entry in rows], axis=0),
                    axis=0,
                )
                for label, rows in grouped.items()
            }
            axis_rows = []
            for left_label, right_label in itertools.combinations(
                sorted(vectors), 2
            ):
                left = vectors[left_label]
                right = vectors[right_label]
                finite = np.isfinite(left) & np.isfinite(right)
                if finite.sum() != TOTAL_HEADS:
                    raise RuntimeError(
                        f"non-finite context comparison: {grouping}/{axis}"
                    )
                row = {
                    "axis": axis,
                    "grouping": grouping,
                    "left": left_label,
                    "right": right_label,
                    "spearman": V142._spearman(left, right),
                    "sign_agreement": float(
                        np.mean(np.sign(left) == np.sign(right))
                    ),
                    "top25_jaccard": _top_fraction_jaccard(left, right),
                    "median_absolute_delta": float(
                        np.median(np.abs(left - right))
                    ),
                }
                stability_rows.append(row)
                axis_rows.append(row)
            if axis_rows:
                summary[f"{axis}:{grouping}"] = {
                    "pair_count": len(axis_rows),
                    "spearman_median": _median(
                        [row["spearman"] for row in axis_rows]
                    ),
                    "spearman_min": min(
                        row["spearman"] for row in axis_rows
                    ),
                    "sign_agreement_median": _median(
                        [row["sign_agreement"] for row in axis_rows]
                    ),
                    "top25_jaccard_median": _median(
                        [row["top25_jaccard"] for row in axis_rows]
                    ),
                }
    return context_rows, stability_rows, summary


def analyze_ab(
    paths: list[Path],
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    jobs, results, audits, contexts = [], [], [], []
    for path in paths:
        job, result, audit, context = _profile_ab(path)
        jobs.append(job)
        results.append(result)
        audits.append(audit)
        contexts.append(context)
    order = np.argsort([int(job["dataset_index"]) for job in jobs])
    jobs = [jobs[index] for index in order]
    results = [results[index] for index in order]
    audits = [audits[index] for index in order]
    contexts = [contexts[index] for index in order]
    if [int(job["dataset_index"]) for job in jobs] != list(range(32)):
        raise RuntimeError("A-B dataset indices are incomplete")
    keys = sorted(results[0])
    stacked = {
        key: np.stack([result[key] for result in results], axis=0)
        for key in keys
    }
    family = {
        key: _family_matrix(jobs, value) for key, value in stacked.items()
    }
    discovery = np.arange(16) % 2 == 0
    validation = ~discovery
    head_rows = []
    for flat_head in range(TOTAL_HEADS):
        row = {
            "layer": flat_head // HEADS,
            "head": flat_head % HEADS,
        }
        for key in keys:
            row[f"discovery_{key}"] = float(
                np.median(family[key][discovery, flat_head])
            )
            row[f"validation_{key}"] = float(
                np.median(family[key][validation, flat_head])
            )
        head_rows.append(row)
    report = {
        "profile_count": len(paths),
        "switch_frame": AB_SWITCH,
        "purpose": (
            "A-to-B plasticity and stale-A suppression; not A recall"
        ),
        "prompt_history_split_spearman": V142._spearman(
            np.asarray(
                [
                    row["discovery_prompt_history_excess"]
                    for row in head_rows
                ]
            ),
            np.asarray(
                [
                    row["validation_prompt_history_excess"]
                    for row in head_rows
                ]
            ),
        ),
        "persistent_content_split_spearman": V142._spearman(
            np.asarray(
                [row["discovery_persistent_content"] for row in head_rows]
            ),
            np.asarray(
                [row["validation_persistent_content"] for row in head_rows]
            ),
        ),
        "matching_branch_policy_parity_p99": max(
            row["matching_parity_p99"] for row in audits
        ),
        "native_reconstruction_relative_max": max(
            row["reconstruction_max"] for row in audits
        ),
        "native_reconstruction_relative_rms": max(
            row["reconstruction_rms"] for row in audits
        ),
    }
    context_rows, stability_rows, context_summary = _ab_context_analysis(
        jobs, contexts
    )
    report["context_conditioning"] = context_summary
    report["correctness_gate"] = bool(
        report["matching_branch_policy_parity_p99"] <= 1e-4
        and report["native_reconstruction_relative_max"] <= 1e-2
        and report["native_reconstruction_relative_rms"] <= 5e-3
    )
    return report, head_rows, audits, context_rows, stability_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-profile-dir", type=Path, required=True)
    parser.add_argument("--ab-profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    natural, natural_heads, natural_audit = analyze_natural(
        _profile_files(args.natural_profile_dir, 128)
    )
    ab, ab_heads, ab_audit, ab_context, ab_stability = analyze_ab(
        _profile_files(args.ab_profile_dir, 32)
    )
    report = {"version": 1, "natural": natural, "ab": ab}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "natural_head_axes.csv", natural_heads)
    _write_csv(args.output_dir / "ab_head_axes.csv", ab_heads)
    _write_csv(args.output_dir / "natural_profile_audit.csv", natural_audit)
    _write_csv(args.output_dir / "ab_profile_audit.csv", ab_audit)
    _write_csv(args.output_dir / "ab_context_axes.csv", ab_context)
    _write_csv(args.output_dir / "ab_context_stability.csv", ab_stability)
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_agreement = natural["output_causal_policy"][
        "split_best_policy_agreement"
    ]
    summary = f"""# v143 Multi-axis Profile Summary

- Natural profiles: `{natural['profile_count']}`
- A-B profiles: `{ab['profile_count']}`
- A-B switch frame: `{ab['switch_frame']}`
- Natural policy split agreement: `{policy_agreement:.4f}`
- A-B prompt/history split rho: `{ab['prompt_history_split_spearman']:.4f}`
- A-B persistent-A split rho: `{ab['persistent_content_split_spearman']:.4f}`
- Correctness gate: `{ab['correctness_gate']}`

The natural temporal axes cover the native 21-frame Self-Forcing window.
The A-B experiment measures scene plasticity and stale-A suppression. The
per-context tables separate switch type, episode, frame, and denoising-state
dependence instead of assuming a static head identity. Persistent A recall
after an intervening episode remains an A-B-A generation question.
"""
    (args.output_dir / "analysis_summary.md").write_text(
        summary, encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
