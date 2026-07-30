#!/usr/bin/env python3
"""Analyze output-causal policy and persistent A-memory head profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


POLICIES = ("recent_budget", "boundary_recent", "uniform_recent")
BRANCHES = ("base", "exact_a", "exact_b", "paraphrase_a", "paraphrase_b")
LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
EPSILON = 1e-6


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float64)


def _median(values, default=float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float(default)


def _quantile(values, q: float, default=float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float(default)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
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
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left_rank = _rankdata(left[finite])
    right_rank = _rankdata(right[finite])
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _normalized_policy_vector(record: dict) -> np.ndarray:
    columns = []
    policy = record.get("causal_policy_metrics") or {}
    for name in POLICIES:
        if name not in policy:
            raise KeyError(f"missing causal policy candidate: {name}")
        columns.append(
            _as_numpy(policy[name]["projected_relative_error"])
        )
    matrix = np.stack(columns, axis=-1)
    return matrix / matrix.sum(axis=-1, keepdims=True).clip(min=EPSILON)


def _raw_policy_vector(record: dict) -> np.ndarray:
    policy = record.get("causal_policy_metrics") or {}
    return np.stack(
        [
            _as_numpy(policy[name]["projected_relative_error"])
            for name in POLICIES
        ],
        axis=-1,
    )


def _profile_files(directory: Path, expected: int) -> list[Path]:
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} profiles in {directory}, found {len(paths)}"
        )
    return paths


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != 6:
        raise RuntimeError(f"{path} is not a v142 version-6 profile")
    metadata = payload.get("metadata") or {}
    if metadata.get("incomplete_calls"):
        raise RuntimeError(f"{path} contains incomplete calls")
    return payload


def _head_slice(layer: int) -> slice:
    start = int(layer) * HEADS
    return slice(start, start + HEADS)


def _profile_natural(
    path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, list[tuple], dict]:
    payload = _load(path)
    job = payload["job"]
    records = payload["records"]
    if {record["branch"] for record in records} != {"base"}:
        raise RuntimeError(f"{path} natural profile contains shadow branches")
    by_state: dict[tuple, np.ndarray] = {}
    parity_max = []
    parity_rms = []
    for record in records:
        key = (
            record["mode"],
            int(record["current_frame"]),
            int(record["nominal_timestep"]),
        )
        matrix = by_state.setdefault(
            key, np.full((TOTAL_HEADS, len(POLICIES)), np.nan)
        )
        layer = int(record["layer"])
        matrix[_head_slice(layer)] = _raw_policy_vector(record)
        metadata = record["causal_policy_metadata"]
        parity_max.append(metadata["native_reconstruction_relative_max"])
        parity_rms.append(metadata["native_reconstruction_relative_rms"])
    state_keys = sorted(by_state)
    states = np.stack([by_state[key] for key in state_keys], axis=0)
    if not np.isfinite(states).all():
        raise RuntimeError(f"{path} has incomplete natural state/head coverage")
    profile_median = np.median(states, axis=0)
    audit = {
        "dataset_index": int(job["dataset_index"]),
        "state_count": int(states.shape[0]),
        "record_count": len(records),
        "parity_max": max(parity_max),
        "parity_rms": max(parity_rms),
    }
    return job, profile_median, states, state_keys, audit


def _bootstrap_sign_reliability(
    family_matrix: np.ndarray,
    *,
    rounds: int = 500,
    seed: int = 20260730,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    full = np.median(family_matrix, axis=0)
    agreement = np.zeros(full.shape, dtype=np.float64)
    for _ in range(rounds):
        indices = rng.integers(0, family_matrix.shape[0], family_matrix.shape[0])
        estimate = np.median(family_matrix[indices], axis=0)
        agreement += (
            (np.sign(estimate) == np.sign(full))
            & (np.sign(estimate) != 0)
            & (np.sign(full) != 0)
        ).astype(np.float64)
    agreement /= rounds
    return agreement, float(np.mean(agreement >= 0.75))


def analyze_natural(
    paths: list[Path],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    jobs = []
    profile_medians = []
    state_arrays = []
    state_keys_reference = None
    audits = []
    for path in paths:
        job, profile_median, states, state_keys, audit = _profile_natural(path)
        if state_keys_reference is None:
            state_keys_reference = state_keys
        elif state_keys != state_keys_reference:
            raise RuntimeError("natural profiles contain different state grids")
        jobs.append(job)
        profile_medians.append(profile_median)
        state_arrays.append(states)
        audits.append(audit)
    order = np.argsort([int(job["dataset_index"]) for job in jobs])
    jobs = [jobs[index] for index in order]
    profiles = np.stack([profile_medians[index] for index in order], axis=0)
    states = np.stack([state_arrays[index] for index in order], axis=0)
    indices = np.asarray([int(job["dataset_index"]) for job in jobs])
    if not np.array_equal(indices, np.arange(128)):
        raise RuntimeError("natural profile dataset indices are incomplete")
    discovery = indices % 2 == 0
    validation = ~discovery
    discovery_loss = np.median(profiles[discovery], axis=0)
    validation_loss = np.median(profiles[validation], axis=0)
    discovery_label = discovery_loss.argmin(axis=-1)
    validation_label = validation_loss.argmin(axis=-1)
    discovery_sorted = np.sort(discovery_loss, axis=-1)
    discovery_margin = (
        discovery_sorted[:, 1] - discovery_sorted[:, 0]
    ) / discovery_loss.mean(axis=-1).clip(min=EPSILON)
    label_agreement = float(np.mean(discovery_label == validation_label))
    discovery_need = discovery_loss[:, 0] - discovery_loss[:, 1:].min(axis=-1)
    validation_need = validation_loss[:, 0] - validation_loss[:, 1:].min(axis=-1)
    split_rho = _spearman(discovery_need, validation_need)

    state_labels = states.argmin(axis=-1)
    modal_fraction = np.zeros(TOTAL_HEADS, dtype=np.float64)
    modal_label = np.zeros(TOTAL_HEADS, dtype=np.int64)
    for head in range(TOTAL_HEADS):
        counts = np.bincount(
            state_labels[:, :, head].reshape(-1), minlength=len(POLICIES)
        )
        modal_label[head] = int(counts.argmax())
        modal_fraction[head] = float(counts.max() / counts.sum())

    validation_states = states[validation]
    static_loss = np.take_along_axis(
        validation_states,
        discovery_label[None, None, :, None],
        axis=-1,
    )[..., 0]
    oracle_loss = validation_states.min(axis=-1)
    scale = validation_states.mean(axis=-1).clip(min=EPSILON)
    normalized_regret = (static_loss - oracle_loss) / scale
    per_head_regret = np.median(normalized_regret, axis=(0, 1))

    context_loss = np.median(states, axis=0)
    context_label = context_loss.argmin(axis=-1)
    context_need = context_loss[:, :, 0] - context_loss[:, :, 1:].min(axis=-1)
    context_rhos = []
    context_agreements = []
    for left in range(context_loss.shape[0]):
        for right in range(left + 1, context_loss.shape[0]):
            context_rhos.append(
                _spearman(context_need[left], context_need[right])
            )
            context_agreements.append(
                float(np.mean(context_label[left] == context_label[right]))
            )
    context_rows = []
    for state_index, state_key in enumerate(state_keys_reference or []):
        mode, current_frame, nominal_timestep = state_key
        for head in range(TOTAL_HEADS):
            row = {
                "mode": mode,
                "current_frame": current_frame,
                "nominal_timestep": nominal_timestep,
                "layer": head // HEADS,
                "head": head % HEADS,
                "best_policy": POLICIES[context_label[state_index, head]],
                "policy_need": float(context_need[state_index, head]),
            }
            for policy_index, policy in enumerate(POLICIES):
                row[f"{policy}_loss"] = float(
                    context_loss[state_index, head, policy_index]
                )
            context_rows.append(row)

    head_rows = []
    for head in range(TOTAL_HEADS):
        row = {
            "layer": head // HEADS,
            "head": head % HEADS,
            "discovery_best_policy": POLICIES[discovery_label[head]],
            "validation_best_policy": POLICIES[validation_label[head]],
            "split_label_agree": int(
                discovery_label[head] == validation_label[head]
            ),
            "discovery_policy_need": float(discovery_need[head]),
            "validation_policy_need": float(validation_need[head]),
            "modal_state_policy": POLICIES[modal_label[head]],
            "modal_state_fraction": float(modal_fraction[head]),
            "validation_static_regret": float(per_head_regret[head]),
            "discovery_policy_margin": float(discovery_margin[head]),
        }
        for policy_index, policy in enumerate(POLICIES):
            row[f"discovery_{policy}_loss"] = float(
                discovery_loss[head, policy_index]
            )
            row[f"validation_{policy}_loss"] = float(
                validation_loss[head, policy_index]
            )
        head_rows.append(row)

    policy_counts = Counter(POLICIES[index] for index in discovery_label)
    report = {
        "profile_count": len(paths),
        "states_per_profile": sorted(
            {int(array.shape[0]) for array in state_arrays}
        ),
        "split_policy_need_spearman": split_rho,
        "split_best_policy_agreement": label_agreement,
        "discovery_policy_counts": dict(policy_counts),
        "discovery_boundary_fraction": float(
            np.mean(discovery_margin < 0.01)
        ),
        "median_modal_state_fraction": float(np.median(modal_fraction)),
        "context_policy_need_spearman_median": _median(context_rhos),
        "context_policy_need_spearman_min": min(context_rhos),
        "context_best_policy_agreement_median": _median(
            context_agreements
        ),
        "context_best_policy_agreement_min": min(context_agreements),
        "validation_static_regret_median": float(
            np.median(normalized_regret)
        ),
        "validation_static_regret_p90": float(
            np.quantile(normalized_regret, 0.9)
        ),
        "native_reconstruction_relative_max": max(
            audit["parity_max"] for audit in audits
        ),
        "native_reconstruction_relative_rms": max(
            audit["parity_rms"] for audit in audits
        ),
    }
    report["static_policy_gate"] = bool(
        split_rho >= 0.6
        and label_agreement >= 0.8
        and report["median_modal_state_fraction"] >= 0.75
        and report["discovery_boundary_fraction"] <= 0.2
    )
    report["online_policy_opportunity"] = bool(
        report["validation_static_regret_median"] >= 0.02
    )
    report["correctness_gate"] = bool(
        report["native_reconstruction_relative_max"] <= 1e-2
        and report["native_reconstruction_relative_rms"] <= 5e-3
    )
    return report, head_rows, audits, context_rows


def _branch_groups(payload: dict) -> dict[tuple, dict[str, dict]]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for record in payload["records"]:
        key = (
            record["mode"],
            int(record["current_frame"]),
            int(record["nominal_timestep"]),
            int(record["layer"]),
        )
        branch = str(record["branch"])
        if branch in groups[key]:
            raise RuntimeError(f"duplicate branch={branch} for state={key}")
        groups[key][branch] = record
    return groups


def _l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left - right).mean(axis=-1)


def _persistent_selectivity(
    branches: dict[str, dict],
    metric: str,
    *,
    lower_is_better: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = {
        branch: _as_numpy(
            branches[branch]["persistent_probe_metrics"][metric]
        )
        for branch in ("exact_a", "exact_b", "paraphrase_a", "paraphrase_b")
    }
    exact = values["exact_a"] - values["exact_b"]
    if lower_is_better:
        exact = -exact
    paraphrase_noise = 0.5 * (
        np.abs(values["exact_a"] - values["paraphrase_a"])
        + np.abs(values["exact_b"] - values["paraphrase_b"])
    )
    return exact, paraphrase_noise


def _profile_aba(path: Path) -> tuple[dict, dict, dict]:
    payload = _load(path)
    job = payload["job"]
    groups = _branch_groups(payload)
    policy_score = [[] for _ in range(TOTAL_HEADS)]
    policy_episode = {
        "A1": [[] for _ in range(TOTAL_HEADS)],
        "B": [[] for _ in range(TOTAL_HEADS)],
        "A2": [[] for _ in range(TOTAL_HEADS)],
    }
    policy_exact = [[] for _ in range(TOTAL_HEADS)]
    policy_paraphrase = [[] for _ in range(TOTAL_HEADS)]
    persistent = {
        "content": {"all": [[] for _ in range(TOTAL_HEADS)]},
        "positioned": {"all": [[] for _ in range(TOTAL_HEADS)]},
        "output": {"all": [[] for _ in range(TOTAL_HEADS)]},
    }
    for axis in persistent.values():
        axis["B"] = [[] for _ in range(TOTAL_HEADS)]
        axis["A2"] = [[] for _ in range(TOTAL_HEADS)]
        axis["noise"] = [[] for _ in range(TOTAL_HEADS)]
    matching_parity = []
    reconstruction_max = []
    reconstruction_rms = []

    for key, branches in groups.items():
        if set(branches) != set(BRANCHES):
            raise RuntimeError(f"{path} state {key} has incomplete branches")
        mode, frame, timestep, layer = key
        vectors = {
            branch: _normalized_policy_vector(record)
            for branch, record in branches.items()
        }
        exact_distance = _l1(vectors["exact_a"], vectors["exact_b"])
        paraphrase_distance = 0.5 * (
            _l1(vectors["exact_a"], vectors["paraphrase_a"])
            + _l1(vectors["exact_b"], vectors["paraphrase_b"])
        )
        score = np.log(
            (exact_distance + EPSILON) / (paraphrase_distance + EPSILON)
        )
        episode = "A1" if frame < 39 else ("B" if frame < 78 else "A2")
        head_slice = _head_slice(layer)
        for local_head, global_head in enumerate(
            range(head_slice.start, head_slice.stop)
        ):
            policy_score[global_head].append(score[local_head])
            policy_episode[episode][global_head].append(score[local_head])
            policy_exact[global_head].append(exact_distance[local_head])
            policy_paraphrase[global_head].append(
                paraphrase_distance[local_head]
            )
        matching = "exact_b" if 39 <= frame < 78 else "exact_a"
        matching_parity.extend(
            np.abs(
                _raw_policy_vector(branches["base"])
                - _raw_policy_vector(branches[matching])
            ).reshape(-1)
        )
        base_metadata = branches["base"]["causal_policy_metadata"]
        reconstruction_max.append(
            base_metadata["native_reconstruction_relative_max"]
        )
        reconstruction_rms.append(
            base_metadata["native_reconstruction_relative_rms"]
        )

        if "persistent_probe_metrics" not in branches["base"]:
            continue
        episode = "B" if 39 <= frame < 78 else "A2"
        metric_specs = {
            "content": ("content_top1_cosine", False),
            "positioned": ("positioned_top1_cosine", False),
            "output": ("output_projected_relative_error", True),
        }
        for axis, (metric, lower_is_better) in metric_specs.items():
            selectivity, noise = _persistent_selectivity(
                branches,
                metric,
                lower_is_better=lower_is_better,
            )
            for local_head, global_head in enumerate(
                range(head_slice.start, head_slice.stop)
            ):
                persistent[axis]["all"][global_head].append(
                    selectivity[local_head]
                )
                persistent[axis][episode][global_head].append(
                    selectivity[local_head]
                )
                persistent[axis]["noise"][global_head].append(
                    noise[local_head]
                )

    def aggregate(values):
        return np.asarray([_median(items) for items in values])

    result = {
        "policy_score": aggregate(policy_score),
        "policy_exact": aggregate(policy_exact),
        "policy_paraphrase": aggregate(policy_paraphrase),
    }
    for episode, values in policy_episode.items():
        result[f"policy_score_{episode}"] = aggregate(values)
    for axis, values in persistent.items():
        for split, rows in values.items():
            result[f"persistent_{axis}_{split}"] = aggregate(rows)
    audit = {
        "dataset_index": int(job["dataset_index"]),
        "family_index": int(job["family_index"]),
        "switch_type": str(job["switch_type"]),
        "group_count": len(groups),
        "matching_parity_p99": _quantile(matching_parity, 0.99),
        "reconstruction_max": max(reconstruction_max),
        "reconstruction_rms": max(reconstruction_rms),
    }
    return job, result, audit


def _family_matrix(jobs: list[dict], values: np.ndarray) -> np.ndarray:
    rows = []
    for family in range(16):
        indices = [
            index
            for index, job in enumerate(jobs)
            if int(job["family_index"]) == family
        ]
        if len(indices) != 2:
            raise RuntimeError(f"family {family} does not have two switch types")
        rows.append(np.median(values[indices], axis=0))
    return np.stack(rows, axis=0)


def analyze_aba(paths: list[Path]) -> tuple[dict, list[dict], list[dict]]:
    jobs = []
    results = []
    audits = []
    for path in paths:
        job, result, audit = _profile_aba(path)
        jobs.append(job)
        results.append(result)
        audits.append(audit)
    order = np.argsort([int(job["dataset_index"]) for job in jobs])
    jobs = [jobs[index] for index in order]
    results = [results[index] for index in order]
    if [int(job["dataset_index"]) for job in jobs] != list(range(32)):
        raise RuntimeError("A-B-A profile dataset indices are incomplete")
    keys = sorted(results[0])
    stacked = {
        key: np.stack([result[key] for result in results], axis=0)
        for key in keys
    }
    family = {key: _family_matrix(jobs, value) for key, value in stacked.items()}
    discovery = np.arange(16) % 2 == 0
    validation = ~discovery

    policy_discovery = np.median(family["policy_score"][discovery], axis=0)
    policy_validation = np.median(family["policy_score"][validation], axis=0)
    policy_rho = _spearman(policy_discovery, policy_validation)
    policy_sign_agreement = float(
        np.mean(np.sign(policy_discovery) == np.sign(policy_validation))
    )
    policy_reliability, policy_reliable_fraction = (
        _bootstrap_sign_reliability(family["policy_score"])
    )

    content_discovery = np.median(
        family["persistent_content_all"][discovery], axis=0
    )
    content_validation = np.median(
        family["persistent_content_all"][validation], axis=0
    )
    content_rho = _spearman(content_discovery, content_validation)
    content_sign_agreement = float(
        np.mean(np.sign(content_discovery) == np.sign(content_validation))
    )
    content_reliability, content_reliable_fraction = (
        _bootstrap_sign_reliability(family["persistent_content_all"])
    )
    policy_episode_rho = _spearman(
        np.median(family["policy_score_B"], axis=0),
        np.median(family["policy_score_A2"], axis=0),
    )
    policy_episode_sign_agreement = float(
        np.mean(
            np.sign(np.median(family["policy_score_B"], axis=0))
            == np.sign(np.median(family["policy_score_A2"], axis=0))
        )
    )

    policy_exact_median = float(np.median(stacked["policy_exact"]))
    policy_paraphrase_median = float(
        np.median(stacked["policy_paraphrase"])
    )
    content_median = float(
        np.median(stacked["persistent_content_all"])
    )
    content_noise_median = float(
        np.median(stacked["persistent_content_noise"])
    )

    head_rows = []
    for head in range(TOTAL_HEADS):
        head_rows.append(
            {
                "layer": head // HEADS,
                "head": head % HEADS,
                "policy_prompt_score_discovery": float(
                    policy_discovery[head]
                ),
                "policy_prompt_score_validation": float(
                    policy_validation[head]
                ),
                "policy_prompt_bootstrap_agreement": float(
                    policy_reliability[head]
                ),
                "policy_prompt_A1": float(
                    np.median(stacked["policy_score_A1"][:, head])
                ),
                "policy_prompt_B": float(
                    np.median(stacked["policy_score_B"][:, head])
                ),
                "policy_prompt_A2": float(
                    np.median(stacked["policy_score_A2"][:, head])
                ),
                "persistent_content_discovery": float(
                    content_discovery[head]
                ),
                "persistent_content_validation": float(
                    content_validation[head]
                ),
                "persistent_content_bootstrap_agreement": float(
                    content_reliability[head]
                ),
                "persistent_content_B": float(
                    np.median(stacked["persistent_content_B"][:, head])
                ),
                "persistent_content_A2": float(
                    np.median(stacked["persistent_content_A2"][:, head])
                ),
                "persistent_positioned": float(
                    np.median(
                        stacked["persistent_positioned_all"][:, head]
                    )
                ),
                "persistent_output": float(
                    np.median(stacked["persistent_output_all"][:, head])
                ),
            }
        )

    report = {
        "profile_count": len(paths),
        "policy_exact_distance_median": policy_exact_median,
        "policy_paraphrase_distance_median": policy_paraphrase_median,
        "policy_prompt_split_spearman": policy_rho,
        "policy_prompt_sign_agreement": policy_sign_agreement,
        "policy_prompt_bootstrap_reliable_fraction": policy_reliable_fraction,
        "policy_prompt_B_A2_spearman": policy_episode_rho,
        "policy_prompt_B_A2_sign_agreement": (
            policy_episode_sign_agreement
        ),
        "policy_prompt_positive_heads": int(
            (np.median(family["policy_score"], axis=0) > 0).sum()
        ),
        "persistent_content_selectivity_median": content_median,
        "persistent_content_paraphrase_noise_median": content_noise_median,
        "persistent_content_split_spearman": content_rho,
        "persistent_content_sign_agreement": content_sign_agreement,
        "persistent_content_bootstrap_reliable_fraction": (
            content_reliable_fraction
        ),
        "persistent_content_positive_heads": int(
            (
                np.median(
                    family["persistent_content_all"], axis=0
                )
                > 0
            ).sum()
        ),
        "persistent_content_B_median": float(
            np.median(stacked["persistent_content_B"])
        ),
        "persistent_content_A2_median": float(
            np.median(stacked["persistent_content_A2"])
        ),
        "matching_branch_policy_parity_p99": max(
            audit["matching_parity_p99"] for audit in audits
        ),
        "native_reconstruction_relative_max": max(
            audit["reconstruction_max"] for audit in audits
        ),
        "native_reconstruction_relative_rms": max(
            audit["reconstruction_rms"] for audit in audits
        ),
    }
    report["prompt_policy_modulation_gate"] = bool(
        policy_exact_median > policy_paraphrase_median
        and policy_rho >= 0.3
        and policy_reliable_fraction >= 0.7
    )
    report["persistent_a_selectivity_gate"] = bool(
        content_median > 0
        and content_rho >= 0.3
        and content_reliable_fraction >= 0.7
    )
    report["correctness_gate"] = bool(
        report["matching_branch_policy_parity_p99"] <= 1e-4
        and report["native_reconstruction_relative_max"] <= 1e-2
        and report["native_reconstruction_relative_rms"] <= 5e-3
    )
    return report, head_rows, audits


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recommendation(natural: dict, aba: dict) -> str:
    if not natural["correctness_gate"] or not aba["correctness_gate"]:
        return "invalid_correctness_contract_do_not_interpret"
    if natural["static_policy_gate"] and aba["prompt_policy_modulation_gate"]:
        return "static_backbone_with_online_prompt_conditioned_override"
    if (
        natural["online_policy_opportunity"]
        or aba["prompt_policy_modulation_gate"]
        or aba["persistent_a_selectivity_gate"]
    ):
        return "online_context_conditioned_output_causal_routing"
    if natural["static_policy_gate"]:
        return "static_output_causal_policy_map"
    return "no_head_policy_supported_redesign_probe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-profile-dir", type=Path, required=True)
    parser.add_argument("--aba-profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    natural, natural_heads, natural_audit, natural_context = analyze_natural(
        _profile_files(args.natural_profile_dir, 128)
    )
    aba, aba_heads, aba_audit = analyze_aba(
        _profile_files(args.aba_profile_dir, 32)
    )
    recommendation = _recommendation(natural, aba)
    report = {
        "version": 1,
        "recommendation": recommendation,
        "natural": natural,
        "aba": aba,
        "gates": {
            "correctness": bool(
                natural["correctness_gate"] and aba["correctness_gate"]
            ),
            "static_policy": natural["static_policy_gate"],
            "online_policy_opportunity": natural[
                "online_policy_opportunity"
            ],
            "prompt_policy_modulation": aba[
                "prompt_policy_modulation_gate"
            ],
            "persistent_a_selectivity": aba[
                "persistent_a_selectivity_gate"
            ],
        },
        "evidence_boundary": [
            "Policy losses approximate full attention within one layer; they "
            "do not establish final video quality.",
            "The natural SF window contains at most 21 latent frames.",
            "The persistent A archive is a bounded diagnostic sample and is "
            "never written into the base trajectory.",
            "Online-oracle regret is an opportunity estimate, not an "
            "implementable routing result.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "natural_head_summary.csv", natural_heads)
    _write_csv(
        args.output_dir / "natural_context_head_summary.csv",
        natural_context,
    )
    _write_csv(args.output_dir / "aba_head_summary.csv", aba_heads)
    _write_csv(args.output_dir / "natural_profile_audit.csv", natural_audit)
    _write_csv(args.output_dir / "aba_profile_audit.csv", aba_audit)
    summary = f"""# v142 Output-Causal Head Profiling

- Recommendation: `{recommendation}`
- Natural profiles: `{natural['profile_count']}`
- Correctness gate: `{report['gates']['correctness']}`
- Static policy gate: `{natural['static_policy_gate']}`
- Online-policy opportunity: `{natural['online_policy_opportunity']}`
- Policy split Spearman / label agreement: `{natural['split_policy_need_spearman']:.4f}` / `{natural['split_best_policy_agreement']:.4f}`
- Validation static-policy regret median: `{natural['validation_static_regret_median']:.6f}`
- Prompt-policy modulation gate: `{aba['prompt_policy_modulation_gate']}`
- Exact-switch / paraphrase policy distance: `{aba['policy_exact_distance_median']:.6g}` / `{aba['policy_paraphrase_distance_median']:.6g}`
- Persistent-A selectivity gate: `{aba['persistent_a_selectivity_gate']}`
- Persistent content selectivity median: `{aba['persistent_content_selectivity_median']:.6g}`
- Persistent content split Spearman: `{aba['persistent_content_split_spearman']:.4f}`

## Interpretation Boundary

This experiment measures per-head attention-output approximation error and a
read-only sampled A-episode archive. It does not modify the generated
trajectory and cannot by itself establish a generation-quality improvement.
"""
    (args.output_dir / "analysis_summary.md").write_text(
        summary, encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
