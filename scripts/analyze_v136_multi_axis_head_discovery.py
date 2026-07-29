#!/usr/bin/env python3
"""Derive independent functional head axes from frozen v134 profiles.

This script is intentionally read-only with respect to v134 artifacts. It
does not choose a cache policy and does not use video metrics, PF labels, or
legacy head counts to construct a classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch


EPS = 1e-4
EXPECTED_LAYERS = 30
EXPECTED_HEADS = 12


def _finite(values: Iterable[float]) -> list[float]:
    return [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]


def _median(values: Iterable[float]) -> float:
    values = _finite(values)
    return float(statistics.median(values)) if values else float("nan")


def _mean(values: Iterable[float]) -> float:
    values = _finite(values)
    return float(statistics.fmean(values)) if values else float("nan")


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(_finite(values))
    if not ordered:
        return float("nan")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1 - alpha) + ordered[upper] * alpha


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return float("nan")
    left_rank = _rank([pair[0] for pair in pairs])
    right_rank = _rank([pair[1] for pair in pairs])
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator > 0 else float("nan")


def _relative_per_head(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            "paired signatures must have identical [heads, features] shape"
        )
    left = left.float()
    right = right.float()
    numerator = (left - right).norm(dim=-1)
    denominator = 0.5 * (left.norm(dim=-1) + right.norm(dim=-1))
    return numerator / denominator.clamp_min(1e-6)


def _signature_rms_energy(signature: torch.Tensor) -> torch.Tensor:
    if signature.ndim != 2 or signature.shape[-1] % 2 != 0:
        raise ValueError("signature must be [heads, 2 * feature_groups]")
    rms = signature.float()[:, signature.shape[-1] // 2 :]
    return rms.square().mean(dim=-1).clamp_min(0).sqrt()


def _normalized_probabilities(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp_min(0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _js_per_head(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("temporal probabilities must align as [heads, frames]")
    left = _normalized_probabilities(left)
    right = _normalized_probabilities(right)
    midpoint = 0.5 * (left + right)
    left_term = (
        left
        * (
            left.clamp_min(1e-8).log()
            - midpoint.clamp_min(1e-8).log()
        )
    ).sum(dim=-1)
    right_term = (
        right
        * (
            right.clamp_min(1e-8).log()
            - midpoint.clamp_min(1e-8).log()
        )
    ).sum(dim=-1)
    return 0.5 * (left_term + right_term)


def _wasserstein_per_head(
    left: torch.Tensor,
    right: torch.Tensor,
    frame_ids: torch.Tensor,
) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("temporal probabilities must align as [heads, frames]")
    if frame_ids.ndim != 1 or frame_ids.numel() != left.shape[-1]:
        raise ValueError("frame ids must align with temporal probabilities")
    order = frame_ids.float().argsort()
    left = _normalized_probabilities(left).index_select(-1, order)
    right = _normalized_probabilities(right).index_select(-1, order)
    ordered_ids = frame_ids.float().index_select(0, order)
    if ordered_ids.numel() <= 1:
        return torch.zeros(left.shape[0], dtype=torch.float32)
    spacing = ordered_ids[1:] - ordered_ids[:-1]
    support_range = (ordered_ids[-1] - ordered_ids[0]).clamp_min(1.0)
    cdf_gap = (left.cumsum(-1) - right.cumsum(-1)).abs()
    return (cdf_gap[:, :-1] * spacing).sum(dim=-1) / support_range


def _record_key(record: dict) -> tuple:
    return (
        str(record["mode"]),
        int(record["current_frame"]),
        int(record["nominal_timestep"]),
        int(record["layer"]),
    )


def _load_profiles(directory: Path) -> list[dict]:
    profiles = []
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) != 2:
            raise ValueError(f"unsupported v134 profile version in {path}")
        if "job" not in payload or "records" not in payload:
            raise ValueError(f"malformed profile payload in {path}")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def _audit_profile_contract(
    profiles: list[dict],
    *,
    kind: str,
    expected_states: int,
) -> list[dict]:
    expected_branches = (
        {"base"}
        if kind == "observational"
        else {"base", "semantic", "null"}
    )
    expected_layers = set(range(EXPECTED_LAYERS))
    output = []
    failures = []
    for profile in profiles:
        job = profile["job"]
        job_id = str(job["job_id"])
        if str(job["kind"]) != kind:
            failures.append(
                f"{job_id}: kind={job['kind']!r}, expected {kind!r}"
            )
            continue
        layer_groups = defaultdict(set)
        branch_states = defaultdict(set)
        tensor_failures = []
        for record in profile["records"]:
            branch = str(record["branch"])
            state = (
                str(record["mode"]),
                int(record["current_frame"]),
                int(record["nominal_timestep"]),
            )
            layer_groups[(branch,) + state].add(int(record["layer"]))
            branch_states[branch].add(state)
            for tensor_name in (
                "residual_signature",
                "native_signature",
                "query_signature",
                "current_key_signature",
                "temporal_logits",
                "temporal_probs",
            ):
                tensor = record[tensor_name]
                if tensor.ndim != 2 or tensor.shape[0] != EXPECTED_HEADS:
                    tensor_failures.append(
                        f"{tensor_name}:{tuple(tensor.shape)}"
                    )
            if (
                record["history_frame_ids"].ndim != 1
                or record["history_frame_ids"].numel()
                != record["temporal_probs"].shape[-1]
            ):
                tensor_failures.append("history_frame_ids")
        observed_branches = set(branch_states)
        bad_layer_groups = [
            key
            for key, layers in layer_groups.items()
            if layers != expected_layers
        ]
        state_sets_match = (
            observed_branches == expected_branches
            and all(
                branch_states[branch] == branch_states["base"]
                for branch in expected_branches
            )
        )
        passed = (
            observed_branches == expected_branches
            and len(branch_states.get("base", ())) == expected_states
            and state_sets_match
            and not bad_layer_groups
            and not tensor_failures
        )
        row = {
            "job_id": job_id,
            "kind": kind,
            "records": len(profile["records"]),
            "base_states": len(branch_states.get("base", ())),
            "observed_branches": ",".join(sorted(observed_branches)),
            "bad_layer_groups": len(bad_layer_groups),
            "tensor_failures": len(tensor_failures),
            "passed": int(passed),
        }
        output.append(row)
        if not passed:
            failures.append(
                f"{job_id}: contract failure {row}; "
                f"layer_examples={bad_layer_groups[:2]}; "
                f"tensor_examples={tensor_failures[:2]}"
            )
    if failures:
        raise ValueError(
            "v134 profile contract audit failed:\n" + "\n".join(failures[:8])
        )
    return output


def _temporal_metrics(record: dict, recent_frames: int) -> dict[str, torch.Tensor]:
    probs = _normalized_probabilities(record["temporal_probs"])
    logits = record["temporal_logits"].float()
    frame_ids = record["history_frame_ids"].float()
    if (
        probs.ndim != 2
        or logits.shape != probs.shape
        or frame_ids.ndim != 1
        or frame_ids.numel() != probs.shape[-1]
    ):
        raise ValueError("malformed temporal tensors")
    current_frame = float(record["current_frame"])
    ages = current_frame - frame_ids
    recent = ages <= float(recent_frames)
    older = ~recent
    history_count = probs.shape[-1]
    entropy_denominator = math.log(max(2, history_count))
    entropy = -(
        probs * probs.clamp_min(1e-8).log()
    ).sum(dim=-1) / entropy_denominator
    expected_age = (probs * ages).sum(dim=-1)
    peak_age = ages[probs.argmax(dim=-1)]
    recent_mass = probs[:, recent].sum(dim=-1)
    old_mass = probs[:, older].sum(dim=-1)
    old12_mass = probs[:, ages > 12].sum(dim=-1)
    uniform_old_mass = float(older.sum().item()) / max(1, history_count)
    centered = logits - logits.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1).clamp_min(1e-8).sqrt()

    if recent.any() and older.any():
        middle_recent_margin = (
            logits[:, older].mean(dim=-1)
            - logits[:, recent].mean(dim=-1)
        ) / scale
    else:
        middle_recent_margin = torch.full(
            (logits.shape[0],), float("nan")
        )

    if history_count > 1:
        signs = logits >= 0
        sign_switch_rate = (
            signs[:, 1:] != signs[:, :-1]
        ).float().mean(dim=-1)
        age_centered = ages - ages.mean()
        age_scale = age_centered.square().sum().clamp_min(1e-8).sqrt()
        logit_scale = centered.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        age_logit_correlation = (
            centered * age_centered
        ).sum(dim=-1) / (age_scale * logit_scale)
    else:
        sign_switch_rate = torch.zeros(logits.shape[0])
        age_logit_correlation = torch.zeros(logits.shape[0])

    dominant_period = torch.full((logits.shape[0],), float("nan"))
    spectral_peak_ratio = torch.full((logits.shape[0],), float("nan"))
    if history_count >= 6:
        power = torch.fft.rfft(centered, dim=-1).abs().square()
        non_dc = power[:, 1:]
        total = non_dc.sum(dim=-1)
        peak, peak_index = non_dc.max(dim=-1)
        valid = total > 1e-8
        spectral_peak_ratio[valid] = peak[valid] / total[valid]
        frequency_index = peak_index.float() + 1.0
        dominant_period[valid] = (
            float(history_count) / frequency_index[valid]
        )

    return {
        "expected_age": expected_age,
        "normalized_expected_age": expected_age
        / ages.max().clamp_min(1.0),
        "recent_mass": recent_mass,
        "old_mass": old_mass,
        "old12_mass": old12_mass,
        "old_mass_excess": old_mass - uniform_old_mass,
        "temporal_entropy": entropy,
        "peak_age": peak_age,
        "positive_logit_fraction": (logits > 0).float().mean(dim=-1),
        "sign_switch_rate": sign_switch_rate,
        "dominant_period": dominant_period,
        "spectral_peak_ratio": spectral_peak_ratio,
        "middle_recent_margin": middle_recent_margin,
        "age_logit_correlation": age_logit_correlation,
        "history_count": torch.full(
            (logits.shape[0],), float(history_count)
        ),
        "eligible_long_history": torch.full(
            (logits.shape[0],), float(bool(recent.any() and older.any()))
        ),
    }


def _validate_branch_alignment(base: dict, other: dict) -> None:
    for key in (
        "mode",
        "current_frame",
        "nominal_timestep",
        "layer",
        "history_frames",
        "recent_frames",
    ):
        if base[key] != other[key]:
            raise ValueError(f"counterfactual branch mismatch for {key}")
    if not torch.equal(
        base["history_frame_ids"].cpu(),
        other["history_frame_ids"].cpu(),
    ):
        raise ValueError("counterfactual branches used different history ids")


def _collect_counterfactual(
    profiles: list[dict],
    recent_frames: int,
) -> tuple[list[dict], list[dict]]:
    rows = []
    audit = []
    for profile in profiles:
        job = profile["job"]
        records: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for record in profile["records"]:
            key = _record_key(record)
            branch = str(record["branch"])
            if branch in records[key]:
                raise ValueError(
                    f"duplicate {branch} record for {job['job_id']} {key}"
                )
            records[key][branch] = record
        for key, branches in records.items():
            missing = {"base", "semantic", "null"} - set(branches)
            if missing:
                audit.append(
                    {
                        "job_id": str(job["job_id"]),
                        "record_key": repr(key),
                        "status": "missing_branch",
                        "detail": ",".join(sorted(missing)),
                    }
                )
                continue
            base = branches["base"]
            semantic = branches["semantic"]
            null = branches["null"]
            _validate_branch_alignment(base, semantic)
            _validate_branch_alignment(base, null)
            base_temporal = _temporal_metrics(base, recent_frames)
            eligible = bool(
                base_temporal["eligible_long_history"][0].item()
            )
            status = "primary" if eligible else "negative_control"
            audit.append(
                {
                    "job_id": str(job["job_id"]),
                    "record_key": repr(key),
                    "status": status,
                    "detail": (
                        ""
                        if eligible
                        else "full history contains no frame older than recent"
                    ),
                }
            )

            signature_names = (
                "residual_signature",
                "native_signature",
                "query_signature",
                "current_key_signature",
            )
            distances = {}
            for name in signature_names:
                distances[f"semantic_{name}"] = _relative_per_head(
                    semantic[name], base[name]
                )
                distances[f"null_{name}"] = _relative_per_head(
                    null[name], base[name]
                )
            semantic_js = _js_per_head(
                semantic["temporal_probs"], base["temporal_probs"]
            )
            null_js = _js_per_head(
                null["temporal_probs"], base["temporal_probs"]
            )
            semantic_w1 = _wasserstein_per_head(
                semantic["temporal_probs"],
                base["temporal_probs"],
                base["history_frame_ids"],
            )
            null_w1 = _wasserstein_per_head(
                null["temporal_probs"],
                base["temporal_probs"],
                base["history_frame_ids"],
            )
            semantic_temporal = _temporal_metrics(semantic, recent_frames)
            null_temporal = _temporal_metrics(null, recent_frames)
            residual_energy = _signature_rms_energy(
                base["residual_signature"]
            )
            native_energy = _signature_rms_energy(base["native_signature"])
            temporal_reach = residual_energy / native_energy.clamp_min(1e-6)

            for head in range(EXPECTED_HEADS):
                semantic_interaction = float(
                    distances["semantic_residual_signature"][head]
                )
                null_interaction = float(
                    distances["null_residual_signature"][head]
                )
                semantic_query = float(
                    distances["semantic_query_signature"][head]
                )
                null_query = float(distances["null_query_signature"][head])
                rows.append(
                    {
                        "dataset_index": int(job["dataset_index"]),
                        "job_id": str(job["job_id"]),
                        "family_id": str(
                            job.get("family_id", job["job_id"])
                        ),
                        "factor": str(job.get("factor", "unknown")),
                        "mode": str(base["mode"]),
                        "current_frame": int(base["current_frame"]),
                        "nominal_timestep": int(base["nominal_timestep"]),
                        "layer": int(base["layer"]),
                        "head": head,
                        "eligible_long_history": int(eligible),
                        "cphi_semantic": semantic_interaction,
                        "cphi_null": null_interaction,
                        "cphi_score": math.log(
                            (semantic_interaction + EPS)
                            / (null_interaction + EPS)
                        ),
                        "age_js_semantic": float(semantic_js[head]),
                        "age_js_null": float(null_js[head]),
                        "age_js_score": math.log(
                            (float(semantic_js[head]) + EPS)
                            / (float(null_js[head]) + EPS)
                        ),
                        "age_w1_semantic": float(semantic_w1[head]),
                        "age_w1_null": float(null_w1[head]),
                        "age_w1_score": math.log(
                            (float(semantic_w1[head]) + EPS)
                            / (float(null_w1[head]) + EPS)
                        ),
                        "semantic_expected_age_delta": float(
                            semantic_temporal["expected_age"][head]
                            - base_temporal["expected_age"][head]
                        ),
                        "null_expected_age_delta": float(
                            null_temporal["expected_age"][head]
                            - base_temporal["expected_age"][head]
                        ),
                        "native_score": math.log(
                            (
                                float(
                                    distances[
                                        "semantic_native_signature"
                                    ][head]
                                )
                                + EPS
                            )
                            / (
                                float(
                                    distances["null_native_signature"][head]
                                )
                                + EPS
                            )
                        ),
                        "query_score": math.log(
                            (semantic_query + EPS) / (null_query + EPS)
                        ),
                        "current_key_score": math.log(
                            (
                                float(
                                    distances[
                                        "semantic_current_key_signature"
                                    ][head]
                                )
                                + EPS
                            )
                            / (
                                float(
                                    distances[
                                        "null_current_key_signature"
                                    ][head]
                                )
                                + EPS
                            )
                        ),
                        "cphi_to_query_ratio": semantic_interaction
                        / (semantic_query + EPS),
                        "temporal_reach_ratio": float(temporal_reach[head]),
                    }
                )
    return rows, audit


TEMPORAL_FIELDS = (
    "temporal_reach_ratio",
    "expected_age",
    "normalized_expected_age",
    "recent_mass",
    "old_mass",
    "old12_mass",
    "old_mass_excess",
    "temporal_entropy",
    "peak_age",
    "positive_logit_fraction",
    "sign_switch_rate",
    "dominant_period",
    "spectral_peak_ratio",
    "middle_recent_margin",
    "age_logit_correlation",
)


def _collect_temporal(
    profiles: list[dict],
    recent_frames: int,
) -> list[dict]:
    rows = []
    for profile in profiles:
        job = profile["job"]
        for record in profile["records"]:
            if str(record["branch"]) != "base":
                continue
            metrics = _temporal_metrics(record, recent_frames)
            residual_energy = _signature_rms_energy(
                record["residual_signature"]
            )
            native_energy = _signature_rms_energy(
                record["native_signature"]
            )
            metrics["temporal_reach_ratio"] = (
                residual_energy / native_energy.clamp_min(1e-6)
            )
            for head in range(EXPECTED_HEADS):
                row = {
                    "dataset_index": int(job["dataset_index"]),
                    "job_id": str(job["job_id"]),
                    "kind": str(job["kind"]),
                    "factor": str(job.get("factor", "natural")),
                    "mode": str(record["mode"]),
                    "current_frame": int(record["current_frame"]),
                    "nominal_timestep": int(record["nominal_timestep"]),
                    "layer": int(record["layer"]),
                    "head": head,
                    "eligible_long_history": int(
                        metrics["eligible_long_history"][head].item()
                    ),
                }
                for field in TEMPORAL_FIELDS:
                    row[field] = float(metrics[field][head])
                rows.append(row)
    return rows


def _aggregate(
    rows: list[dict],
    keys: tuple[str, ...],
    fields: tuple[str, ...],
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[name] for name in keys)
        for field in fields:
            grouped[key][field].append(row[field])
    output = []
    for key, values in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        for field in fields:
            row[field] = _median(values[field])
            row[f"{field}_samples"] = len(_finite(values[field]))
        output.append(row)
    return output


def _two_stage_aggregate(
    rows: list[dict],
    *,
    keys: tuple[str, ...],
    cluster_key: str,
    fields: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    within = _aggregate(rows, (cluster_key,) + keys, fields)
    across = _aggregate(within, keys, fields)
    return across, within


def _bootstrap_sign(
    rows: list[dict],
    *,
    cluster_key: str,
    fields: tuple[str, ...],
    rounds: int,
    seed: int,
) -> dict[tuple[int, int], dict]:
    family_rows = _aggregate(
        rows,
        (cluster_key, "layer", "head"),
        fields,
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for row in family_rows:
        key = (int(row["layer"]), int(row["head"]))
        for field in fields:
            value = float(row[field])
            if math.isfinite(value):
                grouped[key][field].append(value)
    output = defaultdict(dict)
    for key in (
        (layer, head)
        for layer in range(EXPECTED_LAYERS)
        for head in range(EXPECTED_HEADS)
    ):
        for field_index, field in enumerate(fields):
            values = grouped[key][field]
            samples: list[float] = []
            if values:
                generator = torch.Generator()
                generator.manual_seed(
                    seed
                    + key[0] * 1009
                    + key[1] * 97
                    + field_index * 7919
                )
                value_tensor = torch.tensor(values, dtype=torch.float64)
                indices = torch.randint(
                    0,
                    len(values),
                    (rounds, len(values)),
                    generator=generator,
                )
                # The lower median is sufficient for sign confidence and is
                # substantially faster than sorting every bootstrap draw in
                # Python.
                samples = (
                    value_tensor[indices]
                    .median(dim=1)
                    .values.cpu()
                    .tolist()
                )
            probability = _mean(value > 0 for value in samples)
            output[key][f"{field}_bootstrap_low"] = _percentile(
                samples, 0.025
            )
            output[key][f"{field}_bootstrap_high"] = _percentile(
                samples, 0.975
            )
            output[key][f"{field}_bootstrap_p_positive"] = probability
            output[key][f"{field}_bootstrap_confidence"] = (
                max(probability, 1.0 - probability)
                if math.isfinite(probability)
                else float("nan")
            )
            output[key][f"{field}_cluster_count"] = len(values)
    return dict(output)


def _split_half_reproducibility(
    rows: list[dict],
    *,
    cluster_key: str,
    fields: tuple[str, ...],
) -> dict[str, dict]:
    clusters = sorted({str(row[cluster_key]) for row in rows})
    left_clusters = set(clusters[::2])
    right_clusters = set(clusters[1::2])
    output = {}
    for field in fields:
        left = _aggregate(
            [row for row in rows if str(row[cluster_key]) in left_clusters],
            ("layer", "head"),
            (field,),
        )
        right = _aggregate(
            [row for row in rows if str(row[cluster_key]) in right_clusters],
            ("layer", "head"),
            (field,),
        )
        left_map = {
            (int(row["layer"]), int(row["head"])): float(row[field])
            for row in left
        }
        right_map = {
            (int(row["layer"]), int(row["head"])): float(row[field])
            for row in right
        }
        keys = [
            (layer, head)
            for layer in range(EXPECTED_LAYERS)
            for head in range(EXPECTED_HEADS)
        ]
        left_values = [left_map.get(key, float("nan")) for key in keys]
        right_values = [right_map.get(key, float("nan")) for key in keys]
        agreements = [
            int(a > 0) == int(b > 0)
            for a, b in zip(left_values, right_values)
            if math.isfinite(a) and math.isfinite(b)
        ]
        output[field] = {
            "left_cluster_count": len(left_clusters),
            "right_cluster_count": len(right_clusters),
            "spearman": _spearman(left_values, right_values),
            "zero_label_agreement": _mean(agreements),
        }
    return output


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _fit_gmm_1d(values: list[float], components: int) -> dict:
    ordered = sorted(_finite(values))
    count = len(ordered)
    if count < components:
        raise ValueError("not enough samples for GMM")
    global_mean = sum(ordered) / count
    global_variance = max(
        1e-4,
        sum((value - global_mean) ** 2 for value in ordered) / count,
    )
    means = [
        ordered[min(count - 1, int((index + 0.5) * count / components))]
        for index in range(components)
    ]
    variances = [global_variance] * components
    weights = [1.0 / components] * components

    previous = None
    for _ in range(200):
        responsibilities = []
        log_likelihood = 0.0
        for value in ordered:
            logs = [
                math.log(max(weights[index], 1e-12))
                - 0.5 * math.log(2.0 * math.pi * variances[index])
                - 0.5 * (value - means[index]) ** 2 / variances[index]
                for index in range(components)
            ]
            normalizer = _logsumexp(logs)
            responsibilities.append(
                [math.exp(item - normalizer) for item in logs]
            )
            log_likelihood += normalizer
        if previous is not None and abs(log_likelihood - previous) < 1e-8:
            break
        previous = log_likelihood
        for index in range(components):
            mass = max(
                1e-8,
                sum(row[index] for row in responsibilities),
            )
            mean = sum(
                row[index] * value
                for row, value in zip(responsibilities, ordered)
            ) / mass
            variance = sum(
                row[index] * (value - mean) ** 2
                for row, value in zip(responsibilities, ordered)
            ) / mass
            weights[index] = mass / count
            means[index] = mean
            variances[index] = max(1e-4, variance)

    order = sorted(range(components), key=lambda index: means[index])
    means = [means[index] for index in order]
    variances = [variances[index] for index in order]
    weights = [weights[index] for index in order]
    log_likelihood = sum(
        _logsumexp(
            [
                math.log(max(weights[index], 1e-12))
                - 0.5 * math.log(2.0 * math.pi * variances[index])
                - 0.5 * (value - means[index]) ** 2 / variances[index]
                for index in range(components)
            ]
        )
        for value in ordered
    )
    return {
        "components": components,
        "means": means,
        "variances": variances,
        "weights": weights,
        "log_likelihood": log_likelihood,
        "bic": -2.0 * log_likelihood
        + (3 * components - 1) * math.log(count),
    }


def _gmm_threshold(model: dict) -> float:
    if int(model["components"]) != 2:
        raise ValueError("threshold requires a two-component GMM")
    low, high = model["means"]
    if high <= low:
        return float(low)
    best_value = 0.5 * (low + high)
    best_gap = float("inf")
    for index in range(4097):
        value = low + (high - low) * index / 4096
        logs = [
            math.log(max(model["weights"][component], 1e-12))
            - 0.5 * math.log(
                2.0 * math.pi * model["variances"][component]
            )
            - 0.5
            * (value - model["means"][component]) ** 2
            / model["variances"][component]
            for component in range(2)
        ]
        gap = abs(logs[0] - logs[1])
        if gap < best_gap:
            best_gap = gap
            best_value = value
    return float(best_value)


def _otsu_threshold(values: list[float]) -> float:
    ordered = sorted(_finite(values))
    if len(ordered) < 2:
        return float("nan")
    total_mean = sum(ordered) / len(ordered)
    best_score = -1.0
    best = 0.5 * (ordered[0] + ordered[-1])
    prefix = 0.0
    for index in range(len(ordered) - 1):
        prefix += ordered[index]
        if ordered[index] == ordered[index + 1]:
            continue
        left_count = index + 1
        right_count = len(ordered) - left_count
        left_mean = prefix / left_count
        right_mean = (
            total_mean * len(ordered) - prefix
        ) / right_count
        score = left_count * right_count * (left_mean - right_mean) ** 2
        if score > best_score:
            best_score = score
            best = 0.5 * (ordered[index] + ordered[index + 1])
    return float(best)


def _axis_diagnostics(
    rows: list[dict],
    fields: tuple[str, ...],
) -> tuple[list[dict], dict]:
    table = []
    models = {}
    for field in fields:
        values = _finite(row.get(field, float("nan")) for row in rows)
        if len(values) < 3:
            table.append(
                {
                    "axis": field,
                    "count": len(values),
                    "median": _median(values),
                    "q25": _percentile(values, 0.25),
                    "q75": _percentile(values, 0.75),
                    "zero_positive": sum(value > 0 for value in values),
                    "zero_non_positive": sum(value <= 0 for value in values),
                    "otsu_threshold": float("nan"),
                    "gmm2_threshold": float("nan"),
                    "gmm2_high": 0,
                    "gmm2_low": 0,
                    "bic1": float("nan"),
                    "bic2": float("nan"),
                    "bic3": float("nan"),
                    "bic1_minus_bic2": float("nan"),
                    "bic3_minus_bic2": float("nan"),
                }
            )
            models[field] = {"status": "insufficient_finite_values"}
            continue
        gmms = [_fit_gmm_1d(values, count) for count in (1, 2, 3)]
        threshold = _gmm_threshold(gmms[1])
        zero_positive = sum(value > 0 for value in values)
        gmm_positive = sum(value > threshold for value in values)
        row = {
            "axis": field,
            "count": len(values),
            "median": _median(values),
            "q25": _percentile(values, 0.25),
            "q75": _percentile(values, 0.75),
            "zero_positive": zero_positive,
            "zero_non_positive": len(values) - zero_positive,
            "otsu_threshold": _otsu_threshold(values),
            "gmm2_threshold": threshold,
            "gmm2_high": gmm_positive,
            "gmm2_low": len(values) - gmm_positive,
            "bic1": gmms[0]["bic"],
            "bic2": gmms[1]["bic"],
            "bic3": gmms[2]["bic"],
            "bic1_minus_bic2": gmms[0]["bic"] - gmms[1]["bic"],
            "bic3_minus_bic2": gmms[2]["bic"] - gmms[1]["bic"],
        }
        table.append(row)
        models[field] = {
            "gmm": gmms,
            "otsu_threshold": row["otsu_threshold"],
            "gmm2_threshold": threshold,
        }
    return table, models


def _axis_correlations(
    rows: list[dict],
    fields: tuple[str, ...],
) -> list[dict]:
    output = []
    for left_index, left in enumerate(fields):
        for right in fields[left_index + 1 :]:
            output.append(
                {
                    "left_axis": left,
                    "right_axis": right,
                    "spearman": _spearman(
                        [row.get(left, float("nan")) for row in rows],
                        [row.get(right, float("nan")) for row in rows],
                    ),
                }
            )
    return output


def _axis_specialization(
    rows: list[dict],
    *,
    category_fields: tuple[str, ...],
    axes: tuple[str, ...],
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        category = ":".join(str(row[field]) for field in category_fields)
        for axis in axes:
            grouped[key][axis][category] = float(row[axis])
    output = []
    for key, axis_values in sorted(grouped.items()):
        row = {"layer": key[0], "head": key[1]}
        for axis, by_category in axis_values.items():
            finite = {
                category: value
                for category, value in by_category.items()
                if math.isfinite(value)
            }
            values = list(finite.values())
            if not values:
                continue
            dominant_category, dominant_value = max(
                finite.items(), key=lambda item: abs(item[1])
            )
            absolute = [abs(value) for value in values]
            absolute_sum = sum(absolute)
            if absolute_sum > 0 and len(absolute) > 1:
                weights = [value / absolute_sum for value in absolute]
                entropy = -sum(
                    weight * math.log(max(weight, 1e-12))
                    for weight in weights
                ) / math.log(len(weights))
            else:
                entropy = 0.0
            row[f"{axis}_category_count"] = len(values)
            row[f"{axis}_dominant_category"] = dominant_category
            row[f"{axis}_dominant_value"] = dominant_value
            row[f"{axis}_range"] = max(values) - min(values)
            row[f"{axis}_std"] = (
                float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            )
            row[f"{axis}_positive_fraction"] = _mean(
                value > 0 for value in values
            )
            row[f"{axis}_absolute_entropy"] = entropy
        output.append(row)
    return output


def _context_stability(
    rows: list[dict],
    *,
    context_fields: tuple[str, ...],
    axes: tuple[str, ...],
) -> list[dict]:
    contexts = sorted(
        {
            tuple(str(row[field]) for field in context_fields)
            for row in rows
        }
    )
    maps = defaultdict(dict)
    for row in rows:
        context = tuple(str(row[field]) for field in context_fields)
        key = (int(row["layer"]), int(row["head"]))
        for axis in axes:
            maps[(context, axis)][key] = float(row[axis])
    head_keys = [
        (layer, head)
        for layer in range(EXPECTED_LAYERS)
        for head in range(EXPECTED_HEADS)
    ]
    output = []
    for left_index, left in enumerate(contexts):
        for right in contexts[left_index + 1 :]:
            for axis in axes:
                left_values = [
                    maps[(left, axis)].get(key, float("nan"))
                    for key in head_keys
                ]
                right_values = [
                    maps[(right, axis)].get(key, float("nan"))
                    for key in head_keys
                ]
                agreements = [
                    int(a > 0) == int(b > 0)
                    for a, b in zip(left_values, right_values)
                    if math.isfinite(a) and math.isfinite(b)
                ]
                output.append(
                    {
                        "context_fields": ":".join(context_fields),
                        "left_context": ":".join(left),
                        "right_context": ":".join(right),
                        "axis": axis,
                        "spearman": _spearman(left_values, right_values),
                        "zero_label_agreement": _mean(agreements),
                    }
                )
    return output


def _merge_head_rows(
    prompt_rows: list[dict],
    temporal_rows: list[dict],
    prompt_bootstrap: dict[tuple[int, int], dict],
    temporal_bootstrap: dict[tuple[int, int], dict],
) -> list[dict]:
    prompt_map = {
        (int(row["layer"]), int(row["head"])): row for row in prompt_rows
    }
    temporal_map = {
        (int(row["layer"]), int(row["head"])): row for row in temporal_rows
    }
    output = []
    for layer in range(EXPECTED_LAYERS):
        for head in range(EXPECTED_HEADS):
            key = (layer, head)
            if key not in prompt_map or key not in temporal_map:
                raise ValueError(f"incomplete head grid at {key}")
            row = {"layer": layer, "head": head}
            for source in (prompt_map[key], temporal_map[key]):
                for field, value in source.items():
                    if field not in {"layer", "head"}:
                        row[field] = value
            row.update(prompt_bootstrap.get(key, {}))
            row.update(temporal_bootstrap.get(key, {}))

            prompt_conditional = float(row["cphi_score"]) > 0
            age_conditional = float(row["age_js_score"]) > 0
            history_supportive = float(row["middle_recent_margin"]) >= 0
            long_range_attending = float(row["old_mass_excess"]) >= 0
            long_range_consensus = (
                history_supportive and long_range_attending
            )
            row["prompt_label"] = (
                "prompt_conditional"
                if prompt_conditional
                else "prompt_invariant"
            )
            row["age_routing_label"] = (
                "age_conditional"
                if age_conditional
                else "age_invariant"
            )
            row["history_polarity_label"] = (
                "history_supportive"
                if history_supportive
                else "recent_preferred"
            )
            row["long_range_label"] = (
                "long_range"
                if long_range_consensus
                else "local_or_mixed"
            )
            row["exploratory_joint_role"] = (
                ("conditional" if prompt_conditional else "invariant")
                + "_"
                + ("long" if long_range_consensus else "local")
            )
            row["prompt_age_label_agreement"] = int(
                prompt_conditional == age_conditional
            )
            row["history_axis_label_agreement"] = int(
                history_supportive == long_range_attending
            )
            output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, float)
                        and not math.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


PROMPT_FIELDS = (
    "cphi_semantic",
    "cphi_null",
    "cphi_score",
    "age_js_semantic",
    "age_js_null",
    "age_js_score",
    "age_w1_semantic",
    "age_w1_null",
    "age_w1_score",
    "semantic_expected_age_delta",
    "null_expected_age_delta",
    "native_score",
    "query_score",
    "current_key_score",
    "cphi_to_query_ratio",
    "temporal_reach_ratio",
)


def analyze(
    observational_profiles: list[dict],
    counterfactual_profiles: list[dict],
    *,
    output_dir: Path,
    recent_frames: int,
    expected_count: int,
    expected_states: int,
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> dict:
    if len(observational_profiles) != expected_count:
        raise ValueError(
            "observational profile count mismatch: "
            f"{len(observational_profiles)} != {expected_count}"
        )
    if len(counterfactual_profiles) != expected_count:
        raise ValueError(
            "counterfactual profile count mismatch: "
            f"{len(counterfactual_profiles)} != {expected_count}"
        )
    profile_contract = _audit_profile_contract(
        observational_profiles,
        kind="observational",
        expected_states=expected_states,
    ) + _audit_profile_contract(
        counterfactual_profiles,
        kind="counterfactual",
        expected_states=expected_states,
    )
    controlled, state_audit = _collect_counterfactual(
        counterfactual_profiles, recent_frames
    )
    natural = _collect_temporal(observational_profiles, recent_frames)
    eligible_controlled = [
        row for row in controlled if row["eligible_long_history"]
    ]
    negative_control = [
        row for row in controlled if not row["eligible_long_history"]
    ]
    eligible_natural = [
        row for row in natural if row["eligible_long_history"]
    ]
    if not eligible_controlled or not eligible_natural:
        raise ValueError("no eligible long-history observations")

    prompt_head, prompt_job = _two_stage_aggregate(
        eligible_controlled,
        keys=("layer", "head"),
        cluster_key="job_id",
        fields=PROMPT_FIELDS,
    )
    temporal_head, temporal_job = _two_stage_aggregate(
        eligible_natural,
        keys=("layer", "head"),
        cluster_key="job_id",
        fields=TEMPORAL_FIELDS,
    )
    prompt_bootstrap = _bootstrap_sign(
        eligible_controlled,
        cluster_key="family_id",
        fields=("cphi_score", "age_js_score", "age_w1_score"),
        rounds=bootstrap_rounds,
        seed=bootstrap_seed,
    )
    temporal_bootstrap = _bootstrap_sign(
        eligible_natural,
        cluster_key="job_id",
        fields=("middle_recent_margin", "old_mass_excess"),
        rounds=bootstrap_rounds,
        seed=bootstrap_seed + 100_000,
    )
    head_axes = _merge_head_rows(
        prompt_head,
        temporal_head,
        prompt_bootstrap,
        temporal_bootstrap,
    )

    factor_axes, _ = _two_stage_aggregate(
        eligible_controlled,
        keys=("factor", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "cphi_score",
            "age_js_score",
            "age_w1_score",
            "native_score",
            "query_score",
            "current_key_score",
        ),
    )
    timestep_axes, _ = _two_stage_aggregate(
        eligible_controlled,
        keys=("mode", "nominal_timestep", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "cphi_score",
            "age_js_score",
            "age_w1_score",
            "native_score",
            "query_score",
        ),
    )
    ar_axes, _ = _two_stage_aggregate(
        eligible_controlled,
        keys=("mode", "current_frame", "layer", "head"),
        cluster_key="family_id",
        fields=(
            "cphi_score",
            "age_js_score",
            "age_w1_score",
            "temporal_reach_ratio",
        ),
    )
    natural_timestep_axes, _ = _two_stage_aggregate(
        eligible_natural,
        keys=("mode", "nominal_timestep", "layer", "head"),
        cluster_key="job_id",
        fields=TEMPORAL_FIELDS,
    )
    natural_ar_axes, _ = _two_stage_aggregate(
        eligible_natural,
        keys=("mode", "current_frame", "layer", "head"),
        cluster_key="job_id",
        fields=TEMPORAL_FIELDS,
    )
    factor_specialization = _axis_specialization(
        factor_axes,
        category_fields=("factor",),
        axes=("cphi_score", "age_js_score", "age_w1_score"),
    )
    timestep_specialization = _axis_specialization(
        timestep_axes,
        category_fields=("mode", "nominal_timestep"),
        axes=("cphi_score", "age_js_score", "age_w1_score"),
    )
    ar_specialization = _axis_specialization(
        ar_axes,
        category_fields=("mode", "current_frame"),
        axes=("cphi_score", "age_js_score", "age_w1_score"),
    )
    context_stability = (
        _context_stability(
            factor_axes,
            context_fields=("factor",),
            axes=("cphi_score", "age_js_score"),
        )
        + _context_stability(
            timestep_axes,
            context_fields=("mode", "nominal_timestep"),
            axes=("cphi_score", "age_js_score"),
        )
        + _context_stability(
            ar_axes,
            context_fields=("mode", "current_frame"),
            axes=("cphi_score", "age_js_score"),
        )
    )

    diagnostic_fields = (
        "cphi_score",
        "age_js_score",
        "age_w1_score",
        "middle_recent_margin",
        "old_mass_excess",
        "temporal_reach_ratio",
        "sign_switch_rate",
        "spectral_peak_ratio",
        "age_logit_correlation",
    )
    diagnostics, mixture_models = _axis_diagnostics(
        head_axes, diagnostic_fields
    )
    correlations = _axis_correlations(head_axes, diagnostic_fields)
    prompt_split = _split_half_reproducibility(
        eligible_controlled,
        cluster_key="family_id",
        fields=("cphi_score", "age_js_score", "age_w1_score"),
    )
    temporal_split = _split_half_reproducibility(
        eligible_natural,
        cluster_key="job_id",
        fields=("middle_recent_margin", "old_mass_excess"),
    )

    label_counts = {
        field: dict(
            sorted(
                {
                    str(value): sum(
                        row[field] == value for row in head_axes
                    )
                    for value in {row[field] for row in head_axes}
                }.items()
            )
        )
        for field in (
            "prompt_label",
            "age_routing_label",
            "history_polarity_label",
            "long_range_label",
            "exploratory_joint_role",
        )
    }
    cphi_reliable = _mean(
        row.get("cphi_score_bootstrap_confidence", 0.0) >= 0.80
        for row in head_axes
    )
    temporal_reliable = _mean(
        row.get("middle_recent_margin_bootstrap_confidence", 0.0) >= 0.80
        for row in head_axes
    )
    prompt_minority = min(label_counts["prompt_label"].values()) / len(
        head_axes
    )
    temporal_minority = min(
        label_counts["history_polarity_label"].values()
    ) / len(head_axes)
    cphi_split = prompt_split["cphi_score"]["spearman"]
    temporal_split_score = temporal_split["middle_recent_margin"]["spearman"]
    prompt_gate = (
        _median(row["cphi_semantic"] for row in head_axes)
        > _median(row["cphi_null"] for row in head_axes)
        and cphi_split >= 0.30
        and cphi_reliable >= 0.70
        and prompt_minority >= 0.10
    )
    temporal_gate = (
        temporal_split_score >= 0.30
        and temporal_reliable >= 0.70
        and temporal_minority >= 0.10
    )
    cphi_temporal_correlation = next(
        row["spearman"]
        for row in correlations
        if {
            row["left_axis"],
            row["right_axis"],
        }
        == {"cphi_score", "middle_recent_margin"}
    )
    if prompt_gate and temporal_gate and abs(cphi_temporal_correlation) < 0.85:
        recommendation = "dual_axis_prompt_and_temporal"
    elif prompt_gate:
        recommendation = "prompt_axis_with_continuous_temporal_diagnostics"
    elif temporal_gate:
        recommendation = "temporal_axis_only_prompt_taxonomy_rejected"
    else:
        recommendation = "continuous_scores_only_no_static_taxonomy"

    negative_summary = {
        "row_count": len(negative_control),
        "state_count": sum(
            item["status"] == "negative_control" for item in state_audit
        ),
        "median_cphi_semantic": _median(
            row["cphi_semantic"] for row in negative_control
        ),
        "median_cphi_null": _median(
            row["cphi_null"] for row in negative_control
        ),
        "median_temporal_reach_ratio": _median(
            row["temporal_reach_ratio"] for row in negative_control
        ),
    }
    report = {
        "method": "v136_multi_axis_analysis_of_frozen_v134_profiles",
        "profile_counts": {
            "observational": len(observational_profiles),
            "counterfactual": len(counterfactual_profiles),
        },
        "profile_contract_passed": all(
            bool(row["passed"]) for row in profile_contract
        ),
        "observation_counts": {
            "controlled_all": len(controlled),
            "controlled_primary": len(eligible_controlled),
            "controlled_negative": len(negative_control),
            "natural_all": len(natural),
            "natural_primary": len(eligible_natural),
        },
        "recent_frames": recent_frames,
        "head_count": len(head_axes),
        "label_counts": label_counts,
        "negative_control": negative_summary,
        "split_half": {
            "prompt": prompt_split,
            "temporal": temporal_split,
        },
        "reliable_fraction": {
            "cphi": cphi_reliable,
            "middle_recent_margin": temporal_reliable,
        },
        "minority_fraction": {
            "prompt": prompt_minority,
            "history_polarity": temporal_minority,
        },
        "cphi_temporal_spearman": cphi_temporal_correlation,
        "context_stability": {
            "pair_count": len(context_stability),
            "median_spearman": _median(
                row["spearman"] for row in context_stability
            ),
            "minimum_spearman": min(
                _finite(row["spearman"] for row in context_stability),
                default=float("nan"),
            ),
            "median_zero_label_agreement": _median(
                row["zero_label_agreement"] for row in context_stability
            ),
        },
        "gates": {
            "prompt_axis": prompt_gate,
            "temporal_axis": temporal_gate,
        },
        "recommendation": recommendation,
        "mixture_models_are_diagnostic_only": True,
        "mixture_models": mixture_models,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "head_axes.csv", head_axes)
    _write_csv(output_dir / "head_prompt_job_axes.csv", prompt_job)
    _write_csv(output_dir / "head_temporal_job_axes.csv", temporal_job)
    _write_csv(output_dir / "head_factor_axes.csv", factor_axes)
    _write_csv(output_dir / "head_timestep_axes.csv", timestep_axes)
    _write_csv(output_dir / "head_ar_axes.csv", ar_axes)
    _write_csv(
        output_dir / "head_natural_timestep_axes.csv",
        natural_timestep_axes,
    )
    _write_csv(output_dir / "head_natural_ar_axes.csv", natural_ar_axes)
    _write_csv(
        output_dir / "head_factor_specialization.csv",
        factor_specialization,
    )
    _write_csv(
        output_dir / "head_timestep_specialization.csv",
        timestep_specialization,
    )
    _write_csv(
        output_dir / "head_ar_specialization.csv",
        ar_specialization,
    )
    _write_csv(output_dir / "context_stability.csv", context_stability)
    _write_csv(output_dir / "axis_diagnostics.csv", diagnostics)
    _write_csv(output_dir / "axis_correlations.csv", correlations)
    _write_csv(output_dir / "state_eligibility_audit.csv", state_audit)
    _write_csv(output_dir / "profile_contract_audit.csv", profile_contract)
    _write_json(output_dir / "multi_axis_report.json", report)

    summary = [
        "# v136 Multi-Axis Head Analysis",
        "",
        "This report analyzes frozen v134 profiles. It does not use video "
        "metrics, PF labels, or legacy class counts to select a map.",
        "",
        "## Decision",
        "",
        f"- Recommendation: `{recommendation}`",
        f"- Prompt-axis gate: `{prompt_gate}`",
        f"- Temporal-axis gate: `{temporal_gate}`",
        f"- CPHI/temporal Spearman: `{cphi_temporal_correlation:.4f}`",
        "",
        "## Class Counts",
        "",
    ]
    for field, counts in label_counts.items():
        summary.append(
            f"- `{field}`: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
        )
    summary.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- CPHI split-half Spearman: `{cphi_split:.4f}`",
            f"- CPHI bootstrap-reliable fraction: `{cphi_reliable:.4f}`",
            (
                "- Middle/recent split-half Spearman: "
                f"`{temporal_split_score:.4f}`"
            ),
            (
                "- Middle/recent bootstrap-reliable fraction: "
                f"`{temporal_reliable:.4f}`"
            ),
            "",
            "## Negative Control",
            "",
            (
                "- Ineligible states are excluded from primary scores because "
                "their full history contains no frame older than recent4."
            ),
            (
                "- Median semantic/null residual responses: "
                f"`{negative_summary['median_cphi_semantic']:.6f}` / "
                f"`{negative_summary['median_cphi_null']:.6f}`"
            ),
            "",
            "## Interpretation Boundary",
            "",
            "- `prompt_label` is the zero-threshold CPHI hypothesis.",
            "- `age_routing_label` asks whether prompt semantics change which "
            "history ages receive attention.",
            "- `history_polarity_label` is a native-window middle-vs-recent "
            "diagnostic, not the superseded v98 304/56 map.",
            "- `exploratory_joint_role` must not be used for generation until "
            "the corresponding prompt and temporal gates pass.",
            "- GMM and Otsu thresholds are diagnostics only.",
            "",
        ]
    )
    (output_dir / "multi_axis_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observational-dir", type=Path, required=True)
    parser.add_argument("--counterfactual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument("--expected-states", type=int, default=27)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.recent_frames < 1:
        raise ValueError("--recent-frames must be positive")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    report = analyze(
        _load_profiles(args.observational_dir),
        _load_profiles(args.counterfactual_dir),
        output_dir=args.output_dir,
        recent_frames=args.recent_frames,
        expected_count=args.expected_count,
        expected_states=args.expected_states,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        "[v136] "
        f"heads={report['head_count']} "
        f"recommendation={report['recommendation']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
