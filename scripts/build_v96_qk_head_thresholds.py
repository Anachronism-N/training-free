#!/usr/bin/env python3
"""Build PF-independent binary head maps from frame-level QK profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Pure threshold tests do not require PyTorch.
    torch = None


HeadKey = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--pf-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--strict-gates", action="store_true")
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def mad(values: list[float], center: float | None = None) -> float:
    center = median(values) if center is None else float(center)
    return max(
        1e-6,
        1.4826 * median([abs(value - center) for value in values]),
    )


def read_matrix(path: Path, rows: int, columns: int) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        matrix = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise ValueError(
            f"{path} must have shape {rows}x{columns}, got "
            f"{len(matrix)}x{len(matrix[0]) if matrix else 0}"
        )
    return matrix


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def _record_key(record: dict, include_branch: bool = True) -> tuple:
    key = (
        int(record["layer"]),
        int(record["current_start"]),
        str(record["cache_update_mode"]),
        int(record["call_index"]),
    )
    return key + (str(record["cfg_branch"]),) if include_branch else key


def load_profiles(paths: list[Path]) -> list[dict]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load QK profile tensors")
    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) not in {1, 2, 3}:
            raise ValueError(f"unsupported profile version in {path}")
        metadata = dict(payload.get("metadata") or {})
        records = list(payload.get("records") or [])
        if not records:
            raise ValueError(f"profile has no records: {path}")
        profiles.append(
            {
                "path": str(path),
                "version": int(payload.get("version", 0)),
                "audit": dict(payload.get("audit") or {}),
                "pair_id": str(metadata.get("pair_id") or ""),
                "side": str(metadata.get("side") or ""),
                "seed": int(metadata.get("seed", 0)),
                "records": records,
            }
        )
    return profiles


def _aligned_values(
    left: dict,
    right: dict,
    head: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    left_frames = [int(value) for value in left["key_frames"].tolist()]
    right_frames = [int(value) for value in right["key_frames"].tolist()]
    shared = sorted(set(left_frames) & set(right_frames))
    if len(shared) < 2:
        return None
    left_index = {frame: index for index, frame in enumerate(left_frames)}
    right_index = {frame: index for index, frame in enumerate(right_frames)}
    left_values = left["logits"][head].float()[
        [left_index[frame] for frame in shared]
    ]
    right_values = right["logits"][head].float()[
        [right_index[frame] for frame in shared]
    ]
    return left_values, right_values


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.mean((left - right).square()).sqrt()
    denominator = 0.5 * (
        torch.mean(left.square()).sqrt()
        + torch.mean(right.square()).sqrt()
    )
    return float((numerator / denominator.clamp_min(1e-6)).item())


def collect_cfg_observations(
    profiles: list[dict],
    num_heads: int,
) -> dict[HeadKey, list[float]]:
    result: dict[HeadKey, list[float]] = defaultdict(list)
    for profile in profiles:
        grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for record in profile["records"]:
            key = _record_key(record, include_branch=False)
            grouped[key][str(record["cfg_branch"])] = record
        for (layer, *_), branches in grouped.items():
            if "cond" not in branches or "uncond" not in branches:
                continue
            for head in range(num_heads):
                aligned = _aligned_values(
                    branches["cond"], branches["uncond"], head
                )
                if aligned is not None:
                    result[(layer, head)].append(relative_rms(*aligned))
    return result


def collect_semantic_observations(
    profiles: list[dict],
    num_heads: int,
) -> dict[HeadKey, list[float]]:
    result: dict[HeadKey, list[float]] = defaultdict(list)
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for profile in profiles:
        groups[(profile["pair_id"], profile["seed"])][profile["side"]] = profile
    for sides in groups.values():
        if "a" not in sides or "b" not in sides:
            continue
        left_records = {
            _record_key(record): record
            for record in sides["a"]["records"]
            if str(record["cfg_branch"]) == "cond"
        }
        right_records = {
            _record_key(record): record
            for record in sides["b"]["records"]
            if str(record["cfg_branch"]) == "cond"
        }
        for key in sorted(set(left_records) & set(right_records)):
            layer = int(key[0])
            for head in range(num_heads):
                aligned = _aligned_values(
                    left_records[key], right_records[key], head
                )
                if aligned is not None:
                    result[(layer, head)].append(relative_rms(*aligned))
    return result


def collect_temporal_statistics(
    profiles: list[dict],
    num_heads: int,
) -> dict[HeadKey, dict[str, list[float]]]:
    result: dict[HeadKey, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for profile in profiles:
        for record in profile["records"]:
            if str(record["cfg_branch"]) != "cond":
                continue
            layer = int(record["layer"])
            logits = record["logits"].float()
            for head in range(num_heads):
                values = logits[head]
                if values.numel() < 2:
                    continue
                positive = values > 0
                sign_switch = (positive[1:] != positive[:-1]).float().mean()
                centered = values - values.mean()
                spectrum = torch.fft.rfft(centered)
                power = spectrum.abs().square()
                if power.numel() > 1 and float(power[1:].sum()) > 1e-12:
                    peak_index = int(torch.argmax(power[1:]).item()) + 1
                    dominant_period = float(values.numel() / peak_index)
                    spectral_peak = float(
                        (power[peak_index] / power[1:].sum()).item()
                    )
                else:
                    dominant_period = float(values.numel())
                    spectral_peak = 0.0
                target = result[(layer, head)]
                target["positive_rate"].append(float(positive.float().mean()))
                target["mean_logit"].append(float(values.mean()))
                target["mean_abs_logit"].append(float(values.abs().mean()))
                target["signed_logit_mass"].append(
                    float(
                        (
                            values.sum()
                            / values.abs().sum().clamp_min(1e-6)
                        ).item()
                    )
                )
                target["sign_switch_rate"].append(float(sign_switch))
                target["dominant_period"].append(dominant_period)
                target["spectral_peak_ratio"].append(spectral_peak)
    return result


def summarize_observations(
    observations: dict[HeadKey, list[float]],
    keys: list[HeadKey],
) -> dict[HeadKey, float]:
    missing = [key for key in keys if not observations.get(key)]
    if missing:
        preview = ", ".join(f"L{layer}H{head}" for layer, head in missing[:12])
        raise ValueError(
            f"missing observations for {len(missing)} heads: {preview}. "
            "Profiles must contain all model layers; remapping or padding "
            "missing layers is forbidden."
        )
    return {key: median(observations[key]) for key in keys}


def layer_robust_z(
    values: dict[HeadKey, float],
    num_layers: int,
    num_heads: int,
) -> dict[HeadKey, float]:
    result = {}
    for layer in range(num_layers):
        layer_values = [
            values.get((layer, head))
            for head in range(num_heads)
        ]
        if any(v is None for v in layer_values):
            raise ValueError(f"layer {layer} has incomplete QK observations")
        transformed = [
            math.log1p(max(0.0, v)) for v in layer_values
        ]
        center = median(transformed)
        scale = mad(transformed, center)
        for head, value in enumerate(transformed):
            result[(layer, head)] = (value - center) / scale
    return result


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def fit_gmm_1d(values: list[float], components: int) -> dict:
    ordered = sorted(float(value) for value in values)
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

    def log_density(value: float, index: int) -> float:
        variance = max(1e-6, variances[index])
        return (
            math.log(max(weights[index], 1e-12))
            - 0.5 * math.log(2.0 * math.pi * variance)
            - 0.5 * (value - means[index]) ** 2 / variance
        )

    previous = None
    for _ in range(200):
        responsibilities = []
        log_likelihood = 0.0
        for value in ordered:
            logs = [log_density(value, index) for index in range(components)]
            norm = _logsumexp(logs)
            responsibilities.append([math.exp(item - norm) for item in logs])
            log_likelihood += norm
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
    parameter_count = 3 * components - 1
    bic = -2.0 * log_likelihood + parameter_count * math.log(count)
    return {
        "components": components,
        "means": means,
        "variances": variances,
        "weights": weights,
        "log_likelihood": log_likelihood,
        "bic": bic,
    }


def gmm_threshold(model: dict) -> float:
    if int(model["components"]) != 2:
        raise ValueError("threshold requires a two-component GMM")
    low, high = model["means"]
    if high <= low:
        return float(low)
    best_x = 0.5 * (low + high)
    best_gap = float("inf")
    for index in range(4097):
        value = low + (high - low) * index / 4096
        logs = []
        for component in range(2):
            variance = model["variances"][component]
            logs.append(
                math.log(max(model["weights"][component], 1e-12))
                - 0.5 * math.log(2.0 * math.pi * variance)
                - 0.5
                * (value - model["means"][component]) ** 2
                / variance
            )
        gap = abs(logs[0] - logs[1])
        if gap < best_gap:
            best_gap = gap
            best_x = value
    return float(best_x)


def otsu_threshold(values: list[float]) -> float:
    ordered = sorted(values)
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


def matrix_from_scores(
    scores: dict[HeadKey, float],
    threshold: float,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    return [
        [
            (
                1  # default stable/stride for unprofiled layers
                if (layer, head) not in scores
                else 1 if scores[(layer, head)] <= threshold else -1
            )
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def agreement(left: list[list[int]], right: list[list[int]]) -> dict:
    left_flat = flatten(left)
    right_flat = flatten(right)
    stable_left = {i for i, value in enumerate(left_flat) if value == 1}
    stable_right = {i for i, value in enumerate(right_flat) if value == 1}
    union = stable_left | stable_right
    return {
        "agreement": sum(
            a == b for a, b in zip(left_flat, right_flat)
        ) / len(left_flat),
        "stable_jaccard": (
            len(stable_left & stable_right) / len(union) if union else 1.0
        ),
    }


def threshold_map(
    name: str,
    scores: dict[HeadKey, float],
    num_layers: int,
    num_heads: int,
) -> tuple[list[list[int]], dict]:
    values = [
        scores[(layer, head)]
        for layer in range(num_layers)
        for head in range(num_heads)
        if (layer, head) in scores
    ]
    models = [fit_gmm_1d(values, components) for components in (1, 2, 3)]
    threshold = gmm_threshold(models[1])
    matrix = matrix_from_scores(
        scores, threshold, num_layers, num_heads
    )
    flat = flatten(matrix)
    return matrix, {
        "name": name,
        "threshold": threshold,
        "otsu_threshold": otsu_threshold(values),
        "gmm": models,
        "stable_count": flat.count(1),
        "responsive_count": flat.count(-1),
        "per_layer_stable": [row.count(1) for row in matrix],
    }


def bootstrap_stability(
    cfg_observations: dict[HeadKey, list[float]],
    semantic_observations: dict[HeadKey, list[float]],
    reference: list[list[int]],
    *,
    num_layers: int,
    num_heads: int,
    rounds: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    keys = [
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    ]
    # Only bootstrap keys that have both cfg and semantic observations
    boot_keys = [
        key for key in keys
        if cfg_observations.get(key) and semantic_observations.get(key)
    ]
    matches = [0] * len(keys)
    completed = 0
    for _ in range(max(0, rounds)):
        cfg_sample = {}
        semantic_sample = {}
        for key in boot_keys:
            cfg_values = cfg_observations[key]
            semantic_values = semantic_observations[key]
            cfg_sample[key] = median(
                [rng.choice(cfg_values) for _ in range(len(cfg_values))]
            )
            semantic_sample[key] = median(
                [
                    rng.choice(semantic_values)
                    for _ in range(len(semantic_values))
                ]
            )
        cfg_z = layer_robust_z(cfg_sample, num_layers, num_heads)
        semantic_z = layer_robust_z(
            semantic_sample, num_layers, num_heads
        )
        consensus = {
            key: 0.5 * (cfg_z[key] + semantic_z[key])
            for key in keys
        }
        try:
            sampled, _ = threshold_map(
                "bootstrap", consensus, num_layers, num_heads
            )
        except (ArithmeticError, ValueError):
            continue
        completed += 1
        sampled_flat = flatten(sampled)
        reference_flat = flatten(reference)
        for index, (actual, expected) in enumerate(
            zip(sampled_flat, reference_flat)
        ):
            matches[index] += int(actual == expected)
    agreements = [
        value / completed if completed else 0.0 for value in matches
    ]
    return {
        "requested_rounds": rounds,
        "completed_rounds": completed,
        "mean_head_agreement": (
            sum(agreements) / len(agreements) if agreements else 0.0
        ),
        "stable_head_fraction_at_0_75": (
            sum(value >= 0.75 for value in agreements) / len(agreements)
            if agreements
            else 0.0
        ),
        "per_head_agreement": agreements,
    }


def pf_overlap(
    matrix: list[list[int]],
    pf_labels: list[list[int]],
) -> dict:
    candidate = flatten(matrix)
    pf_flat = flatten(pf_labels)
    stable = {index for index, value in enumerate(candidate) if value == 1}
    anchor = {index for index, value in enumerate(pf_flat) if value == 1}
    intersection = stable & anchor
    cross = {
        str(pf_label): Counter(
            candidate[index]
            for index, label in enumerate(pf_flat)
            if label == pf_label
        )
        for pf_label in (-1, 1, 2)
    }
    return {
        "stable_anchor_precision": (
            len(intersection) / len(stable) if stable else 0.0
        ),
        "anchor_recall": (
            len(intersection) / len(anchor) if anchor else 0.0
        ),
        "stable_anchor_jaccard": (
            len(intersection) / len(stable | anchor)
            if stable | anchor
            else 1.0
        ),
        "cross_tab": {
            pf_label: {
                "stable": int(counts[1]),
                "responsive": int(counts[-1]),
            }
            for pf_label, counts in cross.items()
        },
    }


def random_control(
    reference: list[list[int]],
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    result = []
    for row in reference:
        indices = list(range(len(row)))
        rng.shuffle(indices)
        stable = set(indices[: row.count(1)])
        result.append(
            [1 if index in stable else -1 for index in range(len(row))]
        )
    return result


def aggregate_pf_temporal(
    temporal: dict[HeadKey, dict[str, list[float]]],
    pf_labels: list[list[int]],
) -> dict:
    metrics = (
        "positive_rate",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "sign_switch_rate",
        "dominant_period",
        "spectral_peak_ratio",
    )
    result = {}
    for label, name in ((-1, "wave"), (1, "anchor"), (2, "veil")):
        heads = [
            (layer, head)
            for layer, row in enumerate(pf_labels)
            for head, value in enumerate(row)
            if value == label
        ]
        result[name] = {
            "head_count": len(heads),
            **{
                metric: median(
                    [
                        median(temporal[key][metric])
                        for key in heads
                        if temporal.get(key, {}).get(metric)
                    ]
                )
                for metric in metrics
            },
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profiles)
    pf_labels = read_matrix(
        args.pf_labels, args.num_layers, args.num_heads
    )

    profiled_layers = sorted(
        {int(r["layer"]) for p in profiles for r in p["records"]}
    )
    expected_layers = list(range(args.num_layers))
    if profiled_layers != expected_layers:
        raise ValueError(
            "QK profiles do not contain the exact transformer layer set: "
            f"expected={expected_layers}, observed={profiled_layers}. "
            "Do not remap even capture counters or pad missing layers."
        )
    effective_layers = args.num_layers

    cfg_observations = collect_cfg_observations(
        profiles, args.num_heads
    )
    semantic_observations = collect_semantic_observations(
        profiles, args.num_heads
    )
    temporal = collect_temporal_statistics(profiles, args.num_heads)

    keys = [
        (layer, head)
        for layer in range(effective_layers)
        for head in range(args.num_heads)
    ]
    cfg_raw = summarize_observations(cfg_observations, keys)
    semantic_raw = summarize_observations(semantic_observations, keys)
    cfg_scores = layer_robust_z(
        cfg_raw, effective_layers, args.num_heads
    )
    semantic_scores = layer_robust_z(
        semantic_raw, effective_layers, args.num_heads
    )
    consensus_scores = {
        key: 0.5 * (cfg_scores[key] + semantic_scores[key])
        for key in keys
    }

    maps = {}
    diagnostics = {}
    for name, scores in (
        ("prompt_cfg_threshold", cfg_scores),
        ("prompt_semantic_threshold", semantic_scores),
        ("prompt_consensus_threshold", consensus_scores),
    ):
        matrix, diag = threshold_map(
            name, scores, effective_layers, args.num_heads
        )
        maps[name] = matrix
        diagnostics[name] = diag

    consensus = maps["prompt_consensus_threshold"]
    pf_binary = [
        [1 if value == 1 else -1 for value in row]
        for row in pf_labels
    ]
    maps["pf_binary"] = pf_binary
    maps["prompt_consensus_inverse"] = [
        [-value for value in row] for row in consensus
    ]
    maps["prompt_consensus_random"] = random_control(
        consensus, args.random_seed
    )
    for name, matrix in maps.items():
        write_matrix(args.output_dir / f"{name}.csv", matrix)

    bootstrap = bootstrap_stability(
        cfg_observations,
        semantic_observations,
        consensus,
        num_layers=effective_layers,
        num_heads=args.num_heads,
        rounds=args.bootstrap_rounds,
        seed=args.bootstrap_seed,
    )
    main_diag = diagnostics["prompt_consensus_threshold"]
    bic = [model["bic"] for model in main_diag["gmm"]]
    class_fraction = min(
        main_diag["stable_count"], main_diag["responsive_count"]
    ) / (effective_layers * args.num_heads)
    gates = {
        "two_vs_one_bic": {
            "observed": bic[0] - bic[1],
            "required": 10.0,
            "passed": bic[0] - bic[1] >= 10.0,
        },
        "two_vs_three_bic": {
            "observed": bic[2] - bic[1],
            "required": 0.0,
            "passed": bic[1] <= bic[2],
        },
        "minority_class_fraction": {
            "observed": class_fraction,
            "required": 0.10,
            "passed": class_fraction >= 0.10,
        },
        "bootstrap_stable_fraction": {
            "observed": bootstrap["stable_head_fraction_at_0_75"],
            "required": 0.80,
            "passed": (
                bootstrap["stable_head_fraction_at_0_75"] >= 0.80
            ),
        },
    }
    report = {
        "version": 1,
        "method": "qk_prompt_threshold_binary_head_discovery",
        "profile_count": len(profiles),
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "score_definition": {
            "cfg": "layer-robust-z(log1p(NRMS(QK_cond,QK_uncond)))",
            "semantic": (
                "layer-robust-z(log1p(NRMS(QK_prompt_a,QK_prompt_b)))"
            ),
            "consensus": "0.5 * (cfg_z + semantic_z)",
            "stable": "score <= data-driven GMM intersection",
        },
        "maps": {
            name: {
                **diagnostics.get(name, {}),
                "vs_pf_anchor": pf_overlap(matrix, pf_labels),
                "vs_consensus": agreement(consensus, matrix),
            }
            for name, matrix in maps.items()
        },
        "map_agreement": {
            "cfg_vs_semantic": agreement(
                maps["prompt_cfg_threshold"],
                maps["prompt_semantic_threshold"],
            ),
            "cfg_vs_consensus": agreement(
                maps["prompt_cfg_threshold"], consensus
            ),
            "semantic_vs_consensus": agreement(
                maps["prompt_semantic_threshold"], consensus
            ),
        },
        "pf_temporal_statistics": aggregate_pf_temporal(
            temporal, pf_labels
        ),
        "bootstrap": bootstrap,
        "acceptance_gates": gates,
        "accepted": all(item["passed"] for item in gates.values()),
        "entries": [
            {
                "layer": layer,
                "head": head,
                "pf_label": pf_labels[layer][head],
                "cfg_raw": cfg_raw[(layer, head)],
                "semantic_raw": semantic_raw[(layer, head)],
                "cfg_score": cfg_scores[(layer, head)],
                "semantic_score": semantic_scores[(layer, head)],
                "consensus_score": consensus_scores[(layer, head)],
                "consensus_label": consensus[layer][head],
                "bootstrap_agreement": bootstrap[
                    "per_head_agreement"
                ][layer * args.num_heads + head],
                **{
                    metric: median(temporal[(layer, head)][metric])
                    for metric in (
                        "positive_rate",
                        "mean_logit",
                        "mean_abs_logit",
                        "signed_logit_mass",
                        "sign_switch_rate",
                        "dominant_period",
                        "spectral_peak_ratio",
                    )
                },
            }
            for layer, head in keys
        ],
    }
    report_path = args.output_dir / "qk_head_threshold_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    score_path = args.output_dir / "qk_head_scores.csv"
    score_fields = list(report["entries"][0])
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(report["entries"])

    consensus_overlap = report["maps"][
        "prompt_consensus_threshold"
    ]["vs_pf_anchor"]
    summary = [
        "# QK Head Threshold Summary",
        "",
        f"- profiles: {len(profiles)}",
        f"- accepted: {report['accepted']}",
        (
            "- consensus classes: "
            f"stable={main_diag['stable_count']}, "
            f"responsive={main_diag['responsive_count']}"
        ),
        (
            "- consensus threshold: "
            f"{main_diag['threshold']:.6f} "
            f"(Otsu={main_diag['otsu_threshold']:.6f})"
        ),
        (
            "- stable/PF-Anchor Jaccard: "
            f"{consensus_overlap['stable_anchor_jaccard']:.4f}"
        ),
        (
            "- stable/PF-Anchor precision: "
            f"{consensus_overlap['stable_anchor_precision']:.4f}"
        ),
        (
            "- PF-Anchor recall: "
            f"{consensus_overlap['anchor_recall']:.4f}"
        ),
        (
            "- bootstrap mean head agreement: "
            f"{bootstrap['mean_head_agreement']:.4f}"
        ),
        "",
        "## Threshold Maps",
        "",
        "| Map | stable | responsive | threshold | BIC1-BIC2 | Anchor Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "prompt_cfg_threshold",
        "prompt_semantic_threshold",
        "prompt_consensus_threshold",
    ):
        item = report["maps"][name]
        summary.append(
            f"| {name} | {item['stable_count']} | "
            f"{item['responsive_count']} | {item['threshold']:.6f} | "
            f"{item['gmm'][0]['bic'] - item['gmm'][1]['bic']:.4f} | "
            f"{item['vs_pf_anchor']['stable_anchor_jaccard']:.4f} |"
        )
    summary.extend(
        [
            "",
            "## Consensus/PF Cross-Tab",
            "",
            "| PF label | Prompt-Stable | Prompt-Responsive |",
            "|---:|---:|---:|",
        ]
    )
    for pf_label in ("-1", "1", "2"):
        values = consensus_overlap["cross_tab"][pf_label]
        summary.append(
            f"| {pf_label} | {values['stable']} | "
            f"{values['responsive']} |"
        )
    summary.extend(
        [
        "",
        "## PF Temporal Statistics",
        "",
        "| PF type | heads | positive rate | mean logit | sign switch | period | FFT peak |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in report["pf_temporal_statistics"].items():
        summary.append(
            f"| {name} | {values['head_count']} | "
            f"{values['positive_rate']:.4f} | "
            f"{values['mean_logit']:.4f} | "
            f"{values['sign_switch_rate']:.4f} | "
            f"{values['dominant_period']:.4f} | "
            f"{values['spectral_peak_ratio']:.4f} |"
        )
    summary.extend(
        [
            "",
            "## Gates",
            "",
            *[
                (
                    f"- {name}: observed={item['observed']:.6f}, "
                    f"required={item['required']:.6f}, "
                    f"passed={item['passed']}"
                )
                for name, item in gates.items()
            ],
        ]
    )
    (args.output_dir / "qk_head_threshold_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(
        "[QKHeadThreshold] "
        f"accepted={report['accepted']} "
        f"stable={main_diag['stable_count']} "
        f"responsive={main_diag['responsive_count']} "
        f"threshold={main_diag['threshold']:.6f} "
        f"report={report_path} scores={score_path}",
        flush=True,
    )
    if args.strict_gates and not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
