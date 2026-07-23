#!/usr/bin/env python3
"""Build deterministic persistent/reactive head labels from paired profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--cache-update-mode", default="noisy")
    parser.add_argument("--call-indices", default="0,2,3")
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Exit nonzero when the generated profile fails method acceptance gates.",
    )
    return parser.parse_args()


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mad(values: list[float], center: float | None = None) -> float:
    if not values:
        return 1.0
    center = _median(values) if center is None else float(center)
    return max(1e-6, 1.4826 * _median([abs(value - center) for value in values]))


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired sketches must share [H,D] shape")
    left = left.float()
    right = right.float()
    numerator = (left - right).norm(dim=-1)
    denominator = 0.5 * (left.norm(dim=-1) + right.norm(dim=-1))
    return (numerator / denominator.clamp_min(1e-6)).tolist()


def _record_key(record: dict) -> tuple:
    return (
        int(record["prompt_id"]),
        int(record["layer"]),
        int(record["current_start"]),
        str(record["cache_update_mode"]),
        int(record["call_index"]),
    )


def _load_profiles(paths: list[Path]) -> list[dict]:
    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) != 1:
            raise ValueError(f"unsupported profile version in {path}")
        metadata = dict(payload.get("metadata") or {})
        kind = str(metadata.get("kind") or "")
        side = str(metadata.get("side") or "")
        pair_id = str(metadata.get("pair_id") or "")
        if kind not in {"prompt", "history"}:
            raise ValueError(f"profile {path} has invalid kind={kind!r}")
        if not side or not pair_id:
            raise ValueError(f"profile {path} is missing side or pair_id metadata")
        records = {_record_key(record): record for record in payload.get("records", [])}
        profiles.append(
            {
                "path": str(path),
                "kind": kind,
                "side": side,
                "pair_id": pair_id,
                "seed": int(metadata.get("seed", 0)),
                "records": records,
            }
        )
    return profiles


def _pair_profiles(profiles: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for profile in profiles:
        key = (profile["kind"], profile["pair_id"], profile["seed"])
        if profile["side"] in groups[key]:
            raise ValueError(f"duplicate profile side for {key}: {profile['side']}")
        groups[key][profile["side"]] = profile

    pairs = []
    for (kind, pair_id, seed), sides in sorted(groups.items()):
        expected = ("a", "b") if kind == "prompt" else ("full", "recent")
        missing = [side for side in expected if side not in sides]
        if missing:
            raise ValueError(
                f"incomplete {kind} pair pair_id={pair_id} seed={seed}: missing {missing}"
            )
        pairs.append((sides[expected[0]], sides[expected[1]]))
    return pairs


def collect_observations(
    profiles: list[dict],
    *,
    num_layers: int,
    num_heads: int,
    cache_update_mode: str,
    call_indices: set[int],
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], list[float]], int]:
    prompt: dict[tuple[int, int], list[float]] = defaultdict(list)
    remote: dict[tuple[int, int], list[float]] = defaultdict(list)
    matched_records = 0
    for left, right in _pair_profiles(profiles):
        shared = sorted(set(left["records"]) & set(right["records"]))
        for key in shared:
            _, layer, _, mode, call_index = key
            if mode != cache_update_mode or call_index not in call_indices:
                continue
            left_sketch = left["records"][key]["sketch"]
            right_sketch = right["records"][key]["sketch"]
            if not (0 <= layer < num_layers):
                continue
            differences = _relative_difference(left_sketch, right_sketch)
            if len(differences) != num_heads:
                raise ValueError(
                    f"record layer={layer} has {len(differences)} heads, expected {num_heads}"
                )
            target = prompt if left["kind"] == "prompt" else remote
            for head, value in enumerate(differences):
                if math.isfinite(value):
                    target[(layer, head)].append(float(value))
            matched_records += 1
    return prompt, remote, matched_records


def _normalize_per_layer(
    values: dict[tuple[int, int], float],
    *,
    num_layers: int,
    num_heads: int,
) -> dict[tuple[int, int], float]:
    normalized = {}
    for layer in range(num_layers):
        row = [values[(layer, head)] for head in range(num_heads)]
        center = _median(row)
        scale = _mad(row, center)
        for head, value in enumerate(row):
            normalized[(layer, head)] = (value - center) / scale
    return normalized


def _binary_kmeans(values: list[float]) -> tuple[list[int], tuple[float, float], float]:
    if len(values) < 2:
        raise ValueError("binary clustering requires at least two values")
    low, high = min(values), max(values)
    if abs(high - low) < 1e-8:
        threshold = _median(values)
        labels = [1 if index >= len(values) // 2 else 0 for index in range(len(values))]
        return labels, (low, high), threshold
    labels = [0] * len(values)
    for _ in range(100):
        new_labels = [
            0 if abs(value - low) <= abs(value - high) else 1
            for value in values
        ]
        if not any(label == 0 for label in new_labels) or not any(
            label == 1 for label in new_labels
        ):
            order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
            new_labels = [0] * len(values)
            for idx in order[len(order) // 2 :]:
                new_labels[idx] = 1
        new_low = sum(
            value for value, label in zip(values, new_labels) if label == 0
        ) / sum(label == 0 for label in new_labels)
        new_high = sum(
            value for value, label in zip(values, new_labels) if label == 1
        ) / sum(label == 1 for label in new_labels)
        labels = new_labels
        if abs(new_low - low) < 1e-8 and abs(new_high - high) < 1e-8:
            low, high = new_low, new_high
            break
        low, high = new_low, new_high
    if low > high:
        low, high = high, low
        labels = [1 - label for label in labels]
    return labels, (float(low), float(high)), 0.5 * (low + high)


def build_profile(
    profiles: list[dict],
    *,
    num_layers: int,
    num_heads: int,
    cache_update_mode: str,
    call_indices: set[int],
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> tuple[list[list[int]], dict]:
    prompt_obs, remote_obs, matched = collect_observations(
        profiles,
        num_layers=num_layers,
        num_heads=num_heads,
        cache_update_mode=cache_update_mode,
        call_indices=call_indices,
    )
    keys = [
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    ]
    missing_prompt = [key for key in keys if not prompt_obs.get(key)]
    missing_remote = [key for key in keys if not remote_obs.get(key)]
    if missing_prompt or missing_remote:
        raise ValueError(
            "insufficient paired records: "
            f"missing_prompt_heads={len(missing_prompt)} "
            f"missing_remote_heads={len(missing_remote)}"
        )

    prompt_median = {key: _median(prompt_obs[key]) for key in keys}
    remote_median = {key: _median(remote_obs[key]) for key in keys}
    prompt_z = _normalize_per_layer(
        prompt_median,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    remote_z = _normalize_per_layer(
        remote_median,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    role_scores = [remote_z[key] - prompt_z[key] for key in keys]
    cluster, centers, threshold = _binary_kmeans(role_scores)
    labels_flat = [1 if label == 1 else -1 for label in cluster]

    rng = random.Random(bootstrap_seed)
    agreement = [0] * len(keys)
    rounds = max(0, int(bootstrap_rounds))
    for _ in range(rounds):
        sampled_prompt = {
            key: _median(
                [rng.choice(prompt_obs[key]) for _ in range(len(prompt_obs[key]))]
            )
            for key in keys
        }
        sampled_remote = {
            key: _median(
                [rng.choice(remote_obs[key]) for _ in range(len(remote_obs[key]))]
            )
            for key in keys
        }
        sampled_prompt_z = _normalize_per_layer(
            sampled_prompt,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        sampled_remote_z = _normalize_per_layer(
            sampled_remote,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        sampled_scores = [
            sampled_remote_z[key] - sampled_prompt_z[key] for key in keys
        ]
        sampled_cluster, _, _ = _binary_kmeans(sampled_scores)
        sampled_labels = [1 if label == 1 else -1 for label in sampled_cluster]
        for index, (actual, sampled) in enumerate(zip(labels_flat, sampled_labels)):
            agreement[index] += int(actual == sampled)

    score_scale = _mad(role_scores, threshold)
    entries = []
    for index, ((layer, head), role_score, label) in enumerate(
        zip(keys, role_scores, labels_flat)
    ):
        entries.append(
            {
                "layer": layer,
                "head": head,
                "label": label,
                "role": "persistent" if label == 1 else "reactive",
                "prompt_sensitivity": prompt_median[(layer, head)],
                "remote_utility": remote_median[(layer, head)],
                "prompt_z": prompt_z[(layer, head)],
                "remote_z": remote_z[(layer, head)],
                "role_score": role_score,
                "cluster_margin": abs(role_score - threshold) / score_scale,
                "bootstrap_agreement": (
                    agreement[index] / rounds if rounds > 0 else None
                ),
                "prompt_samples": len(prompt_obs[(layer, head)]),
                "remote_samples": len(remote_obs[(layer, head)]),
            }
        )
    matrix = [
        labels_flat[layer * num_heads : (layer + 1) * num_heads]
        for layer in range(num_layers)
    ]
    persistent_entries = [entry for entry in entries if entry["label"] == 1]
    reactive_entries = [entry for entry in entries if entry["label"] == -1]
    total_heads = len(entries)
    cluster_fraction = min(
        len(persistent_entries), len(reactive_entries)
    ) / max(1, total_heads)
    stable_fraction = (
        sum(
            float(entry["bootstrap_agreement"] or 0.0) >= 0.75
            for entry in entries
        )
        / max(1, total_heads)
        if rounds > 0
        else 0.0
    )
    persistent_remote = _median(
        [entry["remote_utility"] for entry in persistent_entries]
    )
    reactive_remote = _median(
        [entry["remote_utility"] for entry in reactive_entries]
    )
    persistent_prompt = _median(
        [entry["prompt_sensitivity"] for entry in persistent_entries]
    )
    reactive_prompt = _median(
        [entry["prompt_sensitivity"] for entry in reactive_entries]
    )
    gate_rows = {
        "cluster_fraction": {
            "observed": cluster_fraction,
            "required": 0.10,
            "passed": cluster_fraction >= 0.10,
        },
        "bootstrap_stable_fraction": {
            "observed": stable_fraction,
            "required": 0.80,
            "passed": stable_fraction >= 0.80,
        },
        "persistent_remote_direction": {
            "persistent_median": persistent_remote,
            "reactive_median": reactive_remote,
            "passed": persistent_remote > reactive_remote,
        },
        "reactive_prompt_direction": {
            "reactive_median": reactive_prompt,
            "persistent_median": persistent_prompt,
            "passed": reactive_prompt > persistent_prompt,
        },
    }
    report = {
        "version": 1,
        "method": "counterfactual_output_prompt_remote_binary_kmeans",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "cache_update_mode": cache_update_mode,
        "call_indices": sorted(call_indices),
        "matched_record_pairs": matched,
        "bootstrap_rounds": rounds,
        "cluster_centers": {
            "reactive": centers[0],
            "persistent": centers[1],
        },
        "cluster_threshold": threshold,
        "label_counts": {
            "reactive": labels_flat.count(-1),
            "persistent": labels_flat.count(1),
        },
        "acceptance_gates": {
            "accepted": all(row["passed"] for row in gate_rows.values()),
            "checks": gate_rows,
        },
        "entries": entries,
    }
    return matrix, report


def main() -> None:
    args = parse_args()
    call_indices = {
        int(value.strip())
        for value in args.call_indices.split(",")
        if value.strip()
    }
    profiles = _load_profiles(args.profiles)
    matrix, report = build_profile(
        profiles,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cache_update_mode=args.cache_update_mode,
        call_indices=call_indices,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[ProbeCacheProfile] "
        f"pairs={report['matched_record_pairs']} "
        f"labels={report['label_counts']} "
        f"accepted={report['acceptance_gates']['accepted']} "
        f"csv={args.output_csv} json={args.output_json}"
    )
    if args.strict_gates and not report["acceptance_gates"]["accepted"]:
        failed = [
            name
            for name, row in report["acceptance_gates"]["checks"].items()
            if not row["passed"]
        ]
        raise SystemExit(
            "ProbeCache profile failed strict acceptance gates: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
