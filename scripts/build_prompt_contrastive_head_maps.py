#!/usr/bin/env python3
"""Build causal binary read-policy maps from prompt-contrastive profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


PROMPT_STABLE = 1
PROMPT_RESPONSIVE = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-report", required=True, type=Path)
    parser.add_argument("--replica-report", type=Path)
    parser.add_argument("--pf-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_matrix(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row if value.strip()]
            for row in csv.reader(handle)
            if row
        ]
    if not rows or not rows[0]:
        raise ValueError(f"empty label matrix: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"ragged label matrix: {path}")
    return rows


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def _validate_binary(matrix: list[list[int]], *, name: str) -> None:
    unexpected = sorted(set(flatten(matrix)) - {PROMPT_RESPONSIVE, PROMPT_STABLE})
    if unexpected:
        raise ValueError(f"{name} contains non-binary labels: {unexpected}")


def _load_report(
    path: Path,
    *,
    rows: int,
    columns: int,
) -> tuple[dict, dict[tuple[int, int], dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    expected = {(layer, head) for layer in range(rows) for head in range(columns)}
    indexed: dict[tuple[int, int], dict] = {}
    for entry in entries:
        coordinate = (int(entry["layer"]), int(entry["head"]))
        if coordinate in indexed:
            raise ValueError(f"duplicate profile entry at {coordinate}: {path}")
        indexed[coordinate] = entry
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"profile report coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return payload, indexed


def _score_rows(
    entries: dict[tuple[int, int], dict],
    *,
    score_key: str,
    rows: int,
    columns: int,
) -> list[list[float]]:
    output = []
    for layer in range(rows):
        row = []
        for head in range(columns):
            value = float(entries[(layer, head)][score_key])
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite {score_key} at layer={layer} head={head}"
                )
            row.append(value)
        output.append(row)
    return output


def _rank_per_layer(
    scores: list[list[float]],
    stable_budgets: list[int],
    *,
    stable_low: bool,
) -> list[list[int]]:
    if len(scores) != len(stable_budgets):
        raise ValueError("score rows and stable budgets have different lengths")
    output = []
    for layer, (row, budget) in enumerate(zip(scores, stable_budgets)):
        if not 0 <= budget <= len(row):
            raise ValueError(f"invalid stable budget at layer {layer}: {budget}")
        ordered = sorted(
            range(len(row)),
            key=lambda head: (
                row[head] if stable_low else -row[head],
                head,
            ),
        )
        stable = set(ordered[:budget])
        output.append(
            [
                PROMPT_STABLE if head in stable else PROMPT_RESPONSIVE
                for head in range(len(row))
            ]
        )
    return output


def _binary_kmeans(values: list[float]) -> tuple[list[int], tuple[float, float], float]:
    if len(values) < 2:
        raise ValueError("binary k-means requires at least two values")
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        raise ValueError("prompt sensitivity is constant; natural partition is undefined")
    labels = [0] * len(values)
    for _ in range(100):
        updated = [
            0 if abs(value - low) <= abs(value - high) else 1
            for value in values
        ]
        if not any(label == 0 for label in updated) or not any(
            label == 1 for label in updated
        ):
            raise ValueError("binary k-means collapsed to one cluster")
        next_low = statistics.fmean(
            value for value, label in zip(values, updated) if label == 0
        )
        next_high = statistics.fmean(
            value for value, label in zip(values, updated) if label == 1
        )
        labels = updated
        if abs(next_low - low) < 1e-10 and abs(next_high - high) < 1e-10:
            low, high = next_low, next_high
            break
        low, high = next_low, next_high
    if low > high:
        low, high = high, low
        labels = [1 - label for label in labels]
    return labels, (float(low), float(high)), float(0.5 * (low + high))


def _kmeans_map(
    scores: list[list[float]],
) -> tuple[list[list[int]], dict]:
    rows = len(scores)
    columns = len(scores[0])
    labels, centers, threshold = _binary_kmeans(flatten(scores))
    matrix = [
        [
            PROMPT_STABLE
            if labels[layer * columns + head] == 0
            else PROMPT_RESPONSIVE
            for head in range(columns)
        ]
        for layer in range(rows)
    ]
    return matrix, {
        "stable_center": centers[0],
        "responsive_center": centers[1],
        "threshold": threshold,
    }


def _random_per_layer(
    stable_budgets: list[int],
    *,
    columns: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    output = []
    for budget in stable_budgets:
        indices = list(range(columns))
        rng.shuffle(indices)
        stable = set(indices[:budget])
        output.append(
            [
                PROMPT_STABLE if head in stable else PROMPT_RESPONSIVE
                for head in range(columns)
            ]
        )
    return output


def _agreement(reference: list[list[int]], candidate: list[list[int]]) -> dict:
    left = flatten(reference)
    right = flatten(candidate)
    if len(left) != len(right):
        raise ValueError("cannot compare maps with different sizes")
    stable_left = {index for index, value in enumerate(left) if value == PROMPT_STABLE}
    stable_right = {
        index for index, value in enumerate(right) if value == PROMPT_STABLE
    }
    union = stable_left | stable_right
    return {
        "agreement": sum(a == b for a, b in zip(left, right)) / max(1, len(left)),
        "stable_jaccard": (
            len(stable_left & stable_right) / len(union) if union else 1.0
        ),
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must be non-empty and aligned")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 0 else 0.0


def build_maps(
    pf: list[list[int]],
    primary_entries: dict[tuple[int, int], dict],
    *,
    replica_entries: dict[tuple[int, int], dict] | None,
    random_seed: int,
) -> tuple[dict[str, list[list[int]]], dict]:
    rows = len(pf)
    columns = len(pf[0])
    pf_binary = [
        [
            PROMPT_STABLE if value == 1 else PROMPT_RESPONSIVE
            for value in row
        ]
        for row in pf
    ]
    stable_budgets = [row.count(PROMPT_STABLE) for row in pf_binary]
    prompt = _score_rows(
        primary_entries,
        score_key="prompt_z",
        rows=rows,
        columns=columns,
    )
    remote = _score_rows(
        primary_entries,
        score_key="remote_z",
        rows=rows,
        columns=columns,
    )
    role_score = _score_rows(
        primary_entries,
        score_key="role_score",
        rows=rows,
        columns=columns,
    )
    learned = [
        [
            int(primary_entries[(layer, head)]["label"])
            for head in range(columns)
        ]
        for layer in range(rows)
    ]
    prompt_kmeans, kmeans = _kmeans_map(prompt)
    maps = {
        "pf_binary": pf_binary,
        "prompt_pfcount": _rank_per_layer(
            prompt, stable_budgets, stable_low=True
        ),
        "prompt_kmeans": prompt_kmeans,
        "prompt_inverse_pfcount": _rank_per_layer(
            prompt, stable_budgets, stable_low=False
        ),
        "prompt_random_pfcount": _random_per_layer(
            stable_budgets,
            columns=columns,
            seed=random_seed,
        ),
        "remote_pfcount": _rank_per_layer(
            remote, stable_budgets, stable_low=False
        ),
        "role_score_pfcount": _rank_per_layer(
            role_score, stable_budgets, stable_low=False
        ),
        "learned_original": learned,
    }
    if replica_entries is not None:
        replica_prompt = _score_rows(
            replica_entries,
            score_key="prompt_z",
            rows=rows,
            columns=columns,
        )
        maps["prompt_replica_pfcount"] = _rank_per_layer(
            replica_prompt, stable_budgets, stable_low=True
        )
        mean_prompt = [
            [
                0.5 * (prompt[layer][head] + replica_prompt[layer][head])
                for head in range(columns)
            ]
            for layer in range(rows)
        ]
        maps["prompt_consensus_pfcount"] = _rank_per_layer(
            mean_prompt, stable_budgets, stable_low=True
        )

    for name, matrix in maps.items():
        _validate_binary(matrix, name=name)
    metadata = {
        "stable_label": PROMPT_STABLE,
        "responsive_label": PROMPT_RESPONSIVE,
        "stable_budgets_per_layer": stable_budgets,
        "prompt_kmeans": kmeans,
        "prompt_remote_pearson": _pearson(flatten(prompt), flatten(remote)),
    }
    return maps, metadata


def main() -> None:
    args = parse_args()
    pf = read_matrix(args.pf_csv)
    rows = len(pf)
    columns = len(pf[0])
    _, primary_entries = _load_report(
        args.profile_report,
        rows=rows,
        columns=columns,
    )
    replica_entries = None
    if args.replica_report is not None:
        _, replica_entries = _load_report(
            args.replica_report,
            rows=rows,
            columns=columns,
        )
    maps, metadata = build_maps(
        pf,
        primary_entries,
        replica_entries=replica_entries,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = maps["pf_binary"]
    manifest = {
        "version": 1,
        "method": "prompt_contrastive_binary_read_policy_controls",
        "semantics": {
            "1": "prompt-stable: long-horizon stride/anchor policy",
            "-1": "prompt-responsive: cyclic/recent policy",
        },
        "classification_claim": (
            "prompt response only; PF labels and remote utility are controls"
        ),
        "profile_report": str(args.profile_report.resolve()),
        "profile_report_sha256": _sha256(args.profile_report),
        "replica_report": (
            str(args.replica_report.resolve())
            if args.replica_report is not None
            else None
        ),
        "replica_report_sha256": (
            _sha256(args.replica_report)
            if args.replica_report is not None
            else None
        ),
        "pf_csv": str(args.pf_csv.resolve()),
        "pf_csv_sha256": _sha256(args.pf_csv),
        "random_seed": args.random_seed,
        **metadata,
        "maps": {},
    }
    for name, matrix in maps.items():
        path = args.output_dir / f"{name}.csv"
        write_matrix(path, matrix)
        flat = flatten(matrix)
        manifest["maps"][name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "stable_count": flat.count(PROMPT_STABLE),
            "responsive_count": flat.count(PROMPT_RESPONSIVE),
            "per_layer_stable": [
                row.count(PROMPT_STABLE) for row in matrix
            ],
            "vs_pf_binary": _agreement(reference, matrix),
        }
    if "prompt_replica_pfcount" in maps:
        manifest["replica_agreement"] = _agreement(
            maps["prompt_pfcount"],
            maps["prompt_replica_pfcount"],
        )
    manifest_path = args.output_dir / "prompt_contrastive_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[PromptContrastiveMaps] "
        f"maps={len(maps)} "
        f"prompt_pf_agreement="
        f"{manifest['maps']['prompt_pfcount']['vs_pf_binary']['agreement']:.4f} "
        f"prompt_remote_r={manifest['prompt_remote_pearson']:.4f} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
