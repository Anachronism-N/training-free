#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LABEL_NAMES = {-1: "oscillating", 1: "stable", 2: "stable_sparse"}
LABELS = tuple(LABEL_NAMES)


def entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def counts_dict(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {LABEL_NAMES[label]: counts.get(label, 0) for label in LABELS}


def load_matrix(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle) if row]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Expected a non-empty rectangular CSV: {path}")
    unexpected = sorted({value for row in rows for value in row} - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    return rows


def analyze(matrix: list[list[int]]) -> dict:
    num_layers = len(matrix)
    num_heads = len(matrix[0])
    per_layer = [
        {"layer": index, "counts": counts_dict(row), "entropy_bits": entropy(row)}
        for index, row in enumerate(matrix)
    ]

    band_edges = [0, num_layers // 3, 2 * num_layers // 3, num_layers]
    bands = []
    for band_index in range(3):
        start, end = band_edges[band_index], band_edges[band_index + 1]
        labels = [value for row in matrix[start:end] for value in row]
        bands.append(
            {
                "name": ("early", "middle", "late")[band_index],
                "start_layer": start,
                "end_layer_exclusive": end,
                "counts": counts_dict(labels),
                "entropy_bits": entropy(labels),
            }
        )

    transition_counts = {
        LABEL_NAMES[source]: {LABEL_NAMES[target]: 0 for target in LABELS}
        for source in LABELS
    }
    adjacent_same = 0
    adjacent_total = 0
    for previous, current in zip(matrix, matrix[1:]):
        for source, target in zip(previous, current):
            transition_counts[LABEL_NAMES[source]][LABEL_NAMES[target]] += 1
            adjacent_same += int(source == target)
            adjacent_total += 1

    per_head_index = []
    for head_index in range(num_heads):
        labels = [matrix[layer][head_index] for layer in range(num_layers)]
        counts = Counter(labels)
        dominant_label, dominant_count = counts.most_common(1)[0]
        transitions = sum(left != right for left, right in zip(labels, labels[1:]))
        per_head_index.append(
            {
                "head_index": head_index,
                "counts": counts_dict(labels),
                "dominant": LABEL_NAMES[dominant_label],
                "dominant_fraction": dominant_count / num_layers,
                "depth_transitions": transitions,
            }
        )

    return {
        "shape": {"layers": num_layers, "heads": num_heads},
        "global_counts": counts_dict([value for row in matrix for value in row]),
        "depth_bands": bands,
        "adjacent_layer_label_persistence": adjacent_same / max(adjacent_total, 1),
        "transition_counts": transition_counts,
        "per_layer": per_layer,
        "per_head_index": per_head_index,
        "highest_entropy_layers": sorted(
            per_layer, key=lambda item: item["entropy_bits"], reverse=True
        )[:8],
        "most_depth_variable_head_indices": sorted(
            per_head_index, key=lambda item: item["depth_transitions"], reverse=True
        ),
    }


def markdown(result: dict) -> str:
    lines = [
        "# PF Head Depth Analysis",
        "",
        f"Shape: {result['shape']['layers']} layers x {result['shape']['heads']} heads.",
        "",
        f"Adjacent-layer label persistence: {result['adjacent_layer_label_persistence']:.3f}.",
        "",
        "## Depth Bands",
        "",
        "| Band | Layers | Oscillating | Stable | Stable-sparse | Entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for band in result["depth_bands"]:
        counts = band["counts"]
        lines.append(
            f"| {band['name']} | {band['start_layer']}-{band['end_layer_exclusive'] - 1} "
            f"| {counts['oscillating']} | {counts['stable']} | {counts['stable_sparse']} "
            f"| {band['entropy_bits']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Entropy Layers",
            "",
            "| Layer | Oscillating | Stable | Stable-sparse | Entropy |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["highest_entropy_layers"]:
        counts = item["counts"]
        lines.append(
            f"| {item['layer']} | {counts['oscillating']} | {counts['stable']} "
            f"| {counts['stable_sparse']} | {item['entropy_bits']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Head-Index Depth Variability",
            "",
            "| Head index | Dominant | Dominant fraction | Depth transitions |",
            "|---:|---|---:|---:|",
        ]
    )
    for item in result["most_depth_variable_head_indices"]:
        lines.append(
            f"| {item['head_index']} | {item['dominant']} "
            f"| {item['dominant_fraction']:.3f} | {item['depth_transitions']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    result = analyze(load_matrix(args.csv_path))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
