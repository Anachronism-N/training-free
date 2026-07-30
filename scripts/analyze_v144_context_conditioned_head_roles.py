#!/usr/bin/env python3
"""Audit whether v143 A/B axes are static head roles or state-conditioned programs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
EPSILON = 1e-12
AXES = (
    "prompt_history_excess",
    "policy_prompt_score",
    "stale_a_mass",
    "persistent_content",
    "persistent_positioned",
    "persistent_output",
)
SPLITS = ("all", "discovery", "validation")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty context table: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
    if left_rank.std() <= EPSILON or right_rank.std() <= EPSILON:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _layer_residual(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (TOTAL_HEADS,) or not np.isfinite(array).all():
        raise ValueError("context vector must contain 360 finite head values")
    matrix = array.reshape(LAYERS, HEADS)
    return (matrix - np.median(matrix, axis=1, keepdims=True)).reshape(-1)


def _layer_eta_squared(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    grand = float(array.mean())
    total = float(np.square(array - grand).sum())
    if total <= EPSILON:
        return 0.0
    means = array.reshape(LAYERS, HEADS).mean(axis=1)
    between = float(HEADS * np.square(means - grand).sum())
    return between / total


def _top_jaccard(
    left: np.ndarray, right: np.ndarray, *, fraction: float = 0.25
) -> float:
    count = max(1, int(math.ceil(left.size * fraction)))
    left_top = set(np.argsort(left, kind="mergesort")[-count:].tolist())
    right_top = set(np.argsort(right, kind="mergesort")[-count:].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def _context_key(row: dict) -> tuple[str, str, int, int]:
    return (
        str(row["switch_type"]),
        str(row["mode"]),
        int(row["current_frame"]),
        int(row["nominal_timestep"]),
    )


def _head_index(row: dict) -> int:
    layer, head = int(row["layer"]), int(row["head"])
    if not 0 <= layer < LAYERS or not 0 <= head < HEADS:
        raise ValueError(f"invalid head coordinate: {layer}/{head}")
    return layer * HEADS + head


def _load_contexts(rows: list[dict]) -> dict:
    required = {
        "prompt_split",
        "switch_type",
        "mode",
        "current_frame",
        "nominal_timestep",
        "episode",
        "stale_a_visible",
        "layer",
        "head",
        *AXES,
    }
    if not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(
            "v143 context table lacks held-out prompt splits; rerun the "
            f"updated v143 analyzer. Missing: {missing}"
        )
    values = defaultdict(
        lambda: {
            split: {
                axis: np.full(TOTAL_HEADS, np.nan, dtype=np.float64)
                for axis in AXES
            }
            for split in SPLITS
        }
    )
    metadata = {}
    for row in rows:
        split = str(row["prompt_split"])
        if split not in SPLITS:
            raise ValueError(f"unexpected prompt split: {split}")
        key = _context_key(row)
        index = _head_index(row)
        current_metadata = (
            str(row["episode"]),
            bool(int(row["stale_a_visible"])),
        )
        previous = metadata.setdefault(key, current_metadata)
        if previous != current_metadata:
            raise ValueError(f"context metadata differs across splits: {key}")
        for axis in AXES:
            values[key][split][axis][index] = float(row[axis])
    for key, split_values in values.items():
        for split in SPLITS:
            for axis in AXES:
                if not np.isfinite(split_values[split][axis]).all():
                    raise ValueError(
                        f"incomplete context vector: {key}/{split}/{axis}"
                    )
    return {
        key: {
            "episode": metadata[key][0],
            "stale_a_visible": metadata[key][1],
            "values": split_values,
        }
        for key, split_values in values.items()
    }


def analyze(
    context_csv: Path,
    output_dir: Path,
    *,
    minimum_split_rho: float,
    minimum_context_rho: float,
) -> dict:
    contexts = _load_contexts(_read_csv(context_csv))
    context_rows = []
    axis_context_vectors = defaultdict(list)
    for key, payload in sorted(contexts.items()):
        switch_type, mode, frame, timestep = key
        for axis in AXES:
            if axis == "stale_a_mass" and not (
                payload["episode"] == "B" and payload["stale_a_visible"]
            ):
                continue
            all_values = payload["values"]["all"][axis]
            discovery = payload["values"]["discovery"][axis]
            validation = payload["values"]["validation"][axis]
            all_residual = _layer_residual(all_values)
            discovery_residual = _layer_residual(discovery)
            validation_residual = _layer_residual(validation)
            row = {
                "axis": axis,
                "switch_type": switch_type,
                "mode": mode,
                "current_frame": frame,
                "nominal_timestep": timestep,
                "episode": payload["episode"],
                "stale_a_visible": int(payload["stale_a_visible"]),
                "raw_split_spearman": _spearman(discovery, validation),
                "layer_residual_split_spearman": _spearman(
                    discovery_residual, validation_residual
                ),
                "raw_split_top25_jaccard": _top_jaccard(
                    discovery, validation
                ),
                "layer_residual_split_top25_jaccard": _top_jaccard(
                    discovery_residual, validation_residual
                ),
                "layer_eta_squared": _layer_eta_squared(all_values),
                "raw_iqr": float(
                    np.quantile(all_values, 0.75)
                    - np.quantile(all_values, 0.25)
                ),
                "layer_residual_iqr": float(
                    np.quantile(all_residual, 0.75)
                    - np.quantile(all_residual, 0.25)
                ),
            }
            context_rows.append(row)
            axis_context_vectors[axis].append(
                (key, all_values, all_residual)
            )

    summary_rows = []
    head_rows = []
    for axis in AXES:
        rows = [row for row in context_rows if row["axis"] == axis]
        vectors = axis_context_vectors[axis]
        raw_context_rho = []
        residual_context_rho = []
        raw_context_top = []
        residual_context_top = []
        for (_, left, left_residual), (
            _,
            right,
            right_residual,
        ) in itertools.combinations(vectors, 2):
            raw_context_rho.append(_spearman(left, right))
            residual_context_rho.append(
                _spearman(left_residual, right_residual)
            )
            raw_context_top.append(_top_jaccard(left, right))
            residual_context_top.append(
                _top_jaccard(left_residual, right_residual)
            )
        residual_matrix = np.stack(
            [vector[2] for vector in vectors], axis=0
        )
        rank_matrix = np.stack(
            [_rankdata(vector) / (TOTAL_HEADS - 1) for vector in residual_matrix],
            axis=0,
        )
        top_count = max(1, int(math.ceil(TOTAL_HEADS * 0.25)))
        top_frequency = np.zeros(TOTAL_HEADS, dtype=np.float64)
        for vector in residual_matrix:
            top_frequency[
                np.argsort(vector, kind="mergesort")[-top_count:]
            ] += 1.0
        top_frequency /= residual_matrix.shape[0]
        for flat_head in range(TOTAL_HEADS):
            head_rows.append(
                {
                    "axis": axis,
                    "layer": flat_head // HEADS,
                    "head": flat_head % HEADS,
                    "context_count": residual_matrix.shape[0],
                    "residual_median": float(
                        np.median(residual_matrix[:, flat_head])
                    ),
                    "residual_iqr": float(
                        np.quantile(residual_matrix[:, flat_head], 0.75)
                        - np.quantile(residual_matrix[:, flat_head], 0.25)
                    ),
                    "rank_median": float(
                        np.median(rank_matrix[:, flat_head])
                    ),
                    "rank_iqr": float(
                        np.quantile(rank_matrix[:, flat_head], 0.75)
                        - np.quantile(rank_matrix[:, flat_head], 0.25)
                    ),
                    "top25_context_frequency": float(
                        top_frequency[flat_head]
                    ),
                    "positive_context_frequency": float(
                        np.mean(residual_matrix[:, flat_head] > 0)
                    ),
                }
            )
        median_split = float(
            np.median(
                [row["layer_residual_split_spearman"] for row in rows]
            )
        )
        median_context = (
            float(np.median(residual_context_rho))
            if residual_context_rho
            else float("nan")
        )
        static_gate = bool(
            median_split >= minimum_split_rho
            and median_context >= minimum_context_rho
        )
        summary_rows.append(
            {
                "axis": axis,
                "context_count": len(rows),
                "median_raw_split_spearman": float(
                    np.median([row["raw_split_spearman"] for row in rows])
                ),
                "median_layer_residual_split_spearman": median_split,
                "minimum_layer_residual_split_spearman": float(
                    min(
                        row["layer_residual_split_spearman"]
                        for row in rows
                    )
                ),
                "median_layer_eta_squared": float(
                    np.median([row["layer_eta_squared"] for row in rows])
                ),
                "median_raw_cross_context_spearman": (
                    float(np.median(raw_context_rho))
                    if raw_context_rho
                    else float("nan")
                ),
                "median_layer_residual_cross_context_spearman": median_context,
                "median_raw_cross_context_top25_jaccard": (
                    float(np.median(raw_context_top))
                    if raw_context_top
                    else float("nan")
                ),
                "median_layer_residual_cross_context_top25_jaccard": (
                    float(np.median(residual_context_top))
                    if residual_context_top
                    else float("nan")
                ),
                "heads_top25_in_at_least_half_contexts": int(
                    np.sum(top_frequency >= 0.5)
                ),
                "static_head_axis_gate": int(static_gate),
                "interpretation": (
                    "static_head_candidate"
                    if static_gate
                    else "layer_or_state_conditioned"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "context_split_audit.csv", context_rows)
    _write_csv(output_dir / "axis_role_summary.csv", summary_rows)
    _write_csv(output_dir / "head_context_stability.csv", head_rows)
    report = {
        "version": 1,
        "source": str(context_csv),
        "context_count": len(contexts),
        "minimum_layer_residual_split_spearman": minimum_split_rho,
        "minimum_layer_residual_cross_context_spearman": minimum_context_rho,
        "static_head_axis_count": sum(
            int(row["static_head_axis_gate"]) for row in summary_rows
        ),
        "axis_summary": {row["axis"]: row for row in summary_rows},
        "claim_rule": (
            "An axis is a static head candidate only when held-out prompt "
            "stability and cross-context stability both survive within-layer "
            "median removal. Passing is descriptive; functional role names "
            "still require head-selective causal interventions."
        ),
    }
    (output_dir / "context_role_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# v144 Context-conditioned Head-role Audit",
        "",
        "| axis | residual split rho | residual context rho | layer eta2 | "
        "persistent top heads | interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['axis']} | "
            f"{row['median_layer_residual_split_spearman']:.4f} | "
            f"{row['median_layer_residual_cross_context_spearman']:.4f} | "
            f"{row['median_layer_eta_squared']:.4f} | "
            f"{row['heads_top25_in_at_least_half_contexts']} | "
            f"{row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "Passing this audit permits the phrase `static head candidate`, "
            "not a functional role claim. Failed axes should be modeled as "
            "layer/timestep/episode-conditioned routing signals.",
        ]
    )
    (output_dir / "context_role_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-csv",
        type=Path,
        default=Path(
            "runs/v143_multiaxis_profile/analysis/ab_context_axes.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-split-rho", type=float, default=0.50)
    parser.add_argument("--minimum-context-rho", type=float, default=0.30)
    args = parser.parse_args()
    report = analyze(
        args.context_csv,
        args.output_dir,
        minimum_split_rho=args.minimum_split_rho,
        minimum_context_rho=args.minimum_context_rho,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
