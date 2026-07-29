#!/usr/bin/env python3
"""Audit the frozen v98 binary head partition and export paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


PF_NAMES = {-1: "wave", 1: "anchor", 2: "veil"}
DEFAULT_THRESHOLDS = (-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty list")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    return {
        "count": len(data),
        "min": min(data),
        "q10": quantile(data, 0.10),
        "q25": quantile(data, 0.25),
        "median": quantile(data, 0.50),
        "q75": quantile(data, 0.75),
        "q90": quantile(data, 0.90),
        "max": max(data),
        "mean": sum(data) / len(data),
    }


def read_map(path: Path) -> list[list[int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        matrix = [[int(value) for value in row] for row in csv.reader(handle)]
    if len(matrix) != 30 or any(len(row) != 12 for row in matrix):
        raise ValueError(f"expected a 30x12 head map: {path}")
    if set(value for row in matrix for value in row) - {10, 11}:
        raise ValueError(f"head map contains labels outside 10/11: {path}")
    return matrix


def read_assignments(path: Path) -> list[dict[str, int | float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    required = {
        "layer",
        "head",
        "pf_label",
        "signed_logit_mass",
        "positive_rate",
        "history_polarity_zero",
    }
    if not raw or not required <= set(raw[0]):
        raise ValueError(f"assignment columns differ: {path}")
    rows = []
    for item in raw:
        rows.append(
            {
                "layer": int(item["layer"]),
                "head": int(item["head"]),
                "pf_label": int(item["pf_label"]),
                "pf_name": item.get("pf_name") or PF_NAMES[int(item["pf_label"])],
                "score": float(item["signed_logit_mass"]),
                "positive_rate": float(item["positive_rate"]),
                "assignment": int(item["history_polarity_zero"]),
            }
        )
    expected = {(layer, head) for layer in range(30) for head in range(12)}
    observed = {
        (int(row["layer"]), int(row["head"]))
        for row in rows
    }
    if len(rows) != 360 or observed != expected:
        raise ValueError("assignments do not cover the exact 30x12 head grid")
    return sorted(rows, key=lambda row: (int(row["layer"]), int(row["head"])))


def binary_metrics(
    predicted_suppressive: set[tuple[int, int]],
    target_suppressive: set[tuple[int, int]],
    universe: set[tuple[int, int]],
) -> dict[str, float | int]:
    tp = len(predicted_suppressive & target_suppressive)
    fp = len(predicted_suppressive - target_suppressive)
    fn = len(target_suppressive - predicted_suppressive)
    tn = len(universe - predicted_suppressive - target_suppressive)
    tpr = tp / (tp + fn) if tp + fn else 1.0
    tnr = tn / (tn + fp) if tn + fp else 1.0
    union = predicted_suppressive | target_suppressive
    return {
        "agreement": (tp + tn) / len(universe),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "suppressive_jaccard": (
            len(predicted_suppressive & target_suppressive) / len(union)
            if union
            else 1.0
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_report(
    *,
    assignments_path: Path,
    map_path: Path,
    source_manifest_path: Path,
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    rows = read_assignments(assignments_path)
    matrix = read_map(map_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    declared_score_path = Path(str(manifest.get("score_csv", "")))
    fallback_score_path = (
        source_manifest_path.parents[3]
        / "runs/v97_qk_head_scores/scores/qk_head_scores.csv"
    )
    score_path = (
        declared_score_path
        if declared_score_path.is_file()
        else fallback_score_path
    )
    if (
        not score_path.is_file()
        or manifest.get("score_csv_sha256") != sha256(score_path)
    ):
        raise ValueError("source score CSV differs from v98 map manifest")

    mismatches = []
    for row in rows:
        layer, head = int(row["layer"]), int(row["head"])
        expected = 10 if float(row["score"]) >= 0.0 else 11
        actual = matrix[layer][head]
        if int(row["assignment"]) != expected or actual != expected:
            mismatches.append(
                {
                    "layer": layer,
                    "head": head,
                    "score": row["score"],
                    "assignment_csv": row["assignment"],
                    "map": actual,
                    "expected": expected,
                }
            )
    if mismatches:
        raise ValueError(f"frozen map/score mismatches: {mismatches[:5]}")

    universe = {
        (int(row["layer"]), int(row["head"]))
        for row in rows
    }
    base_suppressive = {
        (int(row["layer"]), int(row["head"]))
        for row in rows
        if float(row["score"]) < 0.0
    }
    pf_aw_suppressive = {
        (int(row["layer"]), int(row["head"]))
        for row in rows
        if int(row["pf_label"]) == 2
    }
    pf_ar_suppressive = {
        (int(row["layer"]), int(row["head"]))
        for row in rows
        if int(row["pf_label"]) in {-1, 2}
    }

    sweep = []
    for threshold in thresholds:
        suppressive = {
            (int(row["layer"]), int(row["head"]))
            for row in rows
            if float(row["score"]) < threshold
        }
        sweep.append(
            {
                "threshold": threshold,
                "supportive_heads": len(universe - suppressive),
                "suppressive_heads": len(suppressive),
                "heads_changed_from_zero": len(
                    suppressive.symmetric_difference(base_suppressive)
                ),
                "zero_suppressive_jaccard": (
                    len(suppressive & base_suppressive)
                    / len(suppressive | base_suppressive)
                    if suppressive | base_suppressive
                    else 1.0
                ),
                "posthoc_pf_aw": binary_metrics(
                    suppressive, pf_aw_suppressive, universe
                ),
                "posthoc_pf_ar": binary_metrics(
                    suppressive, pf_ar_suppressive, universe
                ),
            }
        )

    layer_rows = []
    for layer in range(30):
        layer_scores = [
            float(row["score"]) for row in rows if int(row["layer"]) == layer
        ]
        layer_rows.append(
            {
                "layer": layer,
                "supportive_heads": sum(score >= 0.0 for score in layer_scores),
                "suppressive_heads": sum(score < 0.0 for score in layer_scores),
                "score_median": quantile(layer_scores, 0.5),
                "score_min": min(layer_scores),
                "score_max": max(layer_scores),
            }
        )

    class_counts = Counter(
        "supportive" if float(row["score"]) >= 0.0 else "suppressive"
        for row in rows
    )
    cross_tab = {
        name: {
            "heads": sum(int(row["pf_label"]) == label for row in rows),
            "supportive": sum(
                int(row["pf_label"]) == label and float(row["score"]) >= 0.0
                for row in rows
            ),
            "suppressive": sum(
                int(row["pf_label"]) == label and float(row["score"]) < 0.0
                for row in rows
            ),
        }
        for label, name in PF_NAMES.items()
    }
    return {
        "version": 1,
        "method": "v132_frozen_binary_head_partition_audit",
        "classifier": {
            "score": (
                "median_over_records(sum_history_qk_logits / "
                "sum_abs_history_qk_logits)"
            ),
            "rule": "supportive if score >= 0; suppressive otherwise",
            "threshold": 0.0,
            "pf_labels_used_for_classification": False,
            "pf_labels_used_for_posthoc_interpretation_only": True,
        },
        "inputs": {
            "assignments": str(assignments_path.resolve()),
            "assignments_sha256": sha256(assignments_path),
            "frozen_map": str(map_path.resolve()),
            "frozen_map_sha256": sha256(map_path),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_score_csv": str(score_path.resolve()),
            "source_score_csv_sha256": sha256(score_path),
        },
        "head_count": len(rows),
        "class_counts": dict(class_counts),
        "score_distributions": {
            "all": distribution(float(row["score"]) for row in rows),
            "supportive": distribution(
                float(row["score"])
                for row in rows
                if float(row["score"]) >= 0.0
            ),
            "suppressive": distribution(
                float(row["score"])
                for row in rows
                if float(row["score"]) < 0.0
            ),
            **{
                f"pf_{name}_posthoc": distribution(
                    float(row["score"])
                    for row in rows
                    if int(row["pf_label"]) == label
                )
                for label, name in PF_NAMES.items()
            },
        },
        "posthoc_pf_cross_tab": cross_tab,
        "threshold_sweep": sweep,
        "layers": layer_rows,
        "map_score_mismatches": mismatches,
    }


def write_report(report: dict[str, object], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "head_partition_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# v132 binary head-partition audit",
        "",
        "PF labels are used only for post-hoc interpretation, never to form "
        "the proposed partition.",
        "",
        f"- Heads: {report['head_count']}",
        f"- Counts: `{report['class_counts']}`",
        f"- Frozen-map mismatches: {len(report['map_score_mismatches'])}",
        "",
        "## Post-hoc PF cross-tab",
        "",
        "| PF class | Heads | Supportive | Suppressive |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["posthoc_pf_cross_tab"].items():
        lines.append(
            f"| {name} | {row['heads']} | {row['supportive']} | "
            f"{row['suppressive']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold stability",
            "",
            "| Threshold | Supportive | Suppressive | Changed vs 0 | "
            "Jaccard vs 0 | PF-AW agreement (post-hoc) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.3f} | {row['supportive_heads']} | "
            f"{row['suppressive_heads']} | "
            f"{row['heads_changed_from_zero']} | "
            f"{row['zero_suppressive_jaccard']:.4f} | "
            f"{row['posthoc_pf_aw']['agreement']:.4f} |"
        )
    md_path = output_root / "head_partition_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[v132-head-report] json={json_path} md={md_path}", flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=root / "runs/v98_history_polarity/maps/head_assignments.csv",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=root / "configs/head_maps/legacy_v98_absolute_sign_304_56.csv",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(
            root
            / "runs/v98_history_polarity/maps/history_polarity_manifest.json"
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "runs/v132_head_partition_evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = tuple(
        sorted({float(value.strip()) for value in args.thresholds.split(",")})
    )
    if not thresholds or 0.0 not in thresholds:
        raise SystemExit("threshold sweep must include 0")
    report = build_report(
        assignments_path=args.assignments.resolve(),
        map_path=args.map.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        thresholds=thresholds,
    )
    write_report(report, args.output_root.resolve())


if __name__ == "__main__":
    main()
