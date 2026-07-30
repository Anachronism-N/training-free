#!/usr/bin/env python3
"""Summarize v143 feature-threshold and feature-family clustering sensitivity."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


TAXONOMY_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "lifecycle_kv"
    / "head_taxonomy.py"
)
TAXONOMY_SPEC = importlib.util.spec_from_file_location(
    "v143_sensitivity_head_taxonomy", TAXONOMY_PATH
)
TAXONOMY = importlib.util.module_from_spec(TAXONOMY_SPEC)
assert TAXONOMY_SPEC.loader is not None
sys.modules[TAXONOMY_SPEC.name] = TAXONOMY
TAXONOMY_SPEC.loader.exec_module(TAXONOMY)

adjusted_rand_index = TAXONOMY.adjusted_rand_index
normalized_mutual_information = TAXONOMY.normalized_mutual_information

LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
EXPECTED_THRESHOLDS = (0.30, 0.50, 0.70)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assignments(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    rows = _read_csv(path)
    by_head = {
        int(row["layer"]) * HEADS + int(row["head"]): int(row["cluster"])
        for row in rows
    }
    if set(by_head) != set(range(TOTAL_HEADS)):
        raise RuntimeError(f"{path} does not contain exactly 360 heads")
    return np.asarray([by_head[index] for index in range(TOTAL_HEADS)])


def _best_diagnostic(report: dict) -> dict:
    diagnostics = list(report.get("diagnostics") or [])
    if not diagnostics:
        raise RuntimeError("clustering report has no k diagnostics")
    selected = report.get("selected_clusters")
    if selected is not None:
        matches = [
            row for row in diagnostics if int(row["clusters"]) == int(selected)
        ]
        if len(matches) != 1:
            raise RuntimeError("selected k is absent from clustering diagnostics")
        return matches[0]
    return max(
        diagnostics,
        key=lambda row: (
            float(row["split_ari"]),
            float(row["bootstrap_ari_median"]),
            float(row["discovery_silhouette"]),
        ),
    )


def _load_variant(name: str, directory: Path) -> dict:
    report_path = directory / "clustering_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = _best_diagnostic(report)
    excluded = list(report.get("excluded_feature_groups") or [])
    return {
        "name": name,
        "directory": str(directory),
        "report": report,
        "assignments": _assignments(
            directory / "head_cluster_assignments.csv"
        ),
        "row": {
            "variant": name,
            "minimum_feature_split_spearman": float(
                report.get("minimum_feature_split_spearman", 0.30)
            ),
            "excluded_feature_groups": ",".join(excluded),
            "selection_status": str(report["selection_status"]),
            "selected_clusters": (
                ""
                if report.get("selected_clusters") is None
                else int(report["selected_clusters"])
            ),
            "accepted_feature_count": int(report["feature_count"]),
            "best_diagnostic_k": int(best["clusters"]),
            "best_split_label_agreement": float(
                best["split_label_agreement"]
            ),
            "best_split_ari": float(best["split_ari"]),
            "best_discovery_silhouette": float(
                best["discovery_silhouette"]
            ),
            "best_bootstrap_ari_median": float(
                best["bootstrap_ari_median"]
            ),
            "best_minimum_cluster_fraction": float(
                best["minimum_cluster_fraction"]
            ),
            "best_passed": int(best["passed"]),
        },
    }


def _finite_min(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else None


def summarize(
    baseline_dir: Path,
    variant_root: Path,
    output_dir: Path,
) -> dict:
    variants = [_load_variant("rho_030", baseline_dir)]
    if not variant_root.is_dir():
        raise FileNotFoundError(variant_root)
    for directory in sorted(path for path in variant_root.iterdir() if path.is_dir()):
        if (directory / "clustering_report.json").is_file():
            variants.append(_load_variant(directory.name, directory))
    names = [variant["name"] for variant in variants]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate clustering sensitivity variant names")

    pairwise_rows = []
    for left_index, left in enumerate(variants):
        if left["assignments"] is None:
            continue
        for right in variants[left_index + 1 :]:
            if right["assignments"] is None:
                continue
            pairwise_rows.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "left_clusters": int(
                        left["report"]["selected_clusters"]
                    ),
                    "right_clusters": int(
                        right["report"]["selected_clusters"]
                    ),
                    "ari": adjusted_rand_index(
                        left["assignments"], right["assignments"]
                    ),
                    "nmi": normalized_mutual_information(
                        left["assignments"], right["assignments"]
                    ),
                }
            )

    threshold_variants = [
        variant
        for variant in variants
        if not variant["report"].get("excluded_feature_groups")
        and any(
            math.isclose(
                float(
                    variant["report"].get(
                        "minimum_feature_split_spearman", 0.30
                    )
                ),
                threshold,
            )
            for threshold in EXPECTED_THRESHOLDS
        )
    ]
    threshold_values = {
        round(
            float(
                variant["report"].get(
                    "minimum_feature_split_spearman", 0.30
                )
            ),
            2,
        )
        for variant in threshold_variants
    }
    expected_values = {round(value, 2) for value in EXPECTED_THRESHOLDS}
    threshold_pairs = [
        row
        for row in pairwise_rows
        if row["left"].startswith("rho_") and row["right"].startswith("rho_")
    ]
    selected_thresholds = [
        variant
        for variant in threshold_variants
        if variant["assignments"] is not None
    ]
    selected_k = {
        int(variant["report"]["selected_clusters"])
        for variant in selected_thresholds
    }
    threshold_ari_min = _finite_min(
        [float(row["ari"]) for row in threshold_pairs]
    )
    threshold_gate = bool(
        threshold_values == expected_values
        and len(selected_thresholds) == len(EXPECTED_THRESHOLDS)
        and len(selected_k) == 1
        and threshold_ari_min is not None
        and threshold_ari_min >= 0.80
    )

    baseline_pairs = [
        row
        for row in pairwise_rows
        if row["left"] == "rho_030" or row["right"] == "rho_030"
    ]
    leave_one_group_pairs = [
        row
        for row in baseline_pairs
        if (
            row["left"].startswith("drop_")
            or row["right"].startswith("drop_")
        )
    ]
    report = {
        "version": 1,
        "variant_count": len(variants),
        "variants": [variant["row"] for variant in variants],
        "pairwise_comparison_count": len(pairwise_rows),
        "thresholds_expected": list(EXPECTED_THRESHOLDS),
        "thresholds_found": sorted(threshold_values),
        "threshold_selected_k": sorted(selected_k),
        "threshold_pairwise_ari_min": threshold_ari_min,
        "threshold_sensitivity_gate": threshold_gate,
        "threshold_gate_definition": (
            "all rho thresholds produce a stable map with the same k and "
            "minimum pairwise ARI >= 0.80"
        ),
        "leave_one_group_pair_count": len(leave_one_group_pairs),
        "leave_one_group_ari_min": _finite_min(
            [float(row["ari"]) for row in leave_one_group_pairs]
        ),
        "claim_boundary": (
            "sensitivity supports descriptive taxonomy robustness only; "
            "functional role names still require cache-policy interventions"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "cluster_sensitivity_variants.csv",
        [variant["row"] for variant in variants],
    )
    _write_csv(
        output_dir / "cluster_sensitivity_pairwise.csv",
        pairwise_rows,
    )
    (output_dir / "cluster_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# v143 Cluster Sensitivity",
        "",
        f"- Variants: `{len(variants)}`",
        f"- Threshold sensitivity gate: `{threshold_gate}`",
        f"- Threshold pairwise ARI minimum: `{threshold_ari_min}`",
        f"- Leave-one-group ARI minimum: "
        f"`{report['leave_one_group_ari_min']}`",
        "",
        "This is a descriptive stability check. It does not assign functional "
        "roles or validate a cache policy.",
    ]
    (output_dir / "cluster_sensitivity_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.baseline_dir,
        args.variant_root,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
