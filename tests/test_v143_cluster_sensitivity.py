from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_v143_cluster_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("v143_cluster_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_variant(
    directory: Path,
    *,
    threshold: float,
    excluded: list[str] | None = None,
) -> None:
    directory.mkdir(parents=True)
    report = {
        "minimum_feature_split_spearman": threshold,
        "excluded_feature_groups": excluded or [],
        "selection_status": "validated_candidate",
        "selected_clusters": 2,
        "feature_count": 8,
        "diagnostics": [
            {
                "clusters": 2,
                "split_label_agreement": 0.95,
                "split_ari": 0.90,
                "discovery_silhouette": 0.30,
                "bootstrap_ari_median": 0.92,
                "minimum_cluster_fraction": 0.40,
                "passed": 1,
            }
        ],
    }
    (directory / "clustering_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with (directory / "head_cluster_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("layer", "head", "cluster")
        )
        writer.writeheader()
        for flat_head in range(MODULE.TOTAL_HEADS):
            writer.writerow(
                {
                    "layer": flat_head // MODULE.HEADS,
                    "head": flat_head % MODULE.HEADS,
                    "cluster": int(flat_head >= MODULE.TOTAL_HEADS // 2),
                }
            )


def test_cluster_sensitivity_requires_stable_threshold_maps(tmp_path):
    baseline = tmp_path / "baseline"
    variants = tmp_path / "variants"
    _write_variant(baseline, threshold=0.30)
    _write_variant(variants / "rho_050", threshold=0.50)
    _write_variant(variants / "rho_070", threshold=0.70)
    _write_variant(
        variants / "drop_prompt_modulation",
        threshold=0.30,
        excluded=["prompt_modulation"],
    )
    report = MODULE.summarize(baseline, variants, tmp_path / "output")
    assert report["threshold_sensitivity_gate"] is True
    assert report["threshold_pairwise_ari_min"] == 1.0
    assert report["leave_one_group_ari_min"] == 1.0
