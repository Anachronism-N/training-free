from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v97_threshold_pf_merge.py"
SPEC = importlib.util.spec_from_file_location("analyze_v97", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v97_analysis_builds_predeclared_decisions(tmp_path, monkeypatch):
    values = {method: 0.5 for method in MODULE.METHODS}
    values.update(
        {
            "prompt_tau_1p0_merge": 0.9,
            "prompt_tau_1p0_cyclic": 0.6,
            "prompt_tau_1p0_recent": 0.5,
            "prompt_tau_1p0_random_merge": 0.4,
            "prompt_tau_1p0_reversed_merge": 0.3,
            "pf_ar_stride_merge": 0.6,
            "pf_aw_stride_merge": 0.8,
            "pf_native": 0.9,
            "pf_anchor_extended_recent": 0.4,
            "pf_wave_extended_recent": 0.7,
            "pf_veil_extended_recent": 0.8,
        }
    )
    comprehensive = {
        "per_method": {
            method: {
                metric: value
                for metric in MODULE.COMPREHENSIVE_METRICS
            }
            for method, value in values.items()
        }
    }
    vbench = {
        "methods": {
            method: {
                "subject_consistency": value,
                "background_consistency": value,
                "aesthetic_quality": value,
                "imaging_quality": value,
                "dynamic_degree": value,
            }
            for method, value in values.items()
        }
    }
    comprehensive_path = tmp_path / "comprehensive.json"
    comprehensive_path.write_text(
        json.dumps(comprehensive), encoding="utf-8"
    )
    vbench_path = tmp_path / "vbench.json"
    vbench_path.write_text(json.dumps(vbench), encoding="utf-8")
    temporal_path = tmp_path / "temporal.csv"
    with temporal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("video", "temporal_jump")
        )
        writer.writeheader()
        for method, value in values.items():
            writer.writerow(
                {
                    "video": str(tmp_path / method / "000.mp4"),
                    "temporal_jump": 1.0 - value,
                }
            )
    classification = {
        "score_csv_sha256": "score-hash",
        "manual_thresholds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "automatic_thresholds": {"prompt_gmm2": 0.75},
        "gmm_gates": {
            "two_components_preferred_to_one": True,
            "two_components_preferred_to_three": True,
        },
        "maps": {
            name: {"label_counts": {"1": 240, "-1": 120}}
            for name in (
                "prompt_tau_0",
                "prompt_tau_0p5",
                "prompt_tau_1",
                "prompt_tau_1p5",
                "prompt_tau_2",
            )
        },
    }
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(
        json.dumps(classification), encoding="utf-8"
    )
    traces_path = tmp_path / "traces.json"
    traces_path.write_text(
        json.dumps({"strict_pass": True}), encoding="utf-8"
    )
    output_json = tmp_path / "analysis.json"
    output_md = tmp_path / "analysis.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--comprehensive",
            str(comprehensive_path),
            "--temporal-jump",
            str(temporal_path),
            "--vbench",
            str(vbench_path),
            "--classification",
            str(classification_path),
            "--policy-traces",
            str(traces_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    MODULE.main()

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["decisions"]["responsive_merge_supported"]
    assert result["decisions"]["prompt_classification_controls_pass"]
    assert result["decisions"]["preferred_pf_merge"] == "anchor_wave_vs_veil"
    assert result["decisions"]["most_important_pf_class"] == "anchor"
    assert len(result["threshold_sweep"]) == 5
    assert output_md.is_file()
