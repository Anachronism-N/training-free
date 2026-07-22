import json

from scripts.compare_hrem_role_ablation import build_comparison


def test_build_comparison_keeps_retrieval_and_role_metrics_separate(tmp_path) -> None:
    metrics = {
        "aggregate": {
            "native_reset": {
                "full": {"return_margin": 0.3},
                "background": {"return_margin": 0.4},
            },
            "role_hybrid_050": {
                "full": {"return_margin": 0.35},
                "background": {"return_margin": 0.42},
            },
        },
        "paired_delta": {
            "role_hybrid_050": {
                "full": {"return_margin_mean": 0.05, "positive_prompts": 2}
            }
        },
    }
    (tmp_path / "traces").mkdir()
    (tmp_path / "metrics_role_ablation.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    diagnosis = {
        "metrics": {
            "retrieval_accepted_head_fraction": 1.0,
            "head_gate_mean": 0.45,
            "role_gate_std": 0.20,
            "role_active_head_fraction": 0.5,
            "role_evidence_spread_median": 0.2,
            "role_calibration_valid_fraction": 1.0,
            "role_active_head_jaccard": 0.8,
            "delta_to_native_rms_median": 0.03,
        },
        "findings": [],
    }
    (tmp_path / "traces" / "role_hybrid_050_diagnosis.json").write_text(
        json.dumps(diagnosis), encoding="utf-8"
    )

    report = build_comparison(tmp_path)
    role = next(row for row in report["rows"] if row["method"] == "role_hybrid_050")
    assert role["retrieval_accepted_head_fraction"] == 1.0
    assert role["role_active_head_fraction"] == 0.5
    assert role["structurally_eligible_for_visual_review"]
    assert report["eligible_methods"] == ["role_hybrid_050"]


def test_build_comparison_rejects_uncontrolled_config_drift(tmp_path) -> None:
    metrics = {
        "aggregate": {
            "dual_all_heads": {"full": {}, "background": {}},
            "role_relative_050": {"full": {}, "background": {}},
        },
        "paired_delta": {},
    }
    (tmp_path / "traces").mkdir()
    (tmp_path / "metrics_role_ablation.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    common_metrics = {
        "role_active_head_fraction": 0.5,
        "role_calibration_valid_fraction": 1.0,
        "role_active_head_jaccard": 0.8,
    }
    base_config = {
        "active_layers": [15, 18, 20],
        "archive": {"stride": 4},
        "readout": {
            "gate": 0.35,
            "head_routing": "all",
            "role_calibration": "absolute",
        },
    }
    role_config = {
        **base_config,
        "readout": {
            **base_config["readout"],
            "gate": 0.50,
            "head_routing": "persistent",
            "role_calibration": "relative",
        },
    }
    for method, config in {
        "dual_all_heads": base_config,
        "role_relative_050": role_config,
    }.items():
        diagnosis = {
            "config": config,
            "metrics": common_metrics,
            "findings": [],
        }
        (tmp_path / "traces" / f"{method}_diagnosis.json").write_text(
            json.dumps(diagnosis), encoding="utf-8"
        )

    report = build_comparison(tmp_path)
    role = next(row for row in report["rows"] if row["method"] == "role_relative_050")
    assert role["unexpected_config_differences"] == ["readout.gate"]
    assert not role["structurally_eligible_for_visual_review"]
    assert report["eligible_methods"] == []
