from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_v201_fixture(tmp_path: Path, module) -> tuple[Path, Path, Path, Path, Path]:
    source = sys.modules["prepare_v201_head_phase_horizon_screen"]
    development = tmp_path / "development.txt"
    development.write_text(
        "\n".join(f"development prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout32.txt"
    holdout.write_text(
        "\n".join(f"development prompt {index}" for index in range(96, 128))
        + "\n",
        encoding="utf-8",
    )
    bank = tmp_path / "all_heads.csv"
    bank.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )

    def route(classification: str, mode: str) -> tuple[Path, dict]:
        masks = [
            [
                [[False for _ in range(12)] for _ in range(30)]
                for _ in range(4)
            ]
            for _ in range(2)
        ]
        if mode == "all":
            masks = [
                [
                    [[True for _ in range(12)] for _ in range(30)]
                    for _ in range(4)
                ]
                for _ in range(2)
            ]
        elif mode == "one":
            masks[0][0][10][3] = True
            masks[1][0][10][4] = True
        payload = source._map_payload(
            masks,
            operator="retrieval",
            classification=classification,
            current_frames=[8, 16],
            parent_map_id="fixture-parent",
            v200_analysis_sha256="a" * 64,
        )
        path = tmp_path / f"{classification}.json"
        write_json(path, payload)
        return path, payload

    map_specs = {
        "retrieval_all_recent": ("all_recent", "none"),
        "retrieval_all_coverage": ("all_coverage", "all"),
        "retrieval_static_top10": ("static_top10", "one"),
        "retrieval_horizon_top10": ("horizon_top10", "one"),
        "retrieval_horizon_shift_top10": ("horizon_shift_top10", "one"),
    }
    roles = {
        "retrieval_all_recent": "operator_matched_local_control",
        "retrieval_all_coverage": "operator_matched_universal_coverage_control",
        "retrieval_static_top10": "equal_exposure_static_head_phase_control",
        "retrieval_horizon_top10": "primary_head_phase_horizon",
        "retrieval_horizon_shift_top10": "equal_exposure_horizon_alignment_control",
    }
    methods = {
        "sf_native": {
            "runtime": "sf_native",
            "role": "canonical_sf_baseline",
            "operator": None,
        }
    }
    map_rows = {}
    for key, (classification, mode) in map_specs.items():
        path, payload = route(classification, mode)
        map_rows[key] = (path, payload)
        exposure = sum(payload["coverage_count_by_position"])
        methods[key] = {
            "runtime": "head_phase_horizon_cache_runtime",
            "role": roles[key],
            "operator": "retrieval",
            "history_policy": "retrieval",
            "schedule": "head_phase_horizon",
            "horizon_map": str(path),
            "horizon_map_sha256": module.sha256(path),
            "routing_map_id": payload["map_id"],
            "map_classification": classification,
            "current_frames": payload["current_frames"],
            "coverage_count_by_position": payload["coverage_count_by_position"],
            "coverage_count_by_position_call": payload[
                "coverage_count_by_position_call"
            ],
            "coverage_exposure_count": exposure,
            "coverage_exposure_fraction": exposure / (2 * 4 * 30 * 12),
            "head_bank_map": str(bank),
            "head_bank_map_sha256": module.sha256(bank),
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }
    order = ["sf_native", *map_specs]
    selector = [
        "retrieval_static_top10",
        "retrieval_horizon_top10",
        "retrieval_horizon_shift_top10",
    ]
    frozen = {
        "version": 2,
        "experiment": "v201_head_phase_horizon_causal_screen",
        "scope": "classifier_holdout32",
        "development_only": True,
        "prompt_count": 32,
        "source_prompt_count": 128,
        "source_prompt_file": str(development),
        "source_prompt_file_sha256": module.sha256(development),
        "source_indices": list(range(96, 128)),
        "prompt_file": str(holdout),
        "prompt_file_sha256": module.sha256(holdout),
        "prompt_items": [
            {"index": i, "source_index": 96 + i, "text": f"development prompt {96+i}"}
            for i in range(32)
        ],
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 20100,
        "operators": ["retrieval"],
        "method_order": order,
        "methods": methods,
        "operator_contracts": {
            "retrieval": {
                "method_order": order[1:],
                "equal_exposure_selector_methods": selector,
                "coverage_cells_per_position": 1,
                "equal_exposure_verified": True,
                "v200_horizon_gate": True,
            }
        },
        "primary_baseline": "sf_native",
    }
    input_path = tmp_path / "v201_inputs.json"
    write_json(input_path, frozen)

    generation_contract = tmp_path / "v201_generation_contract.json"
    write_json(
        generation_contract,
        {
            "scope": "screen32",
            "primary_baseline": "sf_native",
            "prompt_count": 32,
            "input_manifest_sha256": module.sha256(input_path),
        },
    )
    published_rows = []
    for method in order:
        audit = tmp_path / f"{method}.audit.json"
        write_json(audit, {"ok": True, "method": method})
        published_rows.append(
            {
                "key": method,
                "ok": True,
                "audit": str(audit),
                "audit_sha256": module.sha256(audit),
            }
        )
    published = {
        "ok": True,
        "experiment": "v201_head_phase_horizon_causal_generation",
        "scope": "screen32",
        "methods": published_rows,
        "experiment_contract": str(generation_contract),
        "experiment_contract_sha256": module.sha256(generation_contract),
    }
    published_path = tmp_path / "v201_published.json"
    write_json(published_path, published)

    comparison = {
        "experiment": "v201_head_phase_horizon_causal_vbench_screen32",
        "prompt_count": 32,
        "primary_baseline": "sf_native",
        "methods": [{"key": key} for key in order],
        "source": {
            "published_manifest_sha256": module.sha256(published_path),
            "experiment_contract_sha256": module.sha256(generation_contract),
        },
    }
    comparison_path = tmp_path / "v201_comparison.json"
    write_json(comparison_path, comparison)
    evidence = {}
    for key, content in (
        ("vbench_summary", "{}\n"),
        ("temporal_diagnostics", "method,prompt_index\n"),
        ("temporal_contract", "{}\n"),
    ):
        path = tmp_path / key
        path.write_text(content, encoding="utf-8")
        evidence[key] = str(path)
        evidence[f"{key}_sha256"] = module.sha256(path)
    candidate = "retrieval_horizon_top10"
    status = {
        "operator": "retrieval",
        "sf_efficacy": {
            "noninferiority": {"pass": True},
            "positive_support": {
                "directional_pass": True,
                "directional_axes": [
                    {
                        "window": "full",
                        "metric": "quality_without_dynamic_degree",
                        "mean_delta": 0.2,
                    }
                ],
            },
            "temporal_guard": {"automatic_safety_pass": True},
            "directional_screen_pass": True,
            "interval_supported_screen_pass": True,
        },
        "mechanism_attribution": {"interval_supported_pass": True},
        "selected_for_fresh128": True,
    }
    decision = {
        "version": 2,
        "experiment": "v201_head_phase_horizon_causal_vbench_screen32",
        "development_only": True,
        "recommendation": "advance_sf_significant_horizon_method_to_fresh128",
        "primary_baseline": "sf_native",
        "selected_for_fresh128": [candidate],
        "sf_interval_supported_candidates": [candidate],
        "mechanism_supported_candidates": [candidate],
        "candidate_status": {candidate: status},
        "source": {
            "comparison_manifest": str(comparison_path),
            "comparison_manifest_sha256": module.sha256(comparison_path),
            **evidence,
        },
    }
    decision_path = tmp_path / "v201_decision.json"
    write_json(decision_path, decision)

    fresh_prompts = tmp_path / "moviegen_fresh_0128_0255.txt"
    fresh_prompts.write_text(
        "\n".join(f"fresh prompt {index}" for index in range(128, 256)) + "\n",
        encoding="utf-8",
    )
    fresh_manifest = {
        "experiment": "v180_rccp_fresh128_inputs",
        "prompt_count": 128,
        "prompt_file": str(fresh_prompts),
        "prompt_file_sha256": module.sha256(fresh_prompts),
        "prompt_source_indices": list(range(128, 256)),
        "evaluation_source_index_range": [128, 255],
        "evaluation_prompts_used_for_membership": False,
        "exact_text_overlap_with_calibration": 0,
        "decoded_video_contract": frozen["decoded_video_contract"],
    }
    fresh_manifest_path = tmp_path / "fresh_manifest.json"
    write_json(fresh_manifest_path, fresh_manifest)
    return (
        decision_path,
        input_path,
        published_path,
        comparison_path,
        fresh_manifest_path,
    )


def test_v205_preparer_is_sha_gated_and_prompt_disjoint(tmp_path: Path) -> None:
    module = load_module(
        "v205_prepare", SCRIPTS / "prepare_v205_horizon_confirmation.py"
    )
    inputs = build_v201_fixture(tmp_path, module)
    output = tmp_path / "v205"
    payload = module.prepare(*inputs, output)
    assert module.verify(output / "manifest.json") == payload
    assert payload["method_order"] == ["sf_native", "retrieval_horizon_top10"]
    assert payload["prompt_source_indices"] == list(range(128, 256))
    assert payload["methods"]["retrieval_horizon_top10"][
        "v201_evidence_tier"
    ] == "interval_supported"

    decision = json.loads(inputs[0].read_text(encoding="utf-8"))
    decision["source"]["comparison_manifest_sha256"] = "0" * 64
    write_json(inputs[0], decision)
    with pytest.raises(ValueError, match="SHA-bound"):
        module.prepare(*inputs, tmp_path / "rejected")


def test_v205_analysis_confirms_significant_safe_gain() -> None:
    module = load_module(
        "v205_analysis", SCRIPTS / "analyze_v205_horizon_confirmation.py"
    )
    candidate = "retrieval_horizon_top10"
    methods = ("sf_native", candidate)
    manifest = {
        "selected_v201_candidates": [candidate],
        "v201_provenance": {
            "candidate_status": {
                candidate: {
                    "sf_efficacy": {
                        "positive_support": {
                            "directional_axes": [
                                {
                                    "window": "full",
                                    "metric": "quality_without_dynamic_degree",
                                    "mean_delta": 0.2,
                                }
                            ]
                        }
                    }
                }
            }
        },
        "prompt_items": [
            {"source_index": 128 + prompt, "text": f"prompt {prompt}"}
            for prompt in range(128)
        ],
        "methods": [
            {
                "key": "sf_native",
                "runtime": "sf_native",
                "video_dir": "/videos/sf_native",
            },
            {
                "key": candidate,
                "operator": "retrieval",
                "v201_evidence_tier": "interval_supported",
                "v201_mechanism_supported": True,
                "video_dir": f"/videos/{candidate}",
            },
        ],
        "claim_boundary": "unit-test boundary",
    }
    rows = {}
    for method in methods:
        for prompt in range(128):
            gain = 0.0 if method == "sf_native" else 1.0
            rows[(method, prompt)] = {
                "quality_without_dynamic_degree": 80.0 + gain,
                "official_quality_score": 80.0 + gain,
                "identity_background": 0.97 + gain * 0.002,
                "temporal_mechanics": 0.98 + gain * 0.002,
                "semantic_alignment": 0.55 + gain * 0.003,
                "visual_quality": 0.65 + gain * 0.003,
                "dynamic_degree": 1.0,
            }
    rows_by_window = {window: rows for window in module.WINDOWS}
    defaults = {feature: 0.0 for feature in module.v190.TEMPORAL_FEATURES}
    defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal = {
        (method, prompt): dict(defaults)
        for method in methods
        for prompt in range(128)
    }
    report = module.analyze_from_rows(manifest, rows_by_window, temporal)
    assert report["confirmed_candidates"] == [candidate]
    assert report["long_horizon_supported_candidates"] == [candidate]
    assert report["paper_efficacy_support"] is True
    assert report["manual_review_required_for_decision"] is False
    assert len(report["targeted_review_queue"]) <= 4


def test_v201_temporal_binding_order_and_decision_provenance_are_fixed() -> None:
    source = (SCRIPTS / "analyze_v201_head_phase_horizon.py").read_text(
        encoding="utf-8"
    )
    assert (
        "verify_temporal_contract(\n"
        "        args.temporal_contract,\n"
        "        manifest_path,\n"
        "        args.temporal_csv,\n"
        "    )"
    ) in source
    assert 'report["source"] = {' in source
    assert '"comparison_manifest_sha256": sha256(manifest_path)' in source


def test_v205_runners_are_dynamic_sf_only_confirmations() -> None:
    generation = (SCRIPTS / "run_v205_horizon_confirmation_32gpu.sh").read_text(
        encoding="utf-8"
    )
    evaluation = (SCRIPTS / "run_v205_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "selected_v201_candidates" in (
        SCRIPTS / "prepare_v205_horizon_confirmation.py"
    ).read_text(encoding="utf-8")
    assert "generate128 is frozen to 4 nodes x 8 GPUs" in generation
    assert "head_phase_horizon" in generation
    assert 'if [[ "$method" == "sf_native" ]]' in generation
    assert "verify_sampling_parity" in generation
    assert "shared_sampling=true" in generation
    assert "third_party/Pyramid-Forcing/pyramidkv" in generation
    assert "pf_native" not in generation
    assert "motion-compute" in evaluation and "motion-analyze" in evaluation
    assert "--expected-videos 128" in evaluation


def test_v205_motion_analysis_enumerates_every_frozen_candidate() -> None:
    module = load_module(
        "v205_motion_candidates", SCRIPTS / "analyze_v205_continuous_motion.py"
    )
    candidates = ("landmark_horizon_top10", "retrieval_horizon_top10")
    manifest = {
        "experiment": "v205_profile_disjoint_head_phase_horizon_vbench",
        "confirmatory": True,
        "primary_baseline": "sf_native",
        "selected_v201_candidates": list(candidates),
        "methods": [
            {"key": "sf_native", "runtime": "sf_native"},
            *(
                {
                    "key": candidate,
                    "runtime": "head_phase_horizon_cache_runtime",
                    "role": "frozen_v201_horizon_candidate",
                }
                for candidate in candidates
            ),
        ],
    }
    assert module.candidate_controls(manifest) == [
        (candidate, ("sf_native",)) for candidate in candidates
    ]
