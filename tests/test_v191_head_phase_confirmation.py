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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v191_preparer_requires_passing_sha_bound_v190_and_unseen_prompts(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v191_prepare", SCRIPTS / "prepare_v191_head_phase_confirmation.py"
    )
    development_prompts = tmp_path / "development.txt"
    development_prompts.write_text(
        "\n".join(f"development prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    fresh_prompts = tmp_path / "moviegen_fresh_0128_0255.txt"
    fresh_prompts.write_text(
        "\n".join(f"unseen evaluation prompt {index}" for index in range(128, 256))
        + "\n",
        encoding="utf-8",
    )
    bank = tmp_path / "all_heads.csv"
    bank.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )
    masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    masks[0][10][3] = True
    selected_map = {
        "version": 1,
        "map_id": "v190-selected-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
        "coverage_count_by_call": [1, 0, 0, 0],
    }
    selected_map_path = tmp_path / "selected.json"
    write_json(selected_map_path, selected_map)

    selected = "landmark_compatible"
    v190_methods = ("all_recent", selected)
    v190_input = {
        "experiment": "v190_head_phase_causal_screen",
        "scope": "classifier_holdout32",
        "prompt_count": 32,
        "source_prompt_file": str(development_prompts),
        "method_order": list(v190_methods),
        "methods": {
            "all_recent": {"role": "local_control"},
            selected: {
                "role": "primary_head_phase",
                "schedule": "head_phase",
                "operator": "landmark",
                "clean_policy": "recent",
                "read_frame_equivalents": 9,
                "head_phase_map": str(selected_map_path),
                "head_phase_map_sha256": module.sha256(selected_map_path),
                "head_bank_map": str(bank),
                "head_bank_map_sha256": module.sha256(bank),
            },
        },
    }
    input_path = tmp_path / "v190_inputs.json"
    write_json(input_path, v190_input)

    contract_path = tmp_path / "v190_contract.json"
    write_json(contract_path, {"scope": "screen32"})
    published_methods = []
    for method in v190_methods:
        audit = tmp_path / f"{method}.audit.json"
        write_json(audit, {"ok": True})
        published_methods.append(
            {
                "key": method,
                "ok": True,
                "audit": str(audit),
                "audit_sha256": module.sha256(audit),
            }
        )
    published = {
        "ok": True,
        "experiment": "v190_head_phase_causal_generation",
        "scope": "screen32",
        "experiment_contract": str(contract_path),
        "experiment_contract_sha256": module.sha256(contract_path),
        "methods": published_methods,
    }
    published_path = tmp_path / "v190_published.json"
    write_json(published_path, published)

    comparison = {
        "experiment": "v190_head_phase_causal_vbench_screen32",
        "prompt_count": 32,
        "methods": [{"key": method} for method in v190_methods],
    }
    comparison_path = tmp_path / "v190_comparison.json"
    write_json(comparison_path, comparison)
    vbench_summary = tmp_path / "v190_summary.json"
    write_json(vbench_summary, {"complete": True})
    temporal_csv = tmp_path / "v190_temporal.csv"
    temporal_csv.write_text("method,prompt_index\n", encoding="utf-8")
    temporal_contract = tmp_path / "v190_temporal.contract.json"
    write_json(temporal_contract, {"bound": True})
    decision = {
        "version": 6,
        "experiment": "v190_head_phase_causal_vbench_screen32",
        "development_only": True,
        "recommendation": "advance_head_phase_method_to_fresh128",
        "selected_for_fresh128": selected,
        "passing_methods": [selected],
        "methods": list(v190_methods),
        "temporal_diagnostics_available": True,
        "statuses": {
            selected: {
                "full_screen_pass": True,
                "joint_factorization_pass": True,
                "head_phase_attribution_pass": True,
                "selective_exposure_pass": True,
                "baseline_deltas": {
                    "official_quality_score": 0.2,
                    "identity_background": 0.001,
                    "dynamic_degree": 0.0,
                    "temporal_mechanics": 0.001,
                },
            }
        },
        "metric_validity": {
            "dynamic_degree": {"informative": False, "ceiling_nonregression_only": True}
        },
        "source": {
            "comparison_manifest_sha256": module.sha256(comparison_path),
            "vbench_summary": str(vbench_summary),
            "vbench_summary_sha256": module.sha256(vbench_summary),
            "temporal_diagnostics": str(temporal_csv),
            "temporal_diagnostics_sha256": module.sha256(temporal_csv),
            "temporal_contract": str(temporal_contract),
            "temporal_contract_sha256": module.sha256(temporal_contract),
        },
    }
    decision_path = tmp_path / "v190_decision.json"
    write_json(decision_path, decision)

    fresh_manifest = {
        "experiment": "v180_rccp_fresh128_inputs",
        "prompt_count": 128,
        "prompt_file": str(fresh_prompts),
        "prompt_file_sha256": module.sha256(fresh_prompts),
        "prompt_source_indices": list(range(128, 256)),
        "evaluation_source_index_range": [128, 255],
        "evaluation_prompts_used_for_membership": False,
        "exact_text_overlap_with_calibration": 0,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
    }
    fresh_manifest_path = tmp_path / "fresh_manifest.json"
    write_json(fresh_manifest_path, fresh_manifest)

    output = tmp_path / "v191"
    payload = module.prepare(
        decision_path,
        input_path,
        published_path,
        comparison_path,
        fresh_manifest_path,
        output,
    )
    assert module.verify(output / "manifest.json") == payload
    assert payload["prompt_source_indices"] == list(range(128, 256))
    assert payload["method_order"] == list(module.METHODS)
    assert payload["methods"]["head_phase_joint"]["phase_map_id"] == "v190-selected-map"
    assert payload["methods"]["all_recent"]["coverage_cell_count"] == 0
    assert payload["selected_operator"] == "landmark"

    decision["source"]["comparison_manifest_sha256"] = "0" * 64
    write_json(decision_path, decision)
    with pytest.raises(ValueError, match="SHA-bound"):
        module.prepare(
            decision_path,
            input_path,
            published_path,
            comparison_path,
            fresh_manifest_path,
            tmp_path / "rejected",
        )


def test_v191_analysis_accepts_gain_but_never_claims_ceiling_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v191_analysis", SCRIPTS / "analyze_v191_head_phase_confirmation.py"
    )
    values = {
        "sf_native": (79.9, 0.9698, 0.9798, 1.0),
        "all_recent": (80.0, 0.9700, 0.9800, 1.0),
        "head_phase_joint": (80.4, 0.9710, 0.9810, 1.0),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(module.PROMPT_COUNT):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.55,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda *args, **kwargs: rows)
    manifest = {
        "experiment": "v191_unseen128_head_phase_vbench",
        "confirmatory": True,
        "prompt_count": 128,
        "seed": 10000,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "claim_boundary": "unit-test boundary",
        "v190_provenance": {
            "selected_status": {
                "baseline_deltas": {
                    "official_quality_score": 0.2,
                    "identity_background": 0.0005,
                    "dynamic_degree": 0.0,
                    "temporal_mechanics": 0.0005,
                }
            }
        },
        "prompt_items": [
            {"source_index": index + 128, "text": f"prompt {index}"}
            for index in range(128)
        ],
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in module.METHODS
        ],
    }
    summary = {"methods": {method: {} for method in module.METHODS}, "missing": []}
    temporal_defaults = {feature: 0.0 for feature in module.TEMPORAL_FEATURES}
    temporal_defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal_rows = {
        (method, prompt): dict(temporal_defaults)
        for method in module.METHODS
        for prompt in range(128)
    }
    report = module.analyze(
        manifest,
        summary,
        Path("unused"),
        temporal_rows=temporal_rows,
    )
    assert report["head_phase_effect_confirmed"] is True
    assert report["motion_improvement_claim_supported"] is False
    assert report["metric_validity"]["dynamic_degree"][
        "ceiling_nonregression_only"
    ] is True
    assert len(report["targeted_review_queue"]) == 4


def test_temporal_diagnostic_accepts_audited_vbench_names(tmp_path: Path) -> None:
    module = load_module(
        "temporal_diagnostic", SCRIPTS / "compute_temporal_jump_diagnostic.py"
    )
    first = tmp_path / "all_recent"
    second = tmp_path / "head_phase_joint"
    first.mkdir()
    second.mkdir()
    (first / "000000-0.mp4").write_bytes(b"placeholder")
    (second / "000000-0_ema.mp4").write_bytes(b"placeholder")
    rows = module._indexed_video_paths([first, second], expected_videos=1)
    assert [(method, prompt) for method, prompt, _ in rows] == [
        ("all_recent", 0),
        ("head_phase_joint", 0),
    ]


def test_temporal_contract_binds_csv_to_exact_comparison_videos(
    tmp_path: Path,
) -> None:
    module = load_module(
        "temporal_contract", SCRIPTS / "bind_temporal_diagnostics.py"
    )
    methods = ("all_recent", "head_phase_joint")
    method_rows = []
    for method in methods:
        video_dir = tmp_path / method
        video_dir.mkdir()
        (video_dir / "000000-0.mp4").write_bytes(method.encode("ascii"))
        method_rows.append({"key": method, "video_dir": str(video_dir)})
    comparison = tmp_path / "comparison.json"
    write_json(
        comparison,
        {"experiment": "unit", "prompt_count": 1, "methods": method_rows},
    )
    temporal = tmp_path / "temporal.csv"
    temporal.write_text(
        "method,prompt_index,sample_index,video\n"
        + "\n".join(
            f"{method},0,0,{tmp_path / method / '000000-0.mp4'}"
            for method in methods
        )
        + "\n",
        encoding="utf-8",
    )
    contract = tmp_path / "temporal.contract.json"
    payload = module.build_contract(comparison, temporal)
    module.write_contract(contract, payload)
    assert module.verify_contract(contract, comparison, temporal) == payload
    temporal.write_text(
        temporal.read_text(encoding="utf-8").replace(
            str(tmp_path / "head_phase_joint" / "000000-0.mp4"),
            str(tmp_path / "all_recent" / "000000-0.mp4"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not the comparison video"):
        module.verify_contract(contract, comparison, temporal)


def test_v191_vbench_preparer_materializes_complete_audited_grid(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v191_vbench_prepare", SCRIPTS / "prepare_v191_vbench_comparison.py"
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(f"unseen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    method_configs = {
        "sf_native": {
            "role": "native_self_forcing_baseline",
            "runtime": "self_forcing_native",
        },
        "all_recent": {
            "role": "equal_budget_local_control",
            "runtime": "head_phase_cache_runtime",
            "operator": "landmark",
            "phase_map_id": "recent-map",
        },
        "head_phase_joint": {
            "role": "frozen_joint_head_phase_candidate",
            "runtime": "head_phase_cache_runtime",
            "operator": "landmark",
            "phase_map_id": "joint-map",
        },
    }
    frozen = {
        "experiment": "v191_unseen128_head_phase_confirmation",
        "scope": "confirmatory_unseen128",
        "confirmatory": True,
        "method_order": list(module.METHODS),
        "prompt_count": 128,
        "seed": 10000,
        "prompt_file": str(prompts),
        "prompt_file_sha256": module.sha256(prompts),
        "prompt_items": [
            {"index": index, "source_index": index + 128, "text": f"unseen prompt {index}"}
            for index in range(128)
        ],
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "methods": method_configs,
        "claim_boundary": "unit-test boundary",
    }
    input_manifest = tmp_path / "inputs" / "manifest.json"
    write_json(input_manifest, frozen)
    run_root = tmp_path / "confirm128"
    contract = {
        "scope": "confirm128",
        "confirmatory": True,
        "prompt_count": 128,
        "prompt_indices": list(range(128)),
        "seed": 10000,
        "input_manifest_sha256": module.sha256(input_manifest),
        "methods": list(module.METHODS),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    write_json(contract_path, contract)
    published_methods = []
    for method in module.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{method}:{index}".encode("ascii")
            )
        audit = run_root / "audits" / f"{method}.json"
        write_json(audit, {"ok": True, "method": method})
        published_methods.append(
            {
                "key": method,
                "ok": True,
                "video_dir": str(video_dir),
                "audit": str(audit),
                "audit_sha256": module.sha256(audit),
            }
        )
    published = {
        "ok": True,
        "experiment": "v191_unseen128_head_phase_generation",
        "scope": "confirm128",
        "confirmatory": True,
        "experiment_contract_sha256": module.sha256(contract_path),
        "methods": published_methods,
    }
    write_json(run_root / "published_manifest.json", published)
    comparison_root = run_root / "vbench_comparison"
    report = module.prepare(run_root, comparison_root, input_manifest)
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert report["videos"] == 384
    assert sum(report["link_counts"].values()) == 384
    assert tuple(row["key"] for row in manifest["methods"]) == module.METHODS
    assert len(manifest["input_video_sha256"]["head_phase_joint"]) == 128
    assert (comparison_root / "published" / "sf_native" / "000127-0.mp4").is_file()


def test_temporal_guard_scales_warning_allowance_with_prompt_count() -> None:
    module = load_module(
        "v191_temporal_guard", SCRIPTS / "analyze_v190_head_phase_causal_screen.py"
    )
    defaults = {feature: 0.0 for feature in module.TEMPORAL_FEATURES}
    defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    rows = {
        (method, prompt): dict(defaults)
        for method in ("candidate", "all_recent")
        for prompt in range(128)
    }
    for prompt in range(4):
        rows[("candidate", prompt)]["late_motion_ratio"] = 0.2
    report = module.temporal_guard(
        rows,
        candidate="candidate",
        control="all_recent",
        prompt_count=128,
    )
    assert report["allowed_flagged_prompt_count"] == 4
    assert report["flagged_prompt_count"] == 4
    assert report["automatic_safety_pass"] is True
    rows[("candidate", 4)]["late_motion_ratio"] = 0.2
    assert module.temporal_guard(
        rows,
        candidate="candidate",
        control="all_recent",
        prompt_count=128,
    )["automatic_safety_pass"] is False


def test_v191_runners_freeze_three_methods_and_trace_cache_routes() -> None:
    generation = (SCRIPTS / "run_v191_head_phase_confirmation_32gpu.sh").read_text(
        encoding="utf-8"
    )
    vbench = (SCRIPTS / "run_v191_vbench_long.sh").read_text(encoding="utf-8")
    assert 'ALL_METHODS="sf_native,all_recent,head_phase_joint"' in generation
    assert "NUM_NODES * GPUS_PER_NODE" in generation
    assert "4 nodes x 8 GPUs" in generation
    assert "PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH" in generation
    assert 'if [[ "$method" == "sf_native" ]]' in generation
    assert "--temporal-csv" in vbench
    assert "--temporal-contract" in vbench
    assert "bind_temporal_diagnostics.py" in vbench
    assert "--expected-videos 128" in vbench
    assert "compute_temporal_jump_diagnostic.py" in vbench
