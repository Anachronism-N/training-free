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


def build_v191_fixture(tmp_path: Path, module) -> tuple[Path, Path]:
    prompts = tmp_path / "v191_prompts.txt"
    prompts.write_text(
        "\n".join(f"unseen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    bank = tmp_path / "bank.csv"
    bank.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30)) + "\n",
        encoding="ascii",
    )
    recent_masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    joint_masks = json.loads(json.dumps(recent_masks))
    joint_masks[0][10][3] = True
    maps = {}
    for name, masks in (("all_recent", recent_masks), ("head_phase_joint", joint_masks)):
        path = tmp_path / f"{name}.json"
        write_json(
            path,
            {
                "version": 1,
                "map_id": f"{name}-map",
                "coverage_operator": "landmark",
                "call_count": 4,
                "layer_count": 30,
                "head_count": 12,
                "coverage_masks": masks,
                "coverage_count_by_call": [
                    sum(value for layer in call for value in layer) for call in masks
                ],
            },
        )
        maps[name] = path

    methods = {
        "sf_native": {
            "runtime": "self_forcing_native",
            "role": "native_self_forcing_baseline",
        }
    }
    for name, role in (
        ("all_recent", "equal_budget_local_control"),
        ("head_phase_joint", "frozen_joint_head_phase_candidate"),
    ):
        payload = json.loads(maps[name].read_text(encoding="utf-8"))
        methods[name] = {
            "runtime": "head_phase_cache_runtime",
            "role": role,
            "schedule": "head_phase",
            "operator": "landmark",
            "history_policy": "landmark",
            "head_phase_map": str(maps[name]),
            "head_phase_map_sha256": module.sha256(maps[name]),
            "phase_map_id": payload["map_id"],
            "coverage_count_by_call": payload["coverage_count_by_call"],
            "coverage_cell_count": sum(payload["coverage_count_by_call"]),
            "head_bank_map": str(bank),
            "head_bank_map_sha256": module.sha256(bank),
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }
    frozen = {
        "version": 1,
        "experiment": "v191_unseen128_head_phase_confirmation",
        "scope": "confirmatory_unseen128",
        "confirmatory": True,
        "prompt_count": 128,
        "prompt_file": str(prompts),
        "prompt_file_sha256": module.sha256(prompts),
        "prompt_items": [
            {"index": index, "source_index": 128 + index, "text": f"unseen prompt {index}"}
            for index in range(128)
        ],
        "seed": 10000,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "method_order": list(module.METHODS),
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + structured middle4 + recent4",
            "clean_read": "Recent",
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
    }
    input_path = tmp_path / "v191_input.json"
    write_json(input_path, frozen)

    published = tmp_path / "v191_published.json"
    contract = tmp_path / "v191_contract.json"
    write_json(published, {"ok": True})
    write_json(contract, {"ok": True})
    comparison = {
        "experiment": "v191_unseen128_head_phase_vbench",
        "confirmatory": True,
        "prompt_count": 128,
        "seed": 10000,
        "methods": [{"key": method} for method in module.METHODS],
        "source": {
            "input_manifest_sha256": module.sha256(input_path),
            "published_manifest": str(published),
            "published_manifest_sha256": module.sha256(published),
            "generation_contract": str(contract),
            "generation_contract_sha256": module.sha256(contract),
        },
    }
    comparison_path = tmp_path / "v191_comparison.json"
    write_json(comparison_path, comparison)
    summary = tmp_path / "v191_summary.json"
    temporal = tmp_path / "v191_temporal.csv"
    temporal_contract = tmp_path / "v191_temporal.contract.json"
    write_json(summary, {"complete": True})
    temporal.write_text("method,prompt_index\n", encoding="utf-8")
    write_json(temporal_contract, {"bound": True})
    comparisons = [
        {
            "candidate": "head_phase_joint",
            "control": "all_recent",
            "metric": metric,
            "mean_delta": value,
            "bootstrap_ci95": [value / 2.0, value * 1.5],
            "per_prompt_delta": [value for _ in range(128)],
        }
        for metric, value in (
            ("official_quality_score", 0.4),
            ("identity_background", 0.001),
            ("dynamic_degree", 0.0),
            ("temporal_mechanics", 0.001),
        )
    ]
    decision = {
        "version": 1,
        "experiment": "v191_unseen128_head_phase_vbench",
        "confirmatory": True,
        "head_phase_effect_confirmed": True,
        "recommendation": "freeze_head_phase_method_for_seed_length_and_cross_model_replication",
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "confirmation_gates": {
            "equal_budget_noninferiority": True,
            "equal_budget_positive_effect": True,
            "native_noninferiority": True,
            "candidate_on_primary_pareto_front": True,
            "temporal_safety_vs_equal_budget": True,
            "temporal_safety_vs_native": True,
        },
        "positive_effect_vs_equal_budget": {
            "pass": True,
            "ci_lower_gt_zero": {
                "official_quality_score": True,
                "identity_background": False,
                "temporal_mechanics": False,
                "dynamic_degree": False,
            },
        },
        "candidate_delta_vs_all_recent": {
            "official_quality_score": 0.4,
            "identity_background": 0.001,
            "dynamic_degree": 0.0,
            "temporal_mechanics": 0.001,
        },
        "motion_improvement_claim_supported": False,
        "comparisons": comparisons,
        "source": {
            "comparison_manifest": str(comparison_path),
            "comparison_manifest_sha256": module.sha256(comparison_path),
            "vbench_summary": str(summary),
            "vbench_summary_sha256": module.sha256(summary),
            "temporal_diagnostics": str(temporal),
            "temporal_diagnostics_sha256": module.sha256(temporal),
            "temporal_contract": str(temporal_contract),
            "temporal_contract_sha256": module.sha256(temporal_contract),
        },
    }
    decision_path = tmp_path / "v191_decision.json"
    write_json(decision_path, decision)
    return decision_path, input_path


def test_v192_preparer_freezes_new_seed_and_systematic_long_subset(tmp_path: Path) -> None:
    module = load_module(
        "v192_prepare", SCRIPTS / "prepare_v192_head_phase_robustness.py"
    )
    decision, v191_input = build_v191_fixture(tmp_path, module)
    output = tmp_path / "v192"
    payload = module.prepare(decision, v191_input, output)
    assert module.verify(output / "manifest.json") == payload
    assert tuple(row["key"] for row in payload["scopes"]) == module.SCOPE_KEYS
    seed_scope = module.scope_config(payload, "seed2026_30s_128")
    long_scope = module.scope_config(payload, "long60_seed10000_32")
    assert seed_scope["seed"] == 2026
    assert seed_scope["prompt_count"] == 128
    assert long_scope["num_output_frames"] == 240
    assert long_scope["prompt_positions_in_v191"] == list(range(0, 128, 4))
    assert long_scope["prompt_source_indices"] == list(range(128, 256, 4))
    assert payload["v191_positive_metrics_to_replicate"] == [
        "official_quality_score"
    ]


def test_v192_preparer_rejects_any_failed_v191_gate(tmp_path: Path) -> None:
    module = load_module(
        "v192_prepare_failed", SCRIPTS / "prepare_v192_head_phase_robustness.py"
    )
    decision_path, v191_input = build_v191_fixture(tmp_path, module)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["confirmation_gates"]["native_noninferiority"] = False
    write_json(decision_path, decision)
    with pytest.raises(ValueError, match="every frozen v191 confirmation gate"):
        module.prepare(decision_path, v191_input, tmp_path / "rejected")


def test_persistence_gate_detects_late_effect_collapse() -> None:
    module = load_module(
        "v192_analysis_persistence",
        SCRIPTS / "analyze_v192_head_phase_robustness.py",
    )

    def rows(candidate_quality: float) -> dict:
        output = {}
        for method in module.METHODS:
            for prompt in range(32):
                quality = candidate_quality if method == module.CANDIDATE else 0.0
                output[(method, prompt)] = {
                    "official_quality_score": quality,
                    "identity_background": quality / 100.0,
                    "dynamic_degree": quality / 10.0,
                    "temporal_mechanics": quality / 100.0,
                }
        return output

    stable = module.persistence_gate(
        {"early_half": rows(0.2), "late_half": rows(0.2)}, prompt_count=32
    )
    collapsed = module.persistence_gate(
        {"early_half": rows(0.2), "late_half": rows(-0.2)}, prompt_count=32
    )
    assert stable["pass"] is True
    assert collapsed["pass"] is False


def test_v192_seed_scope_requires_and_accepts_replicated_positive_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v192_analysis_seed_scope",
        SCRIPTS / "analyze_v192_head_phase_robustness.py",
    )
    values = {
        "sf_native": (79.9, 0.9698, 0.9798, 1.0),
        "all_recent": (80.0, 0.9700, 0.9800, 1.0),
        "head_phase_joint": (80.4, 0.9710, 0.9810, 1.0),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(128):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.55,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module, "_load_scope_rows", lambda *args, **kwargs: {"full": rows})
    manifest = {
        "experiment": module.EXPERIMENT,
        "scope": "seed2026_30s_128",
        "scope_role": "same_prompt_new_seed_replication",
        "confirmatory": True,
        "prompt_count": 128,
        "num_output_frames": 120,
        "seed": 2026,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "v191_positive_metrics_to_replicate": ["official_quality_score"],
        "prompt_items": [
            {
                "index": index,
                "v191_prompt_index": index,
                "source_index": 128 + index,
                "text": f"prompt {index}",
            }
            for index in range(128)
        ],
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in module.METHODS
        ],
        "vbench_long_dimensions": list(module.DIMENSIONS),
        "claim_boundary": "unit-test boundary",
    }
    summary = {
        "methods": {method: {} for method in module.METHODS},
        "dimensions": list(module.DIMENSIONS),
        "missing": [],
    }
    temporal_features = (
        "flow_speed_median",
        "motion_coverage_fraction",
        "late_motion_ratio",
        "longest_low_motion_run_fraction",
        "temporal_jump",
        "appearance_outlier_fraction",
        "flow_accel_outlier_fraction",
        "dark_frame_fraction",
        "bright_frame_fraction",
        "low_contrast_frame_fraction",
        "edge_density_outlier_fraction",
    )
    defaults = {feature: 0.0 for feature in temporal_features}
    defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal_rows = {
        (method, prompt): dict(defaults)
        for method in module.METHODS
        for prompt in range(128)
    }
    report = module.analyze_scope(
        manifest,
        summary,
        Path("unused"),
        temporal_rows=temporal_rows,
    )
    assert report["scope_pass"] is True
    assert report["positive_effect"]["targets"]["official_quality_score"][
        "replicated"
    ] is True
    assert len(report["targeted_review_queue"]) == 4


def test_v192_vbench_preparer_accepts_only_complete_audited_scope(tmp_path: Path) -> None:
    prepare_module = load_module(
        "v192_prepare_for_vbench",
        SCRIPTS / "prepare_v192_head_phase_robustness.py",
    )
    module = load_module(
        "v192_vbench_prepare", SCRIPTS / "prepare_v192_vbench_comparison.py"
    )
    decision, v191_input = build_v191_fixture(tmp_path, prepare_module)
    input_root = tmp_path / "v192_inputs"
    frozen = prepare_module.prepare(decision, v191_input, input_root)
    input_manifest = input_root / "manifest.json"
    scope = "long60_seed10000_32"
    scope_row = prepare_module.scope_config(frozen, scope)
    run_root = tmp_path / "runs" / scope
    contract = {
        "experiment": "v192_head_phase_robustness_generation",
        "scope": scope,
        "run_kind": "full",
        "prompt_indices": list(range(32)),
        "prompt_file_sha256": scope_row["prompt_file_sha256"],
        "prompt_items": scope_row["prompt_items"],
        "num_output_frames": 240,
        "decoded_video_contract": scope_row["decoded_video_contract"],
        "seed": 10000,
        "selected_v190_method": frozen["selected_v190_method"],
        "selected_operator": frozen["selected_operator"],
        "input_manifest_sha256": module.sha256(input_manifest),
        "methods": list(module.METHODS),
    }
    contract_path = run_root / "contracts" / "experiment.json"
    write_json(contract_path, contract)
    published_methods = []
    for method in module.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(32):
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
    write_json(
        run_root / "published_manifest.json",
        {
            "ok": True,
            "complete": True,
            "experiment": "v192_head_phase_robustness_generation",
            "scope": scope,
            "run_kind": "full",
            "confirmatory": True,
            "prompt_count": 32,
            "experiment_contract_sha256": module.sha256(contract_path),
            "methods": published_methods,
        },
    )
    comparison_root = run_root / "vbench_comparison"
    report = module.prepare(run_root, comparison_root, input_manifest, scope)
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert report["videos"] == 96
    assert manifest["prompt_positions_in_v191"] == list(range(0, 128, 4))
    assert manifest["num_output_frames"] == 240
    assert (comparison_root / "published" / "head_phase_joint" / "000031-0.mp4").is_file()


def test_v192_combined_decision_uses_prompt_paired_two_seed_effect(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v192_analysis_combine",
        SCRIPTS / "analyze_v192_head_phase_robustness.py",
    )

    def evidence(prefix: str) -> dict:
        row = {}
        for key in (
            "comparison_manifest",
            "vbench_summary",
            "temporal_diagnostics",
            "temporal_contract",
        ):
            path = tmp_path / f"{prefix}_{key}.txt"
            path.write_text(f"{prefix}:{key}\n", encoding="utf-8")
            row[key] = str(path)
            row[f"{key}_sha256"] = module.sha256(path)
        return row

    decision = {
        "comparisons": [
            {
                "candidate": module.CANDIDATE,
                "control": module.LOCAL_CONTROL,
                "metric": "official_quality_score",
                "per_prompt_delta": [0.4 for _ in range(128)],
            }
        ]
    }
    decision_path = tmp_path / "v191_decision.json"
    write_json(decision_path, decision)
    input_manifest = {
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "v191_positive_metrics_to_replicate": ["official_quality_score"],
        "v191_motion_improvement_claim_supported": False,
        "v191_provenance": {
            "decision": str(decision_path),
            "decision_sha256": module.sha256(decision_path),
        },
        "claim_boundary": "unit-test boundary",
    }
    seed_report = {
        "scope": "seed2026_30s_128",
        "scope_pass": True,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "source": evidence("seed"),
        "metric_validity": {"dynamic_degree": {"informative": False}},
        "targeted_review_queue": [{"scope": "seed", "prompt_index": 0}],
        "comparisons": [
            {
                "candidate": module.CANDIDATE,
                "control": module.LOCAL_CONTROL,
                "metric": "official_quality_score",
                "window": "full",
                "mean_delta": 0.2,
                "bootstrap_ci95": [0.1, 0.3],
                "per_prompt_delta": [0.2 for _ in range(128)],
            },
            {
                "candidate": module.CANDIDATE,
                "control": module.LOCAL_CONTROL,
                "metric": "dynamic_degree",
                "window": "full",
                "mean_delta": 0.0,
                "bootstrap_ci95": [0.0, 0.0],
                "per_prompt_delta": [0.0 for _ in range(128)],
            },
        ],
    }
    long_report = {
        "scope": "long60_seed10000_32",
        "scope_pass": True,
        "selected_v190_method": "landmark_compatible",
        "selected_operator": "landmark",
        "source": evidence("long"),
        "metric_validity": {"dynamic_degree": {"informative": False}},
        "targeted_review_queue": [{"scope": "long", "prompt_index": 0}],
        "comparisons": [
            {
                "candidate": module.CANDIDATE,
                "control": module.LOCAL_CONTROL,
                "metric": "dynamic_degree",
                "window": "late_half",
                "mean_delta": 0.0,
                "bootstrap_ci95": [0.0, 0.0],
                "per_prompt_delta": [0.0 for _ in range(32)],
            }
        ],
    }
    report = module.combine_reports(input_manifest, seed_report, long_report)
    assert report["within_model_seed_length_robustness_confirmed"] is True
    assert report["recommendation"] == (
        "freeze_within_model_head_phase_method_for_cross_model_transfer"
    )
    pooled = report["two_seed_pooled_effect"]["official_quality_score"]
    assert pooled["both_seed_means_positive"] is True
    assert pooled["pooled_ci_lower_gt_zero"] is True
    assert len(report["targeted_review_queue"]) == 2


def test_v192_vbench_runner_configures_thirty_clips_for_long_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v192_vbench_runner", SCRIPTS / "run_v192_vbench_long.py"
    )
    comparison_root = tmp_path / "comparison"
    write_json(
        comparison_root / "comparison_manifest.json",
        {
            "experiment": "v192_head_phase_robustness_vbench",
            "scope": "long60_seed10000_32",
            "confirmatory": True,
            "prompt_count": 32,
            "num_output_frames": 240,
            "seed": 10000,
            "methods": [{"key": method} for method in module.METHODS],
            "vbench_long_dimensions": list(module.DIMENSIONS),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v192_vbench_long.py", "preflight", "--comparison-root", str(comparison_root)],
    )
    module.configure()
    assert module.base.NUM_OUTPUT_FRAMES == 240
    assert module.base.CLIPS_PER_VIDEO == 30
    assert module.base.PROMPT_COUNT == 32


def test_v192_runners_freeze_three_methods_two_scopes_and_no_pf_baseline() -> None:
    generation = (SCRIPTS / "run_v192_head_phase_robustness_32gpu.sh").read_text(
        encoding="utf-8"
    )
    vbench = (SCRIPTS / "run_v192_vbench_long.sh").read_text(encoding="utf-8")
    assert 'ALL_METHODS="sf_native,all_recent,head_phase_joint"' in generation
    assert 'ALL_SCOPES="seed2026_30s_128,long60_seed10000_32"' in generation
    assert "4 nodes x 8 GPUs" in generation
    assert "PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH" in generation
    assert "pf_native" not in generation
    assert "compute_temporal_jump_diagnostic.py" in vbench
    assert "analyze_v192_head_phase_robustness.py" in vbench
    assert "--long-report" in vbench
