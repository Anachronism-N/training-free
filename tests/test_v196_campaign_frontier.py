from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def source(module, root: Path, prefix: str, keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        path = write_json(root / "evidence" / f"{prefix}_{key}.json", {"key": key})
        result[key] = str(path)
        result[f"{key}_sha256"] = module.sha256(path)
    return result


def artifact(module, repo: Path, key: str) -> Path:
    spec = next(row for row in module.STAGES if row.key == key)
    return repo / spec.artifact


def write_v189(module, repo: Path, *, passed: bool = True) -> None:
    manifest = write_json(repo / "runs/v189/manifest.json", {"frozen": True})
    write_json(
        artifact(module, repo, "v189"),
        {
            "experiment": "v189_structured_head_phase_profile",
            "recommendation": (
                "advance_head_phase_maps_to_causal_screen"
                if passed
                else "do_not_generate_from_v189_classifier"
            ),
            "generation_candidates": ["landmark"] if passed else [],
            "operators": {"landmark": {}},
            "profile_audits": {"landmark": {}},
            "input_manifest": str(manifest),
            "input_manifest_sha256": module.sha256(manifest),
        },
    )


def write_v190(module, repo: Path, *, passed: bool = True) -> None:
    selected = "landmark_compatible" if passed else None
    status = {
        "full_screen_pass": True,
        "joint_factorization_pass": True,
        "head_phase_attribution_pass": True,
        "selective_exposure_pass": True,
    }
    write_json(
        artifact(module, repo, "v190"),
        {
            "version": 6,
            "experiment": "v190_head_phase_causal_vbench_screen32",
            "development_only": True,
            "recommendation": (
                "advance_head_phase_method_to_fresh128"
                if passed
                else "do_not_advance_v190"
            ),
            "selected_for_fresh128": selected,
            "statuses": {selected: status} if selected else {},
            "passing_methods": [selected] if selected else [],
            "temporal_diagnostics_available": True,
            "source": source(
                module,
                repo,
                "v190",
                (
                    "comparison_manifest",
                    "vbench_summary",
                    "temporal_diagnostics",
                    "temporal_contract",
                ),
            ),
        },
    )


def write_confirmatory(
    module,
    repo: Path,
    key: str,
    *,
    passed: bool,
) -> None:
    contracts = {
        "v191": (
            "v191_unseen128_head_phase_vbench",
            "confirmation_gates",
            "head_phase_effect_confirmed",
            "freeze_head_phase_method_for_seed_length_and_cross_model_replication",
            "do_not_freeze_v191_head_phase_method",
            module.V191_GATES,
        ),
        "v192": (
            "v192_head_phase_seed_length_robustness",
            "combined_gates",
            "within_model_seed_length_robustness_confirmed",
            "freeze_within_model_head_phase_method_for_cross_model_transfer",
            "do_not_advance_v192_head_phase_robustness",
            module.V192_GATES,
        ),
        "v194": (
            "v194_causal_checkpoint_transfer_vbench",
            "confirmation_gates",
            "cross_checkpoint_transfer_confirmed",
            "freeze_head_phase_route_as_cross_checkpoint_supported",
            "do_not_claim_v194_checkpoint_transfer",
            module.V194_GATES,
        ),
    }
    experiment, gate_key, confirmed_key, pass_rec, fail_rec, gate_keys = contracts[key]
    if key == "v192":
        input_manifest = write_json(
            repo / "evidence/v192_input_manifest.json", {"frozen": True}
        )
        scope_paths = {}
        for report_key, scope_name in module.V192_SCOPE_REPORTS.items():
            scope_paths[report_key] = write_json(
                repo / "evidence" / f"v192_{report_key}.json",
                {
                    "experiment": "v192_head_phase_robustness_vbench",
                    "scope": scope_name,
                    "confirmatory": True,
                    "source": source(
                        module,
                        repo,
                        f"v192_{report_key}",
                        tuple(sorted(module.STANDARD_GENERATION_SOURCES)),
                    ),
                },
            )
        stage_source = {
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": module.sha256(input_manifest),
        }
        for report_key, report_path in scope_paths.items():
            stage_source[report_key] = str(report_path)
            stage_source[f"{report_key}_sha256"] = module.sha256(report_path)
    else:
        stage_source = source(
            module,
            repo,
            key,
            tuple(sorted(module.STANDARD_GENERATION_SOURCES)),
        )
    camera_context = None
    if key == "v194":
        motion_sources = source(
            module,
            repo,
            "v194_camera",
            ("motion_csv", "motion_contract"),
        )
        motion_sources.update(
            {
                "comparison_manifest": stage_source["comparison_manifest"],
                "comparison_manifest_sha256": stage_source[
                    "comparison_manifest_sha256"
                ],
            }
        )
        camera_report = write_json(
            repo / "evidence/v194_camera_report.json",
            {
                "experiment": "v193_camera_compensated_motion_calibration",
                "source": motion_sources,
            },
        )
        camera_context = {
            "available": True,
            "report": str(camera_report),
            "report_sha256": module.sha256(camera_report),
            "measurement_calibration_pass": True,
            "directional_against_all_controls": True,
            "strong_against_all_controls": True,
            "motion_improvement_claim_supported": True,
        }
    payload = {
        "version": 1,
        "experiment": experiment,
        "confirmatory": True,
        gate_key: {gate: passed for gate in gate_keys},
        confirmed_key: passed,
        "recommendation": pass_rec if passed else fail_rec,
        "selected_operator": "landmark",
        "source": stage_source,
    }
    if camera_context is not None:
        payload["camera_compensated_motion"] = camera_context
    write_json(
        artifact(module, repo, key),
        payload,
    )


def write_v195(
    module,
    repo: Path,
    *,
    v194_passed: bool,
    level: str,
) -> None:
    write_json(
        artifact(module, repo, "v195"),
        {
            "version": 1,
            "experiment": "v195_cross_checkpoint_head_phase_profile",
            "diagnostic": True,
            "mechanism_support_level": level,
            "v194_generation_transfer_confirmed": v194_passed,
            "recommendation": module._expected_v195_recommendation(v194_passed, level),
            "manual_review_required": False,
            "mechanism_gates": {
                gate: level != "unsupported" for gate in module.V195_GATES
            },
            "exact_head_identity_transfer_supported": level == "exact_head_identity",
            "phase_layer_structure_transfer_supported": level
            in {"exact_head_identity", "phase_layer_structure"},
            "operator_compatibility_transfer_supported": level != "unsupported",
            "source": source(
                module,
                repo,
                "v195",
                ("input_manifest", "profile_audit", "sf_cell_scores"),
            ),
        },
    )


def pass_through_v192(module, repo: Path) -> None:
    write_v189(module, repo)
    write_v190(module, repo)
    write_confirmatory(module, repo, "v191", passed=True)
    write_confirmatory(module, repo, "v192", passed=True)


def test_empty_campaign_points_to_v189(tmp_path: Path) -> None:
    module = load_module("v196_empty", SCRIPTS / "inspect_v196_campaign_frontier.py")
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "run_stage"
    assert report["frontier"]["key"] == "v189"
    assert report["stages"][0]["state"] == "missing"
    assert all(row["state"] == "blocked" for row in report["stages"][1:])


def test_failed_confirmatory_stage_stops_ladder(tmp_path: Path) -> None:
    module = load_module("v196_stop", SCRIPTS / "inspect_v196_campaign_frontier.py")
    write_v189(module, tmp_path)
    write_v190(module, tmp_path)
    write_confirmatory(module, tmp_path, "v191", passed=False)
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "stop"
    assert report["frontier"]["key"] == "stop_after_v191"
    assert report["terminal"] is True
    assert report["stages"][3]["state"] == "blocked"


def test_failed_v194_still_requires_v195_diagnosis(tmp_path: Path) -> None:
    module = load_module(
        "v196_v194_fail", SCRIPTS / "inspect_v196_campaign_frontier.py"
    )
    pass_through_v192(module, tmp_path)
    write_confirmatory(module, tmp_path, "v194", passed=False)
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "run_stage"
    assert report["frontier"]["key"] == "v195"
    assert report["stages"][4]["state"] == "failed"
    assert report["terminal"] is False


def test_v195_exact_transfer_routes_to_prompt_switch_design(tmp_path: Path) -> None:
    module = load_module(
        "v196_prompt_switch", SCRIPTS / "inspect_v196_campaign_frontier.py"
    )
    pass_through_v192(module, tmp_path)
    write_confirmatory(module, tmp_path, "v194", passed=True)
    write_v195(
        module,
        tmp_path,
        v194_passed=True,
        level="exact_head_identity",
    )
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "scientific_design"
    assert report["frontier"]["key"] == "prompt_switch_ab_aba_design"
    assert report["stages"][-1]["state"] == "complete"
    assert report["manual_review_requested"] is False


def test_hash_drift_is_invalid_not_missing(tmp_path: Path) -> None:
    module = load_module("v196_drift", SCRIPTS / "inspect_v196_campaign_frontier.py")
    write_v189(module, tmp_path)
    write_v190(module, tmp_path)
    payload = json.loads(artifact(module, tmp_path, "v190").read_text(encoding="utf-8"))
    source_path = Path(payload["source"]["vbench_summary"])
    write_json(source_path, {"changed": True})
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "repair_stage"
    assert report["frontier"]["key"] == "v190"
    assert report["stages"][1]["state"] == "invalid"
    assert "hash drifted" in report["frontier"]["reason"]


def test_nested_v192_hash_drift_is_invalid(tmp_path: Path) -> None:
    module = load_module(
        "v196_v192_nested_drift", SCRIPTS / "inspect_v196_campaign_frontier.py"
    )
    write_v189(module, tmp_path)
    write_v190(module, tmp_path)
    write_confirmatory(module, tmp_path, "v191", passed=True)
    write_confirmatory(module, tmp_path, "v192", passed=True)
    decision = json.loads(
        artifact(module, tmp_path, "v192").read_text(encoding="utf-8")
    )
    seed_report = json.loads(
        Path(decision["source"]["seed_report"]).read_text(encoding="utf-8")
    )
    write_json(Path(seed_report["source"]["vbench_summary"]), {"changed": True})
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "repair_stage"
    assert report["frontier"]["key"] == "v192"
    assert "v192/seed2026_30s_128 source hash drifted" in report["frontier"]["reason"]


def test_v194_requires_hash_bound_camera_report(tmp_path: Path) -> None:
    module = load_module(
        "v196_v194_camera_drift", SCRIPTS / "inspect_v196_campaign_frontier.py"
    )
    pass_through_v192(module, tmp_path)
    write_confirmatory(module, tmp_path, "v194", passed=True)
    decision = json.loads(
        artifact(module, tmp_path, "v194").read_text(encoding="utf-8")
    )
    camera_path = Path(decision["camera_compensated_motion"]["report"])
    write_json(camera_path, {"changed": True})
    report = module.inspect_campaign(tmp_path)
    assert report["frontier"]["kind"] == "repair_stage"
    assert report["frontier"]["key"] == "v194"
    assert "camera-compensated motion report" in report["frontier"]["reason"]


def test_package_contains_state_and_hash_manifest(tmp_path: Path) -> None:
    module = load_module("v196_package", SCRIPTS / "inspect_v196_campaign_frontier.py")
    repo = tmp_path / "repo"
    output = repo / "runs/v196_campaign_frontier"
    report = module.inspect_campaign(repo)
    archive = module.package(repo, output, report)
    assert archive.is_file()
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert "runs/v196_campaign_frontier/campaign_state.json" in names
    assert "runs/v196_campaign_frontier/campaign_state.md" in names
    assert "runs/v196_campaign_frontier/next_commands.txt" in names
    assert "runs/v196_campaign_frontier/evidence_manifest.json" in names


def test_runbooks_audit_before_evaluation_and_package_last() -> None:
    module = load_module("v196_runbooks", SCRIPTS / "inspect_v196_campaign_frontier.py")
    runbooks = module._runbooks()
    assert runbooks["v189"][-3].endswith(
        "v189_structured_head_phase_profile_32gpu.sh analyze"
    )
    assert runbooks["v189"][-2].endswith(
        "v189_structured_head_phase_profile_32gpu.sh package"
    )
    assert runbooks["v189"][-1].endswith("run_v197_head_phase_structure.sh package")
    for stage, audit_action in (
        ("v190", "audit-screen"),
        ("v191", "audit-confirm"),
        ("v192", "audit-all"),
        ("v194", " audit"),
    ):
        commands = runbooks[stage]
        audit_index = next(
            index
            for index, command in enumerate(commands)
            if command.endswith(audit_action)
        )
        eval_index = next(
            index
            for index, command in enumerate(commands)
            if "vbench_long.sh eval" in command
        )
        assert audit_index < eval_index
        assert commands[-1].endswith(" package")
    runner = (SCRIPTS / "run_v196_campaign_frontier.sh").read_text(encoding="utf-8")
    assert "inspect|show|next|package" in runner
    assert "next_commands.txt" in runner
