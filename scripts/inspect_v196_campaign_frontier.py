#!/usr/bin/env python3
"""Locate the first unresolved stage in the v189-v195 evidence ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT = "v196_head_phase_campaign_frontier"
STAGE_ORDER = ("v189", "v190", "v191", "v192", "v194", "v195")
MAX_BUNDLE_FILE_BYTES = 8 << 20
ALLOWED_BUNDLE_SUFFIXES = {".json", ".md", ".csv", ".txt"}

V190_RECOMMENDATIONS = {
    "advance_head_phase_method_to_fresh128",
    "head_phase_effect_not_competitive_with_all_coverage",
    "joint_head_phase_not_supported_over_factorized_controls",
    "operator_effect_without_head_phase_attribution",
    "do_not_advance_v190",
}
V195_LEVELS = {
    "exact_head_identity",
    "phase_layer_structure",
    "operator_only",
    "unsupported",
}
V191_GATES = {
    "equal_budget_noninferiority",
    "equal_budget_positive_effect",
    "native_noninferiority",
    "candidate_on_primary_pareto_front",
    "temporal_safety_vs_equal_budget",
    "temporal_safety_vs_native",
}
V192_GATES = {
    "new_seed_scope_pass",
    "two_seed_pooled_positive_effect",
    "long60_scope_pass",
}
V194_GATES = {
    "equal_budget_noninferiority",
    "native21_noninferiority",
    "frozen_positive_target_replicated",
    "same_prompt_cross_checkpoint_effect",
    "temporal_safety_vs_equal_budget",
    "temporal_safety_vs_native21",
}
V195_GATES = {
    "selected_holdout_gain_positive",
    "selected_enriched_over_complement",
    "phase_layer_allocation_beats_call_count_random",
    "head_identity_beats_call_layer_count_random",
}
STANDARD_GENERATION_SOURCES = {
    "comparison_manifest",
    "vbench_summary",
    "temporal_diagnostics",
    "temporal_contract",
}
V192_SCOPE_REPORTS = {
    "seed_report": "seed2026_30s_128",
    "long_report": "long60_seed10000_32",
}


@dataclass(frozen=True)
class StageSpec:
    key: str
    artifact: str
    inspect: Callable[[Path], dict]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_hashes(payload: dict, *, label: str) -> list[dict]:
    rows = []
    for key, value in payload.items():
        if key.endswith("_sha256") or not isinstance(value, str):
            continue
        if f"{key}_sha256" not in payload:
            continue
        path = Path(value)
        expected = str(payload[f"{key}_sha256"])
        if not path.is_file():
            raise ValueError(f"{label} source is missing: {key}={path}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(
                f"{label} source hash drifted: {key} expected={expected} observed={observed}"
            )
        rows.append({"key": key, "path": str(path.resolve()), "sha256": observed})
    return rows


def _require_source_keys(payload: dict, keys: set[str], *, label: str) -> list[dict]:
    if not keys.issubset(payload) or any(
        f"{key}_sha256" not in payload for key in keys
    ):
        raise ValueError(f"{label} source binding is incomplete")
    return _verify_source_hashes(payload, label=label)


def _base(path: Path, payload: dict, *, passed: bool, summary: dict) -> dict:
    return {
        "state": "passed" if passed else "failed",
        "artifact": str(path.resolve()),
        "artifact_sha256": sha256(path),
        "passed": passed,
        "summary": summary,
        "source_files": [],
    }


def inspect_v189(path: Path) -> dict:
    payload = load_json(path)
    recommendation = payload.get("recommendation")
    candidates = list(payload.get("generation_candidates") or ())
    if (
        payload.get("experiment") != "v189_structured_head_phase_profile"
        or recommendation
        not in {
            "advance_head_phase_maps_to_causal_screen",
            "do_not_generate_from_v189_classifier",
        }
        or not isinstance(payload.get("operators"), dict)
        or not isinstance(payload.get("profile_audits"), dict)
    ):
        raise ValueError("invalid v189 analysis contract")
    input_manifest = Path(str(payload.get("input_manifest", "")))
    if not input_manifest.is_file() or sha256(input_manifest) != payload.get(
        "input_manifest_sha256"
    ):
        raise ValueError("v189 input manifest is missing or hash-drifted")
    passed = recommendation == "advance_head_phase_maps_to_causal_screen"
    if passed and not candidates:
        raise ValueError("passing v189 analysis has no generation candidate")
    result = _base(
        path,
        payload,
        passed=passed,
        summary={
            "recommendation": recommendation,
            "generation_candidates": candidates,
            "operators": sorted(payload["operators"]),
        },
    )
    result["source_files"] = [
        {
            "key": "input_manifest",
            "path": str(input_manifest.resolve()),
            "sha256": sha256(input_manifest),
        }
    ]
    return result


def inspect_v190(path: Path) -> dict:
    payload = load_json(path)
    recommendation = payload.get("recommendation")
    selected = payload.get("selected_for_fresh128")
    if (
        int(payload.get("version", -1)) < 6
        or payload.get("experiment") != "v190_head_phase_causal_vbench_screen32"
        or payload.get("development_only") is not True
        or recommendation not in V190_RECOMMENDATIONS
        or not isinstance(payload.get("statuses"), dict)
        or not isinstance(payload.get("passing_methods"), list)
        or payload.get("temporal_diagnostics_available") is not True
    ):
        raise ValueError("invalid or incomplete v190 decision contract")
    source_files = _require_source_keys(
        payload.get("source") or {},
        {
            "comparison_manifest",
            "vbench_summary",
            "temporal_diagnostics",
            "temporal_contract",
        },
        label="v190",
    )
    passed = recommendation == "advance_head_phase_method_to_fresh128"
    if passed:
        status = payload["statuses"].get(str(selected), {})
        if (
            selected not in payload["passing_methods"]
            or status.get("full_screen_pass") is not True
            or status.get("joint_factorization_pass") is not True
            or status.get("head_phase_attribution_pass") is not True
            or status.get("selective_exposure_pass") is not True
        ):
            raise ValueError("v190 passing recommendation disagrees with method gates")
    result = _base(
        path,
        payload,
        passed=passed,
        summary={
            "recommendation": recommendation,
            "selected_method": selected,
            "passing_methods": payload["passing_methods"],
        },
    )
    result["source_files"] = source_files
    return result


def _confirmatory_stage(
    path: Path,
    *,
    experiment: str,
    gate_key: str,
    confirmed_key: str,
    pass_recommendation: str,
    fail_recommendation: str,
    gate_keys: set[str],
    source_keys: set[str],
) -> dict:
    payload = load_json(path)
    gates = payload.get(gate_key) or {}
    confirmed = payload.get(confirmed_key)
    expected = pass_recommendation if confirmed is True else fail_recommendation
    if (
        payload.get("experiment") != experiment
        or payload.get("confirmatory") is not True
        or set(gates) != gate_keys
        or any(not isinstance(value, bool) for value in gates.values())
        or confirmed is not all(gates.values())
        or payload.get("recommendation") != expected
    ):
        raise ValueError(f"invalid or incomplete {experiment} decision contract")
    source_files = _require_source_keys(
        payload.get("source") or {}, source_keys, label=experiment
    )
    result = _base(
        path,
        payload,
        passed=bool(confirmed),
        summary={
            "recommendation": payload["recommendation"],
            "confirmed": confirmed,
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "selected_operator": payload.get("selected_operator"),
        },
    )
    result["source_files"] = source_files
    return result


def inspect_v191(path: Path) -> dict:
    return _confirmatory_stage(
        path,
        experiment="v191_unseen128_head_phase_vbench",
        gate_key="confirmation_gates",
        confirmed_key="head_phase_effect_confirmed",
        pass_recommendation=(
            "freeze_head_phase_method_for_seed_length_and_cross_model_replication"
        ),
        fail_recommendation="do_not_freeze_v191_head_phase_method",
        gate_keys=V191_GATES,
        source_keys={
            "comparison_manifest",
            "vbench_summary",
            "temporal_diagnostics",
            "temporal_contract",
        },
    )


def inspect_v192(path: Path) -> dict:
    result = _confirmatory_stage(
        path,
        experiment="v192_head_phase_seed_length_robustness",
        gate_key="combined_gates",
        confirmed_key="within_model_seed_length_robustness_confirmed",
        pass_recommendation=(
            "freeze_within_model_head_phase_method_for_cross_model_transfer"
        ),
        fail_recommendation="do_not_advance_v192_head_phase_robustness",
        gate_keys=V192_GATES,
        source_keys={"input_manifest", "seed_report", "long_report"},
    )
    payload = load_json(path)
    source = payload["source"]
    for key, expected_scope in V192_SCOPE_REPORTS.items():
        scope_path = Path(source[key])
        scope_report = load_json(scope_path)
        if (
            scope_report.get("experiment") != "v192_head_phase_robustness_vbench"
            or scope_report.get("scope") != expected_scope
            or scope_report.get("confirmatory") is not True
        ):
            raise ValueError(f"invalid v192 scope report contract: {key}")
        result["source_files"].extend(
            _require_source_keys(
                scope_report.get("source") or {},
                STANDARD_GENERATION_SOURCES,
                label=f"v192/{expected_scope}",
            )
        )
    return result


def inspect_v194(path: Path) -> dict:
    result = _confirmatory_stage(
        path,
        experiment="v194_causal_checkpoint_transfer_vbench",
        gate_key="confirmation_gates",
        confirmed_key="cross_checkpoint_transfer_confirmed",
        pass_recommendation="freeze_head_phase_route_as_cross_checkpoint_supported",
        fail_recommendation="do_not_claim_v194_checkpoint_transfer",
        gate_keys=V194_GATES,
        source_keys=STANDARD_GENERATION_SOURCES,
    )
    payload = load_json(path)
    camera = payload.get("camera_compensated_motion") or {}
    bool_keys = {
        "measurement_calibration_pass",
        "directional_against_all_controls",
        "strong_against_all_controls",
        "motion_improvement_claim_supported",
    }
    if (
        camera.get("available") is not True
        or not isinstance(camera.get("report"), str)
        or not isinstance(camera.get("report_sha256"), str)
        or any(not isinstance(camera.get(key), bool) for key in bool_keys)
    ):
        raise ValueError("v194 final decision lacks camera-compensated motion evidence")
    camera_path = Path(camera["report"])
    if not camera_path.is_file() or sha256(camera_path) != camera["report_sha256"]:
        raise ValueError(
            "v194 camera-compensated motion report is missing or hash-drifted"
        )
    camera_report = load_json(camera_path)
    camera_sources = camera_report.get("source") or {}
    if (
        camera_report.get("experiment") != "v193_camera_compensated_motion_calibration"
        or camera_sources.get("comparison_manifest_sha256")
        != payload["source"]["comparison_manifest_sha256"]
    ):
        raise ValueError("v194 camera-compensated motion report is mismatched")
    result["source_files"].append(
        {
            "key": "camera_motion_report",
            "path": str(camera_path.resolve()),
            "sha256": sha256(camera_path),
        }
    )
    result["source_files"].extend(
        _require_source_keys(
            camera_sources,
            {"comparison_manifest", "motion_csv", "motion_contract"},
            label="v194/camera_motion",
        )
    )
    return result


def _expected_v195_recommendation(v194_confirmed: bool, level: str) -> str:
    if v194_confirmed:
        return {
            "exact_head_identity": "freeze_route_with_cross_checkpoint_mechanistic_support",
            "phase_layer_structure": "limit_transfer_mechanism_claim_to_phase_layer_structure",
            "operator_only": "retain_generation_transfer_without_head_identity_claim",
            "unsupported": "inspect_shadow_metric_mismatch_before_any_new_generation",
        }[level]
    return {
        "exact_head_identity": (
            "shadow_metric_transfers_but_generation_does_not_stop_and_diagnose"
        ),
        "phase_layer_structure": (
            "reprofile_causal_membership_with_frozen_feature_standard"
        ),
        "operator_only": "classifier_is_checkpoint_specific_do_not_advance",
        "unsupported": "stop_cross_checkpoint_route_transfer",
    }[level]


def inspect_v195(path: Path) -> dict:
    payload = load_json(path)
    level = payload.get("mechanism_support_level")
    v194_confirmed = payload.get("v194_generation_transfer_confirmed")
    if (
        payload.get("experiment") != "v195_cross_checkpoint_head_phase_profile"
        or payload.get("diagnostic") is not True
        or level not in V195_LEVELS
        or not isinstance(v194_confirmed, bool)
        or payload.get("recommendation")
        != _expected_v195_recommendation(v194_confirmed, str(level))
        or payload.get("manual_review_required") is not False
        or set(payload.get("mechanism_gates") or {}) != V195_GATES
        or any(
            not isinstance(value, bool)
            for value in (payload.get("mechanism_gates") or {}).values()
        )
        or payload.get("exact_head_identity_transfer_supported")
        is not (level == "exact_head_identity")
        or payload.get("phase_layer_structure_transfer_supported")
        is not (level in {"exact_head_identity", "phase_layer_structure"})
        or payload.get("operator_compatibility_transfer_supported")
        is not (level != "unsupported")
    ):
        raise ValueError("invalid or incomplete v195 analysis contract")
    source_files = _require_source_keys(
        payload.get("source") or {},
        {"input_manifest", "profile_audit", "sf_cell_scores"},
        label="v195",
    )
    result = _base(
        path,
        payload,
        passed=bool(v194_confirmed and level == "exact_head_identity"),
        summary={
            "recommendation": payload["recommendation"],
            "mechanism_support_level": level,
            "v194_generation_transfer_confirmed": v194_confirmed,
            "exact_head_identity_transfer_supported": payload.get(
                "exact_head_identity_transfer_supported"
            ),
            "phase_layer_structure_transfer_supported": payload.get(
                "phase_layer_structure_transfer_supported"
            ),
        },
    )
    result["state"] = "complete"
    result["source_files"] = source_files
    return result


STAGES = (
    StageSpec(
        "v189",
        "runs/v189_structured_head_phase_profile/analysis/analysis.json",
        inspect_v189,
    ),
    StageSpec(
        "v190",
        "runs/v190_head_phase_causal_screen/screen32/analysis/v190_head_phase_causal_screen.json",
        inspect_v190,
    ),
    StageSpec(
        "v191",
        "runs/v191_head_phase_confirmation/confirm128/analysis/v191_head_phase_confirmation.json",
        inspect_v191,
    ),
    StageSpec(
        "v192",
        "runs/v192_head_phase_robustness/analysis/v192_head_phase_robustness.json",
        inspect_v192,
    ),
    StageSpec(
        "v194",
        "runs/v194_cf_checkpoint_transfer/transfer64/analysis/v194_checkpoint_transfer.json",
        inspect_v194,
    ),
    StageSpec(
        "v195",
        "runs/v195_cross_checkpoint_head_phase_profile/analysis/analysis.json",
        inspect_v195,
    ),
)


def _runbooks() -> dict[str, list[str]]:
    return {
        "v189": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh profile128",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh analyze",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh package",
        ],
        "v190": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh smoke",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh generate32",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-screen",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_vbench_long.sh prepare",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v190_vbench_long.sh split",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v190_vbench_long.sh eval",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_vbench_long.sh collect",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_vbench_long.sh decision",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh package",
        ],
        "v191": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh smoke",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh audit-smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh generate128",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh audit-confirm",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_vbench_long.sh prepare",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v191_vbench_long.sh split",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v191_vbench_long.sh eval",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_vbench_long.sh collect",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_vbench_long.sh decision",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v191_head_phase_confirmation_32gpu.sh package",
        ],
        "v192": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh smoke",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh audit-smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh generate-all",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh audit-all",
            "for SCOPE in seed2026_30s_128 long60_seed10000_32; do NODE_RANK=0 NUM_NODES=4 SCOPE=$SCOPE bash scripts/run_v192_vbench_long.sh prepare; done",
            "for SCOPE in seed2026_30s_128 long60_seed10000_32; do NODE_RANK=<0|1|2|3> NUM_NODES=4 SCOPE=$SCOPE bash scripts/run_v192_vbench_long.sh split; done",
            "for SCOPE in seed2026_30s_128 long60_seed10000_32; do NODE_RANK=<0|1|2|3> NUM_NODES=4 SCOPE=$SCOPE bash scripts/run_v192_vbench_long.sh eval; done",
            "for SCOPE in seed2026_30s_128 long60_seed10000_32; do NODE_RANK=0 NUM_NODES=4 SCOPE=$SCOPE bash scripts/run_v192_vbench_long.sh collect; done",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_vbench_long.sh decision",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v192_head_phase_robustness_32gpu.sh package",
        ],
        "v194": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh smoke",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh audit-smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh generate",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh audit",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh prepare",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_vbench_long.sh split",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_vbench_long.sh eval",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh collect",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_vbench_long.sh camera-compute",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh camera-collect",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh decision",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh package",
        ],
        "v195": [
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh prepare",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh smoke",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh audit-smoke",
            "NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh profile128",
            "NODE_RANK=0 NUM_NODES=4 bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh package",
        ],
    }


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _next_after_v195(stage: dict) -> dict:
    summary = stage["summary"]
    level = summary["mechanism_support_level"]
    generation = bool(summary["v194_generation_transfer_confirmed"])
    if generation and level == "exact_head_identity":
        return {
            "kind": "scientific_design",
            "key": "prompt_switch_ab_aba_design",
            "reason": (
                "Generation and exact-head mechanism transfer both passed; method review "
                "must freeze the single-prompt route before a prompt-switch experiment."
            ),
        }
    if generation and level == "phase_layer_structure":
        return {
            "kind": "scientific_design",
            "key": "phase_layer_only_transfer_ablation",
            "reason": (
                "Generation transferred but exact heads did not; test a frozen coarser "
                "phase/layer route before any prompt-switch extension."
            ),
        }
    if generation:
        return {
            "kind": "scientific_redesign",
            "key": "operator_only_story_review",
            "reason": (
                "The generation effect lacks classifier support; do not add ABA until "
                "the paper claim is reduced to an operator-level method or rejected."
            ),
        }
    if level in {"exact_head_identity", "phase_layer_structure"}:
        return {
            "kind": "diagnosis",
            "key": "shadow_generation_objective_mismatch",
            "reason": (
                "Profiling structure transferred while generated-video gates failed; "
                "diagnose the shadow objective instead of refitting from v194 videos."
            ),
        }
    return {
        "kind": "stop",
        "key": "stop_current_cross_checkpoint_route",
        "reason": "Neither generated-video nor mechanism transfer supports this route.",
    }


def inspect_campaign(repo_root: Path) -> dict:
    repo_root = repo_root.resolve()
    runbooks = _runbooks()
    rows = []
    dependency_open = True
    terminal = False
    frontier = None
    alerts = []
    for spec in STAGES:
        path = repo_root / spec.artifact
        if not dependency_open:
            state = "blocked"
            if path.is_file():
                alerts.append(
                    f"{spec.key} artifact exists after a failed/invalid prerequisite; treat it as stale"
                )
            rows.append(
                {
                    "key": spec.key,
                    "state": state,
                    "artifact": str(path.resolve()),
                    "passed": False,
                    "summary": {},
                    "source_files": [],
                }
            )
            continue
        if not path.is_file():
            rows.append(
                {
                    "key": spec.key,
                    "state": "missing",
                    "artifact": str(path.resolve()),
                    "passed": False,
                    "summary": {},
                    "source_files": [],
                }
            )
            if frontier is None:
                frontier = {
                    "kind": "run_stage",
                    "key": spec.key,
                    "reason": f"{spec.key} final artifact is absent",
                    "commands": runbooks[spec.key],
                }
            dependency_open = False
            continue
        try:
            row = spec.inspect(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            row = {
                "key": spec.key,
                "state": "invalid",
                "artifact": str(path.resolve()),
                "passed": False,
                "summary": {"error": str(exc)},
                "source_files": [],
            }
            frontier = {
                "kind": "repair_stage",
                "key": spec.key,
                "reason": str(exc),
                "commands": runbooks[spec.key],
            }
            dependency_open = False
            terminal = True
            rows.append(row)
            continue
        row["key"] = spec.key
        rows.append(row)
        if spec.key == "v194":
            # v195 is required to explain either a passing or failing v194 result.
            continue
        if spec.key == "v195":
            frontier = _next_after_v195(row)
            dependency_open = False
            terminal = frontier["kind"] == "stop"
            continue
        if not row["passed"]:
            frontier = {
                "kind": "stop",
                "key": f"stop_after_{spec.key}",
                "reason": row["summary"].get("recommendation", f"{spec.key} failed"),
            }
            dependency_open = False
            terminal = True

    if frontier is None:
        frontier = {
            "kind": "invalid_campaign",
            "key": "no_frontier",
            "reason": "campaign state did not produce a frontier",
        }
        terminal = True
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "repo_root": str(repo_root),
        "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "stage_order": list(STAGE_ORDER),
        "stages": rows,
        "frontier": frontier,
        "terminal": terminal,
        "alerts": alerts,
        "manual_review_requested": False,
        "policy": (
            "Run only the first unresolved stage. v194 failure still advances to v195 "
            "diagnosis; failures in v189-v192 stop the frozen route."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v196 Campaign Frontier",
        "",
        f"- Git head: `{report['git_head']}`",
        f"- Frontier: `{report['frontier']['key']}`",
        f"- Action kind: `{report['frontier']['kind']}`",
        f"- Reason: {report['frontier']['reason']}",
        f"- Manual review requested: `{str(report['manual_review_requested']).lower()}`",
        "",
        "| Stage | State | Passed | Recommendation / error |",
        "|---|---|---:|---|",
    ]
    for row in report["stages"]:
        summary = row.get("summary") or {}
        detail = summary.get("recommendation") or summary.get("error") or ""
        lines.append(
            f"| {row['key']} | {row['state']} | {str(bool(row['passed'])).lower()} | {detail} |"
        )
    if report["alerts"]:
        lines.extend(["", "## Alerts", ""])
        lines.extend(f"- {value}" for value in report["alerts"])
    commands = report["frontier"].get("commands") or []
    if commands:
        lines.extend(
            [
                "",
                "## Next commands",
                "",
                "Run node-0-only lines once. Run lines containing `<0|1|2|3>` concurrently on all four nodes.",
                "",
                "```bash",
            ]
        )
        lines.extend(commands)
        lines.extend(["```", ""])
    else:
        lines.extend(
            [
                "",
                "## Next decision",
                "",
                report["frontier"]["reason"],
                "",
            ]
        )
    return "\n".join(lines)


def write_report(report: dict, output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "campaign_state.json"
    md_path = output_root / "campaign_state.md"
    command_path = output_root / "next_commands.txt"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render(report), encoding="utf-8")
    commands = report["frontier"].get("commands") or []
    command_path.write_text(
        "\n".join(commands)
        + ("\n" if commands else "No GPU stage is authorized by this state.\n"),
        encoding="utf-8",
    )
    return json_path, md_path, command_path


def _candidate_bundle_files(
    repo_root: Path, report: dict, output_files: tuple[Path, ...]
) -> list[Path]:
    candidates = set(output_files)
    for row in report["stages"]:
        artifact = Path(row["artifact"])
        if artifact.is_file():
            candidates.add(artifact)
            markdown = artifact.with_suffix(".md")
            if markdown.is_file():
                candidates.add(markdown)
        for source in row.get("source_files") or ():
            path = Path(source["path"])
            if path.is_file():
                candidates.add(path)
    extras = (
        "runs/v189_structured_head_phase_profile/profile_audit.json",
        "runs/v189_structured_head_phase_profile/analysis/cell_scores.csv",
        "runs/v195_cross_checkpoint_head_phase_profile/profile_audit.json",
        "runs/v195_cross_checkpoint_head_phase_profile/analysis/cell_transfer.csv",
        "runs/v195_cross_checkpoint_head_phase_profile/analysis/holdout_prompt_effects.csv",
    )
    for relative in extras:
        path = repo_root / relative
        if path.is_file():
            candidates.add(path)
    return sorted(candidates)


def package(repo_root: Path, output_root: Path, report: dict) -> Path:
    output_files = write_report(report, output_root)
    files = _candidate_bundle_files(repo_root, report, output_files)
    manifest_rows = []
    accepted = []
    repo_root = repo_root.resolve()
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError:
            continue
        if (
            resolved.suffix.lower() not in ALLOWED_BUNDLE_SUFFIXES
            or resolved.stat().st_size > MAX_BUNDLE_FILE_BYTES
        ):
            continue
        accepted.append((resolved, relative.as_posix()))
        manifest_rows.append(
            {
                "path": relative.as_posix(),
                "size": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )
    evidence_manifest = output_root / "evidence_manifest.json"
    evidence_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "experiment": EXPERIMENT,
                "files": manifest_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = output_root / "v196_campaign_evidence.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path, relative in accepted:
            handle.write(path, relative)
        try:
            manifest_name = (
                evidence_manifest.resolve().relative_to(repo_root).as_posix()
            )
        except ValueError:
            manifest_name = "v196/evidence_manifest.json"
        handle.write(evidence_manifest, manifest_name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "package"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_campaign(args.repo_root)
    paths = write_report(report, args.output_root)
    print(
        "[v196-frontier] "
        f"kind={report['frontier']['kind']} key={report['frontier']['key']} "
        f"state={paths[0]}"
    )
    if args.action == "package":
        archive = package(args.repo_root, args.output_root, report)
        print(f"[v196-package] {archive}")


if __name__ == "__main__":
    main()
