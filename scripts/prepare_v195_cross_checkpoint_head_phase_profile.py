#!/usr/bin/env python3
"""Freeze a diagnostic Causal-checkpoint profile after v194 completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from prepare_v189_structured_head_phase_profile import verify as verify_v189
from prepare_v190_head_phase_causal_screen import validate_map
from prepare_v194_cf_checkpoint_transfer import (
    CANDIDATE as V194_CANDIDATE,
)
from prepare_v194_cf_checkpoint_transfer import verify as verify_v194

EXPERIMENT = "v195_cross_checkpoint_head_phase_profile_inputs"
PROMPT_COUNT = 128
CALLS = 4
LAYERS = 30
HEADS = 12
SEED = 0
NUM_OUTPUT_FRAMES = 120
PROFILE_CONTRACT = "v189"
PROFILE_VERSION = 4
PROFILE_METHOD = "structured_head_phase_cache_compatibility"
EXPECTED_RECORDS = PROMPT_COUNT * CALLS * LAYERS * HEADS
RANDOM_DRAWS = 10_000
RANDOM_SEED = 1950000
V194_GATE_KEYS = {
    "equal_budget_noninferiority",
    "native21_noninferiority",
    "frozen_positive_target_replicated",
    "same_prompt_cross_checkpoint_effect",
    "temporal_safety_vs_equal_budget",
    "temporal_safety_vs_native21",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frozen(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"frozen v195 artifact differs: {path}")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_json_frozen(path: Path, payload: dict) -> str:
    return _write_frozen(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _bound_path(row: dict, key: str, *, label: str) -> Path:
    path = Path(str(row.get(key, "")))
    if not path.is_file() or sha256(path) != row.get(f"{key}_sha256"):
        raise ValueError(f"v195 prerequisite drifted: {label}/{key}")
    return path


def validate_v194_decision(decision_path: Path, input_path: Path) -> tuple[dict, dict]:
    decision = load_json(decision_path)
    frozen = verify_v194(input_path)
    gates = decision.get("confirmation_gates") or {}
    confirmed = bool(decision.get("cross_checkpoint_transfer_confirmed"))
    expected_recommendation = (
        "freeze_head_phase_route_as_cross_checkpoint_supported"
        if confirmed
        else "do_not_claim_v194_checkpoint_transfer"
    )
    if (
        decision.get("experiment") != "v194_causal_checkpoint_transfer"
        or decision.get("confirmatory") is not True
        or set(gates) != V194_GATE_KEYS
        or any(not isinstance(value, bool) for value in gates.values())
        or confirmed != all(gates.values())
        or decision.get("recommendation") != expected_recommendation
        or tuple(decision.get("methods") or ())
        != ("cf_native_21", "cf_all_recent_9ffe", V194_CANDIDATE)
    ):
        raise ValueError(
            "v195 requires one complete and internally consistent v194 decision"
        )
    source = decision.get("source") or {}
    bound_input = _bound_path(source, "input_manifest", label="v194-decision")
    for key in (
        "comparison_manifest",
        "vbench_summary",
        "temporal_diagnostics",
        "temporal_contract",
    ):
        _bound_path(source, key, label="v194-decision")
    if not bound_input.samefile(input_path):
        raise ValueError("v194 decision is bound to a different input manifest")
    if (
        decision.get("transfer_axis") != frozen.get("transfer_axis")
        or decision.get("candidate") != frozen.get("candidate")
        or decision.get("prompt_count") != frozen.get("prompt_count")
    ):
        raise ValueError("v194 decision and frozen input disagree")
    return decision, frozen


def validate_v189_sources(
    manifest_path: Path,
    analysis_path: Path,
    cell_scores_path: Path,
    *,
    operator: str,
) -> tuple[dict, dict]:
    manifest = verify_v189(manifest_path)
    analysis = load_json(analysis_path)
    if (
        analysis.get("experiment") != "v189_structured_head_phase_profile"
        or analysis.get("input_manifest_sha256") != sha256(manifest_path)
        or not Path(str(analysis.get("input_manifest", ""))).samefile(manifest_path)
        or operator not in (analysis.get("operators") or {})
        or operator not in (analysis.get("generation_candidates") or [])
    ):
        raise ValueError(
            "v195 requires the selected operator in one complete v189 analysis"
        )
    required = {
        "operator",
        "call_index",
        "layer",
        "head",
        "discovery_gain",
        "validation_gain",
        "compatible",
    }
    with cell_scores_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        raise ValueError("v189 cell_scores.csv has an incompatible schema")
    selected = [row for row in rows if row["operator"] == operator]
    keys = {
        (int(row["call_index"]), int(row["layer"]), int(row["head"]))
        for row in selected
    }
    expected = {
        (call, layer, head)
        for call in range(CALLS)
        for layer in range(LAYERS)
        for head in range(HEADS)
    }
    if len(selected) != CALLS * LAYERS * HEADS or keys != expected:
        raise ValueError(
            "v189 cell scores do not contain one complete selected operator"
        )
    return manifest, analysis


def _copy(source: Path, target: Path) -> str:
    return _write_frozen(target, source.read_bytes())


def prepare(
    v194_decision_path: Path,
    v194_input_path: Path,
    v189_manifest_path: Path,
    v189_analysis_path: Path,
    v189_cell_scores_path: Path,
    output_root: Path,
) -> dict:
    decision, v194 = validate_v194_decision(v194_decision_path, v194_input_path)
    operator = str(v194["selected_operator"])
    v189, v189_analysis = validate_v189_sources(
        v189_manifest_path,
        v189_analysis_path,
        v189_cell_scores_path,
        operator=operator,
    )
    source_prompt_path = Path(v189["prompt_file"])
    source_map_path = Path(v189["profile_map"])
    prompts = source_prompt_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not value.strip() for value in prompts):
        raise ValueError("v195 requires the exact 128-prompt v189 suite")
    prompt_path = output_root / "moviegen_128_qwen.txt"
    prompt_sha = _copy(source_prompt_path, prompt_path)
    profile_map_path = output_root / "profile_all_heads.csv"
    profile_map_sha = _copy(source_map_path, profile_map_path)

    route_source = Path(v194["methods"][V194_CANDIDATE]["head_phase_map"])
    route_payload = load_json(route_source)
    validate_map(route_payload, operator=operator)
    route_path = output_root / "frozen_sf_head_phase_route.json"
    route_sha = _copy(route_source, route_path)

    checkpoint = dict(v194["checkpoint"])
    runtime = dict(v194["runtime_contract"])
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "diagnostic": True,
        "prerequisite": "completed_v194_automatic_decision",
        "question": (
            "Does the Self-Forcing-selected Head x Denoising-Phase membership retain "
            "shadow-readout compatibility on the Causal-Forcing checkpoint, and does "
            "transfer reside in exact heads or only in phase/layer structure?"
        ),
        "operator": operator,
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_split": v189["prompt_split"],
        "seed": SEED,
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "profile_map": str(profile_map_path.resolve()),
        "profile_map_sha256": profile_map_sha,
        "profile_contract": PROFILE_CONTRACT,
        "profile_artifact_version": PROFILE_VERSION,
        "profile_method": PROFILE_METHOD,
        "expected_record_count": EXPECTED_RECORDS,
        "execution_contract": {
            "num_nodes": 4,
            "gpus_per_node": 8,
            "world_shards": 32,
            "assignment": "prompt_id modulo 32 equals shard rank",
        },
        "active_trajectory": "all_recent_shadow_readout_only",
        "candidate_reads": {
            "recent": "sink1 + recent8 = 9 FFE",
            "coverage": f"sink1 + {operator} middle4 + recent4 = 9 FFE",
            "union": "representation-complete candidate union <= 13 FFE",
        },
        "checkpoint": checkpoint,
        "runtime_contract": runtime,
        "frozen_sf_route": {
            "path": str(route_path.resolve()),
            "sha256": route_sha,
            "map_id": route_payload["map_id"],
            "coverage_count_by_call": route_payload["coverage_count_by_call"],
            "coverage_cell_count": sum(route_payload["coverage_count_by_call"]),
            "never_refit_on_v195": True,
        },
        "random_controls": {
            "draws": RANDOM_DRAWS,
            "seed": RANDOM_SEED,
            "call_count_matched": (
                "preserve selected count per denoising call and randomize layer/head"
            ),
            "call_layer_count_matched": (
                "preserve selected count per call/layer and randomize head identity"
            ),
        },
        "v194_provenance": {
            "decision": str(v194_decision_path.resolve()),
            "decision_sha256": sha256(v194_decision_path),
            "input_manifest": str(v194_input_path.resolve()),
            "input_manifest_sha256": sha256(v194_input_path),
            "cross_checkpoint_transfer_confirmed": decision[
                "cross_checkpoint_transfer_confirmed"
            ],
            "recommendation": decision["recommendation"],
        },
        "v189_provenance": {
            "input_manifest": str(v189_manifest_path.resolve()),
            "input_manifest_sha256": sha256(v189_manifest_path),
            "analysis": str(v189_analysis_path.resolve()),
            "analysis_sha256": sha256(v189_analysis_path),
            "cell_scores": str(v189_cell_scores_path.resolve()),
            "cell_scores_sha256": sha256(v189_cell_scores_path),
            "recommendation": v189_analysis["recommendation"],
        },
        "claim_boundary": (
            "v195 is a paired shadow-readout mechanism audit within the shared Wan "
            "architecture. It does not alter a latent trajectory, generate videos, "
            "or independently establish perceptual quality. A Causal-refit map is "
            "diagnostic only and may not replace the frozen SF map in v194."
        ),
        "manual_review_required": False,
    }
    _write_json_frozen(output_root / "manifest.json", payload)
    return payload


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment") != EXPERIMENT
        or payload.get("diagnostic") is not True
        or payload.get("prerequisite") != "completed_v194_automatic_decision"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or payload.get("profile_contract") != PROFILE_CONTRACT
        or int(payload.get("profile_artifact_version", -1)) != PROFILE_VERSION
        or payload.get("profile_method") != PROFILE_METHOD
        or int(payload.get("expected_record_count", -1)) != EXPECTED_RECORDS
        or payload.get("active_trajectory") != "all_recent_shadow_readout_only"
        or payload.get("manual_review_required") is not False
    ):
        raise ValueError("invalid v195 input manifest")
    if payload.get("execution_contract") != {
        "num_nodes": 4,
        "gpus_per_node": 8,
        "world_shards": 32,
        "assignment": "prompt_id modulo 32 equals shard rank",
    }:
        raise ValueError("v195 execution contract drifted")

    v194_source = payload.get("v194_provenance") or {}
    v194_decision_path = _bound_path(v194_source, "decision", label="v194")
    v194_input_path = _bound_path(v194_source, "input_manifest", label="v194")
    decision, v194 = validate_v194_decision(v194_decision_path, v194_input_path)
    if (
        bool(v194_source.get("cross_checkpoint_transfer_confirmed"))
        != bool(decision["cross_checkpoint_transfer_confirmed"])
        or v194_source.get("recommendation") != decision.get("recommendation")
        or payload.get("operator") != v194.get("selected_operator")
    ):
        raise ValueError("v195 v194 provenance drifted")

    v189_source = payload.get("v189_provenance") or {}
    v189_manifest_path = _bound_path(v189_source, "input_manifest", label="v189")
    v189_analysis_path = _bound_path(v189_source, "analysis", label="v189")
    v189_scores_path = _bound_path(v189_source, "cell_scores", label="v189")
    v189, analysis = validate_v189_sources(
        v189_manifest_path,
        v189_analysis_path,
        v189_scores_path,
        operator=str(payload["operator"]),
    )
    if v189_source.get("recommendation") != analysis.get("recommendation"):
        raise ValueError("v195 v189 recommendation drifted")

    for key, source_key in (
        ("prompt_file", "prompt_file"),
        ("profile_map", "profile_map"),
    ):
        path = Path(payload[key])
        source = Path(v189[source_key])
        if (
            not path.is_file()
            or sha256(path) != payload[f"{key}_sha256"]
            or path.read_bytes() != source.read_bytes()
        ):
            raise ValueError(f"v195 copied artifact drifted: {key}")
    if payload.get("prompt_split") != v189.get("prompt_split"):
        raise ValueError("v195 prompt split drifted")

    route = payload.get("frozen_sf_route") or {}
    route_path = Path(str(route.get("path", "")))
    source_route = Path(v194["methods"][V194_CANDIDATE]["head_phase_map"])
    if (
        not route_path.is_file()
        or sha256(route_path) != route.get("sha256")
        or route_path.read_bytes() != source_route.read_bytes()
    ):
        raise ValueError("v195 frozen route drifted")
    route_payload = load_json(route_path)
    validate_map(route_payload, operator=str(payload["operator"]))
    if (
        route.get("map_id") != route_payload.get("map_id")
        or route.get("coverage_count_by_call")
        != route_payload.get("coverage_count_by_call")
        or int(route.get("coverage_cell_count", -1))
        != sum(route_payload["coverage_count_by_call"])
        or route.get("never_refit_on_v195") is not True
    ):
        raise ValueError("v195 frozen route metadata drifted")

    if payload.get("checkpoint") != v194.get("checkpoint"):
        raise ValueError("v195 checkpoint contract differs from v194")
    checkpoint = payload["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    if (
        not checkpoint_path.is_file()
        or sha256(checkpoint_path) != checkpoint.get("sha256")
        or checkpoint.get("state_key") != "generator"
        or checkpoint.get("use_ema") is not False
    ):
        raise ValueError("v195 checkpoint drifted")
    if payload.get("runtime_contract") != v194.get("runtime_contract"):
        raise ValueError("v195 runtime contract differs from v194")
    for row in payload["runtime_contract"]["files"]:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"v195 runtime source drifted: {row['relative_path']}")
    random_controls = payload.get("random_controls") or {}
    if (
        int(random_controls.get("draws", -1)) != RANDOM_DRAWS
        or int(random_controls.get("seed", -1)) != RANDOM_SEED
    ):
        raise ValueError("v195 random-control contract drifted")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v194-decision", type=Path, required=True)
    prepare_parser.add_argument("--v194-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v189-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v189-analysis", type=Path, required=True)
    prepare_parser.add_argument("--v189-cell-scores", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.v194_decision,
            args.v194_input_manifest,
            args.v189_input_manifest,
            args.v189_analysis,
            args.v189_cell_scores,
            args.output_root,
        )
        print(
            "[v195-prepare] PASS "
            f"operator={payload['operator']} prompts={payload['prompt_count']} "
            f"v194_confirmed={str(payload['v194_provenance']['cross_checkpoint_transfer_confirmed']).lower()}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v195-verify] PASS "
            f"operator={payload['operator']} prompts={payload['prompt_count']}"
        )


if __name__ == "__main__":
    main()
