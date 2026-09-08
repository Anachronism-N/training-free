#!/usr/bin/env python3
"""Freeze a profiling-disjoint confirmation for v201-selected candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v201_head_phase_horizon_screen import (
    NUM_OUTPUT_FRAMES,
    OPERATORS,
    sha256,
    validate_horizon_map,
    verify as verify_v201_input,
)


EXPERIMENT = "v205_profile_disjoint_head_phase_horizon_confirmation"
PROMPT_COUNT = 128
SEED = 20500
MAX_CANDIDATES = 2
ADVANCE_RECOMMENDATIONS = {
    "advance_sf_significant_horizon_method_to_fresh128",
    "advance_sf_significant_method_to_fresh128_mechanism_unresolved",
    "advance_sf_positive_horizon_method_to_fresh128",
    "advance_sf_positive_method_to_fresh128_mechanism_unresolved",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v205 artifact differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def resolve_declared(path_value: str, fallback: Path) -> Path:
    declared = Path(path_value)
    if declared.is_file():
        return declared
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"missing declared and fallback file: {declared}, {fallback}"
    )


def verify_path_hash(row: dict, key: str, *, label: str) -> Path:
    path = Path(str(row.get(key, "")))
    if not path.is_file() or sha256(path) != row.get(f"{key}_sha256"):
        raise ValueError(f"v205 prerequisite drifted: {label}/{key}")
    return path


def validate_v201(
    decision_path: Path,
    input_manifest_path: Path,
    published_path: Path,
    comparison_manifest_path: Path,
) -> tuple[dict, dict, dict, dict, tuple[str, ...]]:
    decision = load_json(decision_path)
    frozen = verify_v201_input(input_manifest_path)
    published = load_json(published_path)
    comparison = load_json(comparison_manifest_path)
    selected = tuple(str(value) for value in decision.get("selected_for_fresh128") or ())
    source = decision.get("source") or {}
    if (
        int(decision.get("version", -1)) < 2
        or decision.get("experiment")
        != "v201_head_phase_horizon_causal_vbench_screen32"
        or decision.get("development_only") is not True
        or decision.get("recommendation") not in ADVANCE_RECOMMENDATIONS
        or decision.get("primary_baseline") != "sf_native"
        or not selected
        or len(selected) > MAX_CANDIDATES
        or len(set(selected)) != len(selected)
        or source.get("comparison_manifest_sha256")
        != sha256(comparison_manifest_path)
    ):
        raise ValueError("v205 requires a SHA-bound advancing v201 decision")
    for key in ("vbench_summary", "temporal_diagnostics", "temporal_contract"):
        verify_path_hash(source, key, label="v201-decision")

    statuses = decision.get("candidate_status") or {}
    for method in selected:
        status = statuses.get(method) or {}
        efficacy = status.get("sf_efficacy") or {}
        if (
            method not in frozen["methods"]
            or frozen["methods"][method].get("role")
            != "primary_head_phase_horizon"
            or frozen["methods"][method].get("schedule")
            != "head_phase_horizon"
            or status.get("selected_for_fresh128") is not True
            or efficacy.get("directional_screen_pass") is not True
            or (efficacy.get("noninferiority") or {}).get("pass") is not True
            or (efficacy.get("positive_support") or {}).get("directional_pass")
            is not True
            or (efficacy.get("temporal_guard") or {}).get(
                "automatic_safety_pass"
            )
            is not True
        ):
            raise ValueError(f"v205 selected candidate is not eligible: {method}")

    published_rows = {
        str(row.get("key")): row for row in published.get("methods") or ()
    }
    generation_contract = Path(str(published.get("experiment_contract", "")))
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v201_head_phase_horizon_causal_generation"
        or published.get("scope") != "screen32"
        or not generation_contract.is_file()
        or published.get("experiment_contract_sha256")
        != sha256(generation_contract)
        or any(
            method not in published_rows or published_rows[method].get("ok") is not True
            for method in ("sf_native", *selected)
        )
    ):
        raise ValueError("v205 requires complete audited v201 generation evidence")
    contract = load_json(generation_contract)
    if (
        contract.get("scope") != "screen32"
        or contract.get("primary_baseline") != "sf_native"
        or int(contract.get("prompt_count", -1)) != 32
        or contract.get("input_manifest_sha256") != sha256(input_manifest_path)
    ):
        raise ValueError("v201 generation contract does not bind its frozen input")
    for method in ("sf_native", *selected):
        audit_path = Path(str(published_rows[method].get("audit", "")))
        if (
            not audit_path.is_file()
            or sha256(audit_path) != published_rows[method].get("audit_sha256")
        ):
            raise ValueError(f"v201 source audit drifted: {method}")

    comparison_rows = {
        str(row.get("key")): row for row in comparison.get("methods") or ()
    }
    comparison_source = comparison.get("source") or {}
    if (
        comparison.get("experiment")
        != "v201_head_phase_horizon_causal_vbench_screen32"
        or int(comparison.get("prompt_count", -1)) != 32
        or comparison.get("primary_baseline") != "sf_native"
        or any(method not in comparison_rows for method in ("sf_native", *selected))
        or comparison_source.get("published_manifest_sha256")
        != sha256(published_path)
        or comparison_source.get("experiment_contract_sha256")
        != sha256(generation_contract)
    ):
        raise ValueError("v201 decision, comparison, and generation evidence disagree")
    return decision, frozen, published, comparison, selected


def validate_profile_disjoint_prompts(
    source_manifest_path: Path,
    v201_input: dict,
) -> tuple[dict, Path, list[str], list[int]]:
    source = load_json(source_manifest_path)
    prompt_path = resolve_declared(
        str(source.get("prompt_file", "")),
        source_manifest_path.parent / "moviegen_fresh_0128_0255.txt",
    )
    source_indices = [int(value) for value in source.get("prompt_source_indices") or ()]
    if (
        source.get("experiment") != "v180_rccp_fresh128_inputs"
        or int(source.get("prompt_count", -1)) != PROMPT_COUNT
        or source.get("evaluation_source_index_range") != [128, 255]
        or source.get("evaluation_prompts_used_for_membership") is not False
        or int(source.get("exact_text_overlap_with_calibration", -1)) != 0
        or source_indices != list(range(128, 256))
        or sha256(prompt_path) != source.get("prompt_file_sha256")
    ):
        raise ValueError("v205 profile-disjoint prompt source contract is invalid")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not prompt.strip() for prompt in prompts):
        raise ValueError("v205 requires 128 non-empty profile-disjoint prompts")

    development_path = Path(str(v201_input.get("source_prompt_file", "")))
    if (
        not development_path.is_file()
        or sha256(development_path) != v201_input.get("source_prompt_file_sha256")
    ):
        raise ValueError("v205 cannot locate the complete v189 development suite")
    development = {
        normalized_text(prompt)
        for prompt in development_path.read_text(encoding="utf-8").splitlines()
    }
    evaluation = {normalized_text(prompt) for prompt in prompts}
    if (
        len(development) != 128
        or len(evaluation) != PROMPT_COUNT
        or "" in development
        or development & evaluation
    ):
        raise ValueError("v205 prompts overlap the v189/v201 development suite")
    return source, prompt_path, prompts, source_indices


def prepare(
    v201_decision: Path,
    v201_input_manifest: Path,
    v201_published: Path,
    v201_comparison_manifest: Path,
    fresh_prompt_manifest: Path,
    output_root: Path,
) -> dict:
    decision, frozen, _published, _comparison, selected = validate_v201(
        v201_decision,
        v201_input_manifest,
        v201_published,
        v201_comparison_manifest,
    )
    source, source_prompt_path, prompts, source_indices = (
        validate_profile_disjoint_prompts(fresh_prompt_manifest, frozen)
    )

    prompt_path = output_root / "prompts" / "moviegen_profile_disjoint_0128_0255.txt"
    prompt_sha = write_frozen(
        prompt_path, ("\n".join(prompts) + "\n").encode("utf-8")
    )
    bank_hashes = {
        str(frozen["methods"][method]["head_bank_map_sha256"])
        for method in selected
    }
    if len(bank_hashes) != 1:
        raise ValueError("v201 selected candidates use different head-bank maps")
    bank_source = Path(frozen["methods"][selected[0]]["head_bank_map"])
    if not bank_source.is_file() or sha256(bank_source) not in bank_hashes:
        raise ValueError("v201 selected head-bank artifact drifted")
    bank_path = output_root / "maps" / "all_profile_banks.csv"
    bank_sha = write_frozen(bank_path, bank_source.read_bytes())

    methods: dict[str, dict] = {
        "sf_native": {
            "runtime": "sf_native",
            "role": "canonical_sf_baseline",
            "operator": None,
        }
    }
    for method in selected:
        row = frozen["methods"][method]
        operator = str(row["operator"])
        if operator not in OPERATORS:
            raise ValueError(f"unsupported v205 operator: {operator}")
        source_map = Path(row["horizon_map"])
        if not source_map.is_file() or sha256(source_map) != row["horizon_map_sha256"]:
            raise ValueError(f"v201 selected horizon map drifted: {method}")
        map_payload = load_json(source_map)
        validate_horizon_map(map_payload, operator=operator)
        map_path = output_root / "maps" / f"{method}.json"
        map_sha = write_frozen(map_path, source_map.read_bytes())
        methods[method] = {
            "runtime": "head_phase_horizon_cache_runtime",
            "role": "frozen_v201_horizon_candidate",
            "operator": operator,
            "history_policy": operator,
            "schedule": "head_phase_horizon",
            "horizon_map": str(map_path.resolve()),
            "horizon_map_sha256": map_sha,
            "routing_map_id": map_payload["map_id"],
            "map_classification": map_payload["classification"],
            "current_frames": map_payload["current_frames"],
            "coverage_count_by_position": map_payload[
                "coverage_count_by_position"
            ],
            "coverage_count_by_position_call": map_payload[
                "coverage_count_by_position_call"
            ],
            "coverage_exposure_count": int(
                sum(map_payload["coverage_count_by_position"])
            ),
            "coverage_exposure_fraction": row["coverage_exposure_fraction"],
            "head_bank_map": str(bank_path.resolve()),
            "head_bank_map_sha256": bank_sha,
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
            "v201_evidence_tier": (
                "interval_supported"
                if method in decision.get("sf_interval_supported_candidates", ())
                else "directional_screen"
            ),
            "v201_mechanism_supported": method
            in decision.get("mechanism_supported_candidates", ()),
        }

    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": "confirmatory_profile_disjoint128",
        "confirmatory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_source_indices": source_indices,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": index, "source_index": source_index, "text": prompt}
            for index, (source_index, prompt) in enumerate(zip(source_indices, prompts))
        ],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": source["decoded_video_contract"],
        "seed": SEED,
        "primary_baseline": "sf_native",
        "selected_v201_candidates": list(selected),
        "method_order": ["sf_native", *selected],
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + structured middle4 + recent4",
            "clean_read": "Recent for every cache candidate",
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
        "profile_disjoint_prompt_provenance": {
            "source_manifest": str(fresh_prompt_manifest.resolve()),
            "source_manifest_sha256": sha256(fresh_prompt_manifest),
            "source_prompt_file": str(source_prompt_path.resolve()),
            "source_prompt_file_sha256": sha256(source_prompt_path),
            "source_index_range": [128, 255],
            "development_exact_text_overlap": 0,
        },
        "v201_provenance": {
            "decision": str(v201_decision.resolve()),
            "decision_sha256": sha256(v201_decision),
            "input_manifest": str(v201_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v201_input_manifest),
            "published_manifest": str(v201_published.resolve()),
            "published_manifest_sha256": sha256(v201_published),
            "comparison_manifest": str(v201_comparison_manifest.resolve()),
            "comparison_manifest_sha256": sha256(v201_comparison_manifest),
            "recommendation": decision["recommendation"],
            "candidate_status": {
                method: decision["candidate_status"][method] for method in selected
            },
        },
        "manual_review_required_before_generation": False,
        "claim_boundary": (
            "v205 confirms only v201-selected Head x Denoising-Phase x AR-Horizon "
            "candidates against canonical Self-Forcing on 128 prompts excluded from "
            "v189 profiling and v201 screening. These prompts appeared in earlier "
            "project experiments, so this is not a project-history-unseen claim. PF "
            "is not reused as a benchmark, and transfer remains a separate claim."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    order = tuple(str(value) for value in payload.get("method_order") or ())
    selected = tuple(str(value) for value in payload.get("selected_v201_candidates") or ())
    if (
        payload.get("experiment") != EXPERIMENT
        or payload.get("scope") != "confirmatory_profile_disjoint128"
        or payload.get("confirmatory") is not True
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(payload.get("seed", -1)) != SEED
        or payload.get("primary_baseline") != "sf_native"
        or not 1 <= len(selected) <= MAX_CANDIDATES
        or order != ("sf_native", *selected)
        or set(payload.get("methods") or {}) != set(order)
    ):
        raise ValueError("invalid v205 frozen manifest")
    prompt_path = Path(payload["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        sha256(prompt_path) != payload["prompt_file_sha256"]
        or len(prompts) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in payload["prompt_items"]]
        or [int(row["source_index"]) for row in payload["prompt_items"]]
        != list(range(128, 256))
    ):
        raise ValueError("v205 prompt contract drifted")
    for method in selected:
        row = payload["methods"][method]
        map_path = Path(row["horizon_map"])
        bank_path = Path(row["head_bank_map"])
        if (
            row.get("runtime") != "head_phase_horizon_cache_runtime"
            or row.get("role") != "frozen_v201_horizon_candidate"
            or row.get("schedule") != "head_phase_horizon"
            or row.get("clean_policy") != "recent"
            or int(row.get("read_frame_equivalents", -1)) != 9
            or not map_path.is_file()
            or sha256(map_path) != row.get("horizon_map_sha256")
            or not bank_path.is_file()
            or sha256(bank_path) != row.get("head_bank_map_sha256")
        ):
            raise ValueError(f"v205 cache contract drifted: {method}")
        map_payload = load_json(map_path)
        validate_horizon_map(map_payload, operator=str(row["operator"]))
        if (
            map_payload["map_id"] != row["routing_map_id"]
            or map_payload["coverage_count_by_position"]
            != row["coverage_count_by_position"]
        ):
            raise ValueError(f"v205 route map drifted: {method}")
    for section, keys in (
        (
            "profile_disjoint_prompt_provenance",
            ("source_manifest", "source_prompt_file"),
        ),
        (
            "v201_provenance",
            ("decision", "input_manifest", "published_manifest", "comparison_manifest"),
        ),
    ):
        for key in keys:
            verify_path_hash(payload[section], key, label=f"manifest/{section}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v201-decision", type=Path, required=True)
    prepare_parser.add_argument("--v201-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v201-published", type=Path, required=True)
    prepare_parser.add_argument("--v201-comparison-manifest", type=Path, required=True)
    prepare_parser.add_argument("--fresh-prompt-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.v201_decision,
            args.v201_input_manifest,
            args.v201_published,
            args.v201_comparison_manifest,
            args.fresh_prompt_manifest,
            args.output_root,
        )
        print(
            "[v205-prepare] PASS "
            f"candidates={payload['selected_v201_candidates']} "
            f"methods={len(payload['method_order'])} prompts={PROMPT_COUNT}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v205-verify] PASS "
            f"candidates={payload['selected_v201_candidates']}"
        )


if __name__ == "__main__":
    main()
