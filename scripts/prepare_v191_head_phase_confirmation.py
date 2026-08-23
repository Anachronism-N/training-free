#!/usr/bin/env python3
"""Freeze the unseen-128 confirmation for a passing v190 Head x Phase map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v190_head_phase_causal_screen import all_recent_map, validate_map


PROMPT_COUNT = 128
SEED = 10000
METHODS = ("sf_native", "all_recent", "head_phase_joint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v191 input differs: {path}")
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
    raise FileNotFoundError(f"missing declared and fallback file: {declared}, {fallback}")


def validate_v190(
    decision_path: Path,
    input_manifest_path: Path,
    published_path: Path,
    comparison_manifest_path: Path,
) -> tuple[dict, dict, dict, dict, str, dict]:
    decision = load_json(decision_path)
    frozen = load_json(input_manifest_path)
    published = load_json(published_path)
    comparison = load_json(comparison_manifest_path)
    selected = str(decision.get("selected_for_fresh128") or "")
    status = (decision.get("statuses") or {}).get(selected, {})
    source = decision.get("source") or {}
    if (
        int(decision.get("version", -1)) < 6
        or decision.get("experiment")
        != "v190_head_phase_causal_vbench_screen32"
        or decision.get("development_only") is not True
        or decision.get("recommendation")
        != "advance_head_phase_method_to_fresh128"
        or selected not in (decision.get("passing_methods") or ())
        or status.get("full_screen_pass") is not True
        or status.get("joint_factorization_pass") is not True
        or status.get("head_phase_attribution_pass") is not True
        or status.get("selective_exposure_pass") is not True
        or decision.get("temporal_diagnostics_available") is not True
        or source.get("comparison_manifest_sha256")
        != sha256(comparison_manifest_path)
    ):
        raise ValueError("v191 requires a SHA-bound, passing v190 decision")
    for key in (
        "vbench_summary",
        "temporal_diagnostics",
        "temporal_contract",
    ):
        path = Path(str(source.get(key, "")))
        if not path.is_file() or sha256(path) != source.get(f"{key}_sha256"):
            raise ValueError(f"v191 source evidence drifted: {key}")

    methods = frozen.get("methods") or {}
    selected_row = methods.get(selected, {})
    if (
        frozen.get("experiment") != "v190_head_phase_causal_screen"
        or frozen.get("scope") != "classifier_holdout32"
        or int(frozen.get("prompt_count", -1)) != 32
        or selected_row.get("role") != "primary_head_phase"
        or selected_row.get("schedule") != "head_phase"
        or selected_row.get("clean_policy") != "recent"
        or int(selected_row.get("read_frame_equivalents", -1)) != 9
        or selected not in frozen.get("method_order", ())
    ):
        raise ValueError("v190 selected map and input contract disagree")

    comparison_methods = {
        str(row.get("key")): row for row in comparison.get("methods") or ()
    }
    if (
        comparison.get("experiment")
        != "v190_head_phase_causal_vbench_screen32"
        or int(comparison.get("prompt_count", -1)) != 32
        or selected not in comparison_methods
        or tuple(decision.get("methods") or ())
        != tuple(row.get("key") for row in comparison.get("methods") or ())
    ):
        raise ValueError("v190 decision and VBench comparison disagree")

    published_rows = {
        str(row.get("key")): row for row in published.get("methods") or ()
    }
    contract_path = Path(published.get("experiment_contract", ""))
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v190_head_phase_causal_generation"
        or published.get("scope") != "screen32"
        or selected not in published_rows
        or "all_recent" not in published_rows
        or published_rows[selected].get("ok") is not True
        or published_rows["all_recent"].get("ok") is not True
        or not contract_path.is_file()
        or published.get("experiment_contract_sha256") != sha256(contract_path)
    ):
        raise ValueError("v190 selected generation evidence is incomplete")
    for method in ("all_recent", selected):
        audit_path = Path(published_rows[method].get("audit", ""))
        if (
            not audit_path.is_file()
            or sha256(audit_path) != published_rows[method].get("audit_sha256")
        ):
            raise ValueError(f"v190 source audit drifted: {method}")

    map_path = Path(selected_row.get("head_phase_map", ""))
    bank_path = Path(selected_row.get("head_bank_map", ""))
    if (
        not map_path.is_file()
        or sha256(map_path) != selected_row.get("head_phase_map_sha256")
        or not bank_path.is_file()
        or sha256(bank_path) != selected_row.get("head_bank_map_sha256")
    ):
        raise ValueError("v190 selected map artifacts drifted")
    selected_map = load_json(map_path)
    validate_map(selected_map, operator=str(selected_row["operator"]))
    return decision, frozen, published, comparison, selected, selected_map


def validate_fresh_prompts(
    source_manifest_path: Path,
    v190_input: dict,
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
        raise ValueError("v191 unseen prompt source contract is invalid")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not prompt.strip() for prompt in prompts):
        raise ValueError("v191 requires 128 non-empty unseen prompts")

    development_path = Path(v190_input.get("source_prompt_file", ""))
    if not development_path.is_file():
        raise ValueError("v191 cannot locate the complete v189 development prompt suite")
    development = {
        normalized_text(prompt)
        for prompt in development_path.read_text(encoding="utf-8").splitlines()
    }
    unseen = {normalized_text(prompt) for prompt in prompts}
    if (
        len(development) != 128
        or len(unseen) != PROMPT_COUNT
        or "" in development
        or development & unseen
    ):
        raise ValueError("v191 prompts overlap the v189/v190 development suite")
    return source, prompt_path, prompts, source_indices


def prepare(
    v190_decision: Path,
    v190_input_manifest: Path,
    v190_published: Path,
    v190_comparison_manifest: Path,
    fresh_prompt_manifest: Path,
    output_root: Path,
) -> dict:
    decision, v190_input, published, comparison, selected, selected_map = validate_v190(
        v190_decision,
        v190_input_manifest,
        v190_published,
        v190_comparison_manifest,
    )
    source, source_prompt_path, prompts, source_indices = validate_fresh_prompts(
        fresh_prompt_manifest,
        v190_input,
    )
    selected_row = v190_input["methods"][selected]
    operator = str(selected_row["operator"])

    prompt_path = output_root / "prompts" / "moviegen_unseen_0128_0255.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(prompts) + "\n").encode("utf-8"),
    )
    bank_source = Path(selected_row["head_bank_map"])
    bank_path = output_root / "maps" / "all_profile_banks.csv"
    bank_sha = write_frozen(bank_path, bank_source.read_bytes())

    recent_payload = all_recent_map(operator)
    recent_path = output_root / "maps" / "all_recent.json"
    recent_sha = write_frozen(
        recent_path,
        (json.dumps(recent_payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    joint_path = output_root / "maps" / "head_phase_joint.json"
    joint_sha = write_frozen(
        joint_path,
        (json.dumps(selected_map, indent=2, sort_keys=True) + "\n").encode(),
    )

    def cache_method(role: str, map_path: Path, map_sha: str, map_payload: dict) -> dict:
        return {
            "runtime": "head_phase_cache_runtime",
            "role": role,
            "schedule": "head_phase",
            "operator": operator,
            "history_policy": operator,
            "head_phase_map": str(map_path.resolve()),
            "head_phase_map_sha256": map_sha,
            "phase_map_id": map_payload["map_id"],
            "coverage_count_by_call": map_payload["coverage_count_by_call"],
            "coverage_cell_count": int(sum(map_payload["coverage_count_by_call"])),
            "head_bank_map": str(bank_path.resolve()),
            "head_bank_map_sha256": bank_sha,
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }

    methods = {
        "sf_native": {
            "runtime": "self_forcing_native",
            "role": "native_self_forcing_baseline",
        },
        "all_recent": cache_method(
            "equal_budget_local_control", recent_path, recent_sha, recent_payload
        ),
        "head_phase_joint": cache_method(
            "frozen_joint_head_phase_candidate", joint_path, joint_sha, selected_map
        ),
    }
    payload = {
        "version": 1,
        "experiment": "v191_unseen128_head_phase_confirmation",
        "scope": "confirmatory_unseen128",
        "confirmatory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_source_indices": source_indices,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": index, "source_index": source_index, "text": prompt}
            for index, (source_index, prompt) in enumerate(zip(source_indices, prompts))
        ],
        "num_output_frames": 120,
        "decoded_video_contract": source["decoded_video_contract"],
        "seed": SEED,
        "selected_v190_method": selected,
        "selected_operator": operator,
        "method_order": list(METHODS),
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + structured middle4 + recent4",
            "clean_read": "Recent for both cache methods",
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
        "unseen_prompt_provenance": {
            "source_manifest": str(fresh_prompt_manifest.resolve()),
            "source_manifest_sha256": sha256(fresh_prompt_manifest),
            "source_prompt_file": str(source_prompt_path.resolve()),
            "source_prompt_file_sha256": sha256(source_prompt_path),
            "source_index_range": [128, 255],
            "development_exact_text_overlap": 0,
        },
        "v190_provenance": {
            "decision": str(v190_decision.resolve()),
            "decision_sha256": sha256(v190_decision),
            "input_manifest": str(v190_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v190_input_manifest),
            "published_manifest": str(v190_published.resolve()),
            "published_manifest_sha256": sha256(v190_published),
            "comparison_manifest": str(v190_comparison_manifest.resolve()),
            "comparison_manifest_sha256": sha256(v190_comparison_manifest),
            "selected_status": decision["statuses"][selected],
            "dynamic_metric_validity": decision["metric_validity"]["dynamic_degree"],
        },
        "claim_boundary": (
            "This frozen single-model confirmation estimates the effect of a v190-selected "
            "joint Head x Denoising-Phase map on 128 prompts excluded from map fitting and "
            "causal screening. Cross-model and longer-duration transfer are separate claims."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment") != "v191_unseen128_head_phase_confirmation"
        or payload.get("scope") != "confirmatory_unseen128"
        or payload.get("confirmatory") is not True
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
        or tuple(payload.get("method_order") or ()) != METHODS
        or set(payload.get("methods") or {}) != set(METHODS)
    ):
        raise ValueError("invalid v191 manifest")
    prompt_path = Path(payload["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        sha256(prompt_path) != payload["prompt_file_sha256"]
        or len(prompts) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in payload["prompt_items"]]
        or [int(row["source_index"]) for row in payload["prompt_items"]]
        != list(range(128, 256))
    ):
        raise ValueError("v191 prompt contract drifted")
    for method in METHODS[1:]:
        row = payload["methods"][method]
        map_path = Path(row["head_phase_map"])
        bank_path = Path(row["head_bank_map"])
        if (
            sha256(map_path) != row["head_phase_map_sha256"]
            or sha256(bank_path) != row["head_bank_map_sha256"]
            or int(row.get("read_frame_equivalents", -1)) != 9
            or row.get("clean_policy") != "recent"
        ):
            raise ValueError(f"v191 cache contract drifted: {method}")
        map_payload = load_json(map_path)
        validate_map(map_payload, operator=str(row["operator"]))
        if (
            map_payload["map_id"] != row["phase_map_id"]
            or map_payload["coverage_count_by_call"]
            != row["coverage_count_by_call"]
        ):
            raise ValueError(f"v191 route map drifted: {method}")
    for section, keys in (
        ("unseen_prompt_provenance", ("source_manifest", "source_prompt_file")),
        (
            "v190_provenance",
            ("decision", "input_manifest", "published_manifest", "comparison_manifest"),
        ),
    ):
        for key in keys:
            path = Path(payload[section][key])
            if not path.is_file() or sha256(path) != payload[section][f"{key}_sha256"]:
                raise ValueError(f"v191 provenance drifted: {section}/{key}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v190-decision", type=Path, required=True)
    prepare_parser.add_argument("--v190-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v190-published", type=Path, required=True)
    prepare_parser.add_argument("--v190-comparison-manifest", type=Path, required=True)
    prepare_parser.add_argument("--fresh-prompt-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.v190_decision,
            args.v190_input_manifest,
            args.v190_published,
            args.v190_comparison_manifest,
            args.fresh_prompt_manifest,
            args.output_root,
        )
        print(
            "[v191-prepare] "
            f"selected={payload['selected_v190_method']} "
            f"operator={payload['selected_operator']} prompts={payload['prompt_count']}"
        )
    else:
        payload = verify(args.manifest)
        print(f"[v191-verify] PASS methods={len(payload['method_order'])}")


if __name__ == "__main__":
    main()
