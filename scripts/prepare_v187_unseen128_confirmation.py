#!/usr/bin/env python3
"""Freeze the v187 unseen-prompt confirmation after v184/v186 promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPT_COUNT = 128
SEED = 10000
METHODS = (
    "sf_native",
    "all_recent",
    "phase_reservoir",
    "phase_deterministic",
)
SCHEDULED_METHODS = METHODS[1:]
SELECTED_OPERATORS = {
    "phase_landmark": {
        "operator": "landmark",
        "history_policy": "landmark",
        "source_kind": "semantic_landmark",
        "storage_ffe": 4,
    },
    "phase_prototype": {
        "operator": "prototype",
        "history_policy": "prototype",
        "source_kind": "temporal_prototype",
        "storage_ffe": 4,
    },
    "phase_retrieval": {
        "operator": "retrieval",
        "history_policy": "retrieval",
        "source_kind": "semantic_retrieval",
        "storage_ffe": 12,
    },
}
SCHEDULE_CALLS = {
    "early1": (0,),
    "early2": (0, 1),
    "late2": (2, 3),
}


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
        raise RuntimeError(f"frozen v187 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _validate_v186(
    decision_path: Path,
    input_manifest_path: Path,
    published_path: Path,
) -> tuple[dict, dict, dict, str, str, dict]:
    decision = load_json(decision_path)
    selected = decision.get("selected_for_fresh128")
    promoted = tuple(decision.get("promoted_to_fresh128") or ())
    if (
        decision.get("experiment")
        != "v186_phase_conditioned_operator_vbench_screen32"
        or decision.get("development_only") is not True
        or int(decision.get("prompt_count", -1)) != 32
        or decision.get("recommendation")
        != "advance_deterministic_operator_to_fresh128"
        or selected not in SELECTED_OPERATORS
        or selected not in promoted
    ):
        raise ValueError(
            "v187 requires a complete v186 decision with one automatically "
            "selected deterministic operator"
        )

    frozen = load_json(input_manifest_path)
    schedule = str(frozen.get("selected_schedule", ""))
    selected_row = (frozen.get("methods") or {}).get(str(selected), {})
    operator_contract = SELECTED_OPERATORS[str(selected)]
    if (
        frozen.get("experiment") != "v186_phase_conditioned_operator_screen"
        or frozen.get("scope") != "development32"
        or int(frozen.get("prompt_count", -1)) != 32
        or schedule not in SCHEDULE_CALLS
        or decision.get("selected_schedule") != schedule
        or selected_row.get("operator") != operator_contract["operator"]
        or selected_row.get("history_policy")
        != operator_contract["history_policy"]
        or selected_row.get("expected_middle_source_kind")
        != operator_contract["source_kind"]
    ):
        raise ValueError("v186 decision and frozen operator contract disagree")

    published = load_json(published_path)
    contract_path = Path(published.get("experiment_contract", ""))
    if not contract_path.is_file():
        raise ValueError("v186 generation contract is missing")
    generation_contract = load_json(contract_path)
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    selected_evidence = rows.get(str(selected))
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v186_phase_conditioned_operator_generation"
        or published.get("scope") != "screen32"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or generation_contract.get("scope") != "screen32"
        or int(generation_contract.get("prompt_count", -1)) != 32
        or generation_contract.get("input_manifest_sha256")
        != sha256(input_manifest_path)
        or selected_evidence is None
        or selected_evidence.get("ok") is not True
    ):
        raise ValueError("v186 selected-operator generation evidence is invalid")
    audit_path = Path(selected_evidence.get("audit", ""))
    if (
        not audit_path.is_file()
        or sha256(audit_path) != selected_evidence.get("audit_sha256")
    ):
        raise ValueError("v186 selected-operator audit drifted")
    return (
        decision,
        frozen,
        published,
        str(selected),
        schedule,
        selected_evidence,
    )


def _validate_fresh_prompts(
    source_manifest_path: Path,
    v186_input: dict,
) -> tuple[dict, Path, list[str], list[int]]:
    source_manifest = load_json(source_manifest_path)
    prompt_path = Path(source_manifest.get("prompt_file", ""))
    source_indices = [
        int(value) for value in source_manifest.get("prompt_source_indices") or ()
    ]
    if (
        source_manifest.get("experiment") != "v180_rccp_fresh128_inputs"
        or int(source_manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or source_manifest.get("evaluation_source_index_range") != [128, 255]
        or source_manifest.get("evaluation_prompts_used_for_membership") is not False
        or int(source_manifest.get("exact_text_overlap_with_calibration", -1)) != 0
        or source_indices != list(range(128, 256))
        or not prompt_path.is_file()
        or sha256(prompt_path) != source_manifest.get("prompt_file_sha256")
    ):
        raise ValueError("v180 unseen-prompt source contract is invalid")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not prompt.strip() for prompt in prompts):
        raise ValueError("v187 requires exactly 128 non-empty unseen prompts")

    development = {
        normalized_text(row.get("text", ""))
        for row in v186_input.get("prompt_items") or ()
    }
    unseen = {normalized_text(prompt) for prompt in prompts}
    if "" in development or len(development) != 32 or development & unseen:
        raise ValueError("v187 unseen prompts overlap the v186 development suite")
    return source_manifest, prompt_path, prompts, source_indices


def _head_map_payload() -> bytes:
    return (
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30))
        + "\n"
    ).encode("ascii")


def prepare(
    v186_decision: Path,
    v186_input_manifest: Path,
    v186_published: Path,
    fresh_prompt_manifest: Path,
    output_root: Path,
) -> dict:
    (
        decision,
        v186_input,
        published,
        selected,
        schedule,
        selected_evidence,
    ) = _validate_v186(v186_decision, v186_input_manifest, v186_published)
    source_manifest, source_prompt_path, prompts, source_indices = (
        _validate_fresh_prompts(fresh_prompt_manifest, v186_input)
    )

    prompt_path = output_root / "prompts" / "moviegen_unseen_0128_0255.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(prompts) + "\n").encode("utf-8"),
    )
    map_path = output_root / "maps" / "all_profile_banks.csv"
    map_sha = write_frozen(map_path, _head_map_payload())
    selected_contract = SELECTED_OPERATORS[selected]

    methods = {
        "sf_native": {
            "execution": "generate_v187",
            "runtime": "self_forcing_native",
            "role": "native_self_forcing_baseline",
        },
        "all_recent": {
            "execution": "generate_v187",
            "runtime": "phase_cache_runtime",
            "role": "equal_budget_local_control",
            "schedule": "recent",
            "operator": "reservoir",
            "history_policy": "reservoir4_multiscalemotion1",
            "expected_middle_source_kind": "temporal_reservoir",
            "coverage_noisy_calls": [],
            "middle_read_capacity": 0,
            "middle_storage_capacity": 4,
        },
        "phase_reservoir": {
            "execution": "generate_v187",
            "runtime": "phase_cache_runtime",
            "role": "random_coverage_reference",
            "schedule": schedule,
            "operator": "reservoir",
            "history_policy": "reservoir4_multiscalemotion1",
            "expected_middle_source_kind": "temporal_reservoir",
            "coverage_noisy_calls": list(SCHEDULE_CALLS[schedule]),
            "middle_read_capacity": 4,
            "middle_storage_capacity": 4,
        },
        "phase_deterministic": {
            "execution": "generate_v187",
            "runtime": "phase_cache_runtime",
            "role": "selected_deterministic_candidate",
            "source_v186_method": selected,
            "schedule": schedule,
            "operator": selected_contract["operator"],
            "history_policy": selected_contract["history_policy"],
            "expected_middle_source_kind": selected_contract["source_kind"],
            "coverage_noisy_calls": list(SCHEDULE_CALLS[schedule]),
            "middle_read_capacity": 4,
            "middle_storage_capacity": selected_contract["storage_ffe"],
        },
    }
    for method in SCHEDULED_METHODS:
        methods[method].update(
            {
                "head_map": str(map_path.resolve()),
                "head_map_sha256": map_sha,
                "head_route_counts": {"10": 360, "11": 0},
                "read_frame_equivalents": 9,
                "clean_policy": "recent",
            }
        )

    prompt_items = [
        {"index": index, "source_index": source_index, "text": prompt}
        for index, (source_index, prompt) in enumerate(zip(source_indices, prompts))
    ]
    selected_audit = Path(selected_evidence["audit"])
    generation_contract = Path(published["experiment_contract"])
    payload = {
        "version": 1,
        "experiment": "v187_unseen128_phase_operator_confirmation",
        "scope": "confirmatory_unseen128",
        "confirmatory": True,
        "claim_boundary": (
            "This run tests the frozen v184 schedule and v186 deterministic "
            "operator on 128 prompts that were absent from both development "
            "stages. It is single-model evidence, not cross-model validation."
        ),
        "prompt_count": PROMPT_COUNT,
        "prompt_source_indices": source_indices,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": prompt_items,
        "num_output_frames": 120,
        "decoded_video_contract": source_manifest["decoded_video_contract"],
        "seed": SEED,
        "selected_v186_method": selected,
        "selected_schedule": schedule,
        "selected_operator": selected_contract["operator"],
        "method_order": list(METHODS),
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + exactly one middle4 operator + recent4",
            "clean_read": "Recent for every scheduled method",
            "shared_clean_updates": True,
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
        "unseen_prompt_provenance": {
            "source_manifest": str(fresh_prompt_manifest.resolve()),
            "source_manifest_sha256": sha256(fresh_prompt_manifest),
            "source_prompt_file": str(source_prompt_path.resolve()),
            "source_prompt_file_sha256": sha256(source_prompt_path),
            "development_exact_text_overlap": 0,
        },
        "v186_provenance": {
            "decision": str(v186_decision.resolve()),
            "decision_sha256": sha256(v186_decision),
            "input_manifest": str(v186_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v186_input_manifest),
            "published_manifest": str(v186_published.resolve()),
            "published_manifest_sha256": sha256(v186_published),
            "generation_contract": str(generation_contract.resolve()),
            "generation_contract_sha256": sha256(generation_contract),
            "selected_operator_audit": str(selected_audit.resolve()),
            "selected_operator_audit_sha256": sha256(selected_audit),
            "decision_selection_rule": decision.get("selection_rule"),
            "selected_candidate_status": (
                decision.get("candidate_status") or {}
            ).get(selected),
        },
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment")
        != "v187_unseen128_phase_operator_confirmation"
        or payload.get("scope") != "confirmatory_unseen128"
        or payload.get("confirmatory") is not True
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
        or tuple(payload.get("method_order") or ()) != METHODS
        or payload.get("selected_schedule") not in SCHEDULE_CALLS
        or payload.get("selected_v186_method") not in SELECTED_OPERATORS
    ):
        raise ValueError("invalid v187 manifest")
    if set(payload.get("methods") or {}) != set(METHODS):
        raise ValueError("v187 method membership drifted")

    prompt_path = Path(payload["prompt_file"])
    if (
        not prompt_path.is_file()
        or sha256(prompt_path) != payload["prompt_file_sha256"]
        or len(prompt_path.read_text(encoding="utf-8").splitlines())
        != PROMPT_COUNT
    ):
        raise ValueError("v187 prompt file drifted")
    prompt_items = payload.get("prompt_items") or ()
    if (
        len(prompt_items) != PROMPT_COUNT
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(PROMPT_COUNT))
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != list(range(128, 256))
    ):
        raise ValueError("v187 prompt-item provenance drifted")

    prompt_provenance = payload["unseen_prompt_provenance"]
    for key in ("source_manifest", "source_prompt_file"):
        path = Path(prompt_provenance[key])
        if not path.is_file() or sha256(path) != prompt_provenance[f"{key}_sha256"]:
            raise ValueError(f"v187 unseen prompt provenance drifted: {key}")
    v186_provenance = payload["v186_provenance"]
    for key in (
        "decision",
        "input_manifest",
        "published_manifest",
        "generation_contract",
        "selected_operator_audit",
    ):
        path = Path(v186_provenance[key])
        if not path.is_file() or sha256(path) != v186_provenance[f"{key}_sha256"]:
            raise ValueError(f"v187 upstream provenance drifted: {key}")

    map_payload = _head_map_payload()
    for method in SCHEDULED_METHODS:
        row = payload["methods"][method]
        map_path = Path(row["head_map"])
        if (
            row.get("execution") != "generate_v187"
            or row.get("head_route_counts") != {"10": 360, "11": 0}
            or int(row.get("read_frame_equivalents", -1)) != 9
            or not map_path.is_file()
            or sha256(map_path) != row["head_map_sha256"]
            or map_path.read_bytes() != map_payload
        ):
            raise ValueError(f"v187 scheduled method contract drifted: {method}")
    selected = SELECTED_OPERATORS[payload["selected_v186_method"]]
    deterministic = payload["methods"]["phase_deterministic"]
    if (
        deterministic.get("operator") != selected["operator"]
        or deterministic.get("history_policy") != selected["history_policy"]
        or deterministic.get("expected_middle_source_kind")
        != selected["source_kind"]
        or payload.get("selected_operator") != selected["operator"]
    ):
        raise ValueError("v187 selected deterministic operator drifted")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v186-decision", type=Path, required=True)
    prepare_parser.add_argument("--v186-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v186-published", type=Path, required=True)
    prepare_parser.add_argument("--fresh-prompt-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(
            args.v186_decision,
            args.v186_input_manifest,
            args.v186_published,
            args.fresh_prompt_manifest,
            args.output_root,
        )
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v187-inputs] PASS "
        f"schedule={payload['selected_schedule']} "
        f"operator={payload['selected_operator']} "
        f"methods={len(METHODS)} prompts={payload['prompt_count']} "
        f"seed={payload['seed']}"
    )


if __name__ == "__main__":
    main()
