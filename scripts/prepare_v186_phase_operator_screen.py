#!/usr/bin/env python3
"""Freeze the v186 deterministic Coverage-operator screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPT_COUNT = 32
SOURCE_PROMPT_COUNT = 128
REUSED_METHODS = ("all_recent", "phase_reservoir")
GENERATED_METHODS = (
    "phase_landmark",
    "phase_prototype",
    "phase_retrieval",
)
METHODS = REUSED_METHODS + GENERATED_METHODS
OPERATORS = {
    "all_recent": "recent",
    "phase_reservoir": "reservoir",
    "phase_landmark": "landmark",
    "phase_prototype": "prototype",
    "phase_retrieval": "retrieval",
}
HISTORY_POLICIES = {
    "phase_landmark": "landmark",
    "phase_prototype": "prototype",
    "phase_retrieval": "retrieval",
}
SOURCE_KINDS = {
    "phase_landmark": "semantic_landmark",
    "phase_prototype": "temporal_prototype",
    "phase_retrieval": "semantic_retrieval",
}
STORAGE_FFE = {
    "all_recent": 0,
    "phase_reservoir": 4,
    "phase_landmark": 4,
    "phase_prototype": 4,
    "phase_retrieval": 12,
}
V184_METHOD_SCHEDULES = {
    "coverage_early1": "early1",
    "coverage_early2": "early2",
    "coverage_late2": "late2",
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


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v186 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v184_decision(path: Path) -> tuple[dict, str, str]:
    decision = load_json(path)
    selected = decision.get("selected_for_operator_screen")
    promoted = tuple(decision.get("promoted_to_operator_screen") or ())
    if (
        decision.get("experiment")
        != "v184_denoise_phase_coverage_vbench_screen32"
        or decision.get("development_only") is not True
        or int(decision.get("prompt_count", -1)) != PROMPT_COUNT
        or decision.get("recommendation")
        != "advance_phase_schedule_to_operator_screen"
        or selected not in V184_METHOD_SCHEDULES
        or selected not in promoted
    ):
        raise ValueError(
            "v186 requires a complete v184 decision with one automatically "
            "selected phase schedule"
        )
    return decision, str(selected), V184_METHOD_SCHEDULES[str(selected)]


def _validate_v184_generation(
    published_path: Path,
    *,
    selected_method: str,
    source_prompts: Path,
) -> tuple[dict, dict, dict[str, dict]]:
    published = load_json(published_path)
    contract_path = Path(published.get("experiment_contract", ""))
    if not contract_path.is_file():
        raise ValueError("v184 generation contract is missing")
    contract = load_json(contract_path)
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v184_denoise_phase_coverage_generation"
        or published.get("scope") != "screen32"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != "screen32"
        or int(contract.get("prompt_count", -1)) != PROMPT_COUNT
        or contract.get("prompt_indices") != list(range(PROMPT_COUNT))
    ):
        raise ValueError("v184 audited generation provenance is invalid")
    prompt_items = contract.get("prompt_items") or ()
    if (
        len(prompt_items) != PROMPT_COUNT
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(PROMPT_COUNT))
    ):
        raise ValueError("v184 prompt-item contract is incomplete")
    prompt_path = Path(contract["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    source = source_prompts.read_text(encoding="utf-8").splitlines()
    source_indices = [int(row["source_index"]) for row in prompt_items]
    if (
        len(source) != SOURCE_PROMPT_COUNT
        or any(not row.strip() for row in source)
        or sha256(prompt_path) != contract["prompt_file_sha256"]
        or prompts != [str(row["text"]) for row in prompt_items]
        or prompts != [source[index].strip() for index in source_indices]
    ):
        raise ValueError("v184 prompt text or source provenance drifted")
    rows = {str(row["key"]): row for row in published.get("methods") or ()}
    if set(rows) != set(contract.get("methods") or ()):
        raise ValueError("v184 published method membership drifted")
    for key in ("all_recent", selected_method):
        row = rows.get(key)
        if row is None or row.get("ok") is not True:
            raise ValueError(f"v184 reusable method is unavailable: {key}")
        video_dir = Path(row["video_dir"])
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        if {item.name for item in video_dir.glob("*.mp4")} != expected:
            raise ValueError(f"v184 reusable videos are incomplete: {key}")
        audit_path = Path(row["audit"])
        if (
            not audit_path.is_file()
            or sha256(audit_path) != row["audit_sha256"]
        ):
            raise ValueError(f"v184 reusable audit drifted: {key}")
    return published, contract, rows


def _head_map_payload() -> bytes:
    return (
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30))
        + "\n"
    ).encode("ascii")


def prepare(
    source_prompts: Path,
    v184_decision: Path,
    v184_published: Path,
    output_root: Path,
) -> dict:
    decision, selected_method, schedule = _validate_v184_decision(v184_decision)
    published, contract, v184_rows = _validate_v184_generation(
        v184_published,
        selected_method=selected_method,
        source_prompts=source_prompts,
    )
    prompt_items = list(contract["prompt_items"])
    prompts = [str(row["text"]) for row in prompt_items]
    prompt_path = output_root / "prompts" / "moviegen_qwen_systematic32.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(prompts) + "\n").encode("utf-8"),
    )
    map_path = output_root / "maps" / "all_profile_banks.csv"
    map_sha = write_frozen(map_path, _head_map_payload())

    methods: dict[str, dict] = {}
    for method in METHODS:
        row = {
            "operator": OPERATORS[method],
            "schedule": "recent" if method == "all_recent" else schedule,
            "coverage_noisy_calls": (
                []
                if method == "all_recent"
                else list(SCHEDULE_CALLS[schedule])
            ),
            "read_frame_equivalents": 9,
            "middle_read_capacity": 0 if method == "all_recent" else 4,
            "middle_storage_capacity": STORAGE_FFE[method],
            "clean_policy": "recent",
        }
        if method in REUSED_METHODS:
            source_key = "all_recent" if method == "all_recent" else selected_method
            source_row = v184_rows[source_key]
            row.update(
                {
                    "execution": "reuse_v184_audited_video",
                    "source_method": source_key,
                    "source_video_dir": source_row["video_dir"],
                    "source_audit": source_row["audit"],
                    "source_audit_sha256": source_row["audit_sha256"],
                    "source_video_sha256": {
                        f"{index:06d}.mp4": sha256(
                            Path(source_row["video_dir"]) / f"{index:06d}.mp4"
                        )
                        for index in range(PROMPT_COUNT)
                    },
                }
            )
        else:
            row.update(
                {
                    "execution": "generate_v186",
                    "history_policy": HISTORY_POLICIES[method],
                    "expected_middle_source_kind": SOURCE_KINDS[method],
                    "head_map": str(map_path.resolve()),
                    "head_map_sha256": map_sha,
                    "head_route_counts": {"10": 360, "11": 0},
                }
            )
        methods[method] = row

    payload = {
        "version": 1,
        "experiment": "v186_phase_conditioned_operator_screen",
        "scope": "development32",
        "claim_boundary": (
            "This screen compares deterministic four-frame Coverage operators "
            "under one schedule selected by v184. It is not a final benchmark "
            "or a head-specialization claim."
        ),
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": SOURCE_PROMPT_COUNT,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "source_indices": [int(row["source_index"]) for row in prompt_items],
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": prompt_items,
        "num_output_frames": 120,
        "decoded_video_contract": contract["decoded_video_contract"],
        "seed": 0,
        "selected_v184_method": selected_method,
        "selected_schedule": schedule,
        "method_order": list(METHODS),
        "generated_methods": list(GENERATED_METHODS),
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + exactly one middle4 operator + recent4",
            "clean_read": "Recent for every method",
            "shared_clean_updates": True,
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
        "v184_provenance": {
            "decision": str(v184_decision.resolve()),
            "decision_sha256": sha256(v184_decision),
            "published_manifest": str(v184_published.resolve()),
            "published_manifest_sha256": sha256(v184_published),
            "generation_contract": published["experiment_contract"],
            "generation_contract_sha256": published[
                "experiment_contract_sha256"
            ],
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
        payload.get("experiment") != "v186_phase_conditioned_operator_screen"
        or payload.get("scope") != "development32"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(payload.get("method_order") or ()) != METHODS
        or tuple(payload.get("generated_methods") or ()) != GENERATED_METHODS
        or payload.get("selected_schedule") not in {"early1", "early2", "late2"}
    ):
        raise ValueError("invalid v186 manifest")
    if set(payload.get("methods") or {}) != set(METHODS):
        raise ValueError("v186 method membership drifted")
    prompt_path = Path(payload["prompt_file"])
    if (
        not prompt_path.is_file()
        or sha256(prompt_path) != payload["prompt_file_sha256"]
        or len(prompt_path.read_text(encoding="utf-8").splitlines())
        != PROMPT_COUNT
    ):
        raise ValueError("v186 prompt file drifted")
    source_path = Path(payload["source_prompt_file"])
    if sha256(source_path) != payload["source_prompt_file_sha256"]:
        raise ValueError("v186 source prompt file drifted")
    provenance = payload["v184_provenance"]
    for key in ("decision", "published_manifest", "generation_contract"):
        path = Path(provenance[key])
        if not path.is_file() or sha256(path) != provenance[f"{key}_sha256"]:
            raise ValueError(f"v186 upstream provenance drifted: {key}")
    for method in GENERATED_METHODS:
        row = payload["methods"][method]
        map_path = Path(row["head_map"])
        if (
            row.get("execution") != "generate_v186"
            or row.get("operator") != OPERATORS[method]
            or row.get("history_policy") != HISTORY_POLICIES[method]
            or row.get("expected_middle_source_kind") != SOURCE_KINDS[method]
            or row.get("head_route_counts") != {"10": 360, "11": 0}
            or not map_path.is_file()
            or sha256(map_path) != row["head_map_sha256"]
            or map_path.read_bytes() != _head_map_payload()
        ):
            raise ValueError(f"v186 generated method contract drifted: {method}")
    for method in REUSED_METHODS:
        row = payload["methods"][method]
        video_dir = Path(row["source_video_dir"])
        expected_hashes = row.get("source_video_sha256") or {}
        if set(expected_hashes) != {
            f"{index:06d}.mp4" for index in range(PROMPT_COUNT)
        }:
            raise ValueError(f"v186 reused video hash set drifted: {method}")
        for name, digest in expected_hashes.items():
            path = video_dir / name
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"v186 reused video drifted: {method}/{name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--v184-decision", type=Path, required=True)
    prepare_parser.add_argument("--v184-published", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(
            args.source_prompts,
            args.v184_decision,
            args.v184_published,
            args.output_root,
        )
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v186-inputs] PASS "
        f"schedule={payload['selected_schedule']} "
        f"generated={len(payload['generated_methods'])} "
        f"reused={len(REUSED_METHODS)} prompts={payload['prompt_count']}"
    )


if __name__ == "__main__":
    main()
