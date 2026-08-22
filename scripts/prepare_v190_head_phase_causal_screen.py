#!/usr/bin/env python3
"""Freeze classifier-holdout controls for the v190 Head x Phase screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPT_COUNT = 32
CALLS = 4
LAYERS = 30
HEADS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v190 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def validate_map(payload: dict, *, operator: str | None = None) -> None:
    masks = payload.get("coverage_masks")
    if (
        payload.get("version") != 1
        or int(payload.get("call_count", -1)) != CALLS
        or int(payload.get("layer_count", -1)) != LAYERS
        or int(payload.get("head_count", -1)) != HEADS
        or not isinstance(masks, list)
        or len(masks) != CALLS
    ):
        raise ValueError("invalid v190 head-phase map contract")
    for call_rows in masks:
        if not isinstance(call_rows, list) or len(call_rows) != LAYERS:
            raise ValueError("invalid v190 map layer shape")
        for row in call_rows:
            if (
                not isinstance(row, list)
                or len(row) != HEADS
                or any(type(value) is not bool for value in row)
            ):
                raise ValueError("invalid v190 map head mask")
    if operator is not None and payload.get("coverage_operator") != operator:
        raise ValueError("v190 map Coverage operator drift")
    counts = [sum(value for row in call_rows for value in row) for call_rows in masks]
    if payload.get("coverage_count_by_call") != counts:
        raise ValueError("v190 map declared counts drift")


def _new_map(
    masks: list[list[list[bool]]],
    *,
    operator: str,
    classification: str,
    parent_map_id: str | None,
) -> dict:
    counts = [sum(value for row in call_rows for value in row) for call_rows in masks]
    payload = {
        "version": 1,
        "experiment": "v190_head_phase_causal_screen",
        "classification": classification,
        "coverage_operator": operator,
        "call_count": CALLS,
        "layer_count": LAYERS,
        "head_count": HEADS,
        "coverage_masks": masks,
        "coverage_count_by_call": counts,
        "parent_map_id": parent_map_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["map_id"] = (
        f"v190-{operator}-{classification}-"
        f"{hashlib.sha256(canonical).hexdigest()[:12]}"
    )
    return payload


def all_recent_map(operator: str) -> dict:
    return _new_map(
        [[[False for _ in range(HEADS)] for _ in range(LAYERS)] for _ in range(CALLS)],
        operator=operator,
        classification="all_recent",
        parent_map_id=None,
    )


def all_coverage_map(operator: str) -> dict:
    return _new_map(
        [[[True for _ in range(HEADS)] for _ in range(LAYERS)] for _ in range(CALLS)],
        operator=operator,
        classification="all_coverage",
        parent_map_id=None,
    )


def membership_shift_map(primary: dict) -> dict:
    masks = primary["coverage_masks"]
    shifted = [
        [[False for _ in range(HEADS)] for _ in range(LAYERS)]
        for _ in range(CALLS)
    ]
    for call in range(CALLS):
        for layer in range(LAYERS):
            selected = {head for head, value in enumerate(masks[call][layer]) if value}
            if not selected or len(selected) == HEADS:
                replacement = selected
            else:
                candidates = [
                    {(head + offset) % HEADS for head in selected}
                    for offset in range(1, HEADS)
                ]
                replacement = min(
                    candidates,
                    key=lambda values: (len(values & selected), sorted(values)),
                )
            for head in replacement:
                shifted[call][layer][head] = True
    return _new_map(
        shifted,
        operator=str(primary["coverage_operator"]),
        classification="layer_count_matched_membership_shift",
        parent_map_id=str(primary["map_id"]),
    )


def phase_shift_map(primary: dict) -> dict:
    masks = primary["coverage_masks"]
    shifted = [masks[(call - 1) % CALLS] for call in range(CALLS)]
    return _new_map(
        shifted,
        operator=str(primary["coverage_operator"]),
        classification="cyclic_phase_shift",
        parent_map_id=str(primary["map_id"]),
    )


def dense_phase_map(primary: dict) -> dict:
    masks = primary["coverage_masks"]
    active_cells = [
        [any(masks[call][layer]) for layer in range(LAYERS)]
        for call in range(CALLS)
    ]
    dense = [
        [
            [active_cells[call][layer] for _ in range(HEADS)]
            for layer in range(LAYERS)
        ]
        for call in range(CALLS)
    ]
    return _new_map(
        dense,
        operator=str(primary["coverage_operator"]),
        classification="same_active_call_layer_cells_all_heads",
        parent_map_id=str(primary["map_id"]),
    )


def validate_factor_map(payload: dict, factor: str) -> None:
    masks = payload["coverage_masks"]
    if factor == "head_only_compatible":
        if any(masks[call] != masks[0] for call in range(1, CALLS)):
            raise ValueError("v189 head-only map varies across denoising calls")
        return
    if factor == "phase_layer_only_compatible":
        if any(
            len(set(bool(value) for value in masks[call][layer])) != 1
            for call in range(CALLS)
            for layer in range(LAYERS)
        ):
            raise ValueError("v189 phase/layer-only map varies across heads")
        return
    raise ValueError(f"unknown v190 factor map: {factor}")


def prepare(
    v189_manifest_path: Path,
    v189_analysis_path: Path,
    output_root: Path,
) -> dict:
    v189_manifest = json.loads(v189_manifest_path.read_text(encoding="utf-8"))
    analysis = json.loads(v189_analysis_path.read_text(encoding="utf-8"))
    if (
        v189_manifest.get("experiment") != "v189_structured_head_phase_profile"
        or analysis.get("experiment") != "v189_structured_head_phase_profile"
        or analysis.get("recommendation")
        != "advance_head_phase_maps_to_causal_screen"
        or analysis.get("input_manifest_sha256") != sha256(v189_manifest_path)
    ):
        raise ValueError("v190 requires a passing, SHA-bound v189 analysis")
    operators = [str(value) for value in analysis.get("generation_candidates") or ()]
    if not operators or any(value not in {"landmark", "retrieval"} for value in operators):
        raise ValueError("v190 has no supported v189 generation candidate")
    holdout = [
        int(value)
        for value in v189_manifest["prompt_split"]["generation_holdout"]
    ]
    if len(holdout) != PROMPT_COUNT or len(set(holdout)) != PROMPT_COUNT:
        raise ValueError("v190 generation holdout is invalid")
    source_path = Path(v189_manifest["source_prompt_file"])
    source = source_path.read_text(encoding="utf-8").splitlines()
    if (
        len(source) != 128
        or sha256(source_path) != v189_manifest["source_prompt_file_sha256"]
    ):
        raise ValueError("v190 source prompt provenance drift")
    prompts = [source[index].strip() for index in holdout]
    prompt_path = output_root / "prompts" / "moviegen_qwen_holdout32.txt"
    prompt_sha = write_frozen(
        prompt_path, ("\n".join(prompts) + "\n").encode("utf-8")
    )
    profile_map = Path(v189_manifest["profile_map"])
    if sha256(profile_map) != v189_manifest["profile_map_sha256"]:
        raise ValueError("v190 all-head bank map drift")

    methods = {}
    method_order = []

    def add_method(key: str, role: str, operator: str, map_payload: dict) -> None:
        validate_map(map_payload, operator=operator)
        map_path = output_root / "maps" / f"{key}.json"
        map_sha = write_frozen(
            map_path,
            (json.dumps(map_payload, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        methods[key] = {
            "role": role,
            "operator": operator,
            "history_policy": operator,
            "schedule": "head_phase",
            "head_phase_map": str(map_path.resolve()),
            "head_phase_map_sha256": map_sha,
            "phase_map_id": map_payload["map_id"],
            "coverage_count_by_call": map_payload["coverage_count_by_call"],
            "coverage_cell_count": int(
                sum(map_payload["coverage_count_by_call"])
            ),
            "coverage_exposure_fraction": float(
                sum(map_payload["coverage_count_by_call"])
                / (CALLS * LAYERS * HEADS)
            ),
            "head_bank_map": str(profile_map.resolve()),
            "head_bank_map_sha256": v189_manifest["profile_map_sha256"],
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }
        method_order.append(key)

    recent = all_recent_map(operators[0])
    add_method("all_recent", "local_control", operators[0], recent)
    control_diagnostics = {}
    control_aliases = {}
    for operator in operators:
        source_rows = analysis["operators"][operator]["maps"]

        def load_source_map(name: str) -> dict:
            if name not in source_rows:
                raise ValueError(
                    f"v190 requires v189 factor map {operator}/{name}; "
                    "rerun the v189 analyzer with the current code"
                )
            row = source_rows[name]
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"v190 source map hash drift: {operator}/{name}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_map(payload, operator=operator)
            if payload.get("map_id") != row["map_id"]:
                raise ValueError(f"v190 source map id drift: {operator}/{name}")
            return payload

        primary = load_source_map("compatible")
        primary_key = f"{operator}_compatible"
        universal = all_coverage_map(operator)
        add_method(
            f"{operator}_all_coverage",
            "all_head_all_phase_control",
            operator,
            universal,
        )
        add_method(primary_key, "primary_head_phase", operator, primary)
        registered_masks = [
            ("all_recent", recent["coverage_masks"]),
            (f"{operator}_all_coverage", universal["coverage_masks"]),
            (primary_key, primary["coverage_masks"]),
        ]
        factor_diagnostics = {}
        for factor, suffix, role in (
            (
                "head_only_compatible",
                "head_only",
                "call_invariant_head_factor_control",
            ),
            (
                "phase_layer_only_compatible",
                "phase_layer_only",
                "head_invariant_phase_layer_factor_control",
            ),
        ):
            factor_map = load_source_map(factor)
            validate_factor_map(factor_map, factor)
            duplicate_of = next(
                (
                    key
                    for key, masks in registered_masks
                    if factor_map["coverage_masks"] == masks
                ),
                None,
            )
            informative = duplicate_of is None
            method_key = f"{operator}_{suffix}"
            if informative:
                add_method(method_key, role, operator, factor_map)
                registered_masks.append((method_key, factor_map["coverage_masks"]))
            else:
                control_aliases[method_key] = duplicate_of
            factor_diagnostics[factor] = {
                "informative": informative,
                "alias_method": duplicate_of,
                "coverage_cell_count": int(
                    sum(factor_map["coverage_count_by_call"])
                ),
            }

        membership = membership_shift_map(primary)
        membership_informative = membership["coverage_masks"] != primary["coverage_masks"]
        if membership_informative:
            add_method(
                f"{operator}_membership_shift",
                "layer_count_matched_membership_control",
                operator,
                membership,
            )
            registered_masks.append(
                (f"{operator}_membership_shift", membership["coverage_masks"])
            )

        phase = phase_shift_map(primary)
        phase_informative = phase["coverage_masks"] != primary["coverage_masks"]
        if phase_informative:
            add_method(
                f"{operator}_phase_shift",
                "call_count_matched_phase_control",
                operator,
                phase,
            )
            registered_masks.append(
                (f"{operator}_phase_shift", phase["coverage_masks"])
            )

        dense = dense_phase_map(primary)
        dense_informative = (
            all(
                dense["coverage_masks"] != masks
                for _, masks in registered_masks
            )
        )
        if dense_informative:
            add_method(
                f"{operator}_dense_phase",
                "same_active_call_layer_cells_dense_control",
                operator,
                dense,
            )
        control_diagnostics[operator] = {
            "all_coverage_cell_count": int(
                sum(universal["coverage_count_by_call"])
            ),
            "membership_shift_informative": membership_informative,
            "phase_shift_informative": phase_informative,
            "dense_phase_informative": dense_informative,
            "dense_phase_deduplicated_against_all_coverage": bool(
                dense["coverage_masks"] == universal["coverage_masks"]
            ),
            "factor_maps": factor_diagnostics,
        }

    payload = {
        "version": 1,
        "experiment": "v190_head_phase_causal_screen",
        "scope": "classifier_holdout32",
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": 128,
        "source_prompt_file": str(source_path.resolve()),
        "source_prompt_file_sha256": v189_manifest["source_prompt_file_sha256"],
        "source_indices": holdout,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": index, "source_index": source_index, "text": prompts[index]}
            for index, source_index in enumerate(holdout)
        ],
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 19000,
        "operators": operators,
        "method_order": method_order,
        "methods": methods,
        "control_aliases": control_aliases,
        "control_diagnostics": control_diagnostics,
        "source": {
            "v189_manifest": str(v189_manifest_path.resolve()),
            "v189_manifest_sha256": sha256(v189_manifest_path),
            "v189_analysis": str(v189_analysis_path.resolve()),
            "v189_analysis_sha256": sha256(v189_analysis_path),
        },
        "claim_boundary": (
            "This 32-prompt screen tests causal transfer of frozen v189 maps. "
            "It is not a final benchmark or a cross-model claim."
        ),
    }
    manifest_path = output_root / "manifest.json"
    write_frozen(
        manifest_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "v190_head_phase_causal_screen"
        or payload.get("scope") != "classifier_holdout32"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or not payload.get("operators")
        or len(payload.get("method_order") or ())
        != len(set(payload.get("method_order") or ()))
        or set(payload.get("method_order") or ())
        != set(payload.get("methods") or {})
        or payload["method_order"][0] != "all_recent"
    ):
        raise ValueError("invalid v190 manifest")
    aliases = payload.get("control_aliases") or {}
    if any(
        alias in payload["methods"] or target not in payload["methods"]
        for alias, target in aliases.items()
    ):
        raise ValueError("invalid v190 control alias")
    prompt_path = Path(payload["prompt_file"])
    if sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v190 prompt hash drift")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if prompts != [str(row["text"]) for row in payload["prompt_items"]]:
        raise ValueError("v190 prompt item drift")
    for key in payload["method_order"]:
        row = payload["methods"][key]
        map_path = Path(row["head_phase_map"])
        if sha256(map_path) != row["head_phase_map_sha256"]:
            raise ValueError(f"v190 map hash drift: {key}")
        map_payload = json.loads(map_path.read_text(encoding="utf-8"))
        validate_map(map_payload, operator=row["operator"])
        coverage_cells = int(sum(map_payload["coverage_count_by_call"]))
        if (
            map_payload["map_id"] != row["phase_map_id"]
            or map_payload["coverage_count_by_call"]
            != row["coverage_count_by_call"]
            or int(row.get("coverage_cell_count", -1)) != coverage_cells
            or abs(
                float(row.get("coverage_exposure_fraction", -1.0))
                - coverage_cells / (CALLS * LAYERS * HEADS)
            )
            > 1e-12
        ):
            raise ValueError(f"v190 map metadata drift: {key}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v189-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v189-analysis", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(args.v189_manifest, args.v189_analysis, args.output_root)
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v190-inputs] PASS "
        f"methods={len(payload['method_order'])} operators={payload['operators']} "
        f"prompts={payload['prompt_count']}"
    )


if __name__ == "__main__":
    main()
