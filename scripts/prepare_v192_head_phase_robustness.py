#!/usr/bin/env python3
"""Freeze seed and long-horizon robustness scopes after a passing v191."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v190_head_phase_causal_screen import validate_map


METHODS = ("sf_native", "all_recent", "head_phase_joint")
CANDIDATE = "head_phase_joint"
LOCAL_CONTROL = "all_recent"
SCOPE_KEYS = ("seed2026_30s_128", "long60_seed10000_32")
LONG_PROMPT_POSITIONS = tuple(range(0, 128, 4))
SCOPE_SPECS = {
    "seed2026_30s_128": {
        "prompt_positions": tuple(range(128)),
        "prompt_count": 128,
        "num_output_frames": 120,
        "seed": 2026,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "duration_seconds": 29.8125,
            "width": 832,
            "height": 480,
        },
        "role": "same_prompt_new_seed_replication",
    },
    "long60_seed10000_32": {
        "prompt_positions": LONG_PROMPT_POSITIONS,
        "prompt_count": 32,
        "num_output_frames": 240,
        "seed": 10000,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16.0,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "role": "predeclared_long_horizon_replication",
    },
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
        raise RuntimeError(f"frozen v192 artifact differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json_frozen(path: Path, payload: dict) -> str:
    return write_frozen(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _verify_path_hash(row: dict, key: str, *, label: str) -> Path:
    path = Path(str(row.get(key, "")))
    if not path.is_file() or sha256(path) != row.get(f"{key}_sha256"):
        raise ValueError(f"v192 prerequisite drifted: {label}/{key}")
    return path


def validate_v191(decision_path: Path, input_manifest_path: Path) -> tuple[dict, dict]:
    decision = load_json(decision_path)
    frozen = load_json(input_manifest_path)
    gates = decision.get("confirmation_gates") or {}
    if (
        int(decision.get("version", -1)) < 1
        or decision.get("experiment") != "v191_unseen128_head_phase_vbench"
        or decision.get("confirmatory") is not True
        or decision.get("head_phase_effect_confirmed") is not True
        or decision.get("recommendation")
        != "freeze_head_phase_method_for_seed_length_and_cross_model_replication"
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("v192 requires every frozen v191 confirmation gate to pass")
    if (
        frozen.get("experiment") != "v191_unseen128_head_phase_confirmation"
        or frozen.get("scope") != "confirmatory_unseen128"
        or frozen.get("confirmatory") is not True
        or int(frozen.get("prompt_count", -1)) != 128
        or int(frozen.get("seed", -1)) != 10000
        or tuple(frozen.get("method_order") or ()) != METHODS
        or [int(row.get("source_index", -1)) for row in frozen.get("prompt_items") or ()]
        != list(range(128, 256))
        or decision.get("selected_v190_method") != frozen.get("selected_v190_method")
        or decision.get("selected_operator") != frozen.get("selected_operator")
    ):
        raise ValueError("v191 decision and frozen generation input disagree")

    source = decision.get("source") or {}
    comparison_path = _verify_path_hash(
        source, "comparison_manifest", label="v191-decision"
    )
    for key in ("vbench_summary", "temporal_diagnostics", "temporal_contract"):
        _verify_path_hash(source, key, label="v191-decision")
    comparison = load_json(comparison_path)
    comparison_source = comparison.get("source") or {}
    if (
        comparison.get("experiment") != "v191_unseen128_head_phase_vbench"
        or comparison.get("confirmatory") is not True
        or int(comparison.get("prompt_count", -1)) != 128
        or int(comparison.get("seed", -1)) != 10000
        or tuple(row.get("key") for row in comparison.get("methods") or ())
        != METHODS
        or comparison_source.get("input_manifest_sha256")
        != sha256(input_manifest_path)
    ):
        raise ValueError("v191 comparison evidence is mixed or does not bind the input")
    for key in ("published_manifest", "generation_contract"):
        _verify_path_hash(comparison_source, key, label="v191-comparison")

    for method in METHODS[1:]:
        row = frozen["methods"][method]
        map_path = _verify_path_hash(row, "head_phase_map", label=method)
        _verify_path_hash(row, "head_bank_map", label=method)
        payload = load_json(map_path)
        validate_map(payload, operator=str(row["operator"]))
        if (
            row.get("schedule") != "head_phase"
            or row.get("clean_policy") != "recent"
            or int(row.get("read_frame_equivalents", -1)) != 9
            or payload.get("map_id") != row.get("phase_map_id")
        ):
            raise ValueError(f"v191 cache contract drifted: {method}")

    positive = decision.get("positive_effect_vs_equal_budget") or {}
    positive_flags = positive.get("ci_lower_gt_zero") or {}
    motion_valid = decision.get("motion_improvement_claim_supported") is True
    replicated_targets = sorted(
        metric
        for metric, passed in positive_flags.items()
        if passed is True and (metric != "dynamic_degree" or motion_valid)
    )
    if positive.get("pass") is not True or not replicated_targets:
        raise ValueError(
            "v192 requires at least one valid v191 positive-effect target"
        )
    return decision, frozen


def _copy_artifact(source: Path, target: Path) -> str:
    return write_frozen(target, source.read_bytes())


def prepare(
    v191_decision: Path,
    v191_input_manifest: Path,
    output_root: Path,
) -> dict:
    decision, v191 = validate_v191(v191_decision, v191_input_manifest)
    operator = str(v191["selected_operator"])

    bank_source = Path(v191["methods"][CANDIDATE]["head_bank_map"])
    bank_path = output_root / "maps" / "all_profile_banks.csv"
    bank_sha = _copy_artifact(bank_source, bank_path)
    methods = {"sf_native": dict(v191["methods"]["sf_native"])}
    for method in METHODS[1:]:
        source_row = v191["methods"][method]
        source_map = Path(source_row["head_phase_map"])
        map_path = output_root / "maps" / f"{method}.json"
        map_sha = _copy_artifact(source_map, map_path)
        row = dict(source_row)
        row.update(
            {
                "head_phase_map": str(map_path.resolve()),
                "head_phase_map_sha256": map_sha,
                "head_bank_map": str(bank_path.resolve()),
                "head_bank_map_sha256": bank_sha,
            }
        )
        methods[method] = row

    source_items = list(v191["prompt_items"])
    scopes = []
    for scope_key in SCOPE_KEYS:
        spec = SCOPE_SPECS[scope_key]
        positions = list(spec["prompt_positions"])
        selected = [source_items[position] for position in positions]
        prompt_items = [
            {
                "index": index,
                "v191_prompt_index": position,
                "source_index": int(item["source_index"]),
                "text": str(item["text"]),
            }
            for index, (position, item) in enumerate(zip(positions, selected))
        ]
        prompt_path = output_root / "prompts" / f"{scope_key}.txt"
        prompt_sha = write_frozen(
            prompt_path,
            ("\n".join(row["text"] for row in prompt_items) + "\n").encode(
                "utf-8"
            ),
        )
        scopes.append(
            {
                "key": scope_key,
                "role": spec["role"],
                "prompt_count": int(spec["prompt_count"]),
                "prompt_positions_in_v191": positions,
                "prompt_source_indices": [row["source_index"] for row in prompt_items],
                "prompt_items": prompt_items,
                "prompt_file": str(prompt_path.resolve()),
                "prompt_file_sha256": prompt_sha,
                "num_output_frames": int(spec["num_output_frames"]),
                "seed": int(spec["seed"]),
                "decoded_video_contract": spec["decoded_video_contract"],
                "selection_rule": (
                    "all frozen v191 unseen prompts"
                    if scope_key == "seed2026_30s_128"
                    else "every fourth v191 prompt position, fixed before v192 metrics"
                ),
            }
        )

    positive_flags = decision["positive_effect_vs_equal_budget"][
        "ci_lower_gt_zero"
    ]
    motion_valid = decision["motion_improvement_claim_supported"] is True
    replication_targets = sorted(
        metric
        for metric, passed in positive_flags.items()
        if passed is True and (metric != "dynamic_degree" or motion_valid)
    )
    payload = {
        "version": 1,
        "experiment": "v192_head_phase_robustness_inputs",
        "confirmatory": True,
        "prerequisite": "passing_v191_unseen128",
        "method_order": list(METHODS),
        "methods": methods,
        "selected_v190_method": v191["selected_v190_method"],
        "selected_operator": operator,
        "scopes": scopes,
        "v191_positive_metrics_to_replicate": replication_targets,
        "v191_candidate_delta_vs_all_recent": decision[
            "candidate_delta_vs_all_recent"
        ],
        "v191_motion_improvement_claim_supported": decision[
            "motion_improvement_claim_supported"
        ],
        "cache_contract": v191["cache_contract"],
        "v191_provenance": {
            "decision": str(v191_decision.resolve()),
            "decision_sha256": sha256(v191_decision),
            "input_manifest": str(v191_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v191_input_manifest),
            "source": decision["source"],
        },
        "claim_boundary": (
            "v192 tests within-model seed replication and 60-second persistence of "
            "the exact v191-frozen Head x Denoising-Phase route. It does not refit "
            "head membership, tune thresholds, establish cross-model transfer, or "
            "evaluate prompt transitions."
        ),
    }
    write_json_frozen(output_root / "manifest.json", payload)
    return payload


def scope_config(manifest: dict, key: str) -> dict:
    row = next((item for item in manifest.get("scopes") or () if item["key"] == key), None)
    if row is None:
        raise ValueError(f"v192 manifest has no scope {key}")
    return row


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment") != "v192_head_phase_robustness_inputs"
        or payload.get("confirmatory") is not True
        or payload.get("prerequisite") != "passing_v191_unseen128"
        or tuple(payload.get("method_order") or ()) != METHODS
        or set(payload.get("methods") or {}) != set(METHODS)
        or tuple(row.get("key") for row in payload.get("scopes") or ())
        != SCOPE_KEYS
        or not payload.get("v191_positive_metrics_to_replicate")
    ):
        raise ValueError("invalid v192 input manifest")

    provenance = payload.get("v191_provenance") or {}
    decision_path = _verify_path_hash(provenance, "decision", label="v191")
    input_path = _verify_path_hash(provenance, "input_manifest", label="v191")
    decision, upstream = validate_v191(decision_path, input_path)
    if (
        payload.get("selected_v190_method") != upstream.get("selected_v190_method")
        or payload.get("selected_operator") != upstream.get("selected_operator")
        or payload.get("v191_candidate_delta_vs_all_recent")
        != decision.get("candidate_delta_vs_all_recent")
    ):
        raise ValueError("v192 frozen upstream selection drifted")

    upstream_items = list(upstream["prompt_items"])
    for scope_key in SCOPE_KEYS:
        row = scope_config(payload, scope_key)
        spec = SCOPE_SPECS[scope_key]
        positions = list(spec["prompt_positions"])
        expected_items = [upstream_items[position] for position in positions]
        expected_scope_items = [
            {
                "index": index,
                "v191_prompt_index": position,
                "source_index": int(item["source_index"]),
                "text": str(item["text"]),
            }
            for index, (position, item) in enumerate(zip(positions, expected_items))
        ]
        prompt_path = Path(row["prompt_file"])
        prompts = prompt_path.read_text(encoding="utf-8").splitlines()
        if (
            int(row.get("prompt_count", -1)) != spec["prompt_count"]
            or int(row.get("num_output_frames", -1)) != spec["num_output_frames"]
            or int(row.get("seed", -1)) != spec["seed"]
            or row.get("decoded_video_contract") != spec["decoded_video_contract"]
            or row.get("prompt_positions_in_v191") != positions
            or row.get("prompt_source_indices")
            != [int(item["source_index"]) for item in expected_items]
            or sha256(prompt_path) != row.get("prompt_file_sha256")
            or prompts != [str(item["text"]) for item in expected_items]
            or row.get("prompt_items") != expected_scope_items
        ):
            raise ValueError(f"v192 scope contract drifted: {scope_key}")

    for method in METHODS[1:]:
        row = payload["methods"][method]
        map_path = _verify_path_hash(row, "head_phase_map", label=method)
        _verify_path_hash(row, "head_bank_map", label=method)
        map_payload = load_json(map_path)
        validate_map(map_payload, operator=str(row["operator"]))
        if map_payload.get("map_id") != row.get("phase_map_id"):
            raise ValueError(f"v192 phase map id drifted: {method}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v191-decision", type=Path, required=True)
    prepare_parser.add_argument("--v191-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.v191_decision,
            args.v191_input_manifest,
            args.output_root,
        )
        print(
            "[v192-prepare] "
            f"operator={payload['selected_operator']} "
            f"scopes={','.join(row['key'] for row in payload['scopes'])}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v192-verify] PASS "
            f"methods={len(payload['method_order'])} scopes={len(payload['scopes'])}"
        )


if __name__ == "__main__":
    main()
