#!/usr/bin/env python3
"""Freeze new-seed and 60-second scopes after a passing v205 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v201_head_phase_horizon_screen import sha256, validate_horizon_map
from prepare_v205_horizon_confirmation import verify as verify_v205_input


EXPERIMENT = "v206_horizon_robustness_inputs"
SCOPE_KEYS = ("seed20600_30s_128", "long60_seed20500_32")
LONG_PROMPT_POSITIONS = tuple(range(0, 128, 4))
SCOPE_SPECS = {
    "seed20600_30s_128": {
        "prompt_positions": tuple(range(128)),
        "prompt_count": 128,
        "num_output_frames": 120,
        "seed": 20600,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "duration_seconds": 29.8125,
            "width": 832,
            "height": 480,
        },
        "role": "same_prompt_new_seed_replication",
    },
    "long60_seed20500_32": {
        "prompt_positions": LONG_PROMPT_POSITIONS,
        "prompt_count": 32,
        "num_output_frames": 240,
        "seed": 20500,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16.0,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "role": "systematic_long_horizon_replication",
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v206 artifact differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_path_hash(row: dict, key: str, *, label: str) -> Path:
    path = Path(str(row.get(key, "")))
    if not path.is_file() or sha256(path) != row.get(f"{key}_sha256"):
        raise ValueError(f"v206 prerequisite drifted: {label}/{key}")
    return path


def validate_v205(
    decision_path: Path, input_manifest_path: Path
) -> tuple[dict, dict, tuple[str, ...]]:
    decision = load_json(decision_path)
    frozen = verify_v205_input(input_manifest_path)
    confirmed = tuple(str(value) for value in decision.get("confirmed_candidates") or ())
    statuses = decision.get("candidate_status") or {}
    if (
        decision.get("experiment")
        != "v205_profile_disjoint_head_phase_horizon_vbench"
        or decision.get("confirmatory") is not True
        or decision.get("paper_efficacy_support") is not True
        or decision.get("recommendation")
        == "do_not_advance_v205_no_confirmatory_sf_gain"
        or not 1 <= len(confirmed) <= 2
        or len(set(confirmed)) != len(confirmed)
        or any(
            candidate not in frozen["selected_v201_candidates"]
            or (statuses.get(candidate) or {}).get("confirmation_pass") is not True
            for candidate in confirmed
        )
    ):
        raise ValueError("v206 requires one or two confirmed v205 candidates")
    source = decision.get("source") or {}
    comparison_path = verify_path_hash(
        source, "comparison_manifest", label="v205-decision"
    )
    for key in ("vbench_summary", "temporal_diagnostics", "temporal_contract"):
        verify_path_hash(source, key, label="v205-decision")
    comparison = load_json(comparison_path)
    comparison_source = comparison.get("source") or {}
    if (
        comparison.get("experiment")
        != "v205_profile_disjoint_head_phase_horizon_vbench"
        or comparison.get("confirmatory") is not True
        or int(comparison.get("prompt_count", -1)) != 128
        or tuple(str(value) for value in comparison.get("selected_v201_candidates") or ())
        != tuple(frozen["selected_v201_candidates"])
        or comparison_source.get("input_manifest_sha256")
        != sha256(input_manifest_path)
    ):
        raise ValueError("v205 decision and frozen generation input disagree")
    for key in ("published_manifest", "generation_contract"):
        verify_path_hash(comparison_source, key, label="v205-comparison")
    return decision, frozen, confirmed


def prepare(
    v205_decision: Path,
    v205_input_manifest: Path,
    output_root: Path,
) -> dict:
    decision, v205, confirmed = validate_v205(v205_decision, v205_input_manifest)
    source_items = list(v205["prompt_items"])

    bank_hashes = {
        str(v205["methods"][candidate]["head_bank_map_sha256"])
        for candidate in confirmed
    }
    if len(bank_hashes) != 1:
        raise ValueError("confirmed v205 candidates use different head-bank maps")
    bank_source = Path(v205["methods"][confirmed[0]]["head_bank_map"])
    if not bank_source.is_file() or sha256(bank_source) not in bank_hashes:
        raise ValueError("confirmed v205 head-bank artifact drifted")
    bank_path = output_root / "maps" / "all_profile_banks.csv"
    bank_sha = write_frozen(bank_path, bank_source.read_bytes())

    methods = {"sf_native": dict(v205["methods"]["sf_native"])}
    for candidate in confirmed:
        source_row = v205["methods"][candidate]
        source_map = Path(source_row["horizon_map"])
        if not source_map.is_file() or sha256(source_map) != source_row["horizon_map_sha256"]:
            raise ValueError(f"v205 horizon map drifted: {candidate}")
        map_payload = load_json(source_map)
        validate_horizon_map(map_payload, operator=str(source_row["operator"]))
        map_path = output_root / "maps" / f"{candidate}.json"
        map_sha = write_frozen(map_path, source_map.read_bytes())
        row = dict(source_row)
        row.update(
            {
                "horizon_map": str(map_path.resolve()),
                "horizon_map_sha256": map_sha,
                "head_bank_map": str(bank_path.resolve()),
                "head_bank_map_sha256": bank_sha,
                "role": "v205_confirmed_horizon_candidate",
            }
        )
        methods[candidate] = row

    scopes = []
    for scope_key in SCOPE_KEYS:
        spec = SCOPE_SPECS[scope_key]
        positions = list(spec["prompt_positions"])
        prompt_items = [
            {
                "index": index,
                "v205_prompt_index": position,
                "source_index": int(source_items[position]["source_index"]),
                "text": str(source_items[position]["text"]),
            }
            for index, position in enumerate(positions)
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
                "prompt_positions_in_v205": positions,
                "prompt_source_indices": [
                    row["source_index"] for row in prompt_items
                ],
                "prompt_items": prompt_items,
                "prompt_file": str(prompt_path.resolve()),
                "prompt_file_sha256": prompt_sha,
                "num_output_frames": int(spec["num_output_frames"]),
                "seed": int(spec["seed"]),
                "decoded_video_contract": spec["decoded_video_contract"],
                "selection_rule": (
                    "all frozen v205 profiling-disjoint prompts"
                    if scope_key == "seed20600_30s_128"
                    else "every fourth v205 prompt position, fixed before v206 metrics"
                ),
            }
        )

    replication_targets = {}
    for candidate in confirmed:
        positive = decision["candidate_status"][candidate]["positive_support"]
        replication_targets[candidate] = [
            f"{row['window']}:{row['metric']}"
            for row in positive["interval_supported_axes"]
        ]
        if not replication_targets[candidate]:
            raise ValueError(f"v205 candidate lacks a confirmatory target: {candidate}")
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "confirmatory": True,
        "prerequisite": "passing_v205_profile_disjoint128",
        "method_order": ["sf_native", *confirmed],
        "methods": methods,
        "confirmed_v205_candidates": list(confirmed),
        "scopes": scopes,
        "v205_positive_axes_to_replicate": replication_targets,
        "v205_long_horizon_supported_candidates": decision[
            "long_horizon_supported_candidates"
        ],
        "v205_source": {
            "decision": str(v205_decision.resolve()),
            "decision_sha256": sha256(v205_decision),
            "input_manifest": str(v205_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v205_input_manifest),
        },
        "manual_review_required_before_generation": False,
        "claim_boundary": (
            "v206 is conditional robustness evidence for candidates that already "
            "passed v205 against SF. The 30-second scope changes only seed; the "
            "60-second scope changes only duration and uses a systematic prompt "
            "subset. Cross-checkpoint transfer is not claimed by these scopes."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def scope_config(payload: dict, key: str) -> dict:
    rows = [row for row in payload.get("scopes") or () if row.get("key") == key]
    if len(rows) != 1:
        raise ValueError(f"unknown or duplicate v206 scope: {key}")
    return rows[0]


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    confirmed = tuple(str(value) for value in payload.get("confirmed_v205_candidates") or ())
    methods = tuple(str(value) for value in payload.get("method_order") or ())
    if (
        payload.get("experiment") != EXPERIMENT
        or payload.get("confirmatory") is not True
        or not 1 <= len(confirmed) <= 2
        or methods != ("sf_native", *confirmed)
        or set(payload.get("methods") or {}) != set(methods)
        or tuple(row.get("key") for row in payload.get("scopes") or ())
        != SCOPE_KEYS
    ):
        raise ValueError("invalid v206 frozen manifest")
    for candidate in confirmed:
        row = payload["methods"][candidate]
        map_path = Path(row["horizon_map"])
        bank_path = Path(row["head_bank_map"])
        if (
            row.get("role") != "v205_confirmed_horizon_candidate"
            or row.get("schedule") != "head_phase_horizon"
            or row.get("clean_policy") != "recent"
            or int(row.get("read_frame_equivalents", -1)) != 9
            or sha256(map_path) != row["horizon_map_sha256"]
            or sha256(bank_path) != row["head_bank_map_sha256"]
        ):
            raise ValueError(f"v206 cache contract drifted: {candidate}")
        validate_horizon_map(load_json(map_path), operator=str(row["operator"]))
    for scope_key in SCOPE_KEYS:
        scope = scope_config(payload, scope_key)
        spec = SCOPE_SPECS[scope_key]
        prompt_path = Path(scope["prompt_file"])
        if (
            int(scope["prompt_count"]) != spec["prompt_count"]
            or int(scope["num_output_frames"]) != spec["num_output_frames"]
            or int(scope["seed"]) != spec["seed"]
            or scope["prompt_positions_in_v205"] != list(spec["prompt_positions"])
            or sha256(prompt_path) != scope["prompt_file_sha256"]
            or prompt_path.read_text(encoding="utf-8").splitlines()
            != [str(row["text"]) for row in scope["prompt_items"]]
        ):
            raise ValueError(f"v206 scope contract drifted: {scope_key}")
    for key in ("decision", "input_manifest"):
        verify_path_hash(payload["v205_source"], key, label="manifest/v205_source")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v205-decision", type=Path, required=True)
    prepare_parser.add_argument("--v205-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(args.v205_decision, args.v205_input_manifest, args.output_root)
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v206-inputs] PASS "
        f"candidates={payload['confirmed_v205_candidates']} "
        f"scopes={[row['key'] for row in payload['scopes']]}"
    )


if __name__ == "__main__":
    main()
