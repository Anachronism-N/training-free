#!/usr/bin/env python3
"""Freeze the v201 Head x Phase x AR-horizon causal-screen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPERIMENT = "v201_head_phase_horizon_causal_screen"
SOURCE_EXPERIMENT = "v200_head_phase_ar_horizon_audit"
BASELINE_METHOD = "sf_native"
PROMPT_COUNT = 32
SOURCE_PROMPT_COUNT = 128
CALLS = 4
LAYERS = 30
HEADS = 12
NUM_OUTPUT_FRAMES = 120
SEED = 20100
OPERATORS = ("landmark", "retrieval")
SELECTOR_ROLES = (
    "static_top10",
    "horizon_top10",
    "horizon_shift_top10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v201 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def validate_horizon_map(payload: dict, *, operator: str | None = None) -> None:
    masks = payload.get("coverage_masks")
    frames = payload.get("current_frames")
    positions = int(payload.get("position_count", -1))
    if (
        payload.get("version") != 2
        or payload.get("horizon_selection") != "nearest_profile_frame"
        or int(payload.get("call_count", -1)) != CALLS
        or int(payload.get("layer_count", -1)) != LAYERS
        or int(payload.get("head_count", -1)) != HEADS
        or positions <= 1
        or not isinstance(frames, list)
        or len(frames) != positions
        or any(type(value) is not int or value < 0 for value in frames)
        or frames != sorted(set(frames))
        or not isinstance(masks, list)
        or len(masks) != positions
    ):
        raise ValueError("invalid v201 horizon-map contract")
    counts_by_position_call = []
    for position_rows in masks:
        if not isinstance(position_rows, list) or len(position_rows) != CALLS:
            raise ValueError("invalid v201 horizon-map call shape")
        call_counts = []
        for call_rows in position_rows:
            if not isinstance(call_rows, list) or len(call_rows) != LAYERS:
                raise ValueError("invalid v201 horizon-map layer shape")
            for row in call_rows:
                if (
                    not isinstance(row, list)
                    or len(row) != HEADS
                    or any(type(value) is not bool for value in row)
                ):
                    raise ValueError("invalid v201 horizon-map head shape")
            call_counts.append(
                sum(value for layer_row in call_rows for value in layer_row)
            )
        counts_by_position_call.append(call_counts)
    counts_by_position = [sum(row) for row in counts_by_position_call]
    if payload.get("coverage_count_by_position_call") != counts_by_position_call:
        raise ValueError("v201 horizon-map position/call count drift")
    if payload.get("coverage_count_by_position") != counts_by_position:
        raise ValueError("v201 horizon-map position count drift")
    if payload.get("constant_exposure_per_position") != (
        len(set(counts_by_position)) == 1
    ):
        raise ValueError("v201 horizon-map constant-exposure declaration drift")
    if operator is not None and payload.get("coverage_operator") != operator:
        raise ValueError("v201 horizon-map Coverage operator drift")


def _map_payload(
    masks: list,
    *,
    operator: str,
    classification: str,
    current_frames: list[int],
    parent_map_id: str | None,
    v200_analysis_sha256: str,
) -> dict:
    counts_by_position_call = [
        [
            sum(value for layer_row in call_rows for value in layer_row)
            for call_rows in position_rows
        ]
        for position_rows in masks
    ]
    counts_by_position = [sum(row) for row in counts_by_position_call]
    payload = {
        "version": 2,
        "experiment": EXPERIMENT,
        "classification": classification,
        "coverage_operator": operator,
        "call_count": CALLS,
        "layer_count": LAYERS,
        "head_count": HEADS,
        "position_count": len(current_frames),
        "current_frames": list(current_frames),
        "horizon_selection": "nearest_profile_frame",
        "coverage_masks": masks,
        "coverage_count_by_position": counts_by_position,
        "coverage_count_by_position_call": counts_by_position_call,
        "constant_exposure_per_position": len(set(counts_by_position)) == 1,
        "parent_map_id": parent_map_id,
        "source_v200_analysis_sha256": v200_analysis_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["map_id"] = (
        f"v201-{operator}-{classification}-{hashlib.sha256(canonical).hexdigest()[:12]}"
    )
    validate_horizon_map(payload, operator=operator)
    return payload


def endpoint_map(
    template: dict,
    *,
    enabled: bool,
    classification: str,
    v200_analysis_sha256: str,
) -> dict:
    positions = int(template["position_count"])
    masks = [
        [
            [[bool(enabled) for _ in range(HEADS)] for _ in range(LAYERS)]
            for _ in range(CALLS)
        ]
        for _ in range(positions)
    ]
    return _map_payload(
        masks,
        operator=str(template["coverage_operator"]),
        classification=classification,
        current_frames=[int(value) for value in template["current_frames"]],
        parent_map_id=str(template["map_id"]),
        v200_analysis_sha256=v200_analysis_sha256,
    )


def _load_source_map(row: dict, *, operator: str) -> dict:
    path = Path(row["path"])
    if not path.is_file() or sha256(path) != row.get("sha256"):
        raise ValueError(
            f"v200 runtime-map hash drift: {operator}/{row.get('classification')}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_horizon_map(payload, operator=operator)
    if payload.get("map_id") != row.get("map_id"):
        raise ValueError(f"v200 runtime-map id drift: {operator}")
    return payload


def prepare(
    v189_manifest_path: Path,
    v200_analysis_path: Path,
    output_root: Path,
) -> dict:
    v189 = json.loads(v189_manifest_path.read_text(encoding="utf-8"))
    v200 = json.loads(v200_analysis_path.read_text(encoding="utf-8"))
    v200_sha = sha256(v200_analysis_path)
    if (
        v189.get("experiment") != "v189_structured_head_phase_profile"
        or v200.get("version") != 2
        or v200.get("experiment") != SOURCE_EXPERIMENT
        or v200.get("recommendation") != "advance_head_phase_horizon_to_runtime_design"
        or v200.get("source", {}).get("manifest_sha256") != sha256(v189_manifest_path)
        or v200.get("split", {}).get("generation_holdout_used") is not False
    ):
        raise ValueError("v201 requires a passing, SHA-bound v200 analysis")
    candidates = [str(value) for value in v200.get("generation_candidates") or ()]
    if not candidates or any(value not in OPERATORS for value in candidates):
        raise ValueError("v201 has no supported v200 generation candidate")

    split = v189.get("prompt_split") or {}
    holdout = [int(value) for value in split.get("generation_holdout") or ()]
    if (
        len(holdout) != PROMPT_COUNT
        or len(set(holdout)) != PROMPT_COUNT
        or v200.get("split", {}).get("generation_holdout") != holdout
    ):
        raise ValueError("v201 generation holdout drift")
    source_path = Path(v189["source_prompt_file"])
    source_prompts = source_path.read_text(encoding="utf-8").splitlines()
    if (
        len(source_prompts) != SOURCE_PROMPT_COUNT
        or sha256(source_path) != v189["source_prompt_file_sha256"]
    ):
        raise ValueError("v201 source prompt provenance drift")
    prompts = [source_prompts[index].strip() for index in holdout]
    prompt_path = output_root / "prompts" / "moviegen_qwen_holdout32.txt"
    prompt_sha = write_frozen(prompt_path, ("\n".join(prompts) + "\n").encode("utf-8"))
    head_bank_map = Path(v189["profile_map"])
    if sha256(head_bank_map) != v189["profile_map_sha256"]:
        raise ValueError("v201 all-head bank map drift")

    methods: dict[str, dict] = {
        BASELINE_METHOD: {
            "runtime": "sf_native",
            "role": "canonical_sf_baseline",
            "operator": None,
            "schedule": None,
            "read_frame_equivalents": None,
            "clean_policy": None,
        }
    }
    method_order: list[str] = [BASELINE_METHOD]
    operator_contracts = {}

    def add_method(
        key: str,
        role: str,
        operator: str,
        map_payload: dict,
        *,
        source_role: str,
    ) -> None:
        validate_horizon_map(map_payload, operator=operator)
        encoded = (json.dumps(map_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        map_path = output_root / "maps" / f"{key}.json"
        map_sha = write_frozen(map_path, encoded)
        exposure_count = int(sum(map_payload["coverage_count_by_position"]))
        denominator = int(map_payload["position_count"]) * CALLS * LAYERS * HEADS
        methods[key] = {
            "runtime": "head_phase_horizon_cache_runtime",
            "role": role,
            "operator": operator,
            "history_policy": operator,
            "schedule": "head_phase_horizon",
            "horizon_map": str(map_path.resolve()),
            "horizon_map_sha256": map_sha,
            "routing_map_id": map_payload["map_id"],
            "map_classification": map_payload["classification"],
            "map_source_role": source_role,
            "current_frames": map_payload["current_frames"],
            "coverage_count_by_position": map_payload["coverage_count_by_position"],
            "coverage_count_by_position_call": map_payload[
                "coverage_count_by_position_call"
            ],
            "coverage_exposure_count": exposure_count,
            "coverage_exposure_fraction": exposure_count / denominator,
            "head_bank_map": str(head_bank_map.resolve()),
            "head_bank_map_sha256": v189["profile_map_sha256"],
            "read_frame_equivalents": 9,
            "clean_policy": "recent",
        }
        method_order.append(key)

    for operator in candidates:
        source_rows = v200["operators"][operator].get("runtime_maps") or {}
        if set(source_rows) != set(SELECTOR_ROLES):
            raise ValueError(f"v201 requires all v200 selector maps for {operator}")
        source_maps = {
            role: _load_source_map(source_rows[role], operator=operator)
            for role in SELECTOR_ROLES
        }
        counts = {
            role: source_maps[role]["coverage_count_by_position"]
            for role in SELECTOR_ROLES
        }
        if not (
            counts["static_top10"]
            == counts["horizon_top10"]
            == counts["horizon_shift_top10"]
        ):
            raise ValueError(f"v201 selector exposure mismatch for {operator}")
        template = source_maps["horizon_top10"]
        recent = endpoint_map(
            template,
            enabled=False,
            classification="all_recent",
            v200_analysis_sha256=v200_sha,
        )
        coverage = endpoint_map(
            template,
            enabled=True,
            classification="all_coverage",
            v200_analysis_sha256=v200_sha,
        )
        add_method(
            f"{operator}_all_recent",
            "operator_matched_local_control",
            operator,
            recent,
            source_role="v201_endpoint_control",
        )
        add_method(
            f"{operator}_all_coverage",
            "operator_matched_universal_coverage_control",
            operator,
            coverage,
            source_role="v201_endpoint_control",
        )
        add_method(
            f"{operator}_static_top10",
            "equal_exposure_static_head_phase_control",
            operator,
            source_maps["static_top10"],
            source_role="v200_discovery_selector",
        )
        add_method(
            f"{operator}_horizon_top10",
            "primary_head_phase_horizon",
            operator,
            source_maps["horizon_top10"],
            source_role="v200_discovery_selector",
        )
        add_method(
            f"{operator}_horizon_shift_top10",
            "equal_exposure_horizon_alignment_control",
            operator,
            source_maps["horizon_shift_top10"],
            source_role="v200_discovery_selector",
        )
        operator_contracts[operator] = {
            "method_order": method_order[-5:],
            "equal_exposure_selector_methods": method_order[-3:],
            "coverage_cells_per_position": counts["horizon_top10"][0],
            "equal_exposure_verified": True,
            "v200_horizon_gate": True,
        }

    payload = {
        "version": 2,
        "experiment": EXPERIMENT,
        "scope": "classifier_holdout32",
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": SOURCE_PROMPT_COUNT,
        "source_prompt_file": str(source_path.resolve()),
        "source_prompt_file_sha256": v189["source_prompt_file_sha256"],
        "source_indices": holdout,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": index, "source_index": source_index, "text": prompts[index]}
            for index, source_index in enumerate(holdout)
        ],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": SEED,
        "operators": candidates,
        "method_order": method_order,
        "methods": methods,
        "operator_contracts": operator_contracts,
        "source": {
            "v189_manifest": str(v189_manifest_path.resolve()),
            "v189_manifest_sha256": sha256(v189_manifest_path),
            "v200_analysis": str(v200_analysis_path.resolve()),
            "v200_analysis_sha256": v200_sha,
        },
        "manual_review_required_before_generation": False,
        "primary_baseline": BASELINE_METHOD,
        "promotion_target": "paired_improvement_over_canonical_sf",
        "claim_boundary": (
            "v201 is a classifier-holdout causal screen. It tests whether the "
            "frozen method improves canonical SF and separately tests whether "
            "the AR-horizon assignment matters beyond equal-exposure static and "
            "time-misaligned selectors. It is not a final benchmark or "
            "cross-model result."
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
    methods = payload.get("methods") or {}
    order = payload.get("method_order") or []
    if (
        int(payload.get("version", -1)) != 2
        or payload.get("experiment") != EXPERIMENT
        or payload.get("scope") != "classifier_holdout32"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or not payload.get("operators")
        or payload.get("primary_baseline") != BASELINE_METHOD
        or len(order) != len(set(order))
        or set(order) != set(methods)
        or not order
        or order[0] != BASELINE_METHOD
    ):
        raise ValueError("invalid v201 input manifest")
    baseline = methods.get(BASELINE_METHOD) or {}
    if (
        baseline.get("runtime") != "sf_native"
        or baseline.get("role") != "canonical_sf_baseline"
        or baseline.get("operator") is not None
    ):
        raise ValueError("invalid v201 canonical SF baseline")
    prompt_path = Path(payload["prompt_file"])
    if sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v201 prompt hash drift")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if prompts != [str(row["text"]) for row in payload["prompt_items"]]:
        raise ValueError("v201 prompt-item drift")
    for operator in payload["operators"]:
        contract = payload["operator_contracts"][operator]
        expected = [
            f"{operator}_all_recent",
            f"{operator}_all_coverage",
            f"{operator}_static_top10",
            f"{operator}_horizon_top10",
            f"{operator}_horizon_shift_top10",
        ]
        if contract["method_order"] != expected:
            raise ValueError(f"v201 method order drift for {operator}")
        selector_counts = []
        for key in expected:
            row = methods[key]
            path = Path(row["horizon_map"])
            if sha256(path) != row["horizon_map_sha256"]:
                raise ValueError(f"v201 horizon-map hash drift: {key}")
            map_payload = json.loads(path.read_text(encoding="utf-8"))
            validate_horizon_map(map_payload, operator=operator)
            exposure = int(sum(map_payload["coverage_count_by_position"]))
            denominator = map_payload["position_count"] * CALLS * LAYERS * HEADS
            if (
                row["routing_map_id"] != map_payload["map_id"]
                or row["coverage_count_by_position"]
                != map_payload["coverage_count_by_position"]
                or int(row["coverage_exposure_count"]) != exposure
                or abs(row["coverage_exposure_fraction"] - exposure / denominator)
                > 1e-12
                or row["clean_policy"] != "recent"
                or int(row["read_frame_equivalents"]) != 9
            ):
                raise ValueError(f"v201 map metadata drift: {key}")
            if key in contract["equal_exposure_selector_methods"]:
                selector_counts.append(map_payload["coverage_count_by_position"])
        if not selector_counts or any(
            row != selector_counts[0] for row in selector_counts
        ):
            raise ValueError(f"v201 selector exposure drift for {operator}")
    expected_order = [BASELINE_METHOD]
    for operator in payload["operators"]:
        expected_order.extend(payload["operator_contracts"][operator]["method_order"])
    if order != expected_order:
        raise ValueError("v201 global method order drift")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v189-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v200-analysis", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(args.v189_manifest, args.v200_analysis, args.output_root)
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v201-inputs] PASS "
        f"operators={payload['operators']} methods={len(payload['method_order'])} "
        f"prompts={payload['prompt_count']} equal_exposure=true"
    )


if __name__ == "__main__":
    main()
