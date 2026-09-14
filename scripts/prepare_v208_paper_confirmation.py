#!/usr/bin/env python3
"""Freeze the v208 paper-stage 30/60-second confirmation after v207 passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v207_context_budget_phase_screen import (
    EXPERIMENT as V207_INPUT_EXPERIMENT,
    HEADS,
    LAYERS,
    SOURCE_INDICES,
    sha256,
    verify as verify_v207_inputs,
)
from prepare_v207_vbench_comparison import EXPERIMENT as V207_REPORT_EXPERIMENT


EXPERIMENT = "v208_paper_confirmation"
PARITY_EXPERIMENT = "v207_sf_runtime_parity"
PROMPT_COUNT = 128
NUM_OUTPUT_FRAMES = {"main30": 120, "long60": 240}
DECODED_CONTRACT = {
    "main30": {"frames": 477, "fps": 16.0, "width": 832, "height": 480},
    "long60": {"frames": 957, "fps": 16.0, "width": 832, "height": 480},
}
SEED = 20800
METHOD_ORDER = ("sf_native", "recent21", "ours")
ALLOWED_PARENT_METHODS = {
    "retrieval21_early1": "early1",
    "retrieval21_early2": "early2",
}


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v208 input differs: {path}")
    if not path.exists():
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _head_map_payload() -> bytes:
    return (
        "\n".join(",".join("10" for _ in range(HEADS)) for _ in range(LAYERS))
        + "\n"
    ).encode("ascii")


def _load_source(path: Path) -> list[str]:
    prompts = path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not row.strip() for row in prompts):
        raise ValueError("v208 source must contain exactly 128 non-empty prompts")
    return [row.strip() for row in prompts]


def selection_splits(items: list[dict]) -> dict[str, list[int]]:
    if [row.get("index") for row in items] != list(range(PROMPT_COUNT)):
        raise ValueError("v208 prompt index drift")
    result: dict[str, list[int]] = {"v207_development32": [], "v208_holdout96": []}
    for row in items:
        index = int(row["index"])
        split = "v207_development32" if index in SOURCE_INDICES else "v208_holdout96"
        if row.get("source_index") != index or row.get("selection_split") != split:
            raise ValueError("v208 prompt split drift")
        result[split].append(index)
    return result


def prepare(
    source_prompts: Path,
    v207_input_manifest: Path,
    v207_report_path: Path,
    parity_report_path: Path,
    output_root: Path,
) -> dict:
    v207_inputs = verify_v207_inputs(v207_input_manifest)
    report = json.loads(v207_report_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_report_path.read_text(encoding="utf-8"))
    selected = [str(value) for value in report.get("selected_for_fresh128") or ()]
    if (
        parity.get("experiment") != PARITY_EXPERIMENT
        or parity.get("parity_pass") is not True
        or parity.get("decision") != "parity_pass_proceed_to_budget_phase_screen"
    ):
        raise ValueError("v208 requires a passing numerical SF parity report")
    if (
        v207_inputs.get("experiment") != V207_INPUT_EXPERIMENT
        or report.get("experiment") != V207_REPORT_EXPERIMENT
        or report.get("recommendation")
        not in {
            "advance_best_equal_budget_retrieval_to_fresh128",
            "advance_directional_equal_budget_retrieval_to_fresh128",
        }
        or len(selected) != 1
        or selected[0] not in ALLOWED_PARENT_METHODS
        or report.get("manual_review_required_for_decision") is not False
    ):
        raise ValueError("v208 requires one automatically promoted v207 candidate")
    report_source = report.get("source") or {}
    comparison_path = Path(report_source.get("comparison_manifest", ""))
    if (
        not comparison_path.is_file()
        or sha256(comparison_path) != report_source.get("comparison_manifest_sha256")
    ):
        raise ValueError("v208 v207 report provenance drift")
    source = _load_source(source_prompts)
    if sha256(source_prompts) != v207_inputs["source_prompt_file_sha256"]:
        raise ValueError("v208 source prompts differ from the v207 benchmark source")
    prompt_path = output_root / "prompts" / "moviegen_qwen_full128.txt"
    prompt_sha = write_frozen(
        prompt_path, ("\n".join(source) + "\n").encode("utf-8")
    )
    map_path = output_root / "maps" / "all_profile_banks.csv"
    map_sha = write_frozen(map_path, _head_map_payload())
    development_indices = set(int(value) for value in v207_inputs["source_indices"])
    parent = selected[0]
    schedule = ALLOWED_PARENT_METHODS[parent]
    methods = {
        "sf_native": {
            "runtime": "sf_native",
            "role": "canonical_sf_baseline",
            "operator": None,
            "history_policy": None,
            "schedule": None,
            "coverage_noisy_calls": [],
            "read_frame_equivalents": 21,
            "sink_frames": None,
            "middle_frames": None,
            "recent_route_frames": None,
            "coverage_recent_frames": None,
            "head_map": None,
            "archive_capacity": None,
        },
        "recent21": {
            "runtime": "cache",
            "role": "equal_budget_local_control",
            "operator": "retrieval",
            "history_policy": "retrieval",
            "schedule": "recent",
            "coverage_noisy_calls": [],
            "read_frame_equivalents": 21,
            "sink_frames": 1,
            "middle_frames": 4,
            "recent_route_frames": 20,
            "coverage_recent_frames": 16,
            "head_map": str(map_path.resolve()),
            "head_map_sha256": map_sha,
            "archive_capacity": 12,
        },
        "ours": {
            "runtime": "cache",
            "role": "frozen_v207_method",
            "parent_method": parent,
            "operator": "retrieval",
            "history_policy": "retrieval",
            "schedule": schedule,
            "coverage_noisy_calls": [0] if schedule == "early1" else [0, 1],
            "read_frame_equivalents": 21,
            "sink_frames": 1,
            "middle_frames": 4,
            "recent_route_frames": 20,
            "coverage_recent_frames": 16,
            "head_map": str(map_path.resolve()),
            "head_map_sha256": map_sha,
            "archive_capacity": 12,
        },
    }
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "paper_stage": True,
        "prompt_count": PROMPT_COUNT,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {
                "index": index,
                "source_index": index,
                "text": prompt,
                "selection_split": (
                    "v207_development32"
                    if index in development_indices
                    else "v208_holdout96"
                ),
            }
            for index, prompt in enumerate(source)
        ],
        "seed": SEED,
        "method_order": list(METHOD_ORDER),
        "methods": methods,
        "scopes": [
            {
                "key": key,
                "duration_seconds": 30 if key == "main30" else 60,
                "num_output_frames": NUM_OUTPUT_FRAMES[key],
                "decoded_video_contract": DECODED_CONTRACT[key],
                "prompt_count": PROMPT_COUNT,
            }
            for key in ("main30", "long60")
        ],
        "frozen_parent_method": parent,
        "cache_contract": {
            "recent": "sink1 + recent20",
            "ours": "sink1 + retrieval4 + recent16",
            "read_budget_frame_equivalents": 21,
            "retrieval_archive_capacity": 12,
            "clean_read": "Recent",
            "shared_updates": True,
        },
        "source": {
            "v207_input_manifest": str(v207_input_manifest.resolve()),
            "v207_input_manifest_sha256": sha256(v207_input_manifest),
            "v207_report": str(v207_report_path.resolve()),
            "v207_report_sha256": sha256(v207_report_path),
            "v207_comparison_manifest": str(comparison_path.resolve()),
            "v207_comparison_manifest_sha256": sha256(comparison_path),
            "sf_parity_report": str(parity_report_path.resolve()),
            "sf_parity_report_sha256": sha256(parity_report_path),
        },
        "claim_boundary": (
            "v208 replicates the frozen v207 candidate on MovieGen-128 with a "
            "new seed, shared between methods and 30/60-second scopes. Primary "
            "paired tests use the 96 prompts outside v207 selection; full128 "
            "and development32 are descriptive. These 96 prompts may have "
            "been used in older experiments: this is not historically fresh "
            "confirmation. A pass is statistical support on this benchmark, "
            "not evidence of cross-model generalization or improved motion."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("version") != 1
        or payload.get("experiment") != EXPERIMENT
        or payload.get("paper_stage") is not True
        or payload.get("method_order") != list(METHOD_ORDER)
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
        or payload.get("frozen_parent_method") not in ALLOWED_PARENT_METHODS
    ):
        raise ValueError("invalid v208 input manifest")
    prompt_path = Path(payload["prompt_file"])
    prompts = _load_source(prompt_path)
    if sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v208 prompt hash drift")
    if prompts != [str(row["text"]) for row in payload["prompt_items"]]:
        raise ValueError("v208 prompt-item drift")
    scopes = {str(row["key"]): row for row in payload.get("scopes") or ()}
    if set(scopes) != {"main30", "long60"}:
        raise ValueError("v208 scope membership drift")
    for key, row in scopes.items():
        if (
            int(row.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES[key]
            or row.get("decoded_video_contract") != DECODED_CONTRACT[key]
            or int(row.get("prompt_count", -1)) != PROMPT_COUNT
        ):
            raise ValueError(f"v208 scope contract drift: {key}")
    methods = payload.get("methods") or {}
    if set(methods) != set(METHOD_ORDER):
        raise ValueError("v208 method membership drift")
    if methods["ours"].get("schedule") != ALLOWED_PARENT_METHODS[payload["frozen_parent_method"]]:
        raise ValueError("v208 selected schedule drift")
    if (
        methods["recent21"].get("schedule") != "recent"
        or methods["recent21"].get("coverage_noisy_calls") != []
        or methods["ours"].get("coverage_noisy_calls")
        != ([0] if methods["ours"]["schedule"] == "early1" else [0, 1])
    ):
        raise ValueError("v208 phase-call contract drift")
    for method in ("recent21", "ours"):
        row = methods[method]
        map_path = Path(row["head_map"])
        if (
            row.get("operator") != "retrieval"
            or int(row.get("read_frame_equivalents", -1)) != 21
            or int(row.get("archive_capacity", -1)) != 12
            or int(row.get("sink_frames", -1)) != 1
            or int(row.get("middle_frames", -1)) != 4
            or int(row.get("recent_route_frames", -1)) != 20
            or int(row.get("coverage_recent_frames", -1)) != 16
            or sha256(map_path) != row["head_map_sha256"]
            or map_path.read_bytes() != _head_map_payload()
        ):
            raise ValueError(f"v208 cache contract drift: {method}")
    selection_splits(payload["prompt_items"])
    for key in ("v207_input_manifest", "v207_report", "v207_comparison_manifest"):
        path = Path(payload["source"][key])
        if not path.is_file() or sha256(path) != payload["source"][key + "_sha256"]:
            raise ValueError(f"v208 parent provenance drift: {key}")
    parity_path = Path(payload["source"]["sf_parity_report"])
    if (
        not parity_path.is_file()
        or sha256(parity_path) != payload["source"]["sf_parity_report_sha256"]
    ):
        raise ValueError("v208 SF parity provenance drift")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity.get("experiment") != PARITY_EXPERIMENT
        or parity.get("parity_pass") is not True
        or parity.get("decision") != "parity_pass_proceed_to_budget_phase_screen"
    ):
        raise ValueError("v208 SF parity status drift")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--v207-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v207-report", type=Path, required=True)
    prepare_parser.add_argument("--parity-report", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(
            args.source_prompts,
            args.v207_input_manifest,
            args.v207_report,
            args.parity_report,
            args.output_root,
        )
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v208-inputs] PASS "
        f"parent={payload['frozen_parent_method']} methods=3 "
        "scopes=main30,long60 prompts=128"
    )


if __name__ == "__main__":
    main()
