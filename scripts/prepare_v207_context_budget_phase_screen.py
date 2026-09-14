#!/usr/bin/env python3
"""Freeze the v207 context-budget and denoising-phase recovery screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPERIMENT = "v207_context_budget_phase_recovery"
PROMPT_COUNT = 32
SOURCE_PROMPT_COUNT = 128
SOURCE_INDICES = tuple(range(1, SOURCE_PROMPT_COUNT, 4))
NUM_OUTPUT_FRAMES = 120
SEED = 20700
CALL_COUNT = 4
LAYERS = 30
HEADS = 12
METHOD_ORDER = (
    "sf_native",
    "recent21",
    "retrieval21_early1",
    "retrieval21_early2",
    "retrieval21_late2",
    "retrieval21_full",
    "landmark21_early2",
    "recent13",
    "retrieval13_early2",
)
METHOD_SPECS = {
    "sf_native": ("sf_native", None, None, 21, "canonical_sf_baseline"),
    "recent21": ("cache", "retrieval", "recent", 21, "equal_budget_local_control"),
    "retrieval21_early1": ("cache", "retrieval", "early1", 21, "phase_candidate"),
    "retrieval21_early2": ("cache", "retrieval", "early2", 21, "primary_phase_candidate"),
    "retrieval21_late2": ("cache", "retrieval", "late2", 21, "equal_dose_phase_control"),
    "retrieval21_full": ("cache", "retrieval", "coverage", 21, "full_exposure_control"),
    "landmark21_early2": ("cache", "landmark", "early2", 21, "operator_control"),
    "recent13": ("cache", "retrieval", "recent", 13, "compressed_local_control"),
    "retrieval13_early2": ("cache", "retrieval", "early2", 13, "budget_curve_candidate"),
}
COVERAGE_CALLS = {
    None: (),
    "recent": (),
    "early1": (0,),
    "early2": (0, 1),
    "late2": (2, 3),
    "coverage": (0, 1, 2, 3),
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
        raise RuntimeError(f"frozen v207 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _load_source(path: Path) -> list[str]:
    prompts = path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != SOURCE_PROMPT_COUNT or any(not row.strip() for row in prompts):
        raise ValueError("v207 source must contain exactly 128 non-empty prompts")
    return [row.strip() for row in prompts]


def _head_map_payload() -> bytes:
    return (
        "\n".join(",".join("10" for _ in range(HEADS)) for _ in range(LAYERS))
        + "\n"
    ).encode("ascii")


def _method_rows(head_map: Path, head_map_sha: str) -> dict[str, dict]:
    rows = {}
    for method in METHOD_ORDER:
        runtime, operator, schedule, budget, role = METHOD_SPECS[method]
        recent_frames = None if runtime == "sf_native" else budget - 1
        coverage_recent = None if runtime == "sf_native" else budget - 5
        rows[method] = {
            "runtime": runtime,
            "role": role,
            "operator": operator,
            "history_policy": operator,
            "schedule": schedule,
            "coverage_noisy_calls": list(COVERAGE_CALLS[schedule]),
            "clean_policy": None if runtime == "sf_native" else "recent",
            "read_frame_equivalents": budget,
            "sink_frames": None if runtime == "sf_native" else 1,
            "middle_frames": None if runtime == "sf_native" else 4,
            "recent_route_frames": recent_frames,
            "coverage_recent_frames": coverage_recent,
            "head_map": None if runtime == "sf_native" else str(head_map.resolve()),
            "head_map_sha256": None if runtime == "sf_native" else head_map_sha,
            "head_route_counts": None if runtime == "sf_native" else {"10": 360, "11": 0},
            "archive_capacity": 12 if operator == "retrieval" else None,
        }
    return rows


def prepare(source_prompts: Path, output_root: Path) -> dict:
    if len(SOURCE_INDICES) != PROMPT_COUNT:
        raise RuntimeError("v207 systematic prompt split drift")
    source = _load_source(source_prompts)
    selected = [source[index] for index in SOURCE_INDICES]
    prompt_path = output_root / "prompts" / "moviegen_qwen_systematic32.txt"
    prompt_sha = write_frozen(
        prompt_path, ("\n".join(selected) + "\n").encode("utf-8")
    )
    map_path = output_root / "maps" / "all_profile_banks.csv"
    map_sha = write_frozen(map_path, _head_map_payload())
    methods = _method_rows(map_path, map_sha)
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": "development32",
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": SOURCE_PROMPT_COUNT,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "source_indices": list(SOURCE_INDICES),
        "selection_policy": "predeclared systematic sample: source indices 1+4k",
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": i, "source_index": source_i, "text": selected[i]}
            for i, source_i in enumerate(SOURCE_INDICES)
        ],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": SEED,
        "denoising_call_count": CALL_COUNT,
        "method_order": list(METHOD_ORDER),
        "methods": methods,
        "cache_contract": {
            "budget_formula": {
                "recent": "sink1 + recent(B-1)",
                "coverage": "sink1 + middle4 + recent(B-5)",
            },
            "budgets_ffe": [13, 21],
            "clean_read": "Recent for every cache method",
            "shared_updates": True,
            "exact_middle_kv": True,
            "retrieval_archive_capacity": 12,
            "model_local_attention_frames": 21,
        },
        "primary_comparisons": [
            ["retrieval21_early1", "sf_native"],
            ["retrieval21_early2", "sf_native"],
            ["retrieval21_early1", "recent21"],
            ["retrieval21_early2", "recent21"],
        ],
        "mechanism_controls": [
            ["retrieval21_early2", "retrieval21_late2"],
            ["retrieval21_early2", "retrieval21_full"],
            ["retrieval21_early2", "landmark21_early2"],
            ["retrieval13_early2", "recent13"],
        ],
        "promotion_rule": (
            "Automatic paired metrics first. Promote at most one retrieval phase "
            "only if it improves an SF-facing quality/semantic/visual axis while "
            "passing identity and temporal non-inferiority; then confirm on fresh128."
        ),
        "claim_boundary": (
            "v207 isolates context-budget redistribution and denoising-phase "
            "exposure. It is not evidence for a static head taxonomy, a final "
            "benchmark, or cross-model transfer."
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
        or payload.get("scope") != "development32"
        or payload.get("method_order") != list(METHOD_ORDER)
        or payload.get("source_indices") != list(SOURCE_INDICES)
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
    ):
        raise ValueError("invalid v207 input manifest")
    source_path = Path(payload["source_prompt_file"])
    source = _load_source(source_path)
    if sha256(source_path) != payload["source_prompt_file_sha256"]:
        raise ValueError("v207 source prompt hash drift")
    prompt_path = Path(payload["prompt_file"])
    selected = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        sha256(prompt_path) != payload["prompt_file_sha256"]
        or selected != [source[index] for index in SOURCE_INDICES]
    ):
        raise ValueError("v207 prompt suite drift")
    if set(payload.get("methods") or {}) != set(METHOD_ORDER):
        raise ValueError("v207 method membership drift")
    for method in METHOD_ORDER:
        row = payload["methods"][method]
        runtime, operator, schedule, budget, role = METHOD_SPECS[method]
        if (
            row.get("runtime") != runtime
            or row.get("operator") != operator
            or row.get("schedule") != schedule
            or row.get("role") != role
            or int(row.get("read_frame_equivalents", -1)) != budget
            or row.get("coverage_noisy_calls") != list(COVERAGE_CALLS[schedule])
        ):
            raise ValueError(f"v207 method contract drift: {method}")
        if runtime == "sf_native":
            continue
        if (
            int(row["sink_frames"]) != 1
            or int(row["middle_frames"]) != 4
            or int(row["recent_route_frames"]) != budget - 1
            or int(row["coverage_recent_frames"]) != budget - 5
            or 1 + int(row["recent_route_frames"]) != budget
            or 1 + 4 + int(row["coverage_recent_frames"]) != budget
            or row.get("head_route_counts") != {"10": 360, "11": 0}
        ):
            raise ValueError(f"v207 equal-budget formula drift: {method}")
        map_path = Path(row["head_map"])
        if sha256(map_path) != row["head_map_sha256"] or map_path.read_bytes() != _head_map_payload():
            raise ValueError(f"v207 head-bank map drift: {method}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(args.source_prompts, args.output_root)
        if args.action == "prepare"
        else verify(args.manifest)
    )
    print(
        "[v207-inputs] PASS "
        f"methods={len(payload['method_order'])} prompts={payload['prompt_count']} "
        "budgets=13,21 primary=sf_native"
    )


if __name__ == "__main__":
    main()
