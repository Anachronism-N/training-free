#!/usr/bin/env python3
"""Freeze inputs for the v184 denoising-phase Coverage screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPT_COUNT = 32
SOURCE_PROMPT_COUNT = 128
SOURCE_INDICES = tuple(range(2, SOURCE_PROMPT_COUNT, 4))
METHODS = (
    "all_recent",
    "coverage_early1",
    "coverage_early2",
    "coverage_late2",
    "all_coverage_noisy",
)
SCHEDULES = {
    "all_recent": "recent",
    "coverage_early1": "early1",
    "coverage_early2": "early2",
    "coverage_late2": "late2",
    "all_coverage_noisy": "coverage",
}
COVERAGE_CALLS = {
    "all_recent": (),
    "coverage_early1": (0,),
    "coverage_early2": (0, 1),
    "coverage_late2": (2, 3),
    "all_coverage_noisy": (0, 1, 2, 3),
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
        raise RuntimeError(f"frozen v184 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _load_source(path: Path) -> list[str]:
    prompts = path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != SOURCE_PROMPT_COUNT:
        raise ValueError("v184 source must contain exactly 128 prompts")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("v184 source contains an empty prompt")
    return prompts


def _head_map_payload() -> bytes:
    return (
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30))
        + "\n"
    ).encode("ascii")


def prepare(source_prompts: Path, output_root: Path) -> dict:
    if len(SOURCE_INDICES) != PROMPT_COUNT or len(set(SOURCE_INDICES)) != PROMPT_COUNT:
        raise RuntimeError("v184 systematic prompt indices are invalid")
    source = _load_source(source_prompts)
    selected = [source[index].strip() for index in SOURCE_INDICES]
    prompt_path = output_root / "prompts" / "moviegen_qwen_systematic32.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(selected) + "\n").encode("utf-8"),
    )
    map_path = output_root / "maps" / "all_profile_banks.csv"
    map_sha = write_frozen(map_path, _head_map_payload())

    methods = {
        method: {
            "schedule": SCHEDULES[method],
            "coverage_noisy_calls": list(COVERAGE_CALLS[method]),
            "clean_policy": "recent",
            "head_map": str(map_path.resolve()),
            "head_map_sha256": map_sha,
            "head_route_counts": {"10": 360, "11": 0},
            "read_frame_equivalents": 9,
        }
        for method in METHODS
    }
    payload = {
        "version": 1,
        "experiment": "v184_denoise_phase_coverage_screen",
        "scope": "development32",
        "claim_boundary": (
            "This 32-prompt screen identifies a denoising-call intervention. "
            "It is not a final benchmark or a static-head membership claim."
        ),
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": SOURCE_PROMPT_COUNT,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "source_indices": list(SOURCE_INDICES),
        "selection_policy": (
            "predeclared systematic sample: source indices 2+4k for k=0..31"
        ),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {
                "index": index,
                "source_index": source_index,
                "text": selected[index],
            }
            for index, source_index in enumerate(SOURCE_INDICES)
        ],
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "denoising_call_count": 4,
        "method_order": list(METHODS),
        "methods": methods,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + reservoir4 + recent4",
            "clean_read": "Recent for every method",
            "bank_updates": (
                "all methods update Recent, Coverage, and unused Episode shadow "
                "banks from the same clean-pass update rule"
            ),
            "coverage_operator": "TemporalReservoirStrategy(capacity=4, seed=2026)",
            "dynamic_rope": True,
            "read_budget_frame_equivalents": 9,
        },
        "primary_metrics": [
            "official_quality_score",
            "identity_background",
            "dynamic_degree",
            "temporal_mechanics",
        ],
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
        payload.get("experiment") != "v184_denoise_phase_coverage_screen"
        or payload.get("scope") != "development32"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("denoising_call_count", -1)) != 4
        or tuple(payload.get("method_order") or ()) != METHODS
        or payload.get("source_indices") != list(SOURCE_INDICES)
    ):
        raise ValueError("invalid v184 input manifest")
    source_path = Path(payload["source_prompt_file"])
    source = _load_source(source_path)
    if sha256(source_path) != payload["source_prompt_file_sha256"]:
        raise ValueError("v184 source prompt file hash drift")
    prompt_path = Path(payload["prompt_file"])
    selected = prompt_path.read_text(encoding="utf-8").splitlines()
    if (
        len(selected) != PROMPT_COUNT
        or sha256(prompt_path) != payload["prompt_file_sha256"]
        or selected != [source[index].strip() for index in SOURCE_INDICES]
    ):
        raise ValueError("v184 selected prompt suite drift")
    if set(payload.get("methods") or {}) != set(METHODS):
        raise ValueError("v184 method set drift")
    for method in METHODS:
        row = payload["methods"][method]
        if (
            row.get("schedule") != SCHEDULES[method]
            or row.get("coverage_noisy_calls") != list(COVERAGE_CALLS[method])
            or row.get("clean_policy") != "recent"
            or row.get("head_route_counts") != {"10": 360, "11": 0}
        ):
            raise ValueError(f"v184 method contract drift: {method}")
        map_path = Path(row["head_map"])
        if (
            not map_path.is_file()
            or sha256(map_path) != row["head_map_sha256"]
            or map_path.read_bytes() != _head_map_payload()
        ):
            raise ValueError(f"v184 shared bank map drift: {method}")
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
        "[v184-inputs] PASS "
        f"methods={len(payload['methods'])} prompts={payload['prompt_count']} "
        f"scope={payload['scope']}"
    )


if __name__ == "__main__":
    main()
