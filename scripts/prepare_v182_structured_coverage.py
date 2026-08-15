#!/usr/bin/env python3
"""Freeze the v182 structured-Coverage development screen inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


METHODS = (
    "all_recent",
    "strict5_reservoir",
    "strict5_landmark",
    "strict5_prototype",
    "strict5_retrieval",
)
POLICIES = {
    "all_recent": "reservoir",
    "strict5_reservoir": "reservoir",
    "strict5_landmark": "landmark",
    "strict5_prototype": "prototype",
    "strict5_retrieval": "retrieval",
}
STRATEGIES = {
    "all_recent": None,
    "strict5_reservoir": "TemporalReservoirStrategy",
    "strict5_landmark": "SemanticLandmarkStrategy",
    "strict5_prototype": "TemporalPrototypeStrategy",
    "strict5_retrieval": "SemanticRetrievalStrategy",
}
STRICT_HEADS = ((0, 10), (5, 3), (6, 6), (8, 6), (23, 2))
PROMPT_COUNT = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v182 input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_head_map(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[int(value.strip()) for value in row] for row in csv.reader(handle) if row]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"head map must be 30x12: {path}")
    observed = {value for row in rows for value in row}
    if not observed.issubset({20, 21}):
        raise ValueError(f"unsupported labels in {path}: {sorted(observed)}")
    return rows


def encode_head_map(rows: list[list[int]]) -> bytes:
    return ("\n".join(",".join(str(value) for value in row) for row in rows) + "\n").encode()


def validate_strict_map(rows: list[list[int]]) -> None:
    selected = tuple(
        (layer, head)
        for layer, row in enumerate(rows)
        for head, label in enumerate(row)
        if label == 21
    )
    if selected != STRICT_HEADS:
        raise ValueError(
            "v182 requires the frozen v177 strict-five map; "
            f"observed={selected}"
        )
    if Counter(value for row in rows for value in row) != Counter({20: 355, 21: 5}):
        raise ValueError("v182 strict-five route counts drifted")


def validate_prompt_file(path: Path) -> list[str]:
    prompts = path.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not prompt.strip() for prompt in prompts):
        raise ValueError("v182 requires exactly 16 nonempty development prompts")
    return prompts


def prepare(source_prompts: Path, strict_map: Path, output_root: Path) -> dict:
    prompts = validate_prompt_file(source_prompts)
    strict_rows = read_head_map(strict_map)
    validate_strict_map(strict_rows)
    recent_rows = [[20 for _ in range(12)] for _ in range(30)]

    prompts_path = output_root / "prompts" / "moviegen_diverse16.txt"
    prompt_payload = ("\n".join(prompts) + "\n").encode("utf-8")
    prompt_sha = write_frozen(prompts_path, prompt_payload)
    strict_path = output_root / "maps" / "strict5.csv"
    recent_path = output_root / "maps" / "all_recent.csv"
    strict_sha = write_frozen(strict_path, encode_head_map(strict_rows))
    recent_sha = write_frozen(recent_path, encode_head_map(recent_rows))

    method_rows = {}
    for method in METHODS:
        is_recent = method == "all_recent"
        method_rows[method] = {
            "coverage_policy": POLICIES[method],
            "expected_middle_strategy": STRATEGIES[method],
            "head_map": str((recent_path if is_recent else strict_path).resolve()),
            "head_map_sha256": recent_sha if is_recent else strict_sha,
            "route_counts": {"20": 360, "21": 0, "22": 0}
            if is_recent
            else {"20": 355, "21": 5, "22": 0},
            "read_frame_equivalents": 9,
            "middle_read_capacity": 0 if is_recent else 4,
            "middle_storage_capacity": (
                0 if is_recent else 12 if method == "strict5_retrieval" else 4
            ),
        }

    payload = {
        "version": 1,
        "experiment": "v182_structured_coverage_screen",
        "scope": "development_only",
        "claim_boundary": (
            "This screen compares Coverage operators with frozen v177 membership. "
            "It cannot confirm RCCP membership or a final paper method."
        ),
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(prompts_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "method_order": list(METHODS),
        "strict_coverage_heads": [
            {"layer": layer, "head": head} for layer, head in STRICT_HEADS
        ],
        "cache_contract": {
            "recent": "sink1 + recent8",
            "coverage": "sink1 + middle4 + recent4",
            "episode": "unused",
            "exclusive_owner": True,
            "dynamic_rope": True,
        },
        "methods": method_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_payload = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_frozen(manifest_path, manifest_payload)
    return payload


def verify(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "v182_structured_coverage_screen":
        raise ValueError("not a v182 structured-Coverage manifest")
    if tuple(payload.get("method_order") or ()) != METHODS:
        raise ValueError("v182 method order or membership drifted")
    if set(payload.get("methods") or {}) != set(METHODS):
        raise ValueError("v182 method membership drifted")
    prompt_path = Path(payload["prompt_file"])
    validate_prompt_file(prompt_path)
    if sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v182 prompt hash drift")

    for method, row in payload["methods"].items():
        map_path = Path(row["head_map"])
        if not map_path.is_file() or sha256(map_path) != row["head_map_sha256"]:
            raise ValueError(f"{method}: missing or hash-drifted head map")
        rows = read_head_map(map_path)
        if method == "all_recent":
            if any(value != 20 for values in rows for value in values):
                raise ValueError("all_recent map is not all Recent")
        else:
            validate_strict_map(rows)
        if row["coverage_policy"] != POLICIES[method]:
            raise ValueError(f"{method}: Coverage policy drift")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--strict-map", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "prepare":
        payload = prepare(args.source_prompts, args.strict_map, args.output_root)
    else:
        payload = verify(args.manifest)
    print(
        "[v182-inputs] PASS "
        f"methods={len(payload['methods'])} prompts={payload['prompt_count']} "
        f"scope={payload['scope']}"
    )


if __name__ == "__main__":
    main()
