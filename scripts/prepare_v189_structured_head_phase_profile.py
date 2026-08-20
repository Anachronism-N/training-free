#!/usr/bin/env python3
"""Freeze inputs for structured Head x Denoising-Phase profiling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROMPT_COUNT = 128
LAYERS = 30
HEADS = 12
CALLS = 4
DISCOVERY_COUNT = 64
VALIDATION_COUNT = 32
HOLDOUT_COUNT = 32
SPLIT_SEED = 1892026
OPERATORS = ("landmark", "retrieval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_split() -> tuple[list[int], list[int], list[int]]:
    order = np.random.default_rng(SPLIT_SEED).permutation(PROMPT_COUNT).tolist()
    discovery = sorted(int(value) for value in order[:DISCOVERY_COUNT])
    validation = sorted(
        int(value)
        for value in order[DISCOVERY_COUNT : DISCOVERY_COUNT + VALIDATION_COUNT]
    )
    holdout = sorted(int(value) for value in order[-HOLDOUT_COUNT:])
    return discovery, validation, holdout


def _write_frozen(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"frozen v189 input differs: {path}")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def prepare(source_prompts: Path, output_root: Path) -> dict:
    prompts = source_prompts.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not value.strip() for value in prompts):
        raise ValueError("v189 requires exactly 128 non-empty prompts")
    prompt_path = output_root / "moviegen_128_qwen.txt"
    prompt_sha = _write_frozen(
        prompt_path,
        ("\n".join(value.strip() for value in prompts) + "\n").encode("utf-8"),
    )
    map_path = output_root / "profile_all_heads.csv"
    rows = [[10] * HEADS for _ in range(LAYERS)]
    map_content = "".join(
        ",".join(str(value) for value in row) + "\n" for row in rows
    ).encode("ascii")
    map_sha = _write_frozen(map_path, map_content)
    discovery, validation, holdout = frozen_split()
    payload = {
        "version": 1,
        "experiment": "v189_structured_head_phase_profile",
        "profile_contract": "v189",
        "profile_artifact_version": 4,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_count": PROMPT_COUNT,
        "profile_map": str(map_path.resolve()),
        "profile_map_sha256": map_sha,
        "profile_map_shape": [LAYERS, HEADS],
        "operators": list(OPERATORS),
        "operator_history_policies": {
            "landmark": "landmark",
            "retrieval": "retrieval",
        },
        "candidate_budgets_ffe": {"recent": 9, "coverage": 9},
        "teacher_max_budget_ffe": 13,
        "teacher_requires_representation_candidate_superset": True,
        "teacher_candidates": ["recent", "coverage"],
        "calls": list(range(CALLS)),
        "records_per_prompt_layer": 48,
        "split_seed": SPLIT_SEED,
        "prompt_split": {
            "discovery": discovery,
            "validation": validation,
            "generation_holdout": holdout,
        },
        "claim_boundary": (
            "Active generation always reads Recent. Landmark and Retrieval are "
            "shadow candidates; no profiled readout changes the latent trajectory."
        ),
    }
    manifest_path = output_root / "manifest.json"
    _write_frozen(
        manifest_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "v189_structured_head_phase_profile"
        or payload.get("profile_contract") != "v189"
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or payload.get("operators") != list(OPERATORS)
        or payload.get("calls") != list(range(CALLS))
        or int(payload.get("records_per_prompt_layer", -1)) != 48
    ):
        raise ValueError("invalid v189 input manifest")
    prompt_path = Path(payload["prompt_file"])
    map_path = Path(payload["profile_map"])
    if sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v189 prompt hash drift")
    if sha256(map_path) != payload["profile_map_sha256"]:
        raise ValueError("v189 profile map hash drift")
    with map_path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != LAYERS or any(row != ["10"] * HEADS for row in rows):
        raise ValueError("v189 profile map shape or labels drifted")
    discovery, validation, holdout = frozen_split()
    if payload.get("prompt_split") != {
        "discovery": discovery,
        "validation": validation,
        "generation_holdout": holdout,
    }:
        raise ValueError("v189 frozen prompt split drifted")
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
        "[v189-inputs] PASS "
        f"prompts={payload['prompt_count']} operators={payload['operators']}"
    )


if __name__ == "__main__":
    main()
