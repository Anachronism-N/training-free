#!/usr/bin/env python3
"""Freeze untouched prompts and maps for v178 RCCP causal validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


METHODS = (
    "matched",
    "all_recent",
    "hard_negative_0",
    "hard_negative_1",
    "hard_negative_2",
    "hard_negative_3",
)
LABELS = {20, 21, 22}
LAYERS = 30
HEADS = 12
PROMPTS = 128
HOLDOUT_PROMPTS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen artifact differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def read_map(path: Path) -> list[list[int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle)]
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(f"invalid 30x12 head map: {path}")
    observed = {value for row in rows for value in row}
    if not observed <= LABELS:
        raise ValueError(f"unsupported labels in {path}: {sorted(observed)}")
    return rows


def validate_analysis(payload: dict) -> list[int]:
    if payload.get("experiment") != "v177_strict_superset_rccp":
        raise ValueError("v178 requires v177_strict_superset_rccp analysis")
    if payload.get("profile_contract") != "v177":
        raise ValueError("v178 requires the v177 profile contract")
    if payload.get("generation_ready") is not True:
        raise ValueError("v177 did not support any generation candidate")
    if int(payload.get("supported_nonlocal_head_count", 0)) <= 0:
        raise ValueError("v177 has no supported nonlocal heads")

    audit = payload.get("profile_audit") or {}
    if (
        audit.get("profile_contract") != "v177"
        or audit.get("strict") is not True
        or audit.get("complete_profile") is not True
        or int(audit.get("record_count", -1)) != 184_320
        or list(audit.get("records_per_prompt_layer") or ()) != [48]
        or set(audit.get("prompt_ids") or ()) != set(range(PROMPTS))
    ):
        raise ValueError("v177 profile audit is incomplete or mixed")

    teacher = payload.get("teacher_contract") or {}
    if (
        teacher.get("candidate_representation_superset_required") is not True
        or teacher.get("verification_identity")
        != "physical_frame_and_representation_family"
        or int(teacher.get("union_max_ffe", -1)) != 17
    ):
        raise ValueError("v177 representation-superset contract is absent")
    provenance = payload.get("input_provenance") or {}
    prompt_sha = str(provenance.get("prompt_sha256", ""))
    provenance_manifest = Path(str(provenance.get("input_manifest", "")))
    if (
        int(provenance.get("prompt_count", -1)) != PROMPTS
        or len(prompt_sha) != 64
        or any(value not in "0123456789abcdef" for value in prompt_sha)
        or len(str(provenance.get("input_manifest_sha256", ""))) != 64
    ):
        raise ValueError("v177 profiling input provenance is absent")
    if (
        not provenance_manifest.is_file()
        or sha256(provenance_manifest)
        != provenance.get("input_manifest_sha256")
    ):
        raise ValueError("v177 profiling input manifest hash drift")

    split = payload.get("prompt_split") or {}
    discovery = [int(value) for value in split.get("discovery_prompt_ids") or ()]
    validation = [int(value) for value in split.get("validation_prompt_ids") or ()]
    generation = [int(value) for value in split.get("generation_prompt_ids") or ()]
    if split.get("generation_prompts_used_for_membership") is not False:
        raise ValueError("generation prompts were exposed to membership selection")
    if (len(discovery), len(validation), len(generation)) != (64, 32, 32):
        raise ValueError("v177 prompt split has the wrong cardinality")
    split_sets = [set(discovery), set(validation), set(generation)]
    if any(len(values) != size for values, size in zip(split_sets, (64, 32, 32))):
        raise ValueError("v177 prompt split contains duplicate ids")
    if any(split_sets[left] & split_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("v177 prompt split overlaps")
    if set().union(*split_sets) != set(range(PROMPTS)):
        raise ValueError("v177 prompt split is not exhaustive")
    return generation


def validate_maps(payload: dict) -> dict[str, tuple[Path, list[list[int]]]]:
    artifacts = payload.get("maps") or {}
    result = {}
    for method in METHODS:
        artifact = artifacts.get(method) or {}
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or sha256(path) != artifact.get("sha256"):
            raise ValueError(f"{method}: missing or hash-drifted map")
        result[method] = (path, read_map(path))

    matched = result["matched"][1]
    if not any(value != 20 for row in matched for value in row):
        raise ValueError("matched map contains no nonlocal assignments")
    if any(value != 20 for row in result["all_recent"][1] for value in row):
        raise ValueError("all_recent is not uniformly label 20")
    for method in METHODS[2:]:
        control = result[method][1]
        if control == matched:
            raise ValueError(f"{method}: hard negative equals matched membership")
        for layer in range(LAYERS):
            if Counter(control[layer]) != Counter(matched[layer]):
                raise ValueError(
                    f"{method}: layer {layer} does not match policy counts"
                )
    negative_signatures = {
        tuple(tuple(row) for row in result[method][1]) for method in METHODS[2:]
    }
    if len(negative_signatures) != 4:
        raise ValueError("v177 hard-negative maps are not four unique memberships")
    return result


def prepare(
    analysis_path: Path,
    source_prompts_path: Path,
    output_root: Path,
) -> dict:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    generation_ids = validate_analysis(analysis)
    source_prompts = source_prompts_path.read_text(encoding="utf-8").splitlines()
    if len(source_prompts) != PROMPTS or any(not value.strip() for value in source_prompts):
        raise ValueError("source prompt suite must contain 128 nonempty lines")
    if sha256(source_prompts_path) != analysis["input_provenance"]["prompt_sha256"]:
        raise ValueError("source prompt suite differs from v177 profiling input")
    maps = validate_maps(analysis)

    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = output_root / "generation_holdout32.txt"
    prompt_bytes = (
        "\n".join(source_prompts[index] for index in generation_ids) + "\n"
    ).encode("utf-8")
    if prompt_path.exists() and prompt_path.read_bytes() != prompt_bytes:
        raise RuntimeError(f"frozen holdout prompt file differs: {prompt_path}")
    prompt_path.write_bytes(prompt_bytes)

    map_manifest = {}
    for method, (source, rows) in maps.items():
        target = output_root / "maps" / f"{method}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"frozen head map differs: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
        counts = Counter(value for row in rows for value in row)
        map_manifest[method] = {
            "path": str(target.resolve()),
            "sha256": sha256(target),
            "counts": {str(label): counts.get(label, 0) for label in sorted(LABELS)},
            "per_layer": [
                {str(label): row.count(label) for label in sorted(LABELS)}
                for row in rows
            ],
            "source_path": str(source.resolve()),
        }

    manifest = {
        "version": 1,
        "experiment": "v178_rccp_holdout_inputs",
        "profile_contract": "v177",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256(analysis_path),
        "source_prompt_file": str(source_prompts_path.resolve()),
        "source_prompt_sha256": sha256(source_prompts_path),
        "holdout_prompt_file": str(prompt_path.resolve()),
        "holdout_prompt_sha256": sha256(prompt_path),
        "source_prompt_ids": generation_ids,
        "prompt_count": HOLDOUT_PROMPTS,
        "methods": list(METHODS),
        "maps": map_manifest,
        "generation_prompts_used_for_membership": False,
        "num_output_frames": 120,
        "seed": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_sha = write_frozen(manifest_path, manifest)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_count": HOLDOUT_PROMPTS,
        "method_count": len(METHODS),
    }


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "v178_rccp_holdout_inputs"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
        or int(manifest.get("prompt_count", -1)) != HOLDOUT_PROMPTS
        or tuple(manifest.get("methods") or ()) != METHODS
    ):
        raise ValueError("invalid v178 input manifest")
    analysis_path = Path(manifest["analysis"])
    source_path = Path(manifest["source_prompt_file"])
    prompt_path = Path(manifest["holdout_prompt_file"])
    for path, expected in (
        (analysis_path, manifest["analysis_sha256"]),
        (source_path, manifest["source_prompt_sha256"]),
        (prompt_path, manifest["holdout_prompt_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v178 frozen input hash drift: {path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    generation_ids = validate_analysis(analysis)
    if generation_ids != [int(value) for value in manifest["source_prompt_ids"]]:
        raise ValueError("v178 holdout source ids differ from v177")
    source_prompts = source_path.read_text(encoding="utf-8").splitlines()
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if prompts != [source_prompts[index] for index in generation_ids]:
        raise ValueError("v178 holdout prompt text or order drift")
    frozen_maps = {}
    for method in METHODS:
        artifact = manifest["maps"][method]
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"{method}: frozen map hash drift")
        frozen_maps[method] = (path, read_map(path))
    matched = frozen_maps["matched"][1]
    for method in METHODS[2:]:
        for layer, row in enumerate(frozen_maps[method][1]):
            if Counter(row) != Counter(matched[layer]):
                raise ValueError(f"{method}: frozen layer-count drift at L{layer}")
    negative_signatures = {
        tuple(tuple(row) for row in frozen_maps[method][1])
        for method in METHODS[2:]
    }
    if len(negative_signatures) != 4:
        raise ValueError("v178 frozen hard negatives are not unique")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--analysis", type=Path, required=True)
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        report = prepare(args.analysis, args.source_prompts, args.output_root)
        print(
            "[v178-prepare] "
            f"methods={report['method_count']} prompts={report['prompt_count']} "
            f"manifest={report['manifest']}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v178-preflight] frozen inputs: PASS "
            f"methods={len(payload['methods'])} prompts={payload['prompt_count']}"
        )


if __name__ == "__main__":
    main()
