#!/usr/bin/env python3
"""Freeze the paired v199 Retrieval archive-capacity screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPERIMENT = "v199_retrieval_storage_attribution"
PROMPT_COUNT = 32
NUM_OUTPUT_FRAMES = 240
SOURCE_PROMPT_COUNT = 128
SOURCE_INDICES = tuple(range(0, SOURCE_PROMPT_COUNT, 4))
METHODS = (
    "all_recent",
    "retrieval_archive4",
    "retrieval_archive8",
    "retrieval_archive12",
)
ARCHIVE_CAPACITY = {
    "all_recent": 0,
    "retrieval_archive4": 4,
    "retrieval_archive8": 8,
    "retrieval_archive12": 12,
}
RUNTIME_FILES = (
    "third_party/Pyramid-Forcing/inference.py",
    "third_party/Pyramid-Forcing/pipeline/causal_inference.py",
    "third_party/Pyramid-Forcing/pipeline/pyramidkv_config.py",
    "third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py",
    "third_party/Pyramid-Forcing/pyramidkv/base.py",
    "third_party/Pyramid-Forcing/pyramidkv/factory.py",
    "third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py",
    "third_party/Pyramid-Forcing/pyramidkv/role_memory.py",
)
ADVANCE_RECOMMENDATIONS = {
    "promote_retrieval_operator_to_selective_routing_validation",
    "noninferior_but_no_clear_long_history_gain",
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
        raise RuntimeError(f"refusing to replace different frozen v199 input: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: Path, payload: dict) -> str:
    """Freeze scientific inputs while allowing a later v198 gate refresh."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable_keys = set(payload) - {"upstream_v198", "generation_authorized"}
        if immutable_keys != set(previous) - {
            "upstream_v198",
            "generation_authorized",
        } or any(previous.get(key) != payload.get(key) for key in immutable_keys):
            raise RuntimeError(f"refusing to replace different frozen v199 input: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve v199 runtime commit: {completed.stderr}")
    return completed.stdout.strip()


def map_payload(label: int) -> bytes:
    return (
        "\n".join(",".join([str(label)] * 12) for _ in range(30)) + "\n"
    ).encode("ascii")


def upstream_gate(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {
            "available": False,
            "generation_authorized": False,
            "reason": "v198 decision is absent",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "v198_audited_long60_operator_comparison":
        raise ValueError("v199 received a non-v198 upstream decision")
    recommendation = str(payload.get("recommendation", ""))
    return {
        "available": True,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "recommendation": recommendation,
        "generation_authorized": recommendation in ADVANCE_RECOMMENDATIONS,
        "reason": (
            "v198 supports an archive-capacity attribution screen"
            if recommendation in ADVANCE_RECOMMENDATIONS
            else "v198 rejected or did not support further Retrieval development"
        ),
    }


def prepare(
    repo_root: Path,
    source_prompts: Path,
    output_root: Path,
    *,
    v198_decision: Path | None = None,
) -> dict:
    prompts = source_prompts.read_text(encoding="utf-8").splitlines()
    if len(prompts) != SOURCE_PROMPT_COUNT or any(not value.strip() for value in prompts):
        raise ValueError("v199 requires the complete 128-prompt v181 source suite")
    selected = [prompts[index] for index in SOURCE_INDICES]
    if len(selected) != PROMPT_COUNT or len(set(selected)) != PROMPT_COUNT:
        raise ValueError("v199 prompt subset is incomplete or contains exact duplicates")

    prompt_path = output_root / "prompts" / "moviegen_long60_stride4_32.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(selected) + "\n").encode("utf-8"),
    )
    recent_map = output_root / "maps" / "all_recent.csv"
    coverage_map = output_root / "maps" / "all_coverage.csv"
    recent_sha = write_frozen(recent_map, map_payload(20))
    coverage_sha = write_frozen(coverage_map, map_payload(21))

    runtime = {}
    for relative in RUNTIME_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        runtime[relative] = sha256(path)

    methods = []
    for method in METHODS:
        archive = ARCHIVE_CAPACITY[method]
        retrieval = archive > 0
        methods.append(
            {
                "key": method,
                "role": (
                    "same_runtime_equal_read_and_storage_control"
                    if not retrieval
                    else "retrieval_storage_attribution_candidate"
                ),
                "head_map": str((coverage_map if retrieval else recent_map).resolve()),
                "head_map_sha256": coverage_sha if retrieval else recent_sha,
                "route_counts": (
                    {"20": 0, "21": 360, "22": 0}
                    if retrieval
                    else {"20": 360, "21": 0, "22": 0}
                ),
                "read_frame_equivalents": 9,
                "sink_storage_ffe": 1,
                "recent_storage_ffe": 4 if retrieval else 8,
                "retrieval_read_capacity": 4 if retrieval else 0,
                "retrieval_archive_capacity": archive,
                "total_storage_ffe": 1 + (4 if retrieval else 8) + archive,
            }
        )

    gate = upstream_gate(v198_decision)
    manifest = {
        "version": 1,
        "experiment": EXPERIMENT,
        "development_only": True,
        "prompt_count": PROMPT_COUNT,
        "source_prompt_count": SOURCE_PROMPT_COUNT,
        "source_prompt_file": str(source_prompts.resolve()),
        "source_prompt_file_sha256": sha256(source_prompts),
        "source_indices": list(SOURCE_INDICES),
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {
                "evaluation_index": evaluation_index,
                "source_index": source_index,
                "text": selected[evaluation_index],
            }
            for evaluation_index, source_index in enumerate(SOURCE_INDICES)
        ],
        "seed": 0,
        "reseed_per_prompt": True,
        "seed_index_space": "v199 evaluation index 0..31, shared by every method",
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16.0,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "methods": methods,
        "cache_contract": {
            "all_recent": "sink1 + recent8 = read9, store9 FFE",
            "retrieval": "sink1 + top4(archiveN) + recent4 = read9 FFE",
            "exclusive_dynamic_owner": True,
            "dynamic_rope": True,
            "isolated_factor": "retrieval archive candidate capacity N in {4,8,12}",
        },
        "upstream_v198": gate,
        "generation_authorized": bool(gate["generation_authorized"]),
        "runtime": {
            "repo_root": str(repo_root.resolve()),
            "git_commit": git_commit(repo_root),
            "files": runtime,
        },
        "manual_review_required": False,
        "claim_boundary": (
            "v199 is a paired development attribution screen. It can select an "
            "archive capacity and test whether Retrieval survives equal total "
            "storage, but it is not a confirmatory paper result."
        ),
    }
    manifest_path = output_root / "manifest.json"
    write_manifest(manifest_path, manifest)
    return manifest


def verify(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != EXPERIMENT
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or payload.get("source_indices") != list(SOURCE_INDICES)
        or tuple(row.get("key") for row in payload.get("methods") or ()) != METHODS
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
    ):
        raise ValueError("invalid v199 input manifest")
    prompt_path = Path(payload["prompt_file"])
    if not prompt_path.is_file() or sha256(prompt_path) != payload["prompt_file_sha256"]:
        raise ValueError("v199 prompt file is absent or hash-drifted")
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    if prompts != [row["text"] for row in payload["prompt_items"]]:
        raise ValueError("v199 prompt provenance drift")
    for row in payload["methods"]:
        path = Path(row["head_map"])
        if not path.is_file() or sha256(path) != row["head_map_sha256"]:
            raise ValueError(f"v199 map drift: {row['key']}")
        archive = ARCHIVE_CAPACITY[row["key"]]
        if (
            int(row["retrieval_archive_capacity"]) != archive
            or int(row["read_frame_equivalents"]) != 9
            or int(row["total_storage_ffe"])
            != (9 if archive == 0 else 5 + archive)
        ):
            raise ValueError(f"v199 cache budget drift: {row['key']}")
    source = Path(payload["source_prompt_file"])
    if not source.is_file() or sha256(source) != payload["source_prompt_file_sha256"]:
        raise ValueError("v199 source prompt file drift")
    repo_root = Path(payload["runtime"]["repo_root"])
    for relative, expected in payload["runtime"]["files"].items():
        path = repo_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v199 runtime drift: {relative}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, required=True)
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--v198-decision", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.repo_root.resolve(),
            args.source_prompts.resolve(),
            args.output_root.resolve(),
            v198_decision=(
                None if args.v198_decision is None else args.v198_decision.resolve()
            ),
        )
    else:
        payload = verify(args.manifest.resolve())
    print(
        "[v199-inputs] PASS "
        f"prompts={payload['prompt_count']} methods={len(payload['methods'])} "
        f"authorized={str(payload['generation_authorized']).lower()}"
    )


if __name__ == "__main__":
    main()
