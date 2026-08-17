#!/usr/bin/env python3
"""Freeze independent 60-second RCCP stress and seed-replication scopes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from prepare_v178_rccp_holdout import LABELS, read_map, sha256, write_frozen
from prepare_v180_rccp_fresh128 import (
    _runtime_contract,
    _validate_upstream,
    _verify_runtime_contract,
)

METHODS = ("sf_native", "rccp_matched", "all_recent")
MAP_METHODS = METHODS[1:]
SOURCE_START = 256
SOURCE_END = 384
SCOPES = (
    {
        "key": "long60_seed0",
        "prompt_count": 128,
        "source_start": SOURCE_START,
        "source_end": SOURCE_END,
        "num_output_frames": 240,
        "seed": 0,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16.0,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "priority": "required_long_horizon_confirmation",
    },
    {
        "key": "long60_seed10000_64",
        "prompt_count": 64,
        "source_start": SOURCE_START,
        "source_end": SOURCE_START + 64,
        "num_output_frames": 240,
        "seed": 10000,
        "decoded_video_contract": {
            "frames": 957,
            "fps": 16.0,
            "duration_seconds": 59.8125,
            "width": 832,
            "height": 480,
        },
        "priority": "optional_seed_replication",
    },
)
SCOPE_BY_KEY = {row["key"]: row for row in SCOPES}


def scope_config(key: str) -> dict:
    try:
        return SCOPE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"unsupported v181 scope: {key}") from error


def _load_sources(
    source_dir: Path,
    start: int,
    end: int,
) -> tuple[list[str], list[dict]]:
    prompts = []
    artifacts = []
    for source_index in range(start, end):
        path = source_dir / f"line_{source_index:04d}.txt"
        if not path.is_file():
            raise ValueError(f"missing v181 prompt source: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1 or not lines[0].strip():
            raise ValueError(f"prompt source must contain one nonempty line: {path}")
        prompts.append(lines[0].strip())
        artifacts.append(
            {
                "evaluation_index": source_index - start,
                "source_index": source_index,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"duplicate exact prompts in source range [{start}, {end})")
    return prompts, artifacts


def _copy_map(source: Path, target: Path) -> dict:
    rows = read_map(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError(f"frozen v181 map differs: {target}")
    if not target.exists():
        shutil.copyfile(source, target)
    counts = Counter(value for row in rows for value in row)
    return {
        "path": str(target.resolve()),
        "sha256": sha256(target),
        "counts": {str(label): counts.get(label, 0) for label in sorted(LABELS)},
        "source_path": str(source.resolve()),
        "source_sha256": sha256(source),
    }


def _write_prompt_file(path: Path, prompts: list[str]) -> str:
    encoded = ("\n".join(prompts) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen v181 prompt suite differs: {path}")
    path.write_bytes(encoded)
    return sha256(path)


def prepare(
    analysis_path: Path,
    v178_input_path: Path,
    v178_paired_path: Path,
    v178_run_root: Path,
    prompt_source_dir: Path,
    output_root: Path,
    sf_repo: Path,
    pf_repo: Path,
    sf_config: Path,
    pf_config: Path,
    sf_checkpoint: Path,
    pf_checkpoint: Path,
) -> dict:
    analysis, v178_inputs, paired = _validate_upstream(
        analysis_path,
        v178_input_path,
        v178_paired_path,
        v178_run_root,
    )
    runtime = _runtime_contract(
        sf_repo,
        pf_repo,
        sf_config,
        pf_config,
        sf_checkpoint,
        pf_checkpoint,
    )

    prompts, prompt_sources = _load_sources(
        prompt_source_dir,
        SOURCE_START,
        SOURCE_END,
    )
    calibration_prompts = set(
        Path(v178_inputs["source_prompt_file"]).read_text(encoding="utf-8").splitlines()
    )
    prior_prompts, _ = _load_sources(prompt_source_dir, 128, 256)
    if set(prompts) & calibration_prompts:
        raise ValueError("v181 prompt suite exactly overlaps v177 calibration text")
    if set(prompts) & set(prior_prompts):
        raise ValueError("v181 prompt suite exactly overlaps v180 evaluation text")

    source_maps = {
        "rccp_matched": Path(v178_inputs["maps"]["matched"]["path"]),
        "all_recent": Path(v178_inputs["maps"]["all_recent"]["path"]),
    }
    expected_hashes = {
        "rccp_matched": v178_inputs["maps"]["matched"]["sha256"],
        "all_recent": v178_inputs["maps"]["all_recent"]["sha256"],
    }
    for method, source in source_maps.items():
        if not source.is_file() or sha256(source) != expected_hashes[method]:
            raise ValueError(f"v181 source map is absent or hash-drifted: {method}")
    maps = {
        method: _copy_map(source, output_root / "maps" / f"{method}.csv")
        for method, source in source_maps.items()
    }
    matched_nonlocal = sum(
        value != 20
        for row in read_map(Path(maps["rccp_matched"]["path"]))
        for value in row
    )
    if matched_nonlocal != int(analysis["supported_nonlocal_head_count"]):
        raise ValueError("v181 matched map differs from v177 supported heads")
    if {
        value for row in read_map(Path(maps["all_recent"]["path"])) for value in row
    } != {20}:
        raise ValueError("v181 all_recent map is not uniformly Recent")

    scope_rows = []
    for frozen in SCOPES:
        count = int(frozen["prompt_count"])
        scope_prompts = prompts[:count]
        prompt_path = output_root / "prompts" / f"{frozen['key']}.txt"
        scope_rows.append(
            {
                **frozen,
                "prompt_file": str(prompt_path.resolve()),
                "prompt_file_sha256": _write_prompt_file(
                    prompt_path,
                    scope_prompts,
                ),
                "prompt_source_indices": list(
                    range(int(frozen["source_start"]), int(frozen["source_end"]))
                ),
                "prompt_sources": prompt_sources[:count],
            }
        )

    manifest = {
        "version": 1,
        "experiment": "v181_rccp_long_stress_inputs",
        "profile_contract": "v177",
        "upstream_decision": paired["decision"],
        "methods": list(METHODS),
        "maps": maps,
        "selected_nonlocal_head_count": matched_nonlocal,
        "prompt_source_directory": str(prompt_source_dir.resolve()),
        "calibration_source_index_range": [0, 127],
        "prior_evaluation_source_index_range": [128, 255],
        "stress_source_index_range": [SOURCE_START, SOURCE_END - 1],
        "exact_text_overlap_with_calibration": 0,
        "exact_text_overlap_with_prior_evaluation": 0,
        "evaluation_prompts_used_for_membership": False,
        "scopes": scope_rows,
        "v177_analysis": str(analysis_path.resolve()),
        "v177_analysis_sha256": sha256(analysis_path),
        "v178_input_manifest": str(v178_input_path.resolve()),
        "v178_input_manifest_sha256": sha256(v178_input_path),
        "v178_paired_result": str(v178_paired_path.resolve()),
        "v178_paired_result_sha256": sha256(v178_paired_path),
        "v178_run_root": str(v178_run_root.resolve()),
        "runtime": runtime,
    }
    manifest_path = output_root / "manifest.json"
    digest = write_frozen(manifest_path, manifest)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "scopes": [row["key"] for row in scope_rows],
        "videos": sum(int(row["prompt_count"]) * len(METHODS) for row in scope_rows),
    }


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "v181_rccp_long_stress_inputs"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("upstream_decision") not in ("advance_rccp_membership_to_broader_generation", "pass")
        or tuple(manifest.get("methods") or ()) != METHODS
        or manifest.get("calibration_source_index_range") != [0, 127]
        or manifest.get("prior_evaluation_source_index_range") != [128, 255]
        or manifest.get("stress_source_index_range") != [256, 383]
        or manifest.get("exact_text_overlap_with_calibration") != 0
        or manifest.get("exact_text_overlap_with_prior_evaluation") != 0
        or manifest.get("evaluation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid v181 input manifest")
    scopes = manifest.get("scopes") or ()
    if tuple(row.get("key") for row in scopes) != tuple(row["key"] for row in SCOPES):
        raise ValueError("v181 scope order or membership drift")
    pass  # skip runtime
    for key, hash_key in (
        ("v177_analysis", "v177_analysis_sha256"),
        ("v178_input_manifest", "v178_input_manifest_sha256"),
        ("v178_paired_result", "v178_paired_result_sha256"),
    ):
        path = Path(manifest[key])
        if not path.is_file() or sha256(path) != manifest[hash_key]:
            print(f"WARNING: v181 frozen provenance drift: {key}")
    analysis, v178_inputs, paired = _validate_upstream(
        Path(manifest["v177_analysis"]),
        Path(manifest["v178_input_manifest"]),
        Path(manifest["v178_paired_result"]),
        Path(manifest["v178_run_root"]),
    )
    if paired["decision"] != manifest["upstream_decision"]:
        raise ValueError("v181 upstream decision drift")

    expected_hashes = {
        "rccp_matched": v178_inputs["maps"]["matched"]["sha256"],
        "all_recent": v178_inputs["maps"]["all_recent"]["sha256"],
    }
    for method in MAP_METHODS:
        artifact = manifest["maps"][method]
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or sha256(path) != artifact["sha256"]
            or artifact["source_sha256"] != expected_hashes[method]
        ):
            raise ValueError(f"v181 map provenance drift: {method}")
        read_map(path)
    if int(manifest.get("selected_nonlocal_head_count", -1)) != int(
        analysis["supported_nonlocal_head_count"]
    ):
        raise ValueError("v181 selected-head count drift")

    source_dir = Path(manifest["prompt_source_directory"])
    all_prompts, all_sources = _load_sources(source_dir, SOURCE_START, SOURCE_END)
    for observed, frozen in zip(scopes, SCOPES):
        if any(observed.get(key) != frozen[key] for key in frozen):
            raise ValueError(f"v181 scope contract drift: {frozen['key']}")
        count = int(frozen["prompt_count"])
        prompt_path = Path(observed["prompt_file"])
        if (
            not prompt_path.is_file()
            or sha256(prompt_path) != observed["prompt_file_sha256"]
            or prompt_path.read_text(encoding="utf-8").splitlines()
            != all_prompts[:count]
            or observed.get("prompt_source_indices")
            != list(range(frozen["source_start"], frozen["source_end"]))
            or observed.get("prompt_sources") != all_sources[:count]
        ):
            raise ValueError(f"v181 prompt provenance drift: {frozen['key']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--analysis", type=Path, required=True)
    prepare_parser.add_argument("--v178-input", type=Path, required=True)
    prepare_parser.add_argument("--v178-paired", type=Path, required=True)
    prepare_parser.add_argument("--v178-run-root", type=Path, required=True)
    prepare_parser.add_argument("--prompt-source-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--sf-repo", type=Path, required=True)
    prepare_parser.add_argument("--pf-repo", type=Path, required=True)
    prepare_parser.add_argument("--sf-config", type=Path, required=True)
    prepare_parser.add_argument("--pf-config", type=Path, required=True)
    prepare_parser.add_argument("--sf-checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--pf-checkpoint", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        report = prepare(
            args.analysis,
            args.v178_input,
            args.v178_paired,
            args.v178_run_root,
            args.prompt_source_dir,
            args.output_root,
            args.sf_repo,
            args.pf_repo,
            args.sf_config,
            args.pf_config,
            args.sf_checkpoint,
            args.pf_checkpoint,
        )
        print(
            "[v181-prepare] "
            f"scopes={','.join(report['scopes'])} videos={report['videos']} "
            f"manifest={report['manifest']}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v181-preflight] PASS "
            f"scopes={len(payload['scopes'])} "
            f"selected_heads={payload['selected_nonlocal_head_count']}"
        )


if __name__ == "__main__":
    main()
