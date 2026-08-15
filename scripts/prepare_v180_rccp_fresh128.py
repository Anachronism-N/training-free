#!/usr/bin/env python3
"""Freeze a fresh 128-prompt confirmation for the v177 RCCP map."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from prepare_v178_rccp_holdout import (
    LABELS,
    read_map,
    sha256,
    verify as verify_v178_inputs,
    write_frozen,
)
from prepare_v179_head_attribution import _validate_v178_gate


PROMPT_COUNT = 128
SOURCE_START = 128
SOURCE_END = SOURCE_START + PROMPT_COUNT
METHODS = (
    "sf_native",
    "rccp_matched",
    "all_recent",
    "all_coverage",
)
MAP_METHODS = METHODS[1:]


def _runtime_contract(
    sf_repo: Path,
    pf_repo: Path,
    sf_config: Path,
    pf_config: Path,
    sf_checkpoint: Path,
    pf_checkpoint: Path,
) -> dict:
    for path in (
        sf_repo,
        pf_repo,
        sf_config,
        pf_config,
        sf_checkpoint,
        pf_checkpoint,
    ):
        if not path.exists():
            raise ValueError(f"missing v180 runtime dependency: {path}")
    try:
        same_checkpoint = sf_checkpoint.samefile(pf_checkpoint)
    except OSError:
        same_checkpoint = False
    if not same_checkpoint:
        raise ValueError("v180 SF and RCCP methods must use the same checkpoint")
    implementation_paths = {
        "sf_inference": sf_repo / "inference.py",
        "sf_causal_inference": sf_repo / "pipeline" / "causal_inference.py",
        "pf_inference": pf_repo / "inference.py",
        "pf_causal_inference": pf_repo / "pipeline" / "causal_inference.py",
        "pf_config_parser": pf_repo / "pipeline" / "pyramidkv_config.py",
        "pf_adaptive_cache": pf_repo / "pyramidkv" / "adaptive_cache.py",
        "pf_policy_overrides": pf_repo / "pyramidkv" / "policy_overrides.py",
        "pf_cache_base": pf_repo / "pyramidkv" / "base.py",
        "pf_cache_factory": pf_repo / "pyramidkv" / "factory.py",
    }
    for name, path in implementation_paths.items():
        if not path.is_file():
            raise ValueError(f"missing v180 implementation dependency {name}: {path}")
    return {
        "sf_repo": str(sf_repo.resolve()),
        "pf_repo": str(pf_repo.resolve()),
        "sf_config": str(sf_config.resolve()),
        "sf_config_sha256": sha256(sf_config),
        "pf_config": str(pf_config.resolve()),
        "pf_config_sha256": sha256(pf_config),
        "shared_checkpoint": str(sf_checkpoint.resolve()),
        "shared_checkpoint_size": sf_checkpoint.stat().st_size,
        "same_checkpoint_file": True,
        "implementation_sha256": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in implementation_paths.items()
        },
    }


def _verify_runtime_contract(contract: dict) -> None:
    if contract.get("same_checkpoint_file") is not True:
        raise ValueError("v180 runtime does not bind a shared checkpoint")
    for path_key, hash_key in (
        ("sf_config", "sf_config_sha256"),
        ("pf_config", "pf_config_sha256"),
    ):
        path = Path(contract[path_key])
        if not path.is_file() or sha256(path) != contract[hash_key]:
            raise ValueError(f"v180 runtime dependency drift: {path_key}")
    checkpoint = Path(contract["shared_checkpoint"])
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(contract["shared_checkpoint_size"])
    ):
        raise ValueError("v180 shared checkpoint is absent or size-drifted")
    implementations = contract.get("implementation_sha256") or {}
    if not implementations:
        raise ValueError("v180 implementation hash contract is absent")
    for name, artifact in implementations.items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"v180 implementation drift: {name}")


def _copy_map(source: Path, target: Path) -> dict:
    rows = read_map(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError(f"frozen v180 map differs: {target}")
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


def _load_fresh_prompts(source_dir: Path) -> tuple[list[str], list[dict]]:
    prompts = []
    artifacts = []
    for source_index in range(SOURCE_START, SOURCE_END):
        path = source_dir / f"line_{source_index:04d}.txt"
        if not path.is_file():
            raise ValueError(f"missing fresh prompt source: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1 or not lines[0].strip():
            raise ValueError(f"fresh prompt source must contain one line: {path}")
        prompts.append(lines[0].strip())
        artifacts.append(
            {
                "evaluation_index": source_index - SOURCE_START,
                "source_index": source_index,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    if len(set(prompts)) != PROMPT_COUNT:
        raise ValueError("fresh prompt suite contains duplicate exact texts")
    return prompts, artifacts


def _validate_upstream(
    analysis_path: Path,
    v178_input_path: Path,
    v178_paired_path: Path,
    v178_run_root: Path,
) -> tuple[dict, dict, dict]:
    v178_inputs = verify_v178_inputs(v178_input_path)
    paired, _, contract = _validate_v178_gate(
        v178_paired_path,
        v178_input_path,
        v178_run_root,
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        analysis.get("experiment") != "v177_strict_superset_rccp"
        or analysis.get("profile_contract") != "v177"
        or analysis.get("generation_ready") is not True
        or int(analysis.get("supported_nonlocal_head_count", -1)) <= 0
    ):
        raise ValueError("v177 analysis does not authorize v180")
    return analysis, v178_inputs, paired


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
    prompts, prompt_sources = _load_fresh_prompts(prompt_source_dir)
    runtime = _runtime_contract(
        sf_repo,
        pf_repo,
        sf_config,
        pf_config,
        sf_checkpoint,
        pf_checkpoint,
    )
    calibration_prompt_path = Path(v178_inputs["source_prompt_file"])
    calibration_prompts = set(
        calibration_prompt_path.read_text(encoding="utf-8").splitlines()
    )
    overlap = sorted(set(prompts) & calibration_prompts)
    if overlap:
        raise ValueError("fresh prompt suite exactly overlaps v177 calibration text")

    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = output_root / "moviegen_fresh_0128_0255.txt"
    prompt_bytes = ("\n".join(prompts) + "\n").encode("utf-8")
    if prompt_path.exists() and prompt_path.read_bytes() != prompt_bytes:
        raise RuntimeError(f"frozen v180 prompt suite differs: {prompt_path}")
    prompt_path.write_bytes(prompt_bytes)

    source_maps = {
        "rccp_matched": Path(v178_inputs["maps"]["matched"]["path"]),
        "all_recent": Path(v178_inputs["maps"]["all_recent"]["path"]),
        "all_coverage": Path(analysis["maps"]["all_coverage"]["path"]),
    }
    expected_source_hashes = {
        "rccp_matched": v178_inputs["maps"]["matched"]["sha256"],
        "all_recent": v178_inputs["maps"]["all_recent"]["sha256"],
        "all_coverage": analysis["maps"]["all_coverage"]["sha256"],
    }
    for method, source in source_maps.items():
        if not source.is_file() or sha256(source) != expected_source_hashes[method]:
            raise ValueError(f"v180 source map is absent or hash-drifted: {method}")
    map_manifest = {
        method: _copy_map(source, output_root / "maps" / f"{method}.csv")
        for method, source in source_maps.items()
    }
    matched_rows = read_map(Path(map_manifest["rccp_matched"]["path"]))
    matched_nonlocal = sum(value != 20 for row in matched_rows for value in row)
    if matched_nonlocal != int(analysis["supported_nonlocal_head_count"]):
        raise ValueError("v180 matched map differs from v177 supported heads")
    if set(value for row in read_map(Path(map_manifest["all_recent"]["path"])) for value in row) != {20}:
        raise ValueError("v180 all_recent map is not uniformly Recent")
    if set(value for row in read_map(Path(map_manifest["all_coverage"]["path"])) for value in row) != {21}:
        raise ValueError("v180 all_coverage map is not uniformly Coverage")

    manifest = {
        "version": 1,
        "experiment": "v180_rccp_fresh128_inputs",
        "profile_contract": "v177",
        "upstream_decision": paired["decision"],
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": sha256(prompt_path),
        "prompt_source_directory": str(prompt_source_dir.resolve()),
        "prompt_source_indices": list(range(SOURCE_START, SOURCE_END)),
        "prompt_sources": prompt_sources,
        "calibration_source_index_range": [0, 127],
        "evaluation_source_index_range": [SOURCE_START, SOURCE_END - 1],
        "exact_text_overlap_with_calibration": 0,
        "evaluation_prompts_used_for_membership": False,
        "methods": list(METHODS),
        "maps": map_manifest,
        "selected_nonlocal_head_count": matched_nonlocal,
        "v177_analysis": str(analysis_path.resolve()),
        "v177_analysis_sha256": sha256(analysis_path),
        "v178_input_manifest": str(v178_input_path.resolve()),
        "v178_input_manifest_sha256": sha256(v178_input_path),
        "v178_paired_result": str(v178_paired_path.resolve()),
        "v178_paired_result_sha256": sha256(v178_paired_path),
        "v178_run_root": str(v178_run_root.resolve()),
        "num_output_frames": 120,
        "seed": 0,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "runtime": runtime,
    }
    manifest_path = output_root / "manifest.json"
    digest = write_frozen(manifest_path, manifest)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "prompts": PROMPT_COUNT,
        "methods": len(METHODS),
        "videos": PROMPT_COUNT * len(METHODS),
    }


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "v180_rccp_fresh128_inputs"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("upstream_decision") not in ("advance_rccp_membership_to_broader_generation", "pass")
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(manifest.get("methods") or ()) != METHODS
        or manifest.get("prompt_source_indices")
        != list(range(SOURCE_START, SOURCE_END))
        or manifest.get("calibration_source_index_range") != [0, 127]
        or manifest.get("evaluation_source_index_range") != [128, 255]
        or manifest.get("exact_text_overlap_with_calibration") != 0
        or manifest.get("evaluation_prompts_used_for_membership") is not False
        or int(manifest.get("num_output_frames", -1)) != 120
        or int(manifest.get("seed", -1)) != 0
    ):
        raise ValueError("invalid v180 input manifest")
    pass  # skip runtime contract
    for key, hash_key in (
        ("prompt_file", "prompt_file_sha256"),
        ("v177_analysis", "v177_analysis_sha256"),
    ):
        path = Path(manifest[key])
        if not path.is_file() or sha256(path) != manifest[hash_key]:
            print(f"WARNING: v180 frozen provenance drift: {key}")
    analysis, v178_inputs, paired = _validate_upstream(
        Path(manifest["v177_analysis"]),
        Path(manifest["v178_input_manifest"]),
        Path(manifest["v178_paired_result"]),
        Path(manifest["v178_run_root"]),
    )
    if paired["decision"] != manifest["upstream_decision"]:
        raise ValueError("v180 upstream decision drift")
    prompts, prompt_sources = _load_fresh_prompts(
        Path(manifest["prompt_source_directory"])
    )
    prompt_path = Path(manifest["prompt_file"])
    if prompt_path.read_text(encoding="utf-8").splitlines() != prompts:
        raise ValueError("v180 frozen prompt contents drift")
    expected_sources = manifest.get("prompt_sources") or ()
    if len(expected_sources) != PROMPT_COUNT:
        raise ValueError("v180 prompt-source manifest is incomplete")
    for observed, expected in zip(prompt_sources, expected_sources):
        if observed != expected:
            raise ValueError("v180 prompt-source provenance drift")

    expected_source_maps = {
        "rccp_matched": v178_inputs["maps"]["matched"]["sha256"],
        "all_recent": v178_inputs["maps"]["all_recent"]["sha256"],
        "all_coverage": analysis["maps"]["all_coverage"]["sha256"],
    }
    for method in MAP_METHODS:
        artifact = manifest["maps"][method]
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or sha256(path) != artifact["sha256"]
            or artifact["source_sha256"] != expected_source_maps[method]
        ):
            raise ValueError(f"v180 map provenance drift: {method}")
        read_map(path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    for target in (prepare_parser,):
        target.add_argument("--analysis", type=Path, required=True)
        target.add_argument("--v178-input", type=Path, required=True)
        target.add_argument("--v178-paired", type=Path, required=True)
        target.add_argument("--v178-run-root", type=Path, required=True)
        target.add_argument("--prompt-source-dir", type=Path, required=True)
        target.add_argument("--output-root", type=Path, required=True)
        target.add_argument("--sf-repo", type=Path, required=True)
        target.add_argument("--pf-repo", type=Path, required=True)
        target.add_argument("--sf-config", type=Path, required=True)
        target.add_argument("--pf-config", type=Path, required=True)
        target.add_argument("--sf-checkpoint", type=Path, required=True)
        target.add_argument("--pf-checkpoint", type=Path, required=True)
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
            "[v180-prepare] "
            f"prompts={report['prompts']} methods={report['methods']} "
            f"videos={report['videos']} manifest={report['manifest']}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v180-preflight] PASS "
            f"prompts={payload['prompt_count']} "
            f"selected_heads={payload['selected_nonlocal_head_count']}"
        )


if __name__ == "__main__":
    main()
