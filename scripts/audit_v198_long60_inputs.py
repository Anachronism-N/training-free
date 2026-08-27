#!/usr/bin/env python3
"""Audit and materialize the existing v181/v186 60-second video grid.

v198 does not generate videos.  It binds the two newly uploaded v186 methods to
the already audited v181 SF/all-Recent controls, verifies every decoded video,
and creates one prompt-correct comparison directory for downstream metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from audit_indexed_videos import _probe_video

EXPERIMENT = "v198_audited_long60_operator_comparison"
PROMPT_COUNT = 128
NUM_OUTPUT_FRAMES = 240
CLIPS_PER_VIDEO = 30
WORLD_SHARDS = 16
METHODS = (
    "sf_native",
    "all_recent",
    "pf_native",
    "all_coverage_retrieval",
)
V181_METHODS = ("sf_native", "all_recent")
V186_METHODS = ("pf_native", "all_coverage_retrieval")
METHOD_ROLES = {
    "sf_native": "reused_backbone_reference_matched_artifact_runtime",
    "all_recent": "reused_equal_read_budget_control_matched_artifact_runtime",
    "pf_native": "within_v186_campaign_reference_not_required_for_promotion",
    "all_coverage_retrieval": "candidate_all_head_retrieval_9ffe",
}
DECODED_VIDEO_CONTRACT = {
    "frames": 957,
    "fps": 16.0,
    "duration_seconds": 59.8125,
    "width": 832,
    "height": 480,
}
VIDEO_NAMES = (
    re.compile(r"^(\d+)-0_ema\.mp4$"),
    re.compile(r"^(\d{6})\.mp4$"),
)
PROGRESS = re.compile(r"^\[(\d+)/128\]", re.MULTILINE)
FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "AssertionError",
    "Segmentation fault",
    "RuntimeError:",
    "Killed",
)
RUNTIME_FILES = {
    "pf_adaptive_cache": "third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py",
    "pf_cache_base": "third_party/Pyramid-Forcing/pyramidkv/base.py",
    "pf_cache_factory": "third_party/Pyramid-Forcing/pyramidkv/factory.py",
    "pf_causal_inference": "third_party/Pyramid-Forcing/pipeline/causal_inference.py",
    "pf_config_parser": "third_party/Pyramid-Forcing/pipeline/pyramidkv_config.py",
    "pf_inference": "third_party/Pyramid-Forcing/inference.py",
    "pf_policy_overrides": "third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py",
    "sf_causal_inference": "third_party/Self-Forcing/pipeline/causal_inference.py",
    "sf_inference": "third_party/Self-Forcing/inference.py",
}
PF_LOG_COMMIT = "448867aaab871a378ec9c508b57dea8a02005300"
RETRIEVAL_LOG_COMMIT = "237198dfa4daf3663979b5e2f79b59866784dfb4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict, *, frozen: bool = False) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"refusing to replace a different frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v198 comparison input: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def source_video_dir(v181_root: Path, v186_root: Path, method: str) -> Path:
    if method in V181_METHODS:
        scope = v181_root / "scopes" / "long60_seed0"
        raw = scope / "raw" / method
        return raw if raw.is_dir() else scope / "published" / method
    if method in V186_METHODS:
        return v186_root / "raw" / method
    raise ValueError(f"unknown v198 method: {method}")


def _video_grid(video_dir: Path) -> dict[int, Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"missing v198 source video directory: {video_dir}")
    rows: dict[int, Path] = {}
    malformed = []
    for path in sorted(video_dir.glob("*.mp4")):
        match = None
        for pattern in VIDEO_NAMES:
            match = pattern.fullmatch(path.name)
            if match is not None:
                break
        if match is None:
            malformed.append(path.name)
            continue
        index = int(match.group(1))
        if index in rows:
            raise ValueError(f"duplicate prompt index {index} in {video_dir}")
        rows[index] = path
    expected = set(range(PROMPT_COUNT))
    if malformed or set(rows) != expected:
        raise ValueError(
            f"invalid v198 video grid for {video_dir}: "
            f"missing={sorted(expected - set(rows))[:12]} "
            f"extra={sorted(set(rows) - expected)[:12]} malformed={malformed[:12]}"
        )
    return rows


def audit_v186_logs(v186_root: Path, method: str) -> dict:
    if method not in V186_METHODS:
        return {
            "ok": True,
            "not_applicable": True,
            "reason": "v181 source methods are bound through their prior audit",
            "logs": [],
        }
    log_dir = v186_root / "logs" / method
    expected_names = {f"shard{rank:02d}.log" for rank in range(WORLD_SHARDS)}
    paths = sorted(log_dir.glob("*.log"))
    names = {path.name for path in paths}
    errors = []
    if names != expected_names:
        errors.append(
            f"log grid mismatch missing={sorted(expected_names - names)} "
            f"extra={sorted(names - expected_names)}"
        )
    rows = []
    for rank in range(WORLD_SHARDS):
        path = log_dir / f"shard{rank:02d}.log"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [value for value in FAILURE_PATTERNS if value in text]
        observed_progress = [int(value) for value in PROGRESS.findall(text)]
        expected_progress = [rank + 1 + WORLD_SHARDS * step for step in range(8)]
        required = {
            "prompt_count": "Number of prompts: 128" in text,
            "prompt_range": "Prompt index range: [0, 128)" in text,
            "eighty_blocks_per_prompt": text.count("block 80/80 - 238/240") == 8,
            "progress_indices": observed_progress == expected_progress,
        }
        if method == "all_coverage_retrieval":
            required.update(
                {
                    "exclusive_coverage_route": (
                        "[CacheCompatibilityPolicy] recent=20:0 coverage=21:360 "
                        "episode=22:0 coverage_policy=retrieval budget=9FFE "
                        "read_budget=9FFE owner=HeadComposition"
                    )
                    in text,
                    "retrieval_map": (
                        "Loading PyramidKV config from " in text
                        and "all_coverage_retrieval.csv" in text
                    ),
                    "no_pf_default_map": "best_labels.csv" not in text,
                }
            )
        else:
            required.update(
                {
                    "pf_default_map": "best_labels.csv" in text,
                    "no_cache_compatibility_route": "[CacheCompatibilityPolicy]"
                    not in text,
                }
            )
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        if not all(required.values()):
            errors.append(f"{path.name}: required={required}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "progress_indices": observed_progress,
                "required_markers": required,
                "failure_patterns": failures,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def _inspect_video(path: Path, *, decode: bool) -> dict:
    metadata = _probe_video(path, decode=decode)
    errors = []
    expected = DECODED_VIDEO_CONTRACT
    if int(metadata["frames"]) != int(expected["frames"]):
        errors.append(f"frames={metadata['frames']} expected={expected['frames']}")
    if abs(float(metadata["fps"]) - float(expected["fps"])) > 0.05:
        errors.append(f"fps={metadata['fps']} expected={expected['fps']}")
    if int(metadata["width"]) != int(expected["width"]):
        errors.append(f"width={metadata['width']} expected={expected['width']}")
    if int(metadata["height"]) != int(expected["height"]):
        errors.append(f"height={metadata['height']} expected={expected['height']}")
    if path.stat().st_size <= 0:
        errors.append("empty video")
    return {
        "file": path.name,
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": sha256(path),
        "metadata": metadata,
        "errors": errors,
    }


def audit_method(
    v181_root: Path,
    v186_root: Path,
    output_root: Path,
    method: str,
    *,
    workers: int,
    decode: bool,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unsupported v198 method: {method}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    video_dir = source_video_dir(v181_root, v186_root, method)
    grid = _video_grid(video_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            index: executor.submit(_inspect_video, path, decode=decode)
            for index, path in grid.items()
        }
        videos = []
        for index in range(PROMPT_COUNT):
            try:
                row = futures[index].result()
            except Exception as error:  # noqa: BLE001 - retain every media failure
                row = {
                    "file": grid[index].name,
                    "path": str(grid[index].resolve()),
                    "size": int(grid[index].stat().st_size),
                    "sha256": None,
                    "metadata": None,
                    "errors": [str(error)],
                }
            videos.append({"prompt_index": index, **row})
    logs = audit_v186_logs(v186_root, method)
    errors = [
        f"p{row['prompt_index']}:{value}" for row in videos for value in row["errors"]
    ]
    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method": method,
        "role": METHOD_ROLES[method],
        "video_dir": str(video_dir.resolve()),
        "prompt_count": PROMPT_COUNT,
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": DECODED_VIDEO_CONTRACT,
        "full_decode_required": True,
        "full_decode_performed": decode,
        "videos": videos,
        "runtime_logs": logs,
        "errors": errors,
        "ok": bool(not errors and logs["ok"] and decode),
    }
    path = output_root / "audits" / f"{method}.json"
    report["audit_sha256"] = write_json(path, report)
    if not report["ok"]:
        raise RuntimeError(
            f"v198 audit failed for {method}: media={len(errors)} logs={logs['ok']} "
            f"decode={decode}"
        )
    return report


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _current_runtime_hashes(repo_root: Path) -> dict:
    rows = {}
    for key, relative in RUNTIME_FILES.items():
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[key] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return rows


def _git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot read {commit}:{relative}: {detail}")
    return hashlib.sha256(completed.stdout).hexdigest()


def artifact_runtime_contract(
    repo_root: Path,
    old_runtime: dict,
    old_pf_config_sha256: str,
) -> dict:
    commits = (PF_LOG_COMMIT, RETRIEVAL_LOG_COMMIT)
    hashes = {}
    for commit in commits:
        hashes[commit] = {}
        for key, relative in RUNTIME_FILES.items():
            digest = _git_blob_sha256(repo_root, commit, relative)
            hashes[commit][key] = {
                "relative_path": relative,
                "sha256": digest,
                "matches_v181": digest == old_runtime[key]["sha256"],
            }
        config_hash = _git_blob_sha256(
            repo_root,
            commit,
            "third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml",
        )
        hashes[commit]["pf_config"] = {
            "relative_path": "third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml",
            "sha256": config_hash,
            "matches_v181": config_hash == old_pf_config_sha256,
        }
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--name-only",
            PF_LOG_COMMIT,
            RETRIEVAL_LOG_COMMIT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if changed.returncode != 0:
        raise ValueError(f"cannot compare v186 artifact commits: {changed.stderr}")
    changed_paths = [value for value in changed.stdout.splitlines() if value]
    runtime_paths = set(RUNTIME_FILES.values()) | {
        "third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml"
    }
    exact = all(
        row["matches_v181"] for commit in commits for row in hashes[commit].values()
    ) and not (runtime_paths & set(changed_paths))
    return {
        "artifact_commits": list(commits),
        "git_blob_hashes": hashes,
        "changed_paths_between_artifact_commits": changed_paths,
        "tracked_runtime_paths_changed_between_artifacts": sorted(
            runtime_paths & set(changed_paths)
        ),
        "v181_v186_tracked_runtime_exact_match": exact,
        "execution_worktree_cleanliness_recorded": False,
    }


def finalize(
    repo_root: Path,
    v181_root: Path,
    v186_root: Path,
    output_root: Path,
    comparison_root: Path,
    *,
    prompt_file: Path | None,
) -> dict:
    input_manifest_path = v181_root / "inputs" / "manifest.json"
    input_manifest = _load_json(input_manifest_path)
    scopes = [
        row
        for row in input_manifest.get("scopes") or ()
        if row.get("key") == "long60_seed0"
    ]
    if len(scopes) != 1:
        raise ValueError("v181 input manifest has no unique long60_seed0 scope")
    scope = scopes[0]
    prompt_path = prompt_file or Path(str(scope["prompt_file"]))
    if not prompt_path.is_file():
        fallback = v181_root / "inputs" / "prompts" / "long60_seed0.txt"
        prompt_path = fallback if fallback.is_file() else prompt_path
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    source_indices = [int(value) for value in scope["prompt_source_indices"]]
    if (
        len(prompts) != PROMPT_COUNT
        or len(source_indices) != PROMPT_COUNT
        or sha256(prompt_path) != scope["prompt_file_sha256"]
        or int(scope.get("prompt_count", -1)) != PROMPT_COUNT
        or int(scope.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(scope.get("seed", -1)) != 0
        or scope.get("decoded_video_contract") != DECODED_VIDEO_CONTRACT
    ):
        raise ValueError("v198 prompt/seed/duration contract differs from v181")

    v181_scope_root = v181_root / "scopes" / "long60_seed0"
    v181_published_path = v181_scope_root / "published_manifest.json"
    v181_contract_path = v181_scope_root / "contracts" / "experiment.json"
    v181_published = _load_json(v181_published_path)
    v181_contract = _load_json(v181_contract_path)
    published_rows = {
        str(row.get("key")): row for row in v181_published.get("methods") or ()
    }
    if (
        v181_published.get("ok") is not True
        or v181_published.get("complete") is not True
        or v181_published.get("scope") != "long60_seed0"
        or int(v181_published.get("prompt_count", -1)) != PROMPT_COUNT
        or v181_contract.get("scope") != "long60_seed0"
        or int(v181_contract.get("seed", -1)) != 0
        or int(v181_contract.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or v181_contract.get("prompt_file_sha256") != scope["prompt_file_sha256"]
        or not all(method in published_rows for method in V181_METHODS)
    ):
        raise ValueError("v181 reused controls are not backed by a complete audit")
    for method in V181_METHODS:
        row = published_rows[method]
        audit_path = Path(str(row.get("audit", "")))
        if not audit_path.is_file() or sha256(audit_path) != row.get("audit_sha256"):
            raise ValueError(f"v181 published audit drifted for {method}")

    audits = {}
    for method in METHODS:
        path = output_root / "audits" / f"{method}.json"
        payload = _load_json(path)
        if (
            payload.get("experiment") != EXPERIMENT
            or payload.get("method") != method
            or payload.get("ok") is not True
            or payload.get("full_decode_performed") is not True
            or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
            or len(payload.get("videos") or ()) != PROMPT_COUNT
        ):
            raise ValueError(f"invalid or incomplete v198 audit for {method}")
        expected_dir = source_video_dir(v181_root, v186_root, method)
        if Path(payload["video_dir"]).resolve() != expected_dir.resolve():
            raise ValueError(f"v198 source directory drifted for {method}")
        audits[method] = payload

    duplicates = []
    for index in range(PROMPT_COUNT):
        by_hash: dict[str, list[str]] = {}
        for method in METHODS:
            digest = str(audits[method]["videos"][index]["sha256"])
            by_hash.setdefault(digest, []).append(method)
        for digest, methods in by_hash.items():
            if len(methods) > 1:
                duplicates.append(
                    {"prompt_index": index, "sha256": digest, "methods": methods}
                )
    duplicate_path = output_root / "audits" / "exact_video_duplicates.json"
    write_json(
        duplicate_path,
        {
            "version": 1,
            "experiment": EXPERIMENT,
            "duplicates": duplicates,
            "ok": not duplicates,
        },
    )
    if duplicates:
        raise RuntimeError(
            f"v198 found exact cross-method duplicates: {duplicates[:4]}"
        )

    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHODS:
        target_dir = comparison_root / "published" / method
        for row in audits[method]["videos"]:
            index = int(row["prompt_index"])
            source = Path(str(row["path"]))
            links[link_or_validate(source, target_dir / f"{index:06d}-0.mp4")] += 1
        method_rows.append(
            {
                "key": method,
                "role": METHOD_ROLES[method],
                "source_campaign": "v181" if method in V181_METHODS else "v186",
                "source_video_dir": audits[method]["video_dir"],
                "video_dir": str(target_dir.resolve()),
                "audit": str((output_root / "audits" / f"{method}.json").resolve()),
                "audit_sha256": sha256(output_root / "audits" / f"{method}.json"),
                "read_frame_equivalents": (
                    9 if method in {"all_recent", "all_coverage_retrieval"} else None
                ),
                "middle_archive_frame_equivalents": (
                    12
                    if method == "all_coverage_retrieval"
                    else 0
                    if method == "all_recent"
                    else None
                ),
            }
        )

    old_runtime = input_manifest["runtime"]["implementation_sha256"]
    current_runtime = _current_runtime_hashes(repo_root)
    common_runtime_keys = sorted(set(old_runtime) & set(current_runtime))
    current_tree_match = all(
        old_runtime[key]["sha256"] == current_runtime[key]["sha256"]
        for key in common_runtime_keys
    )
    artifact_runtime = artifact_runtime_contract(
        repo_root,
        old_runtime,
        str(input_manifest["runtime"]["pf_config_sha256"]),
    )
    tracked_runtime_match = bool(
        artifact_runtime["v181_v186_tracked_runtime_exact_match"]
    )
    script_path = repo_root / "scripts" / "run_v186_long60_comparison.sh"
    retrieval_map = (
        repo_root
        / "runs"
        / "v182_structured_coverage"
        / "inputs"
        / "maps"
        / "all_coverage_retrieval.csv"
    )
    pf_labels = (
        repo_root
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv"
    )
    pf_config = (
        repo_root
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "pyramid-forcing.yaml"
    )
    for path in (script_path, retrieval_map, pf_labels, pf_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_routes = {
        "generation_script": {
            "commit": PF_LOG_COMMIT,
            "relative_path": "scripts/run_v186_long60_comparison.sh",
            "sha256": _git_blob_sha256(
                repo_root,
                PF_LOG_COMMIT,
                "scripts/run_v186_long60_comparison.sh",
            ),
        },
        "retrieval_map": {
            "commit": RETRIEVAL_LOG_COMMIT,
            "relative_path": (
                "runs/v182_structured_coverage/inputs/maps/all_coverage_retrieval.csv"
            ),
            "sha256": _git_blob_sha256(
                repo_root,
                RETRIEVAL_LOG_COMMIT,
                "runs/v182_structured_coverage/inputs/maps/all_coverage_retrieval.csv",
            ),
        },
        "pf_default_labels": {
            "commit": PF_LOG_COMMIT,
            "relative_path": (
                "third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv"
            ),
            "sha256": _git_blob_sha256(
                repo_root,
                PF_LOG_COMMIT,
                "third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv",
            ),
        },
    }

    source_payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "video_generation_performed": False,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": sha256(prompt_path),
        "v181_input_manifest": str(input_manifest_path.resolve()),
        "v181_input_manifest_sha256": sha256(input_manifest_path),
        "v181_published_manifest": str(v181_published_path.resolve()),
        "v181_published_manifest_sha256": sha256(v181_published_path),
        "v181_generation_contract": str(v181_contract_path.resolve()),
        "v181_generation_contract_sha256": sha256(v181_contract_path),
        "v186_artifact_route_contract": artifact_routes,
        "current_tree_generation_script": str(script_path.resolve()),
        "current_tree_generation_script_sha256": sha256(script_path),
        "v186_pf_log_artifact_commit": PF_LOG_COMMIT,
        "v186_retrieval_log_artifact_commit": RETRIEVAL_LOG_COMMIT,
        "current_tree_retrieval_map": str(retrieval_map.resolve()),
        "current_tree_retrieval_map_sha256": sha256(retrieval_map),
        "current_tree_pf_labels": str(pf_labels.resolve()),
        "current_tree_pf_labels_sha256": sha256(pf_labels),
        "current_tree_pf_config": str(pf_config.resolve()),
        "current_tree_pf_config_sha256": sha256(pf_config),
        "v181_runtime_hashes": old_runtime,
        "artifact_runtime_contract": artifact_runtime,
        "current_tree_runtime_hashes": current_runtime,
        "common_runtime_key_count": len(common_runtime_keys),
        "v181_current_tree_runtime_exact_match": current_tree_match,
        "execution_snapshot_limitation": (
            "The v186 artifact commits contain PF/SF runtime and PF config Git blobs "
            "that exactly match the hashes frozen by v181, and no tracked runtime "
            "path changed between the two v186 artifact commits. The execution logs "
            "do not record clean-worktree status, so this is a matched tracked-runtime "
            "contract rather than proof about uncommitted server files."
        ),
    }
    source_path = output_root / "contracts" / "source_manifest.json"
    source_sha = write_json(source_path, source_payload, frozen=True)

    from prepare_v191_vbench_comparison import DIMENSIONS

    comparison_payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "exploratory": True,
        "prompt_count": PROMPT_COUNT,
        "prompt_file_sha256": sha256(prompt_path),
        "prompt_items": [
            {"index": index, "source_index": source_indices[index], "text": text}
            for index, text in enumerate(prompts)
        ],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "clips_per_video": CLIPS_PER_VIDEO,
        "decoded_video_contract": DECODED_VIDEO_CONTRACT,
        "seed": 0,
        "reseed_per_prompt": True,
        "candidate": "all_coverage_retrieval",
        "local_control": "all_recent",
        "native_control": "sf_native",
        "pf_context": "pf_native",
        "pf_required_for_promotion": False,
        "budget_contract": {
            "all_recent_read": "sink1 + recent8 = 9 FFE",
            "retrieval_read": "sink1 + retrieved-middle4 + recent4 = 9 FFE",
            "retrieval_middle_archive_frame_equivalents": 12,
            "read_budget_matched": True,
            "storage_budget_matched": False,
        },
        "methods": method_rows,
        "vbench_long_dimensions": list(DIMENSIONS),
        "comparison_roles": {
            "within_v186": ["all_coverage_retrieval", "pf_native"],
            "matched_tracked_runtime_reuse": ["all_recent", "sf_native"],
        },
        "matched_tracked_runtime_control_available": tracked_runtime_match,
        "execution_clean_worktree_recorded": False,
        "source": {
            "source_manifest": str(source_path.resolve()),
            "source_manifest_sha256": source_sha,
            "duplicate_audit": str(duplicate_path.resolve()),
            "duplicate_audit_sha256": sha256(duplicate_path),
        },
        "claim_boundary": (
            "v198 evaluates already generated 60-second videos. It can rank the "
            "uploaded candidate and diagnose long-horizon behavior. The committed "
            "runtime/config blobs match v181 exactly, but execution-time worktree "
            "cleanliness was not logged. Retrieval and all-Recent match the 9-FFE read "
            "budget but not archive storage. PF is context only and is not a promotion "
            "gate; this all-head operator result does not validate a head classifier."
        ),
    }
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest_sha = write_json(manifest_path, comparison_payload, frozen=True)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "videos": PROMPT_COUNT * len(METHODS),
        "links": links,
        "runtime_exact_match": tracked_runtime_match,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    audit = subparsers.add_parser("audit-method")
    audit.add_argument("--repo-root", type=Path, required=True)
    audit.add_argument("--v181-root", type=Path, required=True)
    audit.add_argument("--v186-root", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--method", choices=METHODS, required=True)
    audit.add_argument("--workers", type=int, default=8)
    audit.add_argument("--skip-decode", action="store_true")
    final = subparsers.add_parser("finalize")
    final.add_argument("--repo-root", type=Path, required=True)
    final.add_argument("--v181-root", type=Path, required=True)
    final.add_argument("--v186-root", type=Path, required=True)
    final.add_argument("--output-root", type=Path, required=True)
    final.add_argument("--comparison-root", type=Path, required=True)
    final.add_argument("--prompt-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "audit-method":
        report = audit_method(
            args.v181_root,
            args.v186_root,
            args.output_root,
            args.method,
            workers=args.workers,
            decode=not args.skip_decode,
        )
        print(
            f"[v198-audit] PASS method={args.method} "
            f"videos={len(report['videos'])} logs={len(report['runtime_logs']['logs'])}"
        )
        return
    report = finalize(
        args.repo_root,
        args.v181_root,
        args.v186_root,
        args.output_root,
        args.comparison_root,
        prompt_file=args.prompt_file,
    )
    print(
        "[v198-finalize] PASS "
        f"videos={report['videos']} links={report['links']} "
        f"same_runtime={str(report['runtime_exact_match']).lower()}"
    )


if __name__ == "__main__":
    main()
