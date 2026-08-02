#!/usr/bin/env python3
"""Keep VBench-Long split metadata outside clip-enumeration roots."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from prepare_v129_vbench_splits import split_method, validate_split


LEGACY_MANIFEST = ".v129_split_manifest.json"
CLEAN_MANIFEST = ".training_free_vbench_split_manifest.json"


def clean_manifest_path(video_dir: Path) -> Path:
    return video_dir / CLEAN_MANIFEST


def validate_clean_split(
    video_dir: Path,
    *,
    comparison_manifest_sha256: str,
    vbench_commit: str,
    prompt_count: int,
    clips_per_video: int,
) -> dict[str, Any] | None:
    """Validate clips and require a directory-only ``split_clip`` root."""

    video_dir = video_dir.resolve()
    split_root = video_dir / "split_clip"
    contract_path = clean_manifest_path(video_dir)
    if not split_root.is_dir() or not contract_path.is_file():
        return None
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        contract.get("comparison_manifest_sha256")
        != comparison_manifest_sha256
        or contract.get("vbench_commit") != vbench_commit
        or int(contract.get("prompt_count", -1)) != int(prompt_count)
        or int(contract.get("clips_per_video", -1)) != int(clips_per_video)
        or contract.get("video_dir") != str(video_dir)
    ):
        return None

    root_entries = list(split_root.iterdir())
    if any(not path.is_dir() for path in root_entries):
        return None
    expected_stems = {
        f"{index:06d}-0" for index in range(int(prompt_count))
    }
    if {path.name for path in root_entries} != expected_stems:
        return None

    source_rows = contract.get("source_videos")
    if not isinstance(source_rows, list) or len(source_rows) != prompt_count:
        return None
    expected_sources = {
        f"{index:06d}-0.mp4" for index in range(int(prompt_count))
    }
    observed_sources = {
        str(row.get("name"))
        for row in source_rows
        if isinstance(row, dict)
    }
    if observed_sources != expected_sources:
        return None
    for row in source_rows:
        if not isinstance(row, dict):
            return None
        source = video_dir / str(row.get("name"))
        if (
            not source.is_file()
            or source.stat().st_size <= 0
            or int(row.get("size", -1)) != source.stat().st_size
        ):
            return None

    total_bytes = 0
    for stem in sorted(expected_stems):
        folder = split_root / stem
        expected_clips = {
            f"{stem}_{index:03d}.mp4"
            for index in range(int(clips_per_video))
        }
        children = list(folder.iterdir())
        if any(not path.is_file() for path in children):
            return None
        if {path.name for path in children} != expected_clips:
            return None
        sizes = [path.stat().st_size for path in children]
        if not sizes or min(sizes) <= 0:
            return None
        total_bytes += sum(sizes)
    return {
        "prompt_count": int(prompt_count),
        "clips_per_video": int(clips_per_video),
        "clip_count": int(prompt_count) * int(clips_per_video),
        "total_bytes": int(total_bytes),
        "contract": contract,
        "manifest": str(contract_path),
    }


def _migrate_manifest(video_dir: Path) -> Path:
    source = video_dir / "split_clip" / LEGACY_MANIFEST
    target = clean_manifest_path(video_dir)
    if not source.is_file():
        raise RuntimeError(f"missing legacy split manifest: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return target


def prepare_clean_split(
    *,
    method: str,
    video_dir: Path,
    manifest_sha: str,
    vbench_commit: str,
    prompt_count: int,
    clips_per_video: int,
    split_video: Callable[..., Any],
) -> dict[str, Any]:
    """Resume, migrate, or generate a split cache with clean root entries."""

    video_dir = video_dir.resolve()
    clean = validate_clean_split(
        video_dir,
        comparison_manifest_sha256=manifest_sha,
        vbench_commit=vbench_commit,
        prompt_count=prompt_count,
        clips_per_video=clips_per_video,
    )
    if clean is not None:
        return {"method": method, "status": "resumed_clean", **clean}

    split_root = video_dir / "split_clip"
    legacy = validate_split(
        split_root,
        comparison_manifest_sha256=manifest_sha,
        vbench_commit=vbench_commit,
        prompt_count=prompt_count,
        clips_per_video=clips_per_video,
    )
    if legacy is not None:
        _migrate_manifest(video_dir)
        clean = validate_clean_split(
            video_dir,
            comparison_manifest_sha256=manifest_sha,
            vbench_commit=vbench_commit,
            prompt_count=prompt_count,
            clips_per_video=clips_per_video,
        )
        if clean is None:
            raise RuntimeError(f"{method}: migrated split failed clean validation")
        return {"method": method, "status": "migrated", **clean}

    result = split_method(
        method=method,
        video_dir=video_dir,
        manifest_sha=manifest_sha,
        vbench_commit=vbench_commit,
        prompt_count=prompt_count,
        clips_per_video=clips_per_video,
        split_video=split_video,
    )
    _migrate_manifest(video_dir)
    clean = validate_clean_split(
        video_dir,
        comparison_manifest_sha256=manifest_sha,
        vbench_commit=vbench_commit,
        prompt_count=prompt_count,
        clips_per_video=clips_per_video,
    )
    if clean is None:
        raise RuntimeError(f"{method}: generated split failed clean validation")
    return {
        "method": method,
        "status": f"{result['status']}_clean",
        **clean,
    }
