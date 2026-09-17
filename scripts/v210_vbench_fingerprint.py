#!/usr/bin/env python3
"""Compute the source-bound runtime fingerprint for a VBench checkout."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def _git_output(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _nul_paths(value: bytes) -> set[str]:
    return {os.fsdecode(item) for item in value.split(b"\0") if item}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vbench_checkout_fingerprint(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    head = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    status_bytes = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    diff_bytes = _git_output(root, "diff", "--binary", "HEAD")
    changed = _nul_paths(
        _git_output(root, "diff", "--name-only", "-z", "HEAD", "--")
    )
    untracked = _nul_paths(
        _git_output(root, "ls-files", "--others", "--exclude-standard", "-z")
    )
    runtime_hashes: dict[str, str | None] = {}
    for name in sorted(changed | untracked):
        path = root / name
        runtime_hashes[name] = _sha256(path) if path.is_file() else None
    return {
        "version": 1,
        "head": head,
        "dirty": bool(status_bytes),
        "status_porcelain_v1": os.fsdecode(status_bytes),
        "status_porcelain_v1_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "diff_binary_head_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "runtime_path_sha256": runtime_hashes,
    }
