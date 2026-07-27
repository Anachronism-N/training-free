#!/usr/bin/env python3
"""Pre-split v125 videos once so dimension-parallel VBench jobs are race-free."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


PROMPT_COUNT = 128
CLIPS_PER_VIDEO = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("experiment") != "v125_moviebench128_comparison"
        or payload.get("prompt_count") != PROMPT_COUNT
    ):
        raise ValueError(f"invalid v125 comparison manifest: {path}")
    return payload


def expected_video_names() -> list[str]:
    return [f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)]


def expected_clip_names(stem: str) -> set[str]:
    return {
        f"{stem}_{index:03d}.mp4" for index in range(CLIPS_PER_VIDEO)
    }


def validate_split(
    split_root: Path,
    *,
    comparison_manifest_sha256: str,
    vbench_commit: str,
) -> dict[str, Any] | None:
    contract_path = split_root / ".v125_split_manifest.json"
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
        or contract.get("prompt_count") != PROMPT_COUNT
        or contract.get("clips_per_video") != CLIPS_PER_VIDEO
    ):
        return None
    expected_stems = {
        Path(name).stem for name in expected_video_names()
    }
    observed_stems = {
        path.name for path in split_root.iterdir() if path.is_dir()
    }
    if observed_stems != expected_stems:
        return None
    total_bytes = 0
    for stem in sorted(expected_stems):
        folder = split_root / stem
        observed = {path.name for path in folder.glob("*.mp4")}
        if observed != expected_clip_names(stem):
            return None
        sizes = [path.stat().st_size for path in folder.glob("*.mp4")]
        if not sizes or min(sizes) <= 0:
            return None
        total_bytes += sum(sizes)
    return {
        "prompt_count": PROMPT_COUNT,
        "clips_per_video": CLIPS_PER_VIDEO,
        "clip_count": PROMPT_COUNT * CLIPS_PER_VIDEO,
        "total_bytes": total_bytes,
        "contract": contract,
    }


def safe_remove_split(path: Path, method_dir: Path) -> None:
    resolved = path.resolve()
    expected_parent = method_dir.resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith(
        (".split_clip.v125.", ".split_clip.backup.", "split_clip")
    ):
        raise RuntimeError(f"refusing unsafe split removal: {resolved}")
    shutil.rmtree(resolved)


def replace_split_atomically(
    temporary: Path,
    final: Path,
    *,
    method_dir: Path,
) -> None:
    backup = method_dir / f".split_clip.backup.{uuid.uuid4().hex}"
    moved_old = False
    if final.exists():
        os.replace(final, backup)
        moved_old = True
    try:
        os.replace(temporary, final)
    except Exception:
        if moved_old and backup.exists():
            os.replace(backup, final)
        raise
    if moved_old:
        safe_remove_split(backup, method_dir)


def split_method(
    *,
    method: str,
    video_dir: Path,
    comparison_manifest_sha256: str,
    vbench_commit: str,
    split_video: Callable[..., Any],
) -> dict[str, Any]:
    video_dir = video_dir.resolve()
    expected_names = expected_video_names()
    observed_names = {path.name for path in video_dir.glob("*.mp4")}
    if observed_names != set(expected_names):
        raise RuntimeError(
            f"{method}: source video set changed before split"
        )
    final = video_dir / "split_clip"
    valid = validate_split(
        final,
        comparison_manifest_sha256=comparison_manifest_sha256,
        vbench_commit=vbench_commit,
    )
    if valid is not None:
        return {"method": method, "status": "resumed", **valid}

    temporary = video_dir / f".split_clip.v125.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for index, name in enumerate(expected_names):
            source = video_dir / name
            print(
                f"[v125-split] method={method} "
                f"video={index + 1}/{PROMPT_COUNT} name={name}",
                flush=True,
            )
            split_video(str(source), str(temporary), duration=2, fps=8)
        contract = {
            "version": 1,
            "comparison_manifest_sha256": comparison_manifest_sha256,
            "vbench_commit": vbench_commit,
            "method": method,
            "video_dir": str(video_dir),
            "prompt_count": PROMPT_COUNT,
            "clip_seconds": 2,
            "clips_per_video": CLIPS_PER_VIDEO,
            "source_videos": [
                {
                    "name": name,
                    "size": (video_dir / name).stat().st_size,
                }
                for name in expected_names
            ],
        }
        (temporary / ".v125_split_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        valid = validate_split(
            temporary,
            comparison_manifest_sha256=comparison_manifest_sha256,
            vbench_commit=vbench_commit,
        )
        if valid is None:
            raise RuntimeError(f"{method}: split output failed validation")
        replace_split_atomically(
            temporary,
            final,
            method_dir=video_dir,
        )
        return {"method": method, "status": "generated", **valid}
    finally:
        if temporary.exists():
            safe_remove_split(temporary, video_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--vbench-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("require 0 <= node-rank < num-nodes")
    comparison_root = args.comparison_root.resolve()
    vbench_root = args.vbench_root.resolve()
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_sha = sha256(manifest_path)
    completed = subprocess.run(
        ["git", "-C", str(vbench_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    vbench_commit = (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unknown"
    )
    if str(vbench_root) not in sys.path:
        sys.path.insert(0, str(vbench_root))
    from vbench2_beta_long.utils import split_video_into_clips

    all_jobs = [
        {
            "method": str(row["key"]),
            "video_dir": Path(str(row["video_dir"])),
        }
        for row in manifest["methods"]
    ]
    jobs = all_jobs[args.node_rank :: args.num_nodes]
    if not jobs:
        print(
            f"[v125-split] node={args.node_rank}/{args.num_nodes} jobs=0",
            flush=True,
        )
        return
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(
        max_workers=min(args.workers, len(jobs))
    ) as executor:
        futures = {
            executor.submit(
                split_method,
                method=job["method"],
                video_dir=job["video_dir"],
                comparison_manifest_sha256=manifest_sha,
                vbench_commit=vbench_commit,
                split_video=split_video_into_clips,
            ): job["method"]
            for job in jobs
        }
        for future in as_completed(futures):
            method = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[v125-split-complete] method={method} "
                    f"status={result['status']}",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"{method}: {error}")
                print(f"[v125-split-failed] method={method}: {error}", flush=True)

    report = {
        "version": 1,
        "comparison_manifest": str(manifest_path),
        "comparison_manifest_sha256": manifest_sha,
        "vbench_root": str(vbench_root),
        "vbench_commit": vbench_commit,
        "workers": min(args.workers, len(jobs)),
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "methods": sorted(results, key=lambda row: str(row["method"])),
        "failures": failures,
        "ok": not failures and len(results) == len(jobs),
    }
    report_path = (
        comparison_root
        / "metrics"
        / f"vbench_split_node{args.node_rank}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["ok"]:
        raise SystemExit("\n".join(failures))
    print(
        f"[v125-split-all-complete] methods={len(results)} "
        f"node={args.node_rank}/{args.num_nodes} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
