#!/usr/bin/env python3
"""Pre-split v129 comparison videos once for race-free VBench-Long jobs."""

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
        or payload.get("experiment") != "v129_no_pf_paper_comparison_30s"
        or payload.get("prompt_count") != 128
        or payload.get("num_output_frames") != 120
    ):
        raise ValueError(f"invalid v129 comparison manifest: {path}")
    return payload


def validate_split(
    split_root: Path,
    *,
    comparison_manifest_sha256: str,
    vbench_commit: str,
    prompt_count: int,
    clips_per_video: int,
) -> dict[str, Any] | None:
    contract_path = split_root / ".v129_split_manifest.json"
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
        or contract.get("prompt_count") != prompt_count
        or contract.get("clips_per_video") != clips_per_video
        or contract.get("video_dir") != str(split_root.parent.resolve())
    ):
        return None
    source_rows = contract.get("source_videos")
    if not isinstance(source_rows, list) or len(source_rows) != prompt_count:
        return None
    expected_source_names = {
        f"{index:06d}-0.mp4" for index in range(prompt_count)
    }
    observed_source_names = {
        str(row.get("name"))
        for row in source_rows
        if isinstance(row, dict)
    }
    if observed_source_names != expected_source_names:
        return None
    for row in source_rows:
        if not isinstance(row, dict):
            return None
        source = split_root.parent / str(row["name"])
        if (
            not source.is_file()
            or int(row.get("size", -1)) != source.stat().st_size
        ):
            return None
    expected_stems = {
        f"{index:06d}-0" for index in range(prompt_count)
    }
    observed_stems = {
        path.name for path in split_root.iterdir() if path.is_dir()
    }
    if observed_stems != expected_stems:
        return None
    total_bytes = 0
    for stem in sorted(expected_stems):
        folder = split_root / stem
        expected_names = {
            f"{stem}_{index:03d}.mp4"
            for index in range(clips_per_video)
        }
        observed_names = {path.name for path in folder.glob("*.mp4")}
        if observed_names != expected_names:
            return None
        sizes = [path.stat().st_size for path in folder.glob("*.mp4")]
        if not sizes or min(sizes) <= 0:
            return None
        total_bytes += sum(sizes)
    return {
        "prompt_count": prompt_count,
        "clips_per_video": clips_per_video,
        "clip_count": prompt_count * clips_per_video,
        "total_bytes": total_bytes,
        "contract": contract,
    }


def safe_remove(path: Path, method_dir: Path) -> None:
    resolved = path.resolve()
    if (
        resolved.parent != method_dir.resolve()
        or not resolved.name.startswith(
            (".split_clip.v129.", ".split_clip.backup.", "split_clip")
        )
    ):
        raise RuntimeError(f"refusing unsafe split removal: {resolved}")
    shutil.rmtree(resolved)


def replace_atomically(
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
        safe_remove(backup, method_dir)


def split_method(
    *,
    method: str,
    video_dir: Path,
    manifest_sha: str,
    vbench_commit: str,
    prompt_count: int,
    clips_per_video: int,
    split_video: Callable[..., Any],
) -> dict[str, Any]:
    expected_names = {
        f"{index:06d}-0.mp4" for index in range(prompt_count)
    }
    observed_names = {path.name for path in video_dir.glob("*.mp4")}
    if observed_names != expected_names:
        raise RuntimeError(f"{method}: source video set changed before split")
    final = video_dir / "split_clip"
    valid = validate_split(
        final,
        comparison_manifest_sha256=manifest_sha,
        vbench_commit=vbench_commit,
        prompt_count=prompt_count,
        clips_per_video=clips_per_video,
    )
    if valid is not None:
        return {"method": method, "status": "resumed", **valid}
    temporary = video_dir / f".split_clip.v129.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for index in range(prompt_count):
            name = f"{index:06d}-0.mp4"
            print(
                f"[v129-split] method={method} "
                f"video={index + 1}/{prompt_count}",
                flush=True,
            )
            split_video(
                str(video_dir / name),
                str(temporary),
                duration=2,
                fps=8,
            )
        contract = {
            "version": 1,
            "comparison_manifest_sha256": manifest_sha,
            "vbench_commit": vbench_commit,
            "method": method,
            "video_dir": str(video_dir.resolve()),
            "prompt_count": prompt_count,
            "clip_seconds": 2,
            "clips_per_video": clips_per_video,
            "source_videos": [
                {
                    "name": f"{index:06d}-0.mp4",
                    "size": (
                        video_dir / f"{index:06d}-0.mp4"
                    ).stat().st_size,
                }
                for index in range(prompt_count)
            ],
        }
        (temporary / ".v129_split_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        valid = validate_split(
            temporary,
            comparison_manifest_sha256=manifest_sha,
            vbench_commit=vbench_commit,
            prompt_count=prompt_count,
            clips_per_video=clips_per_video,
        )
        if valid is None:
            raise RuntimeError(f"{method}: split output failed validation")
        replace_atomically(temporary, final, method_dir=video_dir)
        return {"method": method, "status": "generated", **valid}
    finally:
        if temporary.exists():
            safe_remove(temporary, video_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--vbench-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
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
    sys.path.insert(0, str(vbench_root))
    from vbench2_beta_long.utils import split_video_into_clips

    prompt_count = int(manifest["prompt_count"])
    clips_per_video = 15
    all_jobs = [
        (str(row["key"]), Path(str(row["video_dir"])).resolve())
        for row in manifest["methods"]
    ]
    jobs = all_jobs[args.node_rank :: args.num_nodes]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(
        max_workers=min(args.workers, max(1, len(jobs)))
    ) as executor:
        futures = {
            executor.submit(
                split_method,
                method=method,
                video_dir=video_dir,
                manifest_sha=manifest_sha,
                vbench_commit=vbench_commit,
                prompt_count=prompt_count,
                clips_per_video=clips_per_video,
                split_video=split_video_into_clips,
            ): method
            for method, video_dir in jobs
        }
        for future in as_completed(futures):
            method = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[v129-split-complete] method={method} "
                    f"status={result['status']}",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"{method}: {error}")
    report = {
        "version": 1,
        "comparison_manifest_sha256": manifest_sha,
        "vbench_commit": vbench_commit,
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


if __name__ == "__main__":
    main()
