#!/usr/bin/env python3
"""Pre-split v154 videos once before dimension-parallel VBench jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from prepare_v129_vbench_splits import sha256
from prepare_v154_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
)
from vbench_long_split_cache import prepare_clean_split


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != COMPARISON_EXPERIMENT
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("num_output_frames", -1)) != 120
        or tuple(row.get("key") for row in payload.get("methods", []))
        != METHODS
    ):
        raise ValueError(f"invalid v154 VBench manifest: {path}")
    return payload


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
        completed.stdout.strip() if completed.returncode == 0 else "unknown"
    )
    sys.path.insert(0, str(vbench_root))
    from vbench2_beta_long.utils import split_video_into_clips

    clips_per_video = int(manifest["num_output_frames"]) // 8
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
                prepare_clean_split,
                method=method,
                video_dir=video_dir,
                manifest_sha=manifest_sha,
                vbench_commit=vbench_commit,
                prompt_count=PROMPT_COUNT,
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
                    f"[v154-split-complete] method={method} "
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
        raise SystemExit("\n".join(failures or ["v154 split count mismatch"]))


if __name__ == "__main__":
    main()
