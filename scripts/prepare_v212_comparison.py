#!/usr/bin/env python3
"""Publish validated v212 videos with same-device pairing and immutable hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import v212_lphc_protocol as p
from prepare_v210_vbench_comparison import DIMENSIONS
from run_v212_lphc import job_path, load_done, require_gate
from run_v211_worker import validate_media
from v210_vbench_fingerprint import vbench_checkout_fingerprint


def validate_pairs(jobs: list[dict], *, protocol=p) -> None:
    p = protocol
    for source in p.SOURCE_INDICES:
        rows = [x for x in jobs if x["source_index"] == source]
        if len(rows) != len(p.METHODS) or {x["method"] for x in rows} != set(p.METHODS):
            raise ValueError(f"incomplete paired prompt: {source}")
        identities = {(x["hostname"], x["gpu_uuid"]) for x in rows}
        if len(identities) != 1 or any(not uuid for _, uuid in identities):
            raise ValueError(f"paired prompt changed physical GPU: {source}")
        timeline = sorted(rows, key=lambda x: x["started_wall_ns"])
        if any(x["finished_wall_ns"] < x["started_wall_ns"] for x in timeline):
            raise ValueError("invalid generation timestamp interval")
        if any(a["finished_wall_ns"] > b["started_wall_ns"] for a, b in zip(timeline, timeline[1:])):
            raise ValueError(f"same-prompt methods overlapped: {source}")


def verify_published(repo: Path, comparison: Path, *, media: bool = True, protocol=p) -> dict:
    p = protocol
    data = json.loads((comparison / "comparison_manifest.json").read_text(encoding="utf-8"))
    out = Path(data["generation_root"])
    generation = p.verify(repo, out)
    if (data.get("experiment") != p.EXPERIMENT or data.get("prompt_count") != 32
            or [x["key"] for x in data["methods"]] != list(p.METHODS)
            or data.get("input_manifest_sha256") != p.sha256(out / "inputs/manifest.json")
            or data.get("prompt_items") != generation["prompt_items"]):
        raise ValueError("v212 published contract mismatch")
    require_gate(out, generation, protocol=p)
    validate_pairs(data["jobs"], protocol=p)
    for job in data["jobs"]:
        if p.sha256(Path(job["done_path"])) != job["done_sha256"]:
            raise ValueError("generation completion changed after publishing")
        if media and p.sha256(Path(job["media_path"])) != job["media_sha256"]:
            raise ValueError("generation media changed after publishing")
    return data


def prepare(repo: Path, out: Path, vbench: Path, *, protocol=p) -> dict:
    p = protocol
    data = p.verify(repo, out)
    require_gate(out, data, protocol=p)
    comparison = out / "evaluation/vbench_comparison"
    jobs = []
    for source in p.SOURCE_INDICES:
        for method in p.METHODS:
            row = load_done(out, data, "screen32", method, source, protocol=p)
            media = Path(row["media"]["path"])
            validate_media(media, 120)
            done = job_path(out, "screen32", method, source) / "done.json"
            jobs.append({"source_index": source, "method": method, "done_path": str(done),
                         "done_sha256": p.sha256(done), "media_path": str(media),
                         "media_sha256": p.sha256(media), **{k: row[k] for k in (
                             "hostname", "gpu_uuid", "started_wall_ns", "finished_wall_ns", "elapsed_seconds")},
                         "cuda_peak_process_memory_mib": row.get("cuda_peak_process_memory_mib")})
    validate_pairs(jobs, protocol=p)
    method_rows = []
    for method in p.METHODS:
        target = comparison / "published" / method
        target.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(p.SOURCE_INDICES):
            src = job_path(out, "screen32", method, source) / "media/0-0_ema.mp4"
            dst = target / f"{index:06d}-0.mp4"
            if dst.is_symlink() or dst.exists():
                if not dst.is_symlink() or dst.resolve() != src.resolve():
                    raise ValueError(f"existing published path differs: {dst}")
            else:
                dst.symlink_to(src)
        method_rows.append({"key": method, "video_dir": str(target), "spec": p.SPECS[method]})
    result = {"version": 1, "experiment": p.EXPERIMENT, "prompt_count": 32,
              "num_output_frames": 120, "prompt_items": data["prompt_items"], "methods": method_rows,
              "vbench_long_dimensions": list(DIMENSIONS), "jobs": jobs,
              "generation_root": str(out), "input_manifest_sha256": p.sha256(out / "inputs/manifest.json"),
              "vbench_fingerprint": vbench_checkout_fingerprint(vbench), "development_only": True}
    p.frozen_json(comparison / "comparison_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("v212", "v213"), default="v212")
    args = parser.parse_args()
    p = p.load_protocol(args.campaign)
    result = prepare(Path(__file__).resolve().parents[1], p.output_root(args.run_root), args.vbench_root.resolve(), protocol=p)
    print(f"[{p.LABEL}-publish] validated={len(result['jobs'])} same_gpu_pairing=true")
