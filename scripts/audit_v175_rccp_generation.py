#!/usr/bin/env python3
"""Decode-audit and publish v175 RCCP causal-validation videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_indexed_videos import audit_interval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen artifact differs: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--prompt-count", type=int, choices=(32, 64), required=True)
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    stability = json.loads(args.stability.read_text(encoding="utf-8"))
    inputs = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not stability.get("generation_ready") or not stability.get("profile_complete"):
        raise ValueError("v175 requires a complete, stable RCCP profile")
    prompts = args.prompts.read_text(encoding="utf-8").splitlines()
    if len(prompts) != args.prompt_count:
        raise ValueError("v175 prompt subset does not match the frozen scope")
    if args.run_root.name == "screen32":
        source_prompt_ids = inputs["screen_source_prompt_ids"]
        expected_prompt_sha = inputs["screen_prompt_sha256"]
    elif args.run_root.name == "confirm64":
        source_prompt_ids = inputs["transfer_source_prompt_ids"]
        expected_prompt_sha = inputs["transfer_prompt_sha256"]
    else:
        raise ValueError(f"unsupported v175 scope: {args.run_root.name}")
    if len(source_prompt_ids) != args.prompt_count or sha256(args.prompts) != expected_prompt_sha:
        raise ValueError("v175 prompt source mapping or hash drift")
    contract = {
        "version": 1,
        "experiment": "v175_rccp_generation",
        "scope": args.run_root.name,
        "prompt_count": args.prompt_count,
        "prompt_file": str(args.prompts.resolve()),
        "prompt_file_sha256": sha256(args.prompts),
        "source_prompt_ids": source_prompt_ids,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477, "fps": 16.0, "width": 832, "height": 480,
        },
        "seed": 0,
        "methods": list(methods),
        "stability_analysis": str(args.stability.resolve()),
        "stability_analysis_sha256": sha256(args.stability),
        "maps": {method: stability["maps"][method] for method in methods},
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_frozen(contract_path, contract)
    method_rows = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in methods:
        map_path = Path(stability["maps"][method]["path"])
        if sha256(map_path) != stability["maps"][method]["sha256"]:
            raise ValueError(f"{method}: head-map hash drift")
        raw_dir = args.run_root / "raw" / method
        report = audit_interval(
            raw_dir, start_idx=0, end_idx=args.prompt_count, sample_idx=0,
            expected_frames=477, expected_fps=16.0, expected_width=832,
            expected_height=480, fps_tolerance=0.05,
            allow_outside_interval=False, decode=True,
        )
        report_path = args.run_root / "audits" / f"{method}.json"
        write_frozen(report_path, report)
        if not report["ok"]:
            raise RuntimeError(f"{method}: media audit failed")
        published = args.run_root / "published" / method
        for item in report["videos"]:
            source = raw_dir / str(item["file"])
            mode = link_or_validate(
                source, published / f"{int(item['prompt_idx']):06d}.mp4"
            )
            link_counts[mode] += 1
        method_rows.append({
            "key": method,
            "role": "matched_hypothesis" if method == "stable_matched" else "mechanism_control",
            "head_map": str(map_path.resolve()),
            "head_map_sha256": sha256(map_path),
            "video_dir": str(published.resolve()),
            "audit": str(report_path.resolve()),
            "audit_sha256": sha256(report_path),
        })
    manifest = {
        "version": 1, "ok": True, "experiment": contract["experiment"],
        "scope": contract["scope"], "prompt_count": args.prompt_count,
        "methods": method_rows, "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha, "link_counts": link_counts,
    }
    write_frozen(args.run_root / "published_manifest.json", manifest)
    print(f"[v175-audit] PASS methods={len(methods)} prompts={args.prompt_count}")


if __name__ == "__main__":
    main()
