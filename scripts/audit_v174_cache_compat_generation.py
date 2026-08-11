#!/usr/bin/env python3
"""Decode-audit and publish v174 cache-compatibility generation grids."""
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
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a non-empty unique list")
    if not 1 <= args.prompt_count <= 128:
        raise ValueError("prompt count must be within [1, 128]")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("experiment") != "v173_residual_cache_compatibility":
        raise ValueError("v174 requires the v173 profiling analysis")
    if not analysis.get("generation_ready"):
        raise ValueError("v173 gates did not authorize generation validation")
    prompt_lines = args.prompts.read_text(encoding="utf-8").splitlines()
    if len(prompt_lines) != 128:
        raise ValueError("v174 requires the frozen 128-prompt suite")

    contract = {
        "version": 1,
        "experiment": "v174_cache_compat_generation",
        "scope": args.run_root.name,
        "prompt_count": int(args.prompt_count),
        "prompt_file": str(args.prompts.resolve()),
        "prompt_file_sha256": sha256(args.prompts),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "methods": list(methods),
        "profile_analysis": str(args.analysis.resolve()),
        "profile_analysis_sha256": sha256(args.analysis),
        "maps": {
            method: analysis["maps"][method]
            for method in methods
        },
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_frozen(contract_path, contract)

    method_rows = []
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    for method in methods:
        map_path = Path(analysis["maps"][method]["path"])
        if sha256(map_path) != analysis["maps"][method]["sha256"]:
            raise ValueError(f"{method}: head-map hash drift")
        raw_dir = args.run_root / "raw" / method
        report = audit_interval(
            raw_dir,
            start_idx=0,
            end_idx=args.prompt_count,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        report_path = args.run_root / "audits" / f"{method}.json"
        write_frozen(report_path, report)
        if not report["ok"]:
            raise RuntimeError(f"{method}: media audit failed")
        published_dir = args.run_root / "published" / method
        for item in report["videos"]:
            prompt_idx = int(item["prompt_idx"])
            source = raw_dir / str(item["file"])
            mode = link_or_validate(
                source,
                published_dir / f"{prompt_idx:06d}.mp4",
            )
            link_counts[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": (
                    "matched_hypothesis"
                    if method == "matched"
                    else "mechanism_control"
                ),
                "head_map": str(map_path.resolve()),
                "head_map_sha256": sha256(map_path),
                "video_dir": str(published_dir.resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": sha256(report_path),
            }
        )

    payload = {
        "version": 1,
        "ok": True,
        "experiment": contract["experiment"],
        "scope": contract["scope"],
        "prompt_count": int(args.prompt_count),
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    manifest_path = args.run_root / "published_manifest.json"
    write_frozen(manifest_path, payload)
    print(
        "[v174-audit] PASS "
        f"scope={contract['scope']} methods={len(methods)} "
        f"videos={len(methods) * args.prompt_count} links={link_counts}"
    )


if __name__ == "__main__":
    main()
