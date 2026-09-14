#!/usr/bin/env python3
"""Audit v208 30/60-second paper-confirmation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from audit_v207_context_budget_phase_screen import (
    audit_cache_logs,
    audit_sf_logs,
    audit_traces,
)
from prepare_v207_context_budget_phase_screen import sha256
from prepare_v208_paper_confirmation import METHOD_ORDER, PROMPT_COUNT, verify


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return sha256(path)


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v208 published video: {target}")
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
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("smoke", "main30", "long60"), required=True
    )
    parser.add_argument("--smoke-prompt-index", type=int, default=3)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    manifest = verify(args.input_manifest)
    runtime_path = args.input_manifest.parent / "runtime_contract.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    input_sha = sha256(args.input_manifest)
    runtime_sha = sha256(runtime_path)
    if runtime.get("input_manifest_sha256") != input_sha:
        raise ValueError("v208 generation runtime/input binding drift")
    scope_key = "main30" if args.scope == "smoke" else args.scope
    scope = next(row for row in manifest["scopes"] if row["key"] == scope_key)
    decoded = scope["decoded_video_contract"]
    if args.scope == "smoke":
        start, end = args.smoke_prompt_index, args.smoke_prompt_index + 1
    else:
        start, end = 0, PROMPT_COUNT
    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHOD_ORDER:
        config = manifest["methods"][method]
        media = audit_interval(
            args.run_root / "raw" / method,
            start_idx=start,
            end_idx=end,
            sample_idx=0,
            expected_frames=int(decoded["frames"]),
            expected_fps=float(decoded["fps"]),
            expected_width=int(decoded["width"]),
            expected_height=int(decoded["height"]),
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        if method == "sf_native":
            logs = audit_sf_logs(args.run_root)
            traces = {"ok": True, "not_applicable": True}
        else:
            logs = audit_cache_logs(args.run_root, method, config)
            traces = audit_traces(
                args.run_root,
                method,
                config,
                scope="smoke" if args.scope == "smoke" else args.scope,
            )
        for log in logs["logs"]:
            text = Path(log["path"]).read_text(encoding="utf-8", errors="replace")
            expected = f"[V208Provenance] runtime_sha256={runtime_sha} input_sha256={input_sha} git="
            if len(re.findall(r"\[V208Provenance\]", text)) != 1 or expected not in text:
                logs["ok"] = False
                logs["errors"].append(f"{log['path']}: missing or mixed v208 runtime binding")
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        if method_ok:
            published = args.run_root / "published" / method
            for video in media["videos"]:
                source = args.run_root / "raw" / method / str(video["file"])
                mode = link_or_validate(
                    source, published / f"{int(video['prompt_idx']):06d}.mp4"
                )
                links[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "parent_method": config.get("parent_method"),
                "operator": config["operator"],
                "schedule": config["schedule"],
                "read_frame_equivalents": config["read_frame_equivalents"],
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )
    contract = {
        "version": 1,
        "experiment": "v208_paper_confirmation_generation",
        "scope": args.scope,
        "paper_stage": args.scope != "smoke",
        "prompt_count": end - start,
        "prompt_indices": list(range(start, end)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "prompt_items": manifest["prompt_items"][start:end],
        "num_output_frames": int(scope["num_output_frames"]),
        "decoded_video_contract": decoded,
        "seed": manifest["seed"],
        "methods": list(METHOD_ORDER),
        "frozen_parent_method": manifest["frozen_parent_method"],
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "runtime_contract": str(runtime_path.resolve()),
        "runtime_contract_sha256": runtime_sha,
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    summary = {
        "version": 1,
        "ok": all_ok,
        "experiment": contract["experiment"],
        "scope": args.scope,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": links,
    }
    write_json(args.run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v208 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v208-audit] PASS "
        f"scope={args.scope} methods=3 videos={3*(end-start)} links={links}"
    )


if __name__ == "__main__":
    main()
