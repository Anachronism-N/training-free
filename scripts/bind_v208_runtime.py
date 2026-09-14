#!/usr/bin/env python3
"""Bind generation and resume to one runtime and the tested parity artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prepare_v207_context_budget_phase_screen import sha256
from prepare_v208_paper_confirmation import verify, write_frozen


def runtime_hashes(root: Path) -> dict[str, str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "third_party/Self-Forcing", "third_party/Pyramid-Forcing",
         "src/lifecycle_kv", "scripts/*v208*"], cwd=root,
    ).decode("utf-8").split("\0")
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".cu", ".cpp", ".h", ".hpp"}
    return {
        name: sha256(root / name)
        for name in sorted(tracked)
        if name and Path(name).suffix in suffixes
    }


def snapshot(path: Path, *, digest: bool) -> dict:
    stat = path.stat()
    row = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        row["sha256"] = sha256(path)
    return row


def bind(
    manifest_path: Path, parity_manifest_path: Path, root: Path,
    checkpoint: Path, sf_config: Path, pf_config: Path, output: Path,
) -> dict:
    manifest = verify(manifest_path)
    parity = json.loads(parity_manifest_path.read_text(encoding="utf-8"))
    if parity.get("experiment") != "v207_sf_runtime_parity":
        raise ValueError("wrong numerical parity input manifest")
    hashes = runtime_hashes(root)
    for name, expected in parity["runtime_paths"].items():
        if name.startswith(("third_party/", "src/")) and hashes.get(name) != expected:
            raise ValueError(f"runtime changed since numerical parity: {name}")
    artifacts = {"checkpoint": checkpoint, "sf_config": sf_config, "pf_adaptive_config": pf_config}
    frozen = {}
    previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    for key, path in artifacts.items():
        # Hash the checkpoint once. Resume validates path, size and mtime; a changed
        # stamp is rejected instead of rehashing a large shared file on every GPU node.
        current = snapshot(path, digest=key != "checkpoint" or previous is None)
        if key == "checkpoint" and previous is not None:
            old = previous["artifacts"][key]
            if any(current[field] != old[field] for field in ("path", "bytes", "mtime_ns")):
                raise ValueError("checkpoint stamp changed; use a new v208 output root")
            current["sha256"] = old["sha256"]
        if current["sha256"] != parity["artifacts"][key]["sha256"]:
            raise ValueError(f"{key} differs from numerical parity")
        frozen[key] = current
    trace_root = parity_manifest_path.parent.parent / "traces"
    if Path(manifest["source"]["sf_parity_report"]).resolve().parent.parent != parity_manifest_path.resolve().parent.parent:
        raise ValueError("parity report and input manifest must belong to the same run root")
    contract_sha = sha256(parity_manifest_path)
    trace_meta = {}
    for run in parity["run_order"]:
        path = trace_root / run / "trace_meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != contract_sha:
            raise ValueError(f"parity trace belongs to a different contract: {run}")
        trace_meta[run] = {"path": str(path.resolve()), "sha256": sha256(path)}
    payload = {
        "version": 1,
        "experiment": "v208_frozen_runtime",
        "input_manifest_sha256": sha256(manifest_path),
        "parity_report_sha256": manifest["source"]["sf_parity_report_sha256"],
        "parity_input_manifest": str(parity_manifest_path.resolve()),
        "parity_input_manifest_sha256": contract_sha,
        "parity_trace_meta": trace_meta,
        "runtime_paths": hashes,
        "artifacts": frozen,
        "measurement_note": "Checkpoint SHA256 is cached behind an unchanged path/size/mtime stamp.",
    }
    write_frozen(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parity-input-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sf-config", type=Path, required=True)
    parser.add_argument("--pf-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = bind(
        args.manifest, args.parity_input_manifest, args.repo_root,
        args.checkpoint, args.sf_config, args.pf_config, args.output,
    )
    print(f"[v208-runtime] PASS files={len(report['runtime_paths'])} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
