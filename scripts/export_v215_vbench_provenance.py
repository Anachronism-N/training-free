#!/usr/bin/env python3
"""Export frozen evaluator modifications and an operator-declared RAFT checksum, not weights."""
import argparse
import hashlib
import json
from pathlib import Path

from v210_vbench_fingerprint import _git_output, vbench_checkout_fingerprint
from v212_lphc_protocol import frozen_json, sha256


def snapshot(root, comparison, checkpoint):
    root = root.resolve()
    before = vbench_checkout_fingerprint(root)
    if before != comparison["vbench_fingerprint"]:
        raise ValueError("evaluator drift: export the exact frozen checkout, not a replacement")
    patch = _git_output(root, "diff", "--binary", "HEAD")
    if hashlib.sha256(patch).hexdigest() != before["diff_binary_head_sha256"]:
        raise ValueError("evaluator changed during export")
    files, omitted = {}, []
    for name, digest in before["runtime_path_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("evaluator path escapes checkout")
        if digest is None:
            continue
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".json"} or path.stat().st_size > 2_000_000:
            omitted.append(name)
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("runtime changed during export")
        files[name] = raw.decode("utf-8")
    if vbench_checkout_fingerprint(root) != before:
        raise ValueError("evaluator changed during export")
    return {"vbench_fingerprint": before, "tracked_patch": patch.decode("utf-8"),
            "modified_runtime_sources": files, "omitted_paths": omitted,
            "operator_declared_raft_checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "boundary": "Checkpoint supplied by operator, not proof of the actual loaded weights. Inspect dynamic_degree.py and preflight/logs. No model weights are exported."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison_manifest.read_text(encoding="utf-8"))
    result = snapshot(args.vbench_root, comparison, args.raft_checkpoint)
    result["comparison_sha256"] = sha256(args.comparison_manifest)
    frozen_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
