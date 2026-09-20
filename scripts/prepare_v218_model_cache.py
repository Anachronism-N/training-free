#!/usr/bin/env python3
"""Make a private RAFT cache overlay without modifying models used by running jobs."""
import argparse
import json
from pathlib import Path

from v212_lphc_protocol import frozen_json, sha256


def prepare(source, target, checkpoint):
    source, target, checkpoint = source.resolve(), target.resolve(), checkpoint.resolve()
    if not source.is_dir() or not checkpoint.is_file():
        raise ValueError("source cache and downloaded upstream RAFT checkpoint must exist")
    if (source == target or source.is_relative_to(target) or target.is_relative_to(source)
            or checkpoint.is_relative_to(target)):
        raise ValueError("use a separate cache overlay, not an ancestor or child of the source")
    links = {p.name: str(p.resolve()) for p in source.iterdir()
             if p.name not in {"raft_model", "v218_cache_overlay.json"}}
    links["raft_model/models/raft-things.pth"] = str(checkpoint)
    receipt = {"version": 1, "source_cache": str(source), "links": links,
               "raft_checkpoint_sha256": sha256(checkpoint), "strict_probe_required": True}
    manifest = target / "v218_cache_overlay.json"
    if target.exists() and not manifest.is_file():
        raise ValueError("overlay exists without a receipt; choose a new path")
    if manifest.is_file() and json.loads(manifest.read_text()) != receipt:
        raise ValueError("cache overlay source or checkpoint changed")
    # Reserve a receipt first so an interrupted symlink setup can resume.
    frozen_json(manifest, receipt)
    for name, destination in links.items():
        path, original = target / name, Path(destination)
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != original:
                raise ValueError(f"cache overlay link changed: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(original, target_is_directory=original.is_dir())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--raft-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_cache, args.output_cache, args.raft_checkpoint), indent=2))


if __name__ == "__main__":
    main()
