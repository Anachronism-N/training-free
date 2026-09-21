#!/usr/bin/env python3
"""Fetch and verify the original chunk-wise Causal Forcing checkpoint, not CF++."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO_ID = "zhuhz22/Causal-Forcing"
REVISION = "373037a987c3e06eaab3ec7b2fc2f5c9c296b649"
FILENAME = "chunkwise/causal_forcing.pt"
EXPECTED_BYTES = 5676282643
EXPECTED_SHA256 = "cf75ee5cc6f4e2e336c59c973f5544655d8f0aa481761efe6de1b9cb2eb0cd9d"


def download_command(root):
    return ["hf", "download", REPO_ID, FILENAME, "--revision", REVISION,
            "--local-dir", str(root)]


def verify_file(path, *, expected_bytes=EXPECTED_BYTES, expected_sha256=EXPECTED_SHA256):
    path = Path(path).resolve()
    before = path.stat()
    if before.st_size != expected_bytes:
        raise ValueError(f"checkpoint size mismatch: {before.st_size} != {expected_bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed during verification")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: {digest.hexdigest()}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("describe", "download", "verify"))
    parser.add_argument("--local-dir", type=Path, default=Path("checkpoints/causal_forcing_v1"))
    parser.add_argument("--checkpoint", type=Path, help="Existing file to verify; never used by download")
    parser.add_argument("--receipt", type=Path, help="Optional immutable JSON verification receipt")
    args = parser.parse_args()
    if args.checkpoint and args.mode != "verify":
        parser.error("--checkpoint is only supported by verify")
    if args.receipt and args.mode == "describe":
        parser.error("describe does not verify a file and cannot write a receipt")
    root = args.local_dir.resolve()
    description = {"repo_id": REPO_ID, "revision": REVISION, "filename": FILENAME,
                   "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256,
                   "variant": "original_chunkwise_4step", "download_command": download_command(root)}
    if args.mode == "describe":
        print(json.dumps(description, indent=2))
        return
    if args.mode == "download":
        subprocess.run(download_command(root), check=True)
    path = args.checkpoint or root / FILENAME
    print(f"[cf-v1-verify] hashing {path}; expected {EXPECTED_BYTES} bytes", flush=True)
    receipt = {"version": 1, "model": description, "file": verify_file(path),
               "checkpoint_verified": True, "inference_parity_verified": False}
    if args.receipt:
        from v212_lphc_protocol import frozen_json
        frozen_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
