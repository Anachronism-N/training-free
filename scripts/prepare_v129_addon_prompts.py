#!/usr/bin/env python3
"""Create a frozen prompt subset for a v129 add-on screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_indices(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one prompt index is required")
    if len(values) != len(set(values)):
        raise ValueError("prompt indices contain duplicates")
    if any(value < 0 or value >= 128 for value in values):
        raise ValueError("prompt indices must be in [0, 128)")
    return values


def write_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite different frozen file: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"missing source prompt file: {source}")
    source_bytes = source.read_bytes()
    prompts = [
        line.strip()
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 128:
        raise SystemExit(f"expected 128 prompts, found {len(prompts)}")
    try:
        indices = parse_indices(args.indices)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    selected = [prompts[index] for index in indices]
    output_bytes = ("\n".join(selected) + "\n").encode("utf-8")
    manifest = {
        "version": 1,
        "kind": "v129_noncache_addon_prompt_subset",
        "source": str(source),
        "source_sha256": sha256_bytes(source_bytes),
        "source_prompt_count": len(prompts),
        "indices": list(indices),
        "prompts": selected,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_bytes(output_bytes),
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    write_frozen(args.output.resolve(), output_bytes)
    write_frozen(args.manifest.resolve(), manifest_bytes)
    print(
        "[v129-addon-prompts] "
        f"count={len(indices)} indices={','.join(map(str, indices))} "
        f"sha256={manifest['output_sha256']}"
    )


if __name__ == "__main__":
    main()
