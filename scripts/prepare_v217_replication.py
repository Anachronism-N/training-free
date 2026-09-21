#!/usr/bin/env python3
"""Inherit v216's selected method and endpoint without looking at its outcomes."""
import argparse
from pathlib import Path

import v212_lphc_protocol as base
import v216_lphc_protocol as parent
import v217_lphc_protocol as current


def freeze(v216_root, out, *, protocol=current):
    current = protocol
    source = parent.load(v216_root)
    out = out.resolve()
    if not out.name.startswith(current.LABEL + "_"):
        raise ValueError(f"use a new {current.LABEL}_ output directory")
    selection = source.out / "inputs/selection.json"
    scope = {**source.scope, "experiment": current.EXPERIMENT,
             "base_seed": current.Protocol.SEED, "confirmation_sources": list(current.SOURCE_INDICES),
             "parent_selection_sha256": base.sha256(selection), "parent_root": str(source.out),
             "rationale": getattr(current, "RATIONALE", "Second-seed replication and matched random-history control; no new tuning"),
             "same_prompts_across_seeds_not_independent": True}
    path = out / "inputs/selection.json"
    if path.exists() and parent.read(path) != scope:
        raise ValueError("v217 selection already frozen")
    for name in scope["evidence_sha256"]:
        base.write_frozen(out / "inputs/development" / name, (source.out / "inputs/development" / name).read_bytes())
    base.write_frozen(out / "inputs/v216_selection.json", selection.read_bytes())
    base.frozen_json(path, scope)
    return current.load(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v216-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    p = freeze(args.v216_root, args.output_root)
    print(f"[v217-frozen] method={p.scope['selected_method']} endpoint={p.PRIMARY_WINDOW}/{p.PRIMARY_METRIC} "
          f"seed={p.SEED}+source prompts=64 videos=192 nodes=8")


if __name__ == "__main__":
    main()
