#!/usr/bin/env python3
"""Freeze the final fixed-seed ablation; never select prompts by their scores."""
import argparse
from pathlib import Path

import export_lphc_horizon_evidence as evidence
import v223_lphc_protocol as protocol


def freeze(reference_root, out):
    root, out = Path(reference_root).resolve(), Path(out).resolve()
    if root == out or not out.name.startswith("v223_"):
        raise ValueError("use a separate v223_ output directory")
    bundle = evidence.load(root, "v219")
    scope = {**bundle["scope"], "experiment": protocol.EXPERIMENT,
             "confirmation_sources": list(protocol.SOURCE_INDICES),
             "reference_root": str(root), "reference_report_sha256": bundle["report_sha256"],
             "reference_input_sha256": bundle["comparison"]["input_manifest_sha256"],
             "reference_comparison_sha256": protocol.base.sha256(root / "evaluation/vbench_comparison/comparison_manifest.json"),
             "rationale": "One final dose/phase ablation; same seed and uniform prompt subset, unchanged model. Reuse SF/Ours, no seed search.",
             "method_selection_allowed": False, "new_video_count": 64}
    for name in scope["evidence_sha256"]:
        protocol.base.write_frozen(out / "inputs/development" / name, (root / "inputs/development" / name).read_bytes())
    protocol.base.frozen_json(out / "inputs/selection.json", scope)
    return protocol.Protocol(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v219-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    p = freeze(args.v219_root, args.output_root)
    print(f"[v223-freeze] prompts=32 new_videos=64 reused_videos=64 seed=21600+source root={p.out}")


if __name__ == "__main__":
    main()
