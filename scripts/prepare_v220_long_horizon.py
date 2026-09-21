#!/usr/bin/env python3
"""Freeze 60-second SF/Ours without changing v219 or requiring its results."""
import argparse
from pathlib import Path

from prepare_v217_replication import freeze
import v220_lphc_protocol as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v216-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    p = freeze(args.v216_root, args.output_root, protocol=current)
    print(f"[v220-frozen] endpoint={p.PRIMARY_WINDOW}/{p.PRIMARY_METRIC} "
          f"latent_frames={p.FRAMES} decoded_frames={4*p.FRAMES-3} fps=16 "
          "prompts=64 videos=128 nodes=8 gpu_slots=64; no parameter retuning")


if __name__ == "__main__":
    main()
