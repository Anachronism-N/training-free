#!/usr/bin/env python3
"""Freeze the four-arm alternative to an unstarted v217; do not run both."""
import argparse
from pathlib import Path

from prepare_v217_replication import freeze
import v219_lphc_protocol as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v216-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    p = freeze(args.v216_root, args.output_root, protocol=current)
    print(f"[v219-frozen] endpoint={p.PRIMARY_WINDOW}/{p.PRIMARY_METRIC} "
          "prompts=64 videos=256 nodes=8 gpu_slots=64; no method/endpoint retuning")


if __name__ == "__main__":
    main()
