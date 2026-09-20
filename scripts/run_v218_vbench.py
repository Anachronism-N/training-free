#!/usr/bin/env python3
"""Run the existing core-9 evaluator against the isolated corrected v215 view."""
import argparse
from pathlib import Path
import sys

import run_v212_vbench as shared
from v218_evaluation_repair import Protocol


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    shared.configure(Protocol(args.run_root))
    shared.base.RUN_LABEL = "v218"
    shared.base.main()


if __name__ == "__main__":
    main()
