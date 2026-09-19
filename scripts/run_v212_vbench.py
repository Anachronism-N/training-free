#!/usr/bin/env python3
"""Use the prompt-aware evaluator with v212 source and VBench fingerprints."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import run_v154_vbench_long as base
import v212_lphc_protocol as p
from prepare_v210_vbench_comparison import DIMENSIONS
from prepare_v212_comparison import verify_published
from v210_vbench_fingerprint import vbench_checkout_fingerprint
from vbench_quality_contract import exclusive_scores, quality_score_with_fixed_dynamic

_runtime_contract = base.runtime_contract
_job_contract = base.job_contract


def runtime_contract(args):
    p.validate_node(args.node_rank, args.num_nodes)
    manifest = verify_published(Path(__file__).resolve().parents[1], args.comparison_root, protocol=p)
    fingerprint = vbench_checkout_fingerprint(args.vbench_root)
    if fingerprint != manifest["vbench_fingerprint"]:
        raise ValueError("VBench checkout drift after publish")
    result = _runtime_contract(args)
    result[p.LABEL + "_fingerprint"] = fingerprint
    return result


def job_contract(context, **kwargs):
    result = _job_contract(context, **kwargs)
    result[p.LABEL + "_fingerprint"] = context[p.LABEL + "_fingerprint"]
    return result


def analyze(summary):
    return {"development_only": True, "paper_claim_ready": False, "methods": {
        method: {"quality_without_dynamic_degree": quality_score_with_fixed_dynamic(row, dynamic_value=1.),
                 **exclusive_scores(row)} for method, row in summary["methods"].items()}}


def main():
    global p
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--campaign", choices=("v212", "v213", "v214", "v215", "v216"), default="v212")
    args, remaining = parser.parse_known_args()
    p = p.load_protocol(args.campaign)
    sys.argv[1:] = remaining
    base.RUN_LABEL = p.LABEL
    base.COMPARISON_EXPERIMENT = base.SUMMARY_EXPERIMENT = p.EXPERIMENT
    base.METHODS = p.METHODS
    base.PROMPT_COUNT = len(p.SOURCE_INDICES)
    base.NUM_OUTPUT_FRAMES = 120
    base.CLIPS_PER_VIDEO = 15
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = lambda index: f"{index:06d}-0.mp4"
    base.runtime_contract = runtime_contract
    base.job_contract = job_contract
    base.analyze = analyze
    base.render_markdown = lambda report: f"# {p.LABEL} Aggregate\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    base.main()


if __name__ == "__main__":
    main()
