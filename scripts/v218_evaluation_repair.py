#!/usr/bin/env python3
"""Immutable evaluation-only fork of completed v215 generation; eight-node evaluation."""
import argparse
import json
import math
import os
import re
from pathlib import Path

import v215_lphc_protocol as original
from v216_lphc_protocol import validate_nodes
from v212_lphc_protocol import runtime_hashes
from prepare_v212_comparison import prepare as publish, verify_published
from run_v211_worker import assert_authorized_node
from v210_vbench_fingerprint import vbench_checkout_fingerprint
from prepare_v210_vbench_comparison import DIMENSIONS


def evaluation_plan():
    return [{"node_rank": i % 8, "gpu": str(i // 8), "method": method, "dimension": dimension}
            for i, (method, dimension) in enumerate((m, d) for d in DIMENSIONS for m in original.METHODS)]


def validate_probe(probe_path, runtime_receipt):
    probe = json.loads(probe_path.read_text())
    if (probe.get("strict_loading_pass") is not True
            or probe.get("runtime_receipt_sha256") != original.sha256(runtime_receipt)
            or not re.fullmatch("[0-9a-f]{64}", probe.get("checkpoint_sha256", ""))):
        raise ValueError("run the strict RAFT GPU probe with this repaired runtime first")
    for name in ("identical", "right_shift4"):
        for field in ("median_x", "median_y", "mean_magnitude"):
            value = probe.get("synthetic_flow", {}).get(name, {}).get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("missing or nonfinite RAFT synthetic probe diagnostic")
    return probe


class Protocol:
    NODE_COUNT = 8
    EXTRA_METRICS = ("imaging_quality",)

    def __init__(self, out):
        self.out = Path(out).resolve()
        self.repair_path = self.out / "evaluation_repair.json"
        self.repair = json.loads(self.repair_path.read_text())
        self.digest = original.sha256(self.repair_path)
        self.source = Path(self.repair["original_run_root"])
        self.AUTHORIZED_NODES = validate_nodes(self.repair["authorized_nodes"])

    def __getattr__(self, name):
        return getattr(original, name)

    def verify(self, repo, out, *, runtime=True):
        if Path(out).resolve() != self.out or original.sha256(self.repair_path) != self.digest:
            raise ValueError("evaluation repair binding changed")
        if original.sha256(self.source / "inputs/manifest.json") != self.repair["original_inputs_sha256"]:
            raise ValueError("source generation manifest changed")
        if original.sha256(self.source / "evaluation/vbench_comparison/comparison_manifest.json") != self.repair["original_comparison_sha256"]:
            raise ValueError("source comparison changed")
        receipt = Path(self.repair["runtime_receipt"])
        probe_path = self.out / "raft_probe.json"
        if (original.sha256(receipt) != self.repair["runtime_receipt_sha256"]
                or original.sha256(probe_path) != self.repair["raft_probe_sha256"]):
            raise ValueError("RAFT runtime/probe receipt changed")
        validate_probe(probe_path, receipt)
        if runtime and self.repair["evaluation_runtime"] != runtime_hashes(repo):
            raise ValueError("evaluation code changed after repair preparation; use frozen checkout")
        data = original.verify(repo, self.source, runtime=False)
        for name in ("inputs", "jobs", "baseline", "decisions"):
            if not (self.out/name).is_symlink() or (self.out/name).resolve() != (self.source/name).resolve():
                raise ValueError("evaluation fork must retain original generation directories")
        comparison_path = self.out / "evaluation/vbench_comparison/comparison_manifest.json"
        if comparison_path.exists():
            original_jobs = json.loads((self.source / "evaluation/vbench_comparison/comparison_manifest.json").read_text())["jobs"]
            repaired_jobs = json.loads(comparison_path.read_text())["jobs"]
            if media_identity(repaired_jobs) != media_identity(original_jobs):
                raise ValueError("evaluation repair changed source media identities")
        return data

    def validate_node(self, rank, num_nodes=8):
        address = os.environ.get("V218_NODE_ADDRESS")
        if num_nodes != 8 or not 0 <= rank < 8 or address != self.AUTHORIZED_NODES[rank]:
            raise PermissionError("V218_NODE_ADDRESS/NODE_RANK must match the eight-node list")
        return assert_authorized_node({"authorized_nodes": list(self.AUTHORIZED_NODES)}, node_address=address)

    def validate_vbench_fingerprint(self, fingerprint):
        if fingerprint != self.repair["corrected_vbench_fingerprint"]:
            raise ValueError("repaired VBench runtime drift")

    def validate_checkpoint(self, digest):
        if digest != self.repair["raft_checkpoint_sha256"]:
            raise ValueError("evaluation RAFT checkpoint differs from the strict GPU probe")


def media_identity(jobs):
    result = {(r["method"], r["source_index"]): (r["done_sha256"], r["media_sha256"]) for r in jobs}
    if len(result) != len(jobs):
        raise ValueError("duplicate source media identity")
    return result


def prepare(repo, source, out, nodes, runtime_receipt):
    source, out = source.resolve(), original.output_root(out)
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("repair output must be separate from original generation")
    nodes = validate_nodes(nodes)
    runtime = json.loads(runtime_receipt.read_text())
    source_manifest = source / "evaluation/vbench_comparison/comparison_manifest.json"
    if runtime["source_comparison_sha256"] != original.sha256(source_manifest):
        raise ValueError("runtime repair was prepared for different videos")
    if runtime["fingerprint"] != vbench_checkout_fingerprint(Path(runtime["target_root"])):
        raise ValueError("corrected evaluator drift")
    probe_path = out / "raft_probe.json"
    probe = validate_probe(probe_path, runtime_receipt)
    original.verify(repo, source, runtime=False)
    binding = {"version": 1, "original_run_root": str(source), "original_inputs_sha256": original.sha256(source/"inputs/manifest.json"),
               "original_comparison_sha256": original.sha256(source_manifest), "authorized_nodes": list(nodes),
               "corrected_vbench_fingerprint": runtime["fingerprint"], "runtime_receipt": str(runtime_receipt.resolve()),
               "runtime_receipt_sha256": original.sha256(runtime_receipt), "evaluation_runtime": runtime_hashes(repo),
               "raft_probe_sha256": original.sha256(probe_path), "raft_checkpoint_sha256": probe["checkpoint_sha256"],
               "generation_reused": True, "old_metrics_reused": False}
    original.frozen_json(out / "evaluation_repair.json", binding)
    for name in ("inputs", "jobs", "baseline", "decisions"):
        target = out/name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != (source/name).resolve():
                raise ValueError(f"existing evaluation fork path differs: {target}")
        else:
            target.symlink_to(source/name, target_is_directory=True)
    protocol = Protocol(out)
    result = publish(repo, out, Path(runtime["target_root"]), protocol=protocol)
    if media_identity(result["jobs"]) != media_identity(json.loads(source_manifest.read_text())["jobs"]):
        raise ValueError("repaired comparison must reuse exactly the original videos")
    # Retain unchanged temporal diagnostics; bind them to the new media-identical comparison.
    from bind_temporal_diagnostics import build_contract
    csv = out / "evaluation/metrics/temporal_diagnostics.csv"
    original.write_frozen(csv, (source / "evaluation/metrics/temporal_diagnostics.csv").read_bytes())
    original.frozen_json(out / "evaluation/metrics/temporal_diagnostics.contract.json",
                         build_contract(out / "evaluation/vbench_comparison/comparison_manifest.json", csv))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(Path(__file__).resolve().parents[1], args.source_root, args.output_root,
                     json.loads(args.nodes.read_text()), args.runtime_receipt)
    print(f"[v218-eval-only] reused_videos={len(result['jobs'])} methods=7 dimensions=9 nodes=8")


if __name__ == "__main__":
    main()
