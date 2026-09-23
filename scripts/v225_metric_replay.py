#!/usr/bin/env python3
"""Paired frozen-input VBench replay. No generation, resplitting, or seed search."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import audit_v224_reused_clips as audit
import export_lphc_paper_evidence as evidence
import run_v154_vbench_long as evaluator
from v210_vbench_fingerprint import vbench_checkout_fingerprint
from vbench_long_split_cache import clean_manifest_path

EXPERIMENT = "v225_fixed_input_metric_replay32"
CONDITIONS = ("reference", "resplit", "repeat")
METHODS = tuple(f"{m}__{c}" for m in audit.REUSED for c in CONDITIONS)
DIMENSIONS = evidence.RAW_METRICS
ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCRIPTS = ("v225_metric_replay.py", "analyze_v225_metric_replay.py",
                  "eval_vbench_long_prompt_aware.py", "run_v154_vbench_long.py")


def read(path):
    return evidence.read(Path(path))


def summarize_audit(root, scope=None):
    root = Path(root)
    shards = [read(root / f"node{i}.json") for i in range(8)]
    report = audit.collect(shards)
    if report != read(root / "reused_clip_audit.json"):
        raise ValueError("v224 collected report disagrees with its rank receipts")
    if scope is not None:
        binding = {k: scope[k] for k in ("v219_comparison_sha256", "v223_comparison_sha256")}
        if (report["binding"] != binding or shards[0]["expected_pair_ids"] != [p["pair_id"] for p in scope["pairs"]]):
            raise ValueError("v224 does not describe these source comparisons")
    clips = [c for r in report["rows"] for c in r["clips"]]
    if len(report["rows"]) != 64 or len(clips) != 960 or not all(r["same_source_bytes"] for r in report["rows"]):
        raise ValueError("need all 64 reused sources and 15 clips each")
    for row in report["rows"]:
        if [c["clip_index"] for c in row["clips"]] != list(range(15)):
            raise ValueError("invalid clip grid")
        for c in row["clips"]:
            for field, key in (("same_decoded_pixels", "pixels_sha256"), ("same_decoded_timing", "timing_sha256")):
                if c[field] != (c["reference"][key] == c["ablation"][key]):
                    raise ValueError("audit equality flag disagrees with fingerprints")
    return {"source_pair_count": 64, "clip_pair_count": 960,
            "encoded_mismatch_count": sum(not c["same_encoded_bytes"] for c in clips),
            "pixel_mismatch_count": sum(not c["same_decoded_pixels"] for c in clips),
            "timing_mismatch_count": sum(not c["same_decoded_timing"] for c in clips),
            "binding": report["binding"], "source_report_sha256": evidence.dev.sha256(root / "reused_clip_audit.json"),
            "boundary": "Input differences confirmed; their size and causal share of score drift are not established."}


def copy_frozen(source, target, expected):
    source, target = Path(source), Path(target)
    evidence.checked_hash(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        evidence.checked_hash(target, expected)
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        evidence.checked_hash(temporary, expected)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args):
    out = args.output_root.resolve()
    for root in (args.v219_root, args.v223_root, args.v224_root):
        source = root.resolve()
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("use a new output root outside source campaigns")
    scope = audit.plan(args.v219_root, args.v223_root)
    diagnosis = summarize_audit(args.v224_root, scope)
    comparison, _ = audit.comparison(args.v223_root, "v223_core_ablation")
    if vbench_checkout_fingerprint(args.vbench_root) != comparison["vbench_fingerprint"]:
        raise ValueError("keep the original repaired VBench checkout")
    if out.joinpath("comparison_manifest.json").exists():
        data = read(out / "comparison_manifest.json")
        if data["source_binding"] != scope or data["audit"] != diagnosis:
            raise ValueError("prepared source contract changed")
        verify_views(out, data)
        finalize_views(out, data)
        print("[v225-prepare] resumed frozen views; no files regenerated", flush=True)
        return data

    # Copy, rather than re-encode or link, so derived evaluator files cannot alter old caches.
    by_source = {r["source_index"]: i for i, r in enumerate(comparison["prompt_items"])}
    audit_rows = {r["pair_id"]: r for r in read(args.v224_root / "reused_clip_audit.json")["rows"]}
    def copy_pair(pair):
        index = by_source[pair["source_index"]]
        stem = f"{index:06d}-0"
        row = audit_rows[pair["pair_id"]]
        inputs = [audit.clip_paths(Path(folder)) for folder in pair["clip_dirs"]]
        hashes = []
        for version, files in enumerate(inputs):
            values = []
            for i, path in enumerate(files):
                digest = evidence.dev.sha256(path)
                expected = row["clips"][i]["reference" if version == 0 else "ablation"]
                if audit.decoded_fingerprint(path, args.ffmpeg) != expected:
                    raise ValueError(f"split pixels changed since v224: {pair['pair_id']} clip={i}")
                evidence.checked_hash(path, digest)
                values.append(digest)
            hashes.append(values)
        records = []
        for condition in CONDITIONS:
            method = f"{pair['method']}__{condition}"
            folder = out / "published" / method
            version = int(condition == "resplit")
            target = folder / f"{stem}.mp4"
            copy_frozen(pair["source_paths"][version], target, pair["source_hashes"][version])
            files = [{"path": str(target.relative_to(out)), "sha256": pair["source_hashes"][version]}]
            for i, path in enumerate(inputs[version]):
                target = folder / "split_clip" / stem / f"{stem}_{i:03d}.mp4"
                copy_frozen(path, target, hashes[version][i])
                files.append({"path": str(target.relative_to(out)), "sha256": hashes[version][i]})
            records.append({"method": method, "source_index": pair["source_index"], "index": index, "files": files})
        print(f"[v225-copy] pair={pair['pair_id']} conditions=3 source_verified=1 clips_verified=30", flush=True)
        return records
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = [r for bundle in pool.map(copy_pair, scope["pairs"]) for r in bundle]
    contract = read(args.v223_root / "evaluation/metrics/vbench_long_parts/sf_fifo21/dynamic_degree/job_contract.json")
    data = {"version": 1, "experiment": EXPERIMENT, "prompt_count": 32, "num_output_frames": 120,
            "prompt_items": comparison["prompt_items"], "vbench_long_dimensions": list(DIMENSIONS),
            "methods": [{"key": m, "video_dir": str(out / "published" / m)} for m in METHODS],
            "vbench_fingerprint": comparison["vbench_fingerprint"], "source_binding": scope,
            "dependencies": contract["dependencies"], "audit": diagnosis, "view_files": records,
            "script_sha256": {s: evidence.dev.sha256(ROOT / "scripts" / s) for s in FROZEN_SCRIPTS},
            "new_video_count": 0, "new_seed_count": 0,
            "evaluation_rng": "unchanged legacy evaluator; repeat estimates its realized variability, no seed search"}
    evidence.dev.frozen_json(out / "comparison_manifest.json", data)
    finalize_views(out, data)
    evidence.dev.frozen_json(out / "v224_diagnosis.json", diagnosis)
    print("[v225-prepare] views=6 sources=64 new_videos=0 tasks=54", flush=True)
    return data


def finalize_views(out, data):
    evidence.dev.frozen_json(out / "schedule.json", {"groups": schedule(), "nodes": 8,
        "active_gpus": 18, "available_gpus": 64, "metric_jobs": 54, "new_videos": 0})
    manifest_sha = evidence.dev.sha256(out / "comparison_manifest.json")
    for method in METHODS:
        folder = out / "published" / method
        evidence.dev.frozen_json(clean_manifest_path(folder), {
            "version": 1, "comparison_manifest_sha256": manifest_sha, "vbench_commit": data["vbench_fingerprint"]["head"],
            "method": method, "video_dir": str(folder), "prompt_count": 32, "clip_seconds": 2,
            "clips_per_video": 15, "source_videos": [
                {"name": f"{i:06d}-0.mp4", "size": (folder / f"{i:06d}-0.mp4").stat().st_size} for i in range(32)]})


def verify_views(out, data):
    if (data["experiment"] != EXPERIMENT or data["prompt_count"] != 32
            or data["num_output_frames"] != 120 or [m["key"] for m in data["methods"]] != list(METHODS)
            or data["vbench_long_dimensions"] != list(DIMENSIONS)):
        raise ValueError("unexpected replay design")
    if ([r["source_index"] for r in data["prompt_items"]] != list(audit.ablation.SOURCE_INDICES)
            or [r["index"] for r in data["prompt_items"]] != list(range(32))
            or any(r["effective_seed"] != 21600+r["source_index"] for r in data["prompt_items"])):
        raise ValueError("replay prompt membership changed")
    if set(data["script_sha256"]) != set(FROZEN_SCRIPTS):
        raise ValueError("incomplete replay script binding")
    for script, digest in data["script_sha256"].items():
        evidence.checked_hash(ROOT / "scripts" / script, digest)
    rows = audit.unique_by(data["view_files"], ("method", "index"))
    if set(rows) != set(itertools.product(METHODS, range(32))):
        raise ValueError("incomplete replay view grid")
    for method in audit.REUSED:
        for index in range(32):
            a = [r["sha256"] for r in rows[(method+"__reference", index)]["files"]]
            b = [r["sha256"] for r in rows[(method+"__repeat", index)]["files"]]
            if a != b:
                raise ValueError("repeat is not a byte-identical input copy")
    for row in data["view_files"]:
        stem = f"{row['index']:06d}-0"
        folder = Path("published") / row["method"]
        expected = [folder / f"{stem}.mp4", *[
            folder / "split_clip" / stem / f"{stem}_{i:03d}.mp4" for i in range(15)]]
        if ([Path(f["path"]) for f in row["files"]] != expected
                or row["source_index"] != audit.ablation.SOURCE_INDICES[row["index"]]):
            raise ValueError("view file identity or clip membership changed")
        for item in row["files"]:
            path = (out / item["path"]).resolve()
            if not path.is_relative_to(out):
                raise ValueError("view escaped replay output")
            evidence.checked_hash(path, item["sha256"])


def schedule():
    orders = tuple(itertools.permutations(CONDITIONS))
    return [{"index": i, "method": m, "dimension": d, "node_rank": i % 8, "gpu": str(i // 8),
             "conditions": list(orders[i % len(orders)])}
            for i, (d, m) in enumerate(itertools.product(DIMENSIONS, audit.REUSED))]


def configure():
    evaluator.RUN_LABEL = "v225"
    evaluator.COMPARISON_EXPERIMENT = evaluator.SUMMARY_EXPERIMENT = EXPERIMENT
    evaluator.PROMPT_COUNT, evaluator.NUM_OUTPUT_FRAMES, evaluator.CLIPS_PER_VIDEO = 32, 120, 15
    evaluator.METHODS, evaluator.DIMENSIONS = METHODS, DIMENSIONS
    evaluator.comparison_name = lambda index: f"{index:06d}-0.mp4"
    evaluator.analyze = lambda summary: {"diagnostic_only": True, "new_videos": 0}
    evaluator.render_markdown = lambda report: "# v225 metric replay\n\nSee v225_analysis.json for paired diagnostics.\n"


def eval_args(args):
    out = args.output_root.resolve()
    args.manifest = out / "comparison_manifest.json"
    args.wrapper = ROOT / "scripts/eval_vbench_long_prompt_aware.py"
    args.full_info = args.vbench_root / "vbench2_beta_long/VBench_full_info.json"
    args.parts_root = out / "metrics/vbench_long_parts"
    args.summary_root, args.analysis_root = out / "metrics", out / "analysis"
    args.summary_stem, args.analysis_stem = "vbench_core9_summary", "v225_aggregate"
    args.summary_title = "v225 fixed-input replay (diagnostic, not independent generation)"
    args.dimensions, args.local_models = DIMENSIONS, True
    return args


def context(args):
    configure()
    eval_args(args)
    for path in (args.vbench_root, args.vbench_cache, args.torch_hub_dir, args.runtime_home):
        if path is None or not path.is_dir():
            raise ValueError(f"restore the existing local evaluation cache/directory: {path}")
    data = read(args.manifest)
    verify_views(args.output_root.resolve(), data)
    if vbench_checkout_fingerprint(args.vbench_root) != data["vbench_fingerprint"]:
        raise ValueError("VBench fingerprint differs from source evaluations")
    result = evaluator.runtime_contract(args)
    for key, value in result["dependencies"].items():
        if value["sha256"] != data["dependencies"][key]["sha256"]:
            raise ValueError(f"evaluation dependency changed: {key}")
    return result


def device(gpu):
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=30)
    uuid = result.stdout.strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise ValueError("expected one physical GPU")
    return {"hostname": socket.gethostname(), "gpu_uuid": uuid, "slot": gpu}


def run_group(args, ctx, group):
    identity = f"{group['method']}__{group['dimension']}"
    folder = args.output_root / "groups" / identity
    folder.mkdir(parents=True, exist_ok=True)
    placement = {"device": device(group["gpu"]), "schedule": group, "manifest_sha256": ctx["manifest_sha256"]}
    evidence.dev.frozen_json(folder / "placement.json", placement)
    versions = {}
    for name in ("torch", "torchvision", "av", "decord", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    evidence.dev.frozen_json(folder / "package_versions.json", versions)
    # One process owns a group; resume failed jobs after removing only a verified stale lock.
    lock = folder / "running.lock"
    with lock.open("x") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "hostname": socket.gethostname()}))
    try:
        started = time.monotonic()
        results = []
        for condition in group["conditions"]:
            method = f"{group['method']}__{condition}"
            print(f"[v225-eval] group={identity} gpu={group['gpu']} condition={condition}", flush=True)
            evaluator.run_job(args, ctx, method=method, dimension=group["dimension"], gpu=group["gpu"])
            marker = args.parts_root / method / group["dimension"] / "done.json"
            results.append({"method": method, "dimension": group["dimension"], "done_sha256": evidence.dev.sha256(marker)})
        evidence.dev.frozen_json(folder / "done.json", {**placement, "jobs": results})
        print(f"[v225-group-done] {identity} elapsed_seconds={time.monotonic()-started:.1f}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def verify_groups(args, ctx):
    devices, nodes = set(), {}
    for group in schedule():
        folder = args.output_root / "groups" / f"{group['method']}__{group['dimension']}"
        placement, done = read(folder / "placement.json"), read(folder / "done.json")
        if (done["schedule"] != group or done["manifest_sha256"] != ctx["manifest_sha256"]
                or placement != {k: done[k] for k in ("device", "schedule", "manifest_sha256")}
                or [r["method"] for r in done["jobs"]] != [f"{group['method']}__{c}" for c in group["conditions"]]):
            raise ValueError("replay group binding changed")
        device_row = done["device"]
        if device_row["gpu_uuid"] in devices or device_row["slot"] != group["gpu"]:
            raise ValueError("groups overlap physical GPUs or use wrong slots")
        devices.add(device_row["gpu_uuid"])
        host = nodes.setdefault(group["node_rank"], device_row["hostname"])
        if host != device_row["hostname"]:
            raise ValueError("one rank ran on multiple hosts")
        for row in done["jobs"]:
            if row["dimension"] != group["dimension"]:
                raise ValueError("replay group dimension mismatch")
            evidence.checked_hash(args.parts_root / row["method"] / row["dimension"] / "done.json", row["done_sha256"])
    if len(set(nodes.values())) != 8:
        raise ValueError("use eight distinct hosts for the frozen replay placement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "eval", "status", "collect", "schedule", "summarize-audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v219-root", type=Path)
    parser.add_argument("--v223-root", type=Path)
    parser.add_argument("--v224-root", type=Path)
    parser.add_argument("--vbench-root", type=Path)
    parser.add_argument("--vbench-cache", type=Path)
    parser.add_argument("--torch-hub-dir", type=Path)
    parser.add_argument("--runtime-home", type=Path)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if not 0 <= args.node_rank < 8 or not 1 <= args.workers <= 32:
        parser.error("require rank 0..7 and workers 1..32")
    if args.action == "schedule":
        evidence.dev.frozen_json(args.output_root / "schedule.json", {"groups": schedule(), "nodes": 8,
            "active_gpus": 18, "available_gpus": 64, "metric_jobs": 54, "new_videos": 0})
        print(json.dumps(schedule(), indent=2))
        return
    if args.action == "summarize-audit":
        if args.v224_root is None:
            parser.error("summarize-audit needs --v224-root")
        result = summarize_audit(args.v224_root)
        evidence.dev.frozen_json(args.output_root / "v224_diagnosis.json", result)
        print(json.dumps(result, indent=2))
        return
    required = ("v219_root", "v223_root", "v224_root", "vbench_root") if args.action == "prepare" else (
        "vbench_root", "vbench_cache", "torch_hub_dir", "runtime_home")
    if any(getattr(args, name) is None for name in required):
        parser.error("required paths: " + ", ".join(required))
    if args.action in ("prepare", "collect") and args.node_rank != 0:
        parser.error("prepare/collect runs once on rank 0")
    if args.action == "prepare":
        prepare(args)
        return
    ctx = context(args)
    if args.action == "status":
        print(json.dumps(evaluator.completion_report(args, ctx, evaluator.all_jobs()), indent=2))
    elif args.action == "eval":
        groups = [g for g in schedule() if g["node_rank"] == args.node_rank]
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            list(pool.map(lambda g: run_group(args, ctx, g), groups))
        verify_views(args.output_root, ctx["manifest"])
    else:
        verify_groups(args, ctx)
        evaluator.collect(args, ctx)
        from analyze_v225_metric_replay import analyze
        analyze(args.output_root)


if __name__ == "__main__":
    main()
