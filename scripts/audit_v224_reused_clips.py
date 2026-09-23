#!/usr/bin/env python3
"""CPU-only comparison of reused SF/Ours source videos and VBench split pixels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import export_lphc_paper_evidence as evidence
import v223_lphc_protocol as ablation

REUSED = ("sf_fifo21", "ours_correct")


def comparison(root, report_name):
    root = Path(root).resolve()
    report = evidence.read(root / f"evaluation/analysis/{report_name}.json")
    locations = (root / "evaluation/vbench_comparison/comparison_manifest.json",
                 root / "inputs/campaign_comparison.json")
    found = [p for p in locations if p.is_file()]
    if not found:
        raise FileNotFoundError(f"missing comparison manifest in {root}; export the original, do not recreate it")
    for path in found:
        evidence.checked_hash(path, report["source"]["manifest_sha256"])
    return evidence.read(found[0]), report["source"]["manifest_sha256"]


def unique_by(rows, fields):
    result = {tuple(row[k] for k in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate identity: {fields}")
    return result


def plan(v219_root, v223_root):
    a, ah = comparison(v219_root, "v219_mechanism")
    b, bh = comparison(v223_root, "v223_core_ablation")
    if a["experiment"] != "v219_frozen_headwise_mechanism64" or b["experiment"] != ablation.EXPERIMENT:
        raise ValueError("wrong campaigns")
    if a["num_output_frames"] != 120 or b["num_output_frames"] != 120:
        raise ValueError("reuse audit requires matching 30s videos")
    if a["vbench_fingerprint"] != b["vbench_fingerprint"]:
        raise ValueError("different evaluator fingerprints; do not treat this as repeat evaluation")
    old_items = unique_by(a["prompt_items"], ("source_index",))
    old_jobs = unique_by(a["jobs"], ("method", "source_index"))
    new_jobs = unique_by(b["jobs"], ("method", "source_index"))
    if [r["source_index"] for r in b["prompt_items"]] != list(ablation.SOURCE_INDICES):
        raise ValueError("ablation subset changed")
    old_index = {r["source_index"]: i for i, r in enumerate(a["prompt_items"])}
    methods_a = {r["key"]: r for r in a["methods"]}
    methods_b = {r["key"]: r for r in b["methods"]}
    pairs = []
    for i, item in enumerate(b["prompt_items"]):
        s = item["source_index"]
        old = old_items[(s,)]
        if any(old[k] != item[k] for k in ("sha256", "text", "effective_seed")):
            raise ValueError("reused prompt/seed differs")
        for method in REUSED:
            if methods_a[method]["spec"] != methods_b[method]["spec"]:
                raise ValueError("reused method specification changed")
            aa, bb = old_jobs[(method, s)], new_jobs[(method, s)]
            pairs.append({"pair_id": f"{method}:{s}", "method": method, "source_index": s,
                          "effective_seed": item["effective_seed"],
                          "source_paths": [aa["media_path"], bb["media_path"]],
                          "source_hashes": [aa["media_sha256"], bb["media_sha256"]],
                          "clip_dirs": [str(Path(methods_a[method]["video_dir"]) / "split_clip" / f"{old_index[s]:06d}-0"),
                                        str(Path(methods_b[method]["video_dir"]) / "split_clip" / f"{i:06d}-0")]})
    return {"version": 1, "v219_comparison_sha256": ah, "v223_comparison_sha256": bh,
            "pairs": pairs, "new_video_count": 0, "new_seed_count": 0}


def clip_paths(folder):
    folder = Path(folder)
    expected = [folder / f"{folder.name}_{i:03d}.mp4" for i in range(15)]
    if set(folder.glob("*.mp4")) != set(expected):
        raise ValueError(f"missing/extra clips in {folder}; preserve original clips for diagnosis")
    return expected


def decoded_fingerprint(path, ffmpeg="ffmpeg"):
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", "rgb24", "-vsync", "0",
               "-threads", "1", "-f", "framemd5", "-"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr[-1000:]}")
    frames, timing, dimensions = [], [], []
    for line in result.stdout.splitlines():
        if line.startswith("#dimensions"):
            dimensions.append(line)
        elif line and not line.startswith("#"):
            fields = [s.strip() for s in line.split(",")]
            if len(fields) != 6:
                raise ValueError("unexpected ffmpeg framemd5 format")
            frames.append(fields[4:])
            timing.append(fields[:4])
        elif line.startswith("#tb"):
            timing.append(line)
    if not frames or not dimensions:
        raise ValueError(f"no decoded frames or dimensions in {path}")
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"frames": len(frames), "pixels_sha256": digest([dimensions, frames]),
            "timing_sha256": digest(timing)}


def audit_pair(pair, ffmpeg="ffmpeg"):
    row = {k: pair[k] for k in ("pair_id", "method", "source_index", "effective_seed")}
    actual = [evidence.dev.sha256(Path(p)) for p in pair["source_paths"]]
    if actual != pair["source_hashes"]:
        raise ValueError(f"source media changed since evaluation: {pair['pair_id']}")
    row["same_source_bytes"] = actual[0] == actual[1]
    if not row["same_source_bytes"]:
        return {**row, "state": "different_source_video", "clips": []}
    clips = []
    for index, (a, b) in enumerate(zip(*map(clip_paths, pair["clip_dirs"]))):
        ah, bh = evidence.dev.sha256(a), evidence.dev.sha256(b)
        same_bytes = ah == bh
        aa = decoded_fingerprint(a, ffmpeg)
        bb = aa if same_bytes else decoded_fingerprint(b, ffmpeg)
        clips.append({"clip_index": index, "same_encoded_bytes": same_bytes,
                      "same_decoded_pixels": aa["pixels_sha256"] == bb["pixels_sha256"],
                      "same_decoded_timing": aa["timing_sha256"] == bb["timing_sha256"],
                      "reference": aa, "ablation": bb})
    same = all(x["same_decoded_pixels"] and x["same_decoded_timing"] for x in clips)
    return {**row, "state": "same_decoded_input" if same else "split_input_changed", "clips": clips}


def run_shard(scope, rank, count, workers, ffmpeg="ffmpeg"):
    if count not in (1, 8) or not 0 <= rank < count or not 1 <= workers <= 8:
        raise ValueError("use one node or eight ranks, with 1..8 CPU workers per node")
    tasks = scope["pairs"][rank::count]
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=20)
    print(f"[v224-clips] rank={rank}/{count} pairs={len(tasks)} workers={workers} new_videos=0", flush=True)
    def work(pair):
        row = audit_pair(pair, ffmpeg)
        print(f"[v224-clips] rank={rank} pair={row['pair_id']} state={row['state']}", flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, tasks))
    return {"version": 1, "node_rank": rank, "num_nodes": count,
            "decoder_version": version.stdout.splitlines()[0],
            "binding": {k: scope[k] for k in ("v219_comparison_sha256", "v223_comparison_sha256")},
            "expected_pair_ids": [r["pair_id"] for r in scope["pairs"]], "rows": rows}


def collect(shards):
    if not shards:
        raise ValueError("no shards")
    first = shards[0]
    n = first["num_nodes"]
    if n not in (1, 8) or len(shards) != n or {r["node_rank"] for r in shards} != set(range(n)):
        raise ValueError("incomplete or duplicate rank set")
    if any(any(r[k] != first[k] for k in ("binding", "num_nodes", "expected_pair_ids")) for r in shards):
        raise ValueError("shards do not share one audit contract")
    for shard in shards:
        expected = first["expected_pair_ids"][shard["node_rank"]::n]
        if [r["pair_id"] for r in shard["rows"]] != expected:
            raise ValueError("rank does not cover its assigned pairs in order")
    rows = [r for shard in sorted(shards, key=lambda r: r["node_rank"]) for r in shard["rows"]]
    ids = [r["pair_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(first["expected_pair_ids"]):
        raise ValueError("pair coverage is incomplete or duplicated")
    states = ("same_decoded_input", "split_input_changed", "different_source_video")
    if any(r["state"] not in states for r in rows):
        raise ValueError("unknown audit state")
    return {"version": 1, "binding": first["binding"], "rows": rows, "pair_count": len(rows),
            "decoder_versions": sorted({s["decoder_version"] for s in shards}),
            "counts": {state: sum(r["state"] == state for r in rows)
                       for state in states},
            "new_video_count": 0, "new_seed_count": 0,
            "boundary": "Identical decoded input localizes remaining score differences to evaluator/runtime behavior, "
                        "not necessarily a bug. Different pixels/timing require preprocessing inspection; "
                        "neither outcome authorizes deleting prompts or regenerating videos."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "collect", "export-contract"))
    parser.add_argument("--v219-root", type=Path)
    parser.add_argument("--v223-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.action == "export-contract":
        original = args.v223_root / "evaluation/vbench_comparison/comparison_manifest.json"
        comparison(args.v223_root, "v223_core_ablation")
        target = args.output_root / "inputs/campaign_comparison.json"
        evidence.dev.write_frozen(target, original.read_bytes())
        print(f"[v224-export] original contract copied, not regenerated: {target}")
    elif args.action == "run":
        if args.v219_root is None:
            parser.error("run requires --v219-root")
        scope = plan(args.v219_root, args.v223_root)
        row = run_shard(scope, args.node_rank, args.num_nodes, args.workers, args.ffmpeg)
        evidence.dev.frozen_json(args.output_root / f"node{args.node_rank}.json", row)
    else:
        rows = [evidence.read(args.output_root / f"node{i}.json") for i in range(args.num_nodes)]
        report = collect(rows)
        evidence.dev.frozen_json(args.output_root / "reused_clip_audit.json", report)
        print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
