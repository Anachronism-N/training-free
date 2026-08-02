#!/usr/bin/env python3
"""Run prompt-correct, dimension-sharded VBench-Long for v154."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from analyze_v154_vbench import analyze, render_markdown
from prepare_v154_vbench_comparison import (
    COMPARISON_EXPERIMENT,
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
    comparison_name,
)
from vbench_long_split_cache import clean_manifest_path, validate_clean_split


CLIPS_PER_VIDEO = 15
RUN_LABEL = "v154"
SUMMARY_EXPERIMENT = "v154_history_critical_moviebench16_vbench"
ANALYSIS_STEM = "v154_vbench_analysis"
SUMMARY_TITLE = "v154 VBench-Long Summary"
FAILURE_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|FileNotFoundError",
    re.IGNORECASE,
)
VIDEO_PATTERN = re.compile(r"^(\d+)-(\d+)(?:_|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomically(path: Path, payload: Any, *, sort_keys: bool) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def write_frozen_json(path: Path, payload: Any) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen VBench job contract differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        for key in ("score", "overall", "mean", "average", "total_score"):
            if key in value:
                found = finite_score(value[key])
                if found is not None:
                    return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = finite_score(item)
            if found is not None:
                return found
    return None


def prompt_indices(value: Any) -> set[int]:
    indices: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"video_path", "video", "path"} and isinstance(
                item, str
            ):
                for part in reversed(Path(item).parts):
                    match = VIDEO_PATTERN.match(Path(part).stem)
                    if match:
                        if int(match.group(2)) != 0:
                            raise ValueError(
                                f"unexpected VBench sample index in {item}"
                            )
                        indices.add(int(match.group(1)))
                        break
            else:
                indices.update(prompt_indices(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            indices.update(prompt_indices(item))
    return indices


def valid_result(path: Path, dimension: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or dimension not in payload:
        return None
    if finite_score(payload[dimension]) is None:
        return None
    if prompt_indices(payload[dimension]) != set(range(PROMPT_COUNT)):
        return None
    return payload


def normalize_result(output: Path, dimension: str) -> dict[str, Any]:
    target = output / "results.json"
    payload = valid_result(target, dimension) if target.is_file() else None
    if payload is None:
        candidates = sorted(
            output.glob("*_eval_results.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for candidate in candidates:
            payload = valid_result(candidate, dimension)
            if payload is not None:
                write_json_atomically(target, payload, sort_keys=False)
                break
    if payload is None:
        raise RuntimeError(
            f"no finite prompt-complete {dimension} result under {output}"
        )
    return payload


def validate_prompt_mapping(path: Path, manifest_sha: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    full_info = Path(str(payload["full_info"]))
    checks = {
        "comparison_manifest_sha256": manifest_sha,
        "prompt_mapping": "comparison_manifest_exact",
        "prompt_count": PROMPT_COUNT,
        "mapped_count": PROMPT_COUNT,
        "indices": list(range(PROMPT_COUNT)),
        "full_info_sha256": sha256(full_info),
    }
    failures = {
        key: {"actual": payload.get(key), "expected": expected}
        for key, expected in checks.items()
        if payload.get(key) != expected
    }
    if failures:
        raise ValueError(f"invalid {RUN_LABEL} prompt mapping: {failures}")
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = tuple(row.get("key") for row in payload.get("methods", []))
    prompt_items = payload.get("prompt_items") or []
    if (
        payload.get("experiment") != COMPARISON_EXPERIMENT
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("num_output_frames", -1)) != 120
        or methods != METHODS
        or tuple(payload.get("vbench_long_dimensions", ())) != DIMENSIONS
        or len(prompt_items) != PROMPT_COUNT
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(PROMPT_COUNT))
    ):
        raise ValueError(
            f"invalid {RUN_LABEL} VBench comparison manifest: {path}"
        )
    expected_names = {
        comparison_name(index) for index in range(PROMPT_COUNT)
    }
    for row in payload["methods"]:
        video_dir = Path(str(row["video_dir"]))
        actual = {item.name for item in video_dir.glob("*.mp4")}
        if actual != expected_names:
            raise ValueError(
                f"{row['key']}: incomplete VBench videos; "
                f"missing={sorted(expected_names - actual)} "
                f"extra={sorted(actual - expected_names)}"
            )
    return payload


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def runtime_contract(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    manifest_sha = sha256(args.manifest)
    vbench_commit = git_commit(args.vbench_root)
    required = {
        "wrapper": args.wrapper,
        "full_info": args.full_info,
        "raft": args.vbench_cache / "raft_model" / "models" / "raft-things.pth",
        "amt": args.vbench_cache / "amt_model" / "amt-s.pth",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing {RUN_LABEL} VBench dependencies: {missing}")
    split_audits = {}
    for row in manifest["methods"]:
        method = str(row["key"])
        video_dir = Path(str(row["video_dir"]))
        audit = validate_clean_split(
            video_dir,
            comparison_manifest_sha256=manifest_sha,
            vbench_commit=vbench_commit,
            prompt_count=PROMPT_COUNT,
            clips_per_video=CLIPS_PER_VIDEO,
        )
        if audit is None:
            raise ValueError(
                f"{method}: missing/stale split cache; run the split action"
            )
        split_audits[method] = {
            "manifest": str(clean_manifest_path(video_dir)),
            "manifest_sha256": sha256(clean_manifest_path(video_dir)),
            "clip_count": int(audit["clip_count"]),
        }
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "vbench_commit": vbench_commit,
        "dependencies": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in required.items()
        },
        "split_audits": split_audits,
    }


def all_jobs() -> list[tuple[str, str]]:
    return [
        (method, dimension)
        for dimension in DIMENSIONS
        for method in METHODS
    ]


def job_contract(
    context: dict[str, Any],
    *,
    method: str,
    dimension: str,
    video_dir: Path,
) -> dict[str, Any]:
    return {
        "version": 1,
        "comparison_manifest_sha256": context["manifest_sha256"],
        "method": method,
        "dimension": dimension,
        "video_dir": str(video_dir.resolve()),
        "vbench_commit": context["vbench_commit"],
        "dependencies": context["dependencies"],
        "split_manifest_sha256": context["split_audits"][method][
            "manifest_sha256"
        ],
        "prompt_mapping": "comparison_manifest_exact",
        "mode": "long_custom_input",
        "dev_flag": True,
        "num_of_samples_per_prompt": 1,
    }


def marker_payload(
    context: dict[str, Any],
    *,
    method: str,
    dimension: str,
    result: Path,
    contract: Path,
    mapping: Path,
) -> dict[str, Any]:
    return {
        "version": 1,
        "comparison_manifest_sha256": context["manifest_sha256"],
        "method": method,
        "dimension": dimension,
        "result_sha256": sha256(result),
        "job_contract_sha256": sha256(contract),
        "prompt_mapping_sha256": sha256(mapping),
        "vbench_commit": context["vbench_commit"],
    }


def marker_is_valid(path: Path, expected: dict[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def run_job(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    method: str,
    dimension: str,
    gpu: str,
) -> dict[str, Any]:
    method_row = next(
        row for row in context["manifest"]["methods"] if row["key"] == method
    )
    video_dir = Path(str(method_row["video_dir"]))
    output = args.parts_root / method / dimension
    result = output / "results.json"
    marker = output / "done.json"
    contract_path = output / "job_contract.json"
    mapping = output / "prompt_mapping.json"
    log_path = output / "run.log"
    output.mkdir(parents=True, exist_ok=True)
    contract = job_contract(
        context,
        method=method,
        dimension=dimension,
        video_dir=video_dir,
    )
    write_frozen_json(contract_path, contract)
    try:
        normalize_result(output, dimension)
        validate_prompt_mapping(mapping, context["manifest_sha256"])
        expected_marker = marker_payload(
            context,
            method=method,
            dimension=dimension,
            result=result,
            contract=contract_path,
            mapping=mapping,
        )
        if marker_is_valid(marker, expected_marker):
            return {
                "method": method,
                "dimension": dimension,
                "gpu": gpu,
                "status": "resumed",
            }
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
        pass

    for path in (result, marker, mapping):
        path.unlink(missing_ok=True)
    for path in output.glob("*_eval_results.json"):
        path.unlink()
    command = [
        sys.executable,
        str(args.wrapper),
        "--vbench-root",
        str(args.vbench_root),
        "--comparison-manifest",
        str(args.manifest),
        "--videos_path",
        str(video_dir),
        "--dimension",
        dimension,
        "--output_path",
        str(output),
        "--full_json_dir",
        str(args.full_info),
        "--num_of_samples_per_prompt",
        "1",
        "--dev_flag",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["VBENCH_CACHE_DIR"] = str(args.vbench_cache)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=args.vbench_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0 or FAILURE_PATTERN.search(log_text):
        raise RuntimeError(
            f"VBench failed method={method} dimension={dimension}; "
            f"see {log_path}"
        )
    normalize_result(output, dimension)
    validate_prompt_mapping(mapping, context["manifest_sha256"])
    expected_marker = marker_payload(
        context,
        method=method,
        dimension=dimension,
        result=result,
        contract=contract_path,
        mapping=mapping,
    )
    write_json_atomically(marker, expected_marker, sort_keys=True)
    return {
        "method": method,
        "dimension": dimension,
        "gpu": gpu,
        "status": "generated",
    }


def run_worker(
    args: argparse.Namespace,
    context: dict[str, Any],
    gpu: str,
    jobs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    results = []
    for method, dimension in jobs:
        print(
            f"[v154-vbench-task] gpu={gpu} method={method} "
            f"dimension={dimension}",
            flush=True,
        )
        try:
            results.append(
                run_job(
                    args,
                    context,
                    method=method,
                    dimension=dimension,
                    gpu=gpu,
                )
            )
        except Exception as error:
            results.append(
                {
                    "method": method,
                    "dimension": dimension,
                    "gpu": gpu,
                    "status": "failed",
                    "error": str(error),
                }
            )
            print(f"[failed] gpu={gpu}: {error}", flush=True)
    return results


def collect(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    sources: dict[str, dict[str, str]] = {}
    for method in METHODS:
        rows[method] = {}
        sources[method] = {}
        method_row = next(
            row
            for row in context["manifest"]["methods"]
            if row["key"] == method
        )
        video_dir = Path(str(method_row["video_dir"]))
        for dimension in DIMENSIONS:
            output = args.parts_root / method / dimension
            result = output / "results.json"
            marker = output / "done.json"
            contract_path = output / "job_contract.json"
            mapping = output / "prompt_mapping.json"
            expected_contract = job_contract(
                context,
                method=method,
                dimension=dimension,
                video_dir=video_dir,
            )
            write_frozen_json(contract_path, expected_contract)
            payload = normalize_result(output, dimension)
            validate_prompt_mapping(mapping, context["manifest_sha256"])
            expected_marker = marker_payload(
                context,
                method=method,
                dimension=dimension,
                result=result,
                contract=contract_path,
                mapping=mapping,
            )
            if not marker_is_valid(marker, expected_marker):
                raise ValueError(f"stale VBench marker: {method}:{dimension}")
            value = finite_score(payload[dimension])
            if value is None:
                raise ValueError(f"missing VBench score: {method}:{dimension}")
            rows[method][dimension] = value
            sources[method][dimension] = str(result)
    summary = {
        "version": 1,
        "experiment": SUMMARY_EXPERIMENT,
        "comparison_manifest": str(args.manifest),
        "comparison_manifest_sha256": context["manifest_sha256"],
        "methods": rows,
        "dimensions": list(DIMENSIONS),
        "sources": sources,
        "missing": [],
    }
    args.summary_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_root / "vbench_long_summary.json"
    write_json_atomically(summary_json, summary, sort_keys=False)
    with (args.summary_root / "vbench_long_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *DIMENSIONS])
        for method in METHODS:
            writer.writerow(
                [method, *(rows[method][dimension] for dimension in DIMENSIONS)]
            )
    markdown = [
        f"# {SUMMARY_TITLE}",
        "",
        "| Method | " + " | ".join(DIMENSIONS) + " |",
        "|---|" + "|".join("---:" for _ in DIMENSIONS) + "|",
    ]
    for method in METHODS:
        values = [f"{rows[method][dimension]:.5f}" for dimension in DIMENSIONS]
        markdown.append(f"| {method} | " + " | ".join(values) + " |")
    (args.summary_root / "vbench_long_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    report = analyze(summary)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_json_atomically(
        args.analysis_root / f"{ANALYSIS_STEM}.json",
        report,
        sort_keys=True,
    )
    (args.analysis_root / f"{ANALYSIS_STEM}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "eval", "collect"))
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--vbench-root", required=True, type=Path)
    parser.add_argument("--vbench-cache", required=True, type=Path)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--summary-root", required=True, type=Path)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=root / "scripts" / "eval_vbench_long_prompt_aware.py",
    )
    args = parser.parse_args()
    args.comparison_root = args.comparison_root.resolve()
    args.manifest = args.comparison_root / "comparison_manifest.json"
    args.vbench_root = args.vbench_root.resolve()
    args.vbench_cache = args.vbench_cache.expanduser().resolve()
    args.parts_root = args.parts_root.resolve()
    args.summary_root = args.summary_root.resolve()
    args.analysis_root = args.analysis_root.resolve()
    args.wrapper = args.wrapper.resolve()
    args.full_info = (
        args.vbench_root / "vbench2_beta_long" / "VBench_full_info.json"
    )
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("require 0 <= node-rank < num-nodes")
    args.gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    return args


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(
            f"missing {args.manifest}; run the VBench prepare action first"
        )
    context = runtime_contract(args)
    jobs = all_jobs()[args.node_rank :: args.num_nodes]
    if args.mode == "preflight":
        print(
            f"[{RUN_LABEL}-vbench-preflight] node={args.node_rank}/{args.num_nodes} "
            f"jobs={len(jobs)} gpus={len(args.gpus)} "
            f"manifest_sha256={context['manifest_sha256']} "
            f"vbench_commit={context['vbench_commit']}",
            flush=True,
        )
        return
    if args.mode == "collect":
        if args.node_rank != 0:
            raise SystemExit("collect must run on node rank 0")
        report = collect(args, context)
        print(
            f"[{RUN_LABEL}-vbench-collect] gate={report['metric_promotion_gate']} "
            f"output={args.summary_root}",
            flush=True,
        )
        return

    worker_jobs = [jobs[index :: len(args.gpus)] for index in range(len(args.gpus))]
    worker_jobs = [items for items in worker_jobs if items]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(worker_jobs))) as executor:
        futures = {
            executor.submit(
                run_worker,
                args,
                context,
                args.gpus[index],
                items,
            ): args.gpus[index]
            for index, items in enumerate(worker_jobs)
        }
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [row for row in results if row["status"] == "failed"]
    summary = {
        "version": 1,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "comparison_manifest_sha256": context["manifest_sha256"],
        "task_count": len(jobs),
        "result_count": len(results),
        "results": sorted(
            results, key=lambda row: (row["dimension"], row["method"])
        ),
        "failures": failures,
        "ok": not failures and len(results) == len(jobs),
    }
    summary_path = args.parts_root / f"node{args.node_rank}.summary.json"
    write_json_atomically(summary_path, summary, sort_keys=True)
    if not summary["ok"]:
        raise SystemExit(
            "\n".join(row["error"] for row in failures)
            or f"{RUN_LABEL} VBench task count mismatch"
        )
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
