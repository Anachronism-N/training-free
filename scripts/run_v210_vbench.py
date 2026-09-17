#!/usr/bin/env python3
"""Run prompt-correct VBench-Long core-9 for a frozen v210 comparison."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_v154_vbench_long as base
from prepare_v210_vbench_comparison import (
    DIMENSIONS,
    EVALUATION_RUNTIME_FILES,
    EXPERIMENT as EXPECTED_EXPERIMENT,
)
from v210_vbench_fingerprint import vbench_checkout_fingerprint
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    exclusive_scores,
    official_quality_score,
    quality_score_with_fixed_dynamic,
)


METHODS: tuple[str, ...] = ()
PROMPT_COUNT = 0
EXPERIMENT = ""
EXPECTED_METHOD_COUNT = 8
EXPECTED_PROMPT_COUNT = 8
NUM_OUTPUT_FRAMES = 120
CLIPS_PER_VIDEO = 15
FINGERPRINT_NAME = "vbench_checkout_fingerprint.json"
EVALUATION_ROOT: Path | None = None
_BASE_RUNTIME_CONTRACT = base.runtime_contract
_BASE_JOB_CONTRACT = base.job_contract
_BASE_COMPLETION_REPORT = base.completion_report
_BASE_COLLECT = base.collect
_BASE_RUN_JOB = base.run_job
SPLIT_PROVENANCE_NAME = ".v210_split_provenance.json"
CAMPAIGN_SPLIT_PROVENANCE_NAME = "v210_split_provenance.json"


def comparison_name(prompt_index: int) -> str:
    return f"{int(prompt_index):06d}-0.mp4"


def _evaluation_root_from_argv() -> Path:
    positions = [
        index for index, value in enumerate(sys.argv) if value == "--evaluation-root"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(sys.argv):
        raise SystemExit("--evaluation-root is required exactly once")
    index = positions[0]
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return Path(value).expanduser().resolve()


def evaluation_source_provenance(
    evaluation_root: Path,
    expected_commit: str,
    expected_runtime_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = evaluation_root.expanduser().resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else ""
    if commit != expected_commit:
        raise ValueError(
            "v210 evaluation checkout commit drift: "
            f"expected={expected_commit} actual={commit or 'unavailable'}"
        )
    runtime_hashes = {}
    for name in EVALUATION_RUNTIME_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing v210 evaluation runtime file: {path}")
        runtime_hashes[name] = base.sha256(path)
    if (
        expected_runtime_sha256 is not None
        and runtime_hashes != expected_runtime_sha256
    ):
        raise ValueError("v210 evaluation runtime file hash drift")
    return {
        "evaluation_root": str(root),
        "evaluation_commit": commit,
        "runtime_file_sha256": runtime_hashes,
    }


def validate_vbench_root(manifest: dict[str, Any], vbench_root: Path) -> Path:
    actual = vbench_root.expanduser().resolve()
    recorded = Path(str(manifest.get("vbench_root", ""))).expanduser().resolve()
    if recorded != actual:
        raise ValueError(
            f"v210 VBench root drift: expected={recorded} actual={actual}"
        )
    return actual


def validate_source_media(
    manifest: dict[str, Any], *, only_method: str | None = None
) -> dict[str, Any]:
    method_rows = manifest.get("methods") or ()
    methods = tuple(str(row.get("key", "")) for row in method_rows)
    prompt_items = manifest.get("prompt_items") or ()
    prompt_count = int(manifest.get("prompt_count", -1))
    expected_pairs = {
        (method, prompt_index)
        for method in methods
        for prompt_index in range(prompt_count)
    }
    jobs: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("jobs") or ():
        key = (str(row.get("method", "")), int(row.get("prompt_index", -1)))
        if key in jobs:
            raise ValueError(f"duplicate v210 source media job: {key}")
        jobs[key] = row
    if set(jobs) != expected_pairs or len(prompt_items) != prompt_count:
        raise ValueError("v210 source media grid is incomplete or mixed")

    if only_method is not None and only_method not in methods:
        raise ValueError(f"unknown v210 source media method: {only_method}")
    audit_rows = []
    per_method: dict[str, dict[str, Any]] = {}
    for method_row in method_rows:
        method = str(method_row["key"])
        if only_method is not None and method != only_method:
            continue
        video_dir = Path(str(method_row["video_dir"]))
        method_audit_rows = []
        for prompt_index in range(prompt_count):
            job = jobs[(method, prompt_index)]
            published = video_dir / comparison_name(prompt_index)
            recorded = Path(str(job.get("media", "")))
            expected_sha256 = str(job.get("media_sha256", ""))
            prompt = prompt_items[prompt_index]
            if (
                job.get("source_index") != prompt.get("source_index")
                or not published.is_file()
                or not recorded.is_file()
            ):
                raise ValueError(
                    f"v210 source media provenance drift: {method}:{prompt_index}"
                )
            try:
                same_file = published.samefile(recorded)
            except OSError:
                same_file = False
            actual_sha256 = base.sha256(published)
            if (
                not same_file
                or len(expected_sha256) != 64
                or actual_sha256 != expected_sha256
            ):
                raise ValueError(
                    f"v210 source media provenance drift: {method}:{prompt_index}"
                )
            row = {
                "method": method,
                "prompt_index": prompt_index,
                "source_index": job["source_index"],
                "media_sha256": actual_sha256,
            }
            audit_rows.append(row)
            method_audit_rows.append(row)
        method_encoded = json.dumps(
            method_audit_rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        per_method[method] = {
            "video_count": len(method_audit_rows),
            "mapping_sha256": hashlib.sha256(method_encoded).hexdigest(),
        }
    encoded = json.dumps(
        audit_rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "version": 1,
        "video_count": len(audit_rows),
        "mapping_sha256": hashlib.sha256(encoded).hexdigest(),
        "methods": per_method,
    }


def _write_frozen_fingerprint(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    if not path.is_file() or path.read_bytes() != encoded:
        raise ValueError("v210 VBench checkout fingerprint drift")


def bind_vbench_checkout_fingerprint(
    manifest: dict[str, Any],
    vbench_root: Path,
    parts_root: Path,
    *,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    actual = vbench_checkout_fingerprint(vbench_root)
    expected = manifest.get("vbench_checkout_fingerprint")
    if expected is not None:
        if not isinstance(expected, dict) or expected != actual:
            raise ValueError("v210 VBench checkout fingerprint drift")
        source = {
            "kind": "comparison_manifest",
            "path": "comparison_manifest.json:vbench_checkout_fingerprint",
        }
    else:
        lock_path = parts_root / FINGERPRINT_NAME
        lock = {
            "version": 1,
            "comparison_manifest_sha256": (
                base.sha256(manifest_path) if manifest_path is not None else None
            ),
            "vbench_checkout_fingerprint": actual,
        }
        _write_frozen_fingerprint(lock_path, lock)
        source = {"kind": "runtime_lock", "path": str(lock_path.resolve())}
    return actual, source


def runtime_contract(args: Any) -> dict[str, Any]:
    context = _BASE_RUNTIME_CONTRACT(args)
    evaluation_root = getattr(args, "evaluation_root", None) or EVALUATION_ROOT
    if evaluation_root is None:
        raise ValueError("v210 --evaluation-root was not configured")
    evaluation_source = evaluation_source_provenance(
        Path(evaluation_root),
        str(context["manifest"].get("evaluation_commit", "")),
        context["manifest"].get("evaluation_runtime_sha256"),
    )
    validate_vbench_root(context["manifest"], args.vbench_root)
    source_media_audit = validate_source_media(context["manifest"])
    fingerprint, source = bind_vbench_checkout_fingerprint(
        context["manifest"],
        args.vbench_root,
        args.parts_root,
        manifest_path=getattr(args, "manifest", None),
    )
    if fingerprint["head"] != context["vbench_commit"]:
        raise ValueError("v210 VBench HEAD disagrees with the base runtime contract")
    split_provenance = validate_split_provenance(
        context["manifest"], args.manifest
    )
    context["vbench_checkout_fingerprint"] = fingerprint
    context["vbench_checkout_fingerprint_source"] = source
    context["vbench_root"] = str(args.vbench_root.resolve())
    context["evaluation_source"] = evaluation_source
    context["source_media_audit"] = source_media_audit
    context["split_provenance"] = split_provenance
    context["comparison_manifest_path"] = str(args.manifest.resolve())
    return context


def job_contract(context: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    current = vbench_checkout_fingerprint(Path(context["vbench_root"]))
    if current != context["vbench_checkout_fingerprint"]:
        raise ValueError("v210 VBench checkout fingerprint drift")
    evaluation_source = evaluation_source_provenance(
        Path(context["evaluation_source"]["evaluation_root"]),
        context["evaluation_source"]["evaluation_commit"],
        context["evaluation_source"]["runtime_file_sha256"],
    )
    if evaluation_source != context["evaluation_source"]:
        raise ValueError("v210 evaluation runtime file drift")
    method = str(kwargs.get("method", ""))
    current_media = validate_source_media(
        context["manifest"], only_method=method
    )
    expected_media = context["source_media_audit"]["methods"].get(method)
    if current_media["methods"].get(method) != expected_media:
        raise ValueError(f"v210 source media changed during evaluation: {method}")
    manifest_path = Path(context["comparison_manifest_path"])
    current_campaign = validate_campaign_split_provenance(
        context["manifest"], manifest_path, context["split_provenance"]
    )
    current_split = validate_split_provenance(
        context["manifest"],
        manifest_path,
        only_method=method,
        require_campaign=False,
    )
    expected_split = context["split_provenance"]["methods"].get(method)
    if current_split["methods"].get(method) != expected_split:
        raise ValueError(f"v210 split content changed during evaluation: {method}")
    contract = _BASE_JOB_CONTRACT(context, **kwargs)
    contract["vbench_checkout_fingerprint"] = current
    contract["vbench_checkout_fingerprint_source"] = context[
        "vbench_checkout_fingerprint_source"
    ]
    contract["evaluation_source"] = evaluation_source
    contract["source_media_audit"] = context["source_media_audit"]
    contract["split_provenance"] = {
        "aggregate_digest": current_campaign["aggregate_digest"],
        "campaign_provenance": current_campaign["campaign_provenance"],
        "campaign_provenance_sha256": current_campaign[
            "campaign_provenance_sha256"
        ],
        "method": current_split["methods"][method],
    }
    return contract


def completion_report(
    args: Any,
    context: dict[str, Any],
    jobs: list[tuple[str, str]],
) -> dict[str, Any]:
    report = _BASE_COMPLETION_REPORT(args, context, jobs)
    report["evaluation_source"] = context["evaluation_source"]
    report["source_media_audit"] = context["source_media_audit"]
    report["vbench_checkout_fingerprint"] = context[
        "vbench_checkout_fingerprint"
    ]
    report["split_provenance"] = context["split_provenance"]
    return report


def collect(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    report = _BASE_COLLECT(args, context)
    provenance = {
        "evaluation_source": context["evaluation_source"],
        "source_media_audit": context["source_media_audit"],
        "vbench_checkout_fingerprint": context[
            "vbench_checkout_fingerprint"
        ],
        "split_provenance": context["split_provenance"],
    }
    report["runtime_provenance"] = provenance
    summary_path = args.summary_root / f"{args.summary_stem}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_provenance"] = provenance
    base.write_json_atomically(summary_path, summary, sort_keys=False)
    base.write_json_atomically(
        args.analysis_root / f"{args.analysis_stem}.json",
        report,
        sort_keys=True,
    )
    return report


def analyze(payload: dict) -> dict:
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if tuple(rows) != METHODS or dimensions != DIMENSIONS or payload.get("missing"):
        raise ValueError("v210 VBench summary violates the frozen grid")
    for method in METHODS:
        if set(rows[method]) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete v210 VBench dimensions")
        if abs(
            float(rows[method]["overall_consistency"])
            - float(rows[method]["temporal_style"])
        ) > 1e-12:
            raise ValueError(f"{method}: duplicate custom-prompt ViCLIP drift")
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "methods": list(METHODS),
        "dimensions": list(DIMENSIONS),
        "raw_core9": {method: dict(rows[method]) for method in METHODS},
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "exclusive_scores": {
            method: exclusive_scores(rows[method]) for method in METHODS
        },
        "official_quality_score": {
            method: official_quality_score(rows[method]) for method in METHODS
        },
        "quality_without_dynamic_degree": {
            method: quality_score_with_fixed_dynamic(
                rows[method], dynamic_value=1.0
            )
            for method in METHODS
        },
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "metric_validity": {
            "primary_quality_metric": "quality_without_dynamic_degree",
            "dynamic_degree_used_for_selection": False,
        },
        "metric_promotion_gate": False,
        "claim_boundary": (
            "The screen8 aggregate metrics are descriptive development results. "
            "Selection requires paired full/late-half comparisons against both "
            "SF21 controls and temporal safety; Dynamic Degree is excluded."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v210 LPHC Screen8 VBench-Long Core-9",
        "",
        (
            "| Method | Quality | Quality w/o Dynamic | Identity/background | "
            "Temporal | Semantic | Visual | Dynamic |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        row = report["exclusive_scores"][method]
        lines.append(
            f"| {method} | {report['official_quality_score'][method]:.4f} | "
            f"{report['quality_without_dynamic_degree'][method]:.4f} | "
            f"{row['identity_background']:.5f} | "
            f"{row['temporal_mechanics']:.5f} | "
            f"{row['semantic_alignment']:.5f} | "
            f"{row['visual_quality']:.5f} | "
            f"{row['dynamic_degree']:.5f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def _path_option_from_argv(name: str) -> Path:
    try:
        index = sys.argv.index(name)
        return Path(sys.argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit(f"{name} is required") from error


def comparison_root_from_argv() -> Path:
    return _path_option_from_argv("--comparison-root")


def source_preflight(
    manifest: dict[str, Any], evaluation_root: Path, vbench_root: Path
) -> dict[str, Any]:
    actual_vbench_root = validate_vbench_root(manifest, vbench_root)
    evaluation_source = evaluation_source_provenance(
        evaluation_root,
        str(manifest.get("evaluation_commit", "")),
        manifest.get("evaluation_runtime_sha256"),
    )
    source_media_audit = validate_source_media(manifest)
    fingerprint = vbench_checkout_fingerprint(actual_vbench_root)
    if fingerprint != manifest.get("vbench_checkout_fingerprint"):
        raise ValueError("v210 VBench checkout fingerprint drift")
    return {
        "version": 1,
        "evaluation_source": evaluation_source,
        "source_media_audit": source_media_audit,
        "vbench_root": str(actual_vbench_root),
        "vbench_checkout_fingerprint": fingerprint,
    }


def _canonical_digest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def split_provenance_path(video_dir: Path) -> Path:
    return video_dir / SPLIT_PROVENANCE_NAME


def campaign_split_provenance_path(manifest_path: Path) -> Path:
    return manifest_path.parent / CAMPAIGN_SPLIT_PROVENANCE_NAME


def build_split_provenance(
    manifest: dict[str, Any],
    manifest_path: Path,
    method: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    method_row = next(
        row for row in manifest["methods"] if str(row["key"]) == method
    )
    video_dir = Path(str(method_row["video_dir"])).resolve()
    prompt_count = int(manifest["prompt_count"])
    clips_per_video = int(manifest["num_output_frames"]) // 8
    source_rows = []
    clip_rows = []
    for prompt_index in range(prompt_count):
        source = video_dir / comparison_name(prompt_index)
        source_rows.append(
            {
                "name": source.name,
                "sha256": base.sha256(source),
                "bytes": source.stat().st_size,
            }
        )
        stem = source.stem
        folder = video_dir / "split_clip" / stem
        expected = [
            folder / f"{stem}_{index:03d}.mp4"
            for index in range(clips_per_video)
        ]
        if any(not path.is_file() or path.stat().st_size <= 0 for path in expected):
            raise ValueError(f"{method}: incomplete v210 split clips")
        if {path.name for path in folder.iterdir()} != {path.name for path in expected}:
            raise ValueError(f"{method}: mixed v210 split clips")
        clip_rows.extend(
            {
                "name": str(path.relative_to(video_dir)),
                "sha256": base.sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in expected
        )
    if len(source_rows) != 8 or len(clip_rows) != 120:
        raise ValueError(f"{method}: invalid v210 split provenance cardinality")
    return {
        "version": 1,
        "comparison_manifest_sha256": base.sha256(manifest_path),
        "method": method,
        "video_dir": str(video_dir),
        "vbench_root": str(Path(str(manifest["vbench_root"])).resolve()),
        "vbench_checkout_fingerprint": fingerprint,
        "source_videos": source_rows,
        "clips": clip_rows,
        "source_digest": _canonical_digest(source_rows),
        "clip_digest": _canonical_digest(clip_rows),
    }


def _write_frozen_split_provenance(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"frozen v210 split provenance differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError(f"frozen v210 split provenance race: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def validate_campaign_split_provenance(
    manifest: dict[str, Any],
    manifest_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    campaign_path = campaign_split_provenance_path(manifest_path).resolve()
    try:
        encoded = campaign_path.read_bytes()
        campaign = json.loads(encoded)
    except FileNotFoundError as error:
        raise ValueError("missing v210 campaign split provenance") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid v210 campaign split provenance") from error

    method_rows = manifest.get("methods") or ()
    methods = [str(row["key"]) for row in method_rows]
    expected_methods = expected.get("methods") or {}
    if len(methods) != 8 or list(expected_methods) != methods:
        raise ValueError("v210 campaign split provenance requires exactly 8 methods")
    aggregate_rows = [
        {
            "method": method,
            "provenance_sha256": expected_methods[method]["provenance_sha256"],
            "source_digest": expected_methods[method]["source_digest"],
            "clip_digest": expected_methods[method]["clip_digest"],
        }
        for method in methods
    ]
    aggregate_digest = _canonical_digest(aggregate_rows)
    expected_campaign = {
        "version": 1,
        "comparison_manifest_sha256": base.sha256(manifest_path),
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
        "aggregate_digest": aggregate_digest,
        "methods": aggregate_rows,
    }
    campaign_sha256 = hashlib.sha256(encoded).hexdigest()
    if (
        campaign != expected_campaign
        or expected.get("aggregate_digest") != aggregate_digest
        or (
            "campaign_provenance" in expected
            and expected["campaign_provenance"] != str(campaign_path)
        )
        or (
            "campaign_provenance_sha256" in expected
            and expected["campaign_provenance_sha256"] != campaign_sha256
        )
    ):
        raise ValueError("v210 campaign split provenance drift")
    return {
        "campaign_provenance": str(campaign_path),
        "campaign_provenance_sha256": campaign_sha256,
        "aggregate_digest": aggregate_digest,
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
    }


def validate_split_provenance(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    only_method: str | None = None,
    require_campaign: bool = True,
) -> dict[str, Any]:
    method_reports = {}
    aggregate_rows = []
    for method_row in manifest.get("methods") or ():
        method = str(method_row["key"])
        if only_method is not None and method != only_method:
            continue
        video_dir = Path(str(method_row["video_dir"])).resolve()
        path = split_provenance_path(video_dir)
        if not path.is_file():
            raise ValueError(f"missing v210 split provenance: {method}")
        try:
            frozen = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid v210 split provenance: {method}") from error
        current = build_split_provenance(
            manifest,
            manifest_path,
            method,
            manifest["vbench_checkout_fingerprint"],
        )
        if frozen != current:
            raise ValueError(f"v210 split content drift: {method}")
        provenance_sha = base.sha256(path)
        method_reports[method] = {
            "provenance": str(path),
            "provenance_sha256": provenance_sha,
            "source_digest": current["source_digest"],
            "clip_digest": current["clip_digest"],
        }
        aggregate_rows.append(
            {
                "method": method,
                "provenance_sha256": provenance_sha,
                "source_digest": current["source_digest"],
                "clip_digest": current["clip_digest"],
            }
        )
    expected_methods = 1 if only_method is not None else 8
    if len(aggregate_rows) != expected_methods:
        raise ValueError(
            f"v210 split provenance requires exactly {expected_methods} methods"
        )
    report = {
        "version": 1,
        "method_count": expected_methods,
        "source_count": expected_methods * 8,
        "clip_count": expected_methods * 120,
        "aggregate_digest": _canonical_digest(aggregate_rows),
        "methods": method_reports,
    }
    if only_method is None and require_campaign:
        report.update(
            validate_campaign_split_provenance(manifest, manifest_path, report)
        )
    return report


def _int_option_from_argv(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    try:
        return int(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit(f"invalid {name}") from error


def finalize_campaign_split_provenance(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any] | None:
    aggregate_rows = []
    for method_row in manifest["methods"]:
        method = str(method_row["key"])
        path = split_provenance_path(Path(str(method_row["video_dir"])).resolve())
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = build_split_provenance(
            manifest,
            manifest_path,
            method,
            manifest["vbench_checkout_fingerprint"],
        )
        if payload != current:
            raise ValueError(f"v210 split content drift before finalization: {method}")
        aggregate_rows.append(
            {
                "method": method,
                "provenance_sha256": base.sha256(path),
                "source_digest": payload["source_digest"],
                "clip_digest": payload["clip_digest"],
            }
        )
    campaign = {
        "version": 1,
        "comparison_manifest_sha256": base.sha256(manifest_path),
        "method_count": 8,
        "source_count": 64,
        "clip_count": 960,
        "aggregate_digest": _canonical_digest(aggregate_rows),
        "methods": aggregate_rows,
    }
    _write_frozen_split_provenance(
        campaign_split_provenance_path(manifest_path), campaign
    )
    return campaign


def guarded_split(
    manifest: dict[str, Any],
    manifest_path: Path,
    evaluation_root: Path,
    vbench_root: Path,
) -> dict[str, Any]:
    before = source_preflight(manifest, evaluation_root, vbench_root)
    node_rank = _int_option_from_argv("--node-rank", 0)
    num_nodes = _int_option_from_argv("--num-nodes", 1)
    if num_nodes <= 0 or not 0 <= node_rank < num_nodes:
        raise ValueError("require 0 <= node-rank < num-nodes")
    original_argv = list(sys.argv)
    try:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        import prepare_v174_vbench_splits as splitter

        splitter.main()
    finally:
        sys.argv = original_argv
    after = source_preflight(manifest, evaluation_root, vbench_root)
    if after != before:
        for method_row in manifest["methods"]:
            split_provenance_path(
                Path(str(method_row["video_dir"])).resolve()
            ).unlink(missing_ok=True)
        raise ValueError("v210 split inputs changed during splitting")
    methods = [str(row["key"]) for row in manifest["methods"]]
    selected = methods[node_rank::num_nodes]
    for method in selected:
        payload = build_split_provenance(
            manifest,
            manifest_path,
            method,
            after["vbench_checkout_fingerprint"],
        )
        _write_frozen_split_provenance(
            split_provenance_path(Path(payload["video_dir"])), payload
        )
    campaign = finalize_campaign_split_provenance(manifest, manifest_path)
    return {
        "version": 1,
        "node_rank": node_rank,
        "num_nodes": num_nodes,
        "methods": selected,
        "source_preflight": after,
        "campaign_finalized": campaign is not None,
    }


def _job_lease_path(parts_root: Path, method: str, dimension: str) -> Path:
    identity = f"{method}\0{dimension}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return parts_root / ".v210_job_leases" / f"{digest}.lock"


def run_job(
    args: Any,
    context: dict[str, Any],
    *,
    method: str,
    dimension: str,
    gpu: str,
) -> dict[str, Any]:
    lock_path = _job_lease_path(args.parts_root, method, dimension)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            return {
                "method": method,
                "dimension": dimension,
                "gpu": gpu,
                "status": "locked",
                "error": (
                    f"v210 VBench job lease locked: {method}:{dimension}; {owner}"
                ),
            }
        owner = {
            "version": 1,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_manifest_sha256": context["manifest_sha256"],
            "method": method,
            "dimension": dimension,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            return _BASE_RUN_JOB(
                args,
                context,
                method=method,
                dimension=dimension,
                gpu=gpu,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def configure() -> dict:
    global METHODS, PROMPT_COUNT, EXPERIMENT, EVALUATION_ROOT
    EVALUATION_ROOT = _evaluation_root_from_argv()
    manifest_path = comparison_root_from_argv() / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing comparison manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = str(manifest.get("experiment", ""))
    methods = tuple(
        str(row.get("key", "")) for row in manifest.get("methods") or ()
    )
    prompt_count = int(manifest.get("prompt_count", -1))
    prompt_items = manifest.get("prompt_items") or ()
    evaluation_commit = str(manifest.get("evaluation_commit", ""))
    evaluation_hashes = manifest.get("evaluation_runtime_sha256")
    if (
        experiment != EXPECTED_EXPERIMENT
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
        or not isinstance(evaluation_hashes, dict)
        or set(evaluation_hashes) != set(EVALUATION_RUNTIME_FILES)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in evaluation_hashes.values()
        )
        or len(methods) != EXPECTED_METHOD_COUNT
        or any(not method for method in methods)
        or len(set(methods)) != len(methods)
        or prompt_count != EXPECTED_PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or len(prompt_items) != prompt_count
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(prompt_count))
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
    ):
        raise ValueError("invalid v210 VBench comparison contract")
    METHODS = methods
    PROMPT_COUNT = prompt_count
    EXPERIMENT = experiment
    base.RUN_LABEL = "v210_lphc_screen8"
    base.SUMMARY_EXPERIMENT = experiment
    base.ANALYSIS_STEM = "v210_vbench_analysis"
    base.SUMMARY_TITLE = "v210 LPHC Screen8 VBench-Long Core-9"
    base.COMPARISON_EXPERIMENT = experiment
    base.METHODS = methods
    base.PROMPT_COUNT = prompt_count
    base.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
    base.CLIPS_PER_VIDEO = CLIPS_PER_VIDEO
    base.DIMENSIONS = DIMENSIONS
    base.comparison_name = comparison_name
    base.analyze = analyze
    base.render_markdown = render
    base.runtime_contract = runtime_contract
    base.job_contract = job_contract
    base.run_job = run_job
    base.completion_report = completion_report
    base.collect = collect
    if sys.argv[1] not in {"source-preflight", "split"} and "--summary-stem" not in sys.argv:
        sys.argv.extend(["--summary-stem", "vbench_core9_summary"])
    return manifest


def main() -> None:
    manifest = configure()
    if len(sys.argv) > 1 and sys.argv[1] in {"source-preflight", "split"}:
        if EVALUATION_ROOT is None:
            raise RuntimeError("v210 evaluation root was not configured")
        manifest_path = comparison_root_from_argv() / "comparison_manifest.json"
        if sys.argv[1] == "source-preflight":
            report = source_preflight(
                manifest,
                EVALUATION_ROOT,
                _path_option_from_argv("--vbench-root"),
            )
        else:
            report = guarded_split(
                manifest,
                manifest_path,
                EVALUATION_ROOT,
                _path_option_from_argv("--vbench-root"),
            )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    base.main()


if __name__ == "__main__":
    main()
