#!/usr/bin/env python3
"""Audit v207 media, runtime isolation, phase routes, and FFE budgets."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v207_context_budget_phase_screen import (
    CALL_COUNT,
    HEADS,
    LAYERS,
    METHOD_ORDER,
    PROMPT_COUNT,
    sha256,
    verify,
)


FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "scheduled cache read budget drift",
    "Recent schedule leaked middle memory",
    "Coverage schedule exceeded",
    "CacheCompatDenoiseTraceWarning",
    "active policy",
)
SOURCE_KIND = {
    "retrieval": "semantic_retrieval",
    "landmark": "semantic_landmark",
}
RUNTIME_LINE = re.compile(
    r"\[V20(?:7|8)Runtime\] method=(\S+) elapsed_seconds=(\d+) videos=(\d+) "
    r"gpu=(\S+) rank=(\d+) stride=(\d+)"
)


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return sha256(path)


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v207 published video: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def audit_sf_logs(run_root: Path) -> dict:
    paths = sorted((run_root / "logs" / "sf_native").glob("*.log"))
    errors = [] if paths else ["no SF logs"]
    rows = []
    forbidden = (
        "[CacheCompatDenoiseSchedule]",
        "[HistoryPolarityPolicy]",
        "[SemanticRetrievalArchive]",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [token for token in FAILURE_PATTERNS if token in text]
        leaked = [token for token in forbidden if token in text]
        runtime = RUNTIME_LINE.findall(text)
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        if leaked:
            errors.append(f"{path.name}: cache runtime leaked={leaked}")
        if len(runtime) != 1 or runtime[0][0] != "sf_native":
            errors.append(f"{path.name}: invalid runtime record={runtime}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "failure_patterns": failures,
                "forbidden_cache_markers": leaked,
                "runtime_record": runtime[0] if len(runtime) == 1 else None,
            }
        )
    traces = sorted((run_root / "traces" / "sf_native").glob("*.jsonl"))
    if traces:
        errors.append("SF native unexpectedly emitted schedule traces")
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_cache_logs(run_root: Path, method: str, row: dict) -> dict:
    paths = sorted((run_root / "logs" / method).glob("*.log"))
    errors = [] if paths else ["no cache logs"]
    records = []
    operator = str(row["operator"])
    schedule = str(row["schedule"])
    budget = int(row["read_frame_equivalents"])
    recent = int(row["recent_route_frames"])
    coverage_recent = int(row["coverage_recent_frames"])
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [token for token in FAILURE_PATTERNS if token in text]
        runtime = RUNTIME_LINE.findall(text)
        required = {
            "local21": "[ModelAttentionContract] local_attn_size=21" in text,
            "history": (
                f"support={operator} suppress={operator}" in text
                and "counts=10:360,11:0" in text
                and "exclusive_owner=true" in text
            ),
            "schedule": f"schedule={schedule}" in text,
            "operator": f"coverage_operator={operator}" in text,
            "clean_recent": "clean_readout=recent" in text,
            "recent_composition": f"recent=sink1+recent{recent}" in text,
            "coverage_composition": (
                f"coverage=sink1+middle4+recent{coverage_recent}" in text
            ),
            "budget": f"read_budget={budget}FFE" in text,
            "retrieval_archive": (
                operator != "retrieval"
                or "[SemanticRetrievalArchive]" in text
                and "archive_capacity=12" in text
            ),
        }
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        if not all(required.values()):
            errors.append(f"{path.name}: runtime contract={required}")
        if len(runtime) != 1 or runtime[0][0] != method:
            errors.append(f"{path.name}: invalid runtime record={runtime}")
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "required_markers": required,
                "failure_patterns": failures,
                "runtime_record": runtime[0] if len(runtime) == 1 else None,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": records}


def audit_traces(run_root: Path, method: str, row: dict, *, scope: str) -> dict:
    paths = sorted((run_root / "traces" / method).glob("*.schedule.jsonl"))
    errors = [] if paths else ["no schedule traces"]
    expected_calls = set(int(value) for value in row["coverage_noisy_calls"])
    schedule = str(row["schedule"])
    operator = str(row["operator"])
    budget = int(row["read_frame_equivalents"])
    recent_limit = int(row["recent_route_frames"])
    coverage_recent_limit = int(row["coverage_recent_frames"])
    expected_source = SOURCE_KIND[operator]
    calls: dict[int, set[str]] = {index: set() for index in range(CALL_COUNT)}
    clean_policies: set[str] = set()
    layers: set[int] = set()
    schedule_records = 0
    readout_records = 0
    coverage_readouts = 0
    middle_frame_equivalents = 0
    max_total = 0
    observed_sources: set[str] = set()
    for path in paths:
        for item in iter_jsonl(path):
            if item.get("schedule") != schedule:
                errors.append(f"{path.name}: schedule drift")
                continue
            if item.get("coverage_operator") != operator:
                errors.append(f"{path.name}: operator drift")
            layer = int(item.get("layer", -1))
            if not 0 <= layer < LAYERS:
                errors.append(f"{path.name}: invalid layer {layer}")
                continue
            layers.add(layer)
            if int(item.get("read_budget_frame_equivalents", -1)) != budget:
                errors.append(f"{path.name}: declared read budget drift")
            event = item.get("event")
            policy = str(item.get("effective_policy", ""))
            mode = str(item.get("update_mode", ""))
            if event == "schedule":
                schedule_records += 1
                if int(item.get("call_count", -1)) != CALL_COUNT:
                    errors.append(f"{path.name}: call-count drift")
                if mode == "clean":
                    clean_policies.add(policy)
                    if policy != "recent" or item.get("clean_policy_is_recent") is not True:
                        errors.append(f"{path.name}: clean pass did not use Recent")
                elif mode == "noisy":
                    call = int(item.get("call_index", -1))
                    if call not in calls:
                        errors.append(f"{path.name}: invalid noisy call {call}")
                    else:
                        calls[call].add(policy)
                        expected = "coverage" if call in expected_calls else "recent"
                        if policy != expected:
                            errors.append(
                                f"{path.name}: call {call} used {policy}, expected {expected}"
                            )
                else:
                    errors.append(f"{path.name}: invalid update mode {mode!r}")
                continue
            if event != "readout":
                errors.append(f"{path.name}: unknown event {event!r}")
                continue
            readout_records += 1
            if item.get("budget_pass") is not True:
                errors.append(f"{path.name}: budget_pass=false")
            observed_total = int(item.get("max_total_frame_equivalents", -1))
            max_total = max(max_total, observed_total)
            if observed_total > budget:
                errors.append(f"{path.name}: readout exceeded {budget} FFE")
            for head in item.get("selected_heads") or ():
                head_policy = str(head.get("effective_policy", ""))
                counts = head.get("counts") or {}
                static = int(counts.get("static", -1))
                dynamic = int(counts.get("dynamic", -1))
                anchor = int(counts.get("anchor", -1))
                total = int(head.get("total_frame_equivalents", -1))
                if static > 1 or total > budget:
                    errors.append(f"{path.name}: per-head budget drift")
                if head_policy == "recent":
                    if anchor != 0 or dynamic > recent_limit:
                        errors.append(f"{path.name}: Recent route leaked middle")
                elif head_policy == "coverage":
                    coverage_readouts += 1
                    if anchor > 4 or dynamic > coverage_recent_limit:
                        errors.append(f"{path.name}: Coverage route budget drift")
                    middle_frame_equivalents += max(0, anchor)
                    for segment in head.get("segments") or ():
                        if segment.get("kind", "").startswith("anchor:"):
                            source = str(segment.get("source_kind", ""))
                            observed_sources.add(source)
                            if source != expected_source:
                                errors.append(
                                    f"{path.name}: anchor source={source}, expected={expected_source}"
                                )
                else:
                    errors.append(f"{path.name}: invalid head policy {head_policy!r}")
    if not schedule_records or not readout_records:
        errors.append("missing schedule or readout trace records")
    if layers != set(range(LAYERS)):
        errors.append(f"trace layers differ: {sorted(layers)}")
    for call, policies in calls.items():
        expected = "coverage" if call in expected_calls else "recent"
        if policies != {expected}:
            errors.append(f"call {call} policies={sorted(policies)}, expected={expected}")
    if clean_policies != {"recent"}:
        errors.append(f"clean policies differ: {sorted(clean_policies)}")
    if expected_calls and not coverage_readouts:
        errors.append("scheduled Coverage produced no traced readout")
    if scope != "smoke" and expected_calls and middle_frame_equivalents <= 0:
        errors.append("Coverage never exposed a populated middle bank")
    if middle_frame_equivalents > 0 and observed_sources != {expected_source}:
        errors.append(f"middle sources differ: {sorted(observed_sources)}")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "files": [str(path.resolve()) for path in paths],
        "schedule_records": schedule_records,
        "readout_records": readout_records,
        "coverage_readouts": coverage_readouts,
        "middle_frame_equivalents": middle_frame_equivalents,
        "observed_sources": sorted(observed_sources),
        "traced_layers": sorted(layers),
        "noisy_policies": {str(key): sorted(value) for key, value in calls.items()},
        "clean_policies": sorted(clean_policies),
        "max_total_frame_equivalents": max_total,
        "declared_budget_frame_equivalents": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "screen32"), required=True)
    parser.add_argument("--smoke-prompt-index", type=int, default=3)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    manifest = verify(args.input_manifest)
    if args.scope == "smoke":
        start, end = args.smoke_prompt_index, args.smoke_prompt_index + 1
    else:
        start, end = 0, PROMPT_COUNT
    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    links = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in METHOD_ORDER:
        config = manifest["methods"][method]
        media = audit_interval(
            args.run_root / "raw" / method,
            start_idx=start,
            end_idx=end,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        if config["runtime"] == "sf_native":
            logs = audit_sf_logs(args.run_root)
            traces = {"ok": True, "not_applicable": True}
        else:
            logs = audit_cache_logs(args.run_root, method, config)
            traces = audit_traces(args.run_root, method, config, scope=args.scope)
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        if method_ok:
            published = args.run_root / "published" / method
            for video in media["videos"]:
                source = args.run_root / "raw" / method / str(video["file"])
                mode = link_or_validate(
                    source, published / f"{int(video['prompt_idx']):06d}.mp4"
                )
                links[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "operator": config["operator"],
                "schedule": config["schedule"],
                "read_frame_equivalents": config["read_frame_equivalents"],
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )
    contract = {
        "version": 1,
        "experiment": f"{manifest['experiment']}_generation",
        "scope": args.scope,
        "development_only": True,
        "prompt_count": end - start,
        "prompt_indices": list(range(start, end)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "prompt_items": manifest["prompt_items"][start:end],
        "num_output_frames": manifest["num_output_frames"],
        "decoded_video_contract": manifest["decoded_video_contract"],
        "seed": manifest["seed"],
        "methods": list(METHOD_ORDER),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    summary = {
        "version": 1,
        "ok": all_ok,
        "experiment": contract["experiment"],
        "scope": args.scope,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": links,
    }
    write_json(args.run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v207 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v207-audit] PASS "
        f"scope={args.scope} methods={len(METHOD_ORDER)} "
        f"videos={len(METHOD_ORDER) * (end-start)} links={links}"
    )


if __name__ == "__main__":
    main()
