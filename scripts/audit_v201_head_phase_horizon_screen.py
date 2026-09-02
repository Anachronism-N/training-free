#!/usr/bin/env python3
"""Audit v201 videos, AR-horizon routes, and equal-read cache budgets."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from audit_indexed_videos import audit_interval
from prepare_v201_head_phase_horizon_screen import (
    CALLS,
    HEADS,
    LAYERS,
    NUM_OUTPUT_FRAMES,
    PROMPT_COUNT,
    SEED,
    sha256,
    validate_horizon_map,
    verify,
)

FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "scheduled cache read budget drift",
    "Recent schedule leaked middle memory",
    "Coverage schedule exceeded its 4+4 budget",
    "CacheCompatDenoiseTraceWarning",
    "horizon map width mismatch",
    "head-phase-horizon routing requires",
    "active policy",
)
SOURCE_KIND = {
    "landmark": "semantic_landmark",
    "retrieval": "semantic_retrieval",
}


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
            raise RuntimeError(f"refusing mixed v201 published video: {target}")
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


def expected_policy(selected: list[int]) -> str:
    return (
        "recent" if not selected else "coverage" if len(selected) == HEADS else "mixed"
    )


def horizon_position(current_frame: int, reference_frames: list[int]) -> int:
    frame = int(current_frame)
    return min(
        range(len(reference_frames)),
        key=lambda index: (abs(int(reference_frames[index]) - frame), index),
    )


def audit_logs(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "logs" / method).glob("*.log"))
    if not paths:
        return {"ok": False, "errors": ["no logs"], "logs": []}
    errors = []
    rows = []
    operator = str(method_row["operator"])
    map_id = str(method_row["routing_map_id"])
    history_token = f"support={operator} suppress={operator}"
    expected_positions = len(method_row["current_frames"])
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
        required = {
            "history": history_token in text
            and "counts=10:360,11:0" in text
            and "exclusive_owner=true" in text,
            "schedule": "schedule=head_phase_horizon" in text,
            "operator": f"coverage_operator={operator}" in text,
            "map_id": f"phase_map_id={map_id}" in text,
            "horizon_positions": f"horizon_positions={expected_positions}" in text,
            "clean_recent": "clean_readout=recent" in text,
            "budget": "read_budget=9FFE" in text,
        }
        if not all(required.values()):
            errors.append(f"{path.name}: runtime contract={required}")
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "required_markers": required,
                "failure_patterns": failures,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_traces(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "traces" / method).glob("*.schedule.jsonl"))
    if not paths:
        return {"ok": False, "errors": ["no traces"], "files": []}
    map_path = Path(method_row["horizon_map"])
    if sha256(map_path) != method_row["horizon_map_sha256"]:
        raise ValueError(f"{method}: horizon-map hash drift")
    routing_map = json.loads(map_path.read_text(encoding="utf-8"))
    operator = str(method_row["operator"])
    validate_horizon_map(routing_map, operator=operator)
    masks = routing_map["coverage_masks"]
    frames = [int(value) for value in routing_map["current_frames"]]
    map_id = str(routing_map["map_id"])
    errors = []
    schedule_cells = Counter()
    clean_layers = set()
    observed_positions = set()
    observed_frames = set()
    readout_records = 0
    coverage_readout_heads = 0
    structured_middle_heads = 0
    max_budget = 0
    observed_sources = set()
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("schedule") != "head_phase_horizon":
                errors.append(f"{path.name}: schedule drift")
                continue
            if row.get("coverage_operator") != operator:
                errors.append(f"{path.name}: operator drift")
            event = row.get("event")
            layer = int(row.get("layer", -1))
            mode = str(row.get("update_mode", ""))
            if not 0 <= layer < LAYERS:
                errors.append(f"{path.name}: invalid layer {layer}")
                continue
            if event == "schedule":
                if int(row.get("call_count", -1)) != CALLS:
                    errors.append(f"{path.name}: call count drift")
                if mode == "clean":
                    clean_layers.add(layer)
                    if (
                        row.get("effective_policy") != "recent"
                        or int(row.get("coverage_heads", -1)) != 0
                        or row.get("coverage_head_indices") != []
                        or row.get("clean_policy_is_recent") is not True
                        or row.get("horizon_position_index") is not None
                        or row.get("horizon_reference_frame") is not None
                    ):
                        errors.append(f"{path.name}: clean route is not all-Recent")
                    continue
                if mode != "noisy":
                    errors.append(f"{path.name}: invalid update mode {mode!r}")
                    continue
                call = int(row.get("call_index", -1))
                current_frame = int(row.get("current_frame", -1))
                if not 0 <= call < CALLS or current_frame < 0:
                    errors.append(f"{path.name}: invalid noisy route location")
                    continue
                position = horizon_position(current_frame, frames)
                reference_frame = frames[position]
                selected = [
                    head
                    for head, value in enumerate(masks[position][call][layer])
                    if value
                ]
                observed = [
                    int(value) for value in row.get("coverage_head_indices") or ()
                ]
                if (
                    observed != selected
                    or int(row.get("coverage_heads", -1)) != len(selected)
                    or int(row.get("recent_heads", -1)) != HEADS - len(selected)
                    or row.get("effective_policy") != expected_policy(selected)
                    or row.get("phase_map_id") != map_id
                    or int(row.get("horizon_position_index", -1)) != position
                    or int(row.get("horizon_reference_frame", -1)) != reference_frame
                ):
                    errors.append(
                        f"{path.name}: horizon route mismatch frame={current_frame} "
                        f"call={call} layer={layer} observed={observed} "
                        f"expected={selected}"
                    )
                schedule_cells[(position, call, layer)] += 1
                observed_positions.add(position)
                observed_frames.add(current_frame)
            elif event == "readout":
                readout_records += 1
                observed_budget = int(row.get("max_total_frame_equivalents", -1))
                max_budget = max(max_budget, observed_budget)
                if row.get("budget_pass") is not True or observed_budget > 9:
                    errors.append(f"{path.name}: readout budget failed")
                call = row.get("call_index")
                current_frame = int(row.get("current_frame", -1))
                if mode != "noisy" or call is None or current_frame < 0:
                    continue
                call = int(call)
                if not 0 <= call < CALLS:
                    continue
                position = horizon_position(current_frame, frames)
                if (
                    int(row.get("horizon_position_index", -1)) != position
                    or int(row.get("horizon_reference_frame", -1)) != frames[position]
                ):
                    errors.append(f"{path.name}: readout horizon metadata drift")
                for head_row in row.get("selected_heads") or ():
                    head = int(head_row.get("head", -1))
                    if not 0 <= head < HEADS:
                        errors.append(f"{path.name}: invalid traced head")
                        continue
                    policy = (
                        "coverage" if masks[position][call][layer][head] else "recent"
                    )
                    if head_row.get("effective_policy") != policy:
                        errors.append(f"{path.name}: traced head policy mismatch")
                    counts = head_row.get("counts") or {}
                    anchor = int(counts.get("anchor", -1))
                    dynamic = int(counts.get("dynamic", -1))
                    total = int(head_row.get("total_frame_equivalents", -1))
                    if total > 9:
                        errors.append(f"{path.name}: traced head exceeded 9 FFE")
                    if policy == "recent" and (anchor != 0 or dynamic > 8):
                        errors.append(f"{path.name}: Recent head leaked middle")
                    if policy == "coverage":
                        coverage_readout_heads += 1
                        if anchor > 4 or dynamic > 4:
                            errors.append(f"{path.name}: Coverage head budget drift")
                        sources = {
                            str(segment.get("source_kind", ""))
                            for segment in head_row.get("segments") or ()
                        }
                        observed_sources.update(sources)
                        if SOURCE_KIND[operator] in sources:
                            structured_middle_heads += 1
            else:
                errors.append(f"{path.name}: unknown event {event!r}")
    expected_cells = {
        (position, call, layer)
        for position in range(len(frames))
        for call in range(CALLS)
        for layer in range(LAYERS)
    }
    if set(schedule_cells) != expected_cells:
        errors.append("noisy schedule did not cover all horizon/call/layer cells")
    if clean_layers != set(range(LAYERS)):
        errors.append("clean trace did not cover all layers")
    if readout_records == 0:
        errors.append("no readout trace records")
    expected_coverage = sum(
        value
        for position_rows in masks
        for call_rows in position_rows
        for layer_row in call_rows
        for value in layer_row
    )
    if expected_coverage > 0 and coverage_readout_heads == 0:
        errors.append("map selects Coverage but no Coverage head was traced")
    if expected_coverage > 0 and structured_middle_heads == 0:
        errors.append("map selects Coverage but no structured middle frame was read")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "files": [str(path.resolve()) for path in paths],
        "map_id": map_id,
        "horizon_position_count": len(frames),
        "observed_horizon_positions": sorted(observed_positions),
        "observed_current_frames": sorted(observed_frames),
        "schedule_cell_count": len(schedule_cells),
        "clean_layers": sorted(clean_layers),
        "readout_records": readout_records,
        "coverage_readout_heads": coverage_readout_heads,
        "structured_middle_heads": structured_middle_heads,
        "observed_source_kinds": sorted(observed_sources),
        "max_total_frame_equivalents": max_budget,
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
    methods = [str(value) for value in manifest["method_order"]]
    start_idx, end_idx = (
        (args.smoke_prompt_index, args.smoke_prompt_index + 1)
        if args.scope == "smoke"
        else (0, PROMPT_COUNT)
    )
    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in methods:
        config = manifest["methods"][method]
        media = audit_interval(
            args.run_root / "raw" / method,
            start_idx=start_idx,
            end_idx=end_idx,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=not args.skip_decode,
        )
        logs = audit_logs(args.run_root, method, config)
        traces = audit_traces(args.run_root, method, config)
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        if method_ok:
            published_dir = args.run_root / "published" / method
            expected_published = {
                f"{int(item['prompt_idx']):06d}.mp4" for item in media["videos"]
            }
            existing_published = {
                path.name for path in published_dir.glob("*.mp4")
            }
            unexpected = existing_published - expected_published
            if unexpected:
                raise RuntimeError(
                    f"refusing stale v201 published videos for {method}: "
                    f"{sorted(unexpected)}"
                )
            for item in media["videos"]:
                source = args.run_root / "raw" / method / str(item["file"])
                mode = link_or_validate(
                    source,
                    published_dir / f"{int(item['prompt_idx']):06d}.mp4",
                )
                link_counts[mode] += 1
            observed_published = {
                path.name for path in published_dir.glob("*.mp4")
            }
            if observed_published != expected_published:
                raise RuntimeError(f"incomplete v201 published set for {method}")
        method_rows.append(
            {
                "key": method,
                "role": config["role"],
                "operator": config["operator"],
                "routing_map_id": config["routing_map_id"],
                "map_classification": config["map_classification"],
                "coverage_count_by_position": config["coverage_count_by_position"],
                "coverage_exposure_count": config["coverage_exposure_count"],
                "coverage_exposure_fraction": config["coverage_exposure_fraction"],
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )
    contract = {
        "version": 1,
        "experiment": "v201_head_phase_horizon_causal_generation",
        "scope": args.scope,
        "development_only": True,
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "prompt_items": manifest["prompt_items"][start_idx:end_idx],
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": manifest["decoded_video_contract"],
        "seed": SEED,
        "methods": methods,
        "operators": manifest["operators"],
        "operator_contracts": manifest["operator_contracts"],
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
    }
    contract_path = args.run_root / "contracts" / "experiment.json"
    contract_sha = write_json(contract_path, contract)
    summary = {
        "version": 1,
        "ok": bool(all_ok),
        "experiment": contract["experiment"],
        "scope": args.scope,
        "methods": method_rows,
        "experiment_contract": str(contract_path.resolve()),
        "experiment_contract_sha256": contract_sha,
        "link_counts": link_counts,
    }
    write_json(args.run_root / "audits" / "summary.json", summary)
    if not all_ok:
        failed = [row["key"] for row in method_rows if not row["ok"]]
        raise RuntimeError(f"v201 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v201-audit] PASS "
        f"scope={args.scope} methods={len(methods)} "
        f"videos={len(methods) * (end_idx - start_idx)} links={link_counts}"
    )


if __name__ == "__main__":
    main()
