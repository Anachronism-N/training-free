#!/usr/bin/env python3
"""Audit v184 media, runtime contracts, and denoising schedule traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_indexed_videos import audit_interval

from prepare_v184_denoise_phase_screen import METHODS


FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "scheduled cache read budget drift",
    "Recent schedule leaked middle memory",
    "Coverage schedule exceeded its 4+4 budget",
    "CacheCompatDenoiseTraceWarning",
)
HISTORY_LINE = re.compile(
    r"\[HistoryPolarityPolicy\].*support=reservoir4_multiscalemotion1"
    r".*suppress=reservoir4_multiscalemotion1.*counts=10:360,11:0"
    r".*exclusive_owner=true"
)
SCHEDULE_LINE = re.compile(
    r"\[CacheCompatDenoiseSchedule\].*schedule=(\w+)"
    r".*clean_readout=recent.*read_budget=9FFE"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"refusing mixed v184 published video: {target}")
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


def audit_logs(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "logs" / method).glob("*.log"))
    if not paths:
        return {"ok": False, "errors": ["no logs"], "logs": []}
    errors = []
    rows = []
    expected_schedule = method_row["schedule"]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
        schedules = SCHEDULE_LINE.findall(text)
        history_ok = bool(HISTORY_LINE.search(text))
        schedule_ok = bool(schedules) and set(schedules) == {expected_schedule}
        if not history_ok:
            errors.append(f"{path.name}: missing shared-bank route contract")
        if not schedule_ok:
            errors.append(
                f"{path.name}: schedule={schedules}, expected={expected_schedule}"
            )
        if failures:
            errors.append(f"{path.name}: failures={failures}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "history_contract": history_ok,
                "schedules": schedules,
                "failure_patterns": failures,
            }
        )
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_schedule_traces(run_root: Path, method: str, method_row: dict) -> dict:
    paths = sorted((run_root / "traces" / method).glob("*.schedule.jsonl"))
    if not paths:
        return {"ok": False, "errors": ["no schedule traces"], "files": []}
    expected_schedule = method_row["schedule"]
    expected_calls = set(method_row["coverage_noisy_calls"])
    errors = []
    schedule_records = 0
    readout_records = 0
    noisy_policies: dict[int, set[str]] = {index: set() for index in range(4)}
    clean_policies: set[str] = set()
    traced_layers: set[int] = set()
    max_budget = 0
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("schedule") != expected_schedule:
                errors.append(f"{path.name}: trace schedule drift")
                continue
            event = row.get("event")
            layer = int(row.get("layer", -1))
            traced_layers.add(layer)
            policy = str(row.get("effective_policy", ""))
            mode = str(row.get("update_mode", ""))
            if event == "schedule":
                schedule_records += 1
                if int(row.get("call_count", -1)) != 4:
                    errors.append(f"{path.name}: denoise call count drift")
                if mode == "clean":
                    clean_policies.add(policy)
                    if policy != "recent" or row.get("clean_policy_is_recent") is not True:
                        errors.append(f"{path.name}: clean pass did not use Recent")
                elif mode == "noisy":
                    call_index = int(row.get("call_index", -1))
                    if call_index not in noisy_policies:
                        errors.append(f"{path.name}: invalid noisy call index")
                    else:
                        noisy_policies[call_index].add(policy)
                        expected = "coverage" if call_index in expected_calls else "recent"
                        if policy != expected:
                            errors.append(
                                f"{path.name}: call {call_index} used {policy}, expected {expected}"
                            )
                else:
                    errors.append(f"{path.name}: invalid update mode {mode!r}")
            elif event == "readout":
                readout_records += 1
                if row.get("budget_pass") is not True:
                    errors.append(f"{path.name}: readout budget flag failed")
                observed_budget = int(row.get("max_total_frame_equivalents", -1))
                max_budget = max(max_budget, observed_budget)
                if observed_budget > 9:
                    errors.append(f"{path.name}: readout exceeded 9 FFE")
                for head in row.get("selected_heads") or ():
                    counts = head.get("counts") or {}
                    anchor = int(counts.get("anchor", -1))
                    dynamic = int(counts.get("dynamic", -1))
                    total = int(head.get("total_frame_equivalents", -1))
                    if total > 9:
                        errors.append(f"{path.name}: traced head exceeded 9 FFE")
                    if policy == "recent" and (anchor != 0 or dynamic > 8):
                        errors.append(f"{path.name}: Recent trace leaked Coverage")
                    if policy == "coverage" and (anchor > 4 or dynamic > 4):
                        errors.append(f"{path.name}: Coverage trace budget drift")
            else:
                errors.append(f"{path.name}: unknown trace event {event!r}")
    if not schedule_records or not readout_records:
        errors.append("missing schedule or readout trace records")
    if traced_layers != {0, 10, 20, 29}:
        errors.append(f"trace layers differ: {sorted(traced_layers)}")
    for call_index, policies in noisy_policies.items():
        expected = "coverage" if call_index in expected_calls else "recent"
        if policies != {expected}:
            errors.append(
                f"call {call_index} policies={sorted(policies)}, expected={expected}"
            )
    if clean_policies != {"recent"}:
        errors.append(f"clean policies differ: {sorted(clean_policies)}")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "files": [str(path.resolve()) for path in paths],
        "schedule_records": schedule_records,
        "readout_records": readout_records,
        "traced_layers": sorted(traced_layers),
        "noisy_policies": {
            str(index): sorted(values) for index, values in noisy_policies.items()
        },
        "clean_policies": sorted(clean_policies),
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

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "v184_denoise_phase_coverage_screen":
        raise ValueError("v184 audit received the wrong input manifest")
    methods = tuple(manifest["method_order"])
    if methods != METHODS or set(methods) != set(manifest["methods"]):
        raise ValueError("v184 manifest method order/membership mismatch")
    if args.scope == "smoke":
        start_idx, end_idx = args.smoke_prompt_index, args.smoke_prompt_index + 1
    else:
        start_idx, end_idx = 0, int(manifest["prompt_count"])

    published_path = args.run_root / "published_manifest.json"
    published_path.unlink(missing_ok=True)
    all_ok = True
    link_counts = {"existing": 0, "hardlink": 0, "symlink": 0}
    method_rows = []
    for method in methods:
        method_config = manifest["methods"][method]
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
        logs = audit_logs(args.run_root, method, method_config)
        traces = audit_schedule_traces(args.run_root, method, method_config)
        report = {"media": media, "logs": logs, "schedule_traces": traces}
        report_path = args.run_root / "audits" / f"{method}.json"
        report_sha = write_json(report_path, report)
        method_ok = bool(media["ok"] and logs["ok"] and traces["ok"])
        all_ok = all_ok and method_ok
        if method_ok:
            published_dir = args.run_root / "published" / method
            for item in media["videos"]:
                source = args.run_root / "raw" / method / str(item["file"])
                mode = link_or_validate(
                    source,
                    published_dir / f"{int(item['prompt_idx']):06d}.mp4",
                )
                link_counts[mode] += 1
        method_rows.append(
            {
                "key": method,
                "role": "local_control" if method == "all_recent" else "phase_candidate",
                "schedule": method_config["schedule"],
                "coverage_noisy_calls": method_config["coverage_noisy_calls"],
                "video_dir": str((args.run_root / "published" / method).resolve()),
                "audit": str(report_path.resolve()),
                "audit_sha256": report_sha,
                "ok": method_ok,
            }
        )

    contract = {
        "version": 1,
        "experiment": "v184_denoise_phase_coverage_generation",
        "scope": args.scope,
        "development_only": True,
        "prompt_count": end_idx - start_idx,
        "prompt_indices": list(range(start_idx, end_idx)),
        "prompt_file": manifest["prompt_file"],
        "prompt_file_sha256": manifest["prompt_file_sha256"],
        "prompt_items": manifest["prompt_items"][start_idx:end_idx],
        "num_output_frames": 120,
        "decoded_video_contract": manifest["decoded_video_contract"],
        "methods": list(methods),
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
        raise RuntimeError(f"v184 audit failed: {failed}")
    write_json(published_path, summary)
    print(
        "[v184-audit] PASS "
        f"scope={args.scope} methods={len(methods)} "
        f"videos={len(methods) * (end_idx - start_idx)} links={link_counts}"
    )


if __name__ == "__main__":
    main()
