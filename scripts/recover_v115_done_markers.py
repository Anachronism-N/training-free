#!/usr/bin/env python3
"""Recover missing v115 done markers from complete existing artifacts.

This command never launches inference. It writes a marker only after the
frozen config, experiment contract, log, video, policy trace, and role-memory
trace all pass the same audits used by the original runner.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_v100_fast_selection_1video import (
    Cell,
    audit_motion_trace,
    audit_policy_trace,
    audit_role_event_trace,
    audit_scene_trace,
    audit_video,
    sha256,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)


EXPERIMENT = "v115_role_memory_cache_1video"
FAILURE_SIGNATURES = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "PyramidKVPolicyTraceError",
    "PyramidKVMotionTraceError",
    "PyramidKVRoleEventTraceError",
)


def cell_from_payload(payload: dict[str, Any]) -> Cell:
    allowed = {field.name for field in fields(Cell)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown Cell fields: {unknown}")
    return Cell(**payload)


def matching_contract(
    run_root: Path,
    expected_sha256: str,
) -> Path:
    matches = [
        path
        for path in sorted((run_root / "contracts").glob("*.json"))
        if sha256(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one frozen contract with SHA {expected_sha256}, "
            f"found {matches}"
        )
    return matches[0]


def validate_log(log: Path, cell: Cell) -> None:
    if not log.is_file():
        raise FileNotFoundError(f"missing inference log: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = [value for value in FAILURE_SIGNATURES if value in text]
    if hits:
        raise ValueError(f"failure signatures in log: {hits}")
    required = (
        "[PyramidKVRuntimePolicy]",
        "[HistoryPolarityPolicy]",
        "legacy_pf_labels=false",
        "exclusive_owner=true",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(f"runtime markers missing from log: {missing}")
    if cell.uses_motion and "[PyramidKVMotionEvent]" not in text:
        raise ValueError("motion-event runtime marker is missing")
    if cell.uses_role_event and "[PyramidKVRoleEvent]" not in text:
        raise ValueError("role-event runtime marker is missing")
    if cell.scene_cache and "[SceneCacheSwitch]" not in text:
        raise ValueError("scene-cache runtime marker is missing")


def recover_cell(
    *,
    repo_root: Path,
    run_root: Path,
    config_path: Path,
    legacy_map: Path,
    dry_run: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != EXPERIMENT:
        raise ValueError(
            f"{config_path}: expected experiment {EXPERIMENT}, "
            f"got {config.get('experiment')!r}"
        )
    cell = cell_from_payload(config["cell"])
    if config_path.stem != cell.name:
        raise ValueError("config filename and cell name disagree")
    if config.get("head_map_sha256") != EXPECTED_MAP_SHA256:
        raise ValueError("config does not use the frozen old-v98 map")
    map_audit = validate_frozen_map(
        legacy_map,
        repo_root
        / "third_party"
        / "Pyramid-Forcing"
        / "configs"
        / "head_configs"
        / "best_labels.csv",
    )
    if map_audit["sha256"] != config["head_map_sha256"]:
        raise ValueError("current legacy map differs from frozen cell config")
    contract_sha = str(config["experiment_contract_sha256"])
    contract_path = matching_contract(run_root, contract_sha)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("experiment") != EXPERIMENT:
        raise ValueError("matched contract has the wrong experiment")
    contract_cells = {
        str(row["name"]) for row in contract.get("cells", [])
        if isinstance(row, dict) and "name" in row
    }
    if cell.name not in contract_cells:
        raise ValueError("cell is absent from its frozen experiment contract")

    output = run_root / "videos" / cell.name
    log = run_root / "logs" / f"{cell.name}.log"
    policy_trace = run_root / "traces" / f"{cell.name}.policy.jsonl"
    motion_trace = run_root / "traces" / f"{cell.name}.motion.jsonl"
    role_event_trace = (
        run_root / "traces" / f"{cell.name}.role_event.jsonl"
    )
    scene_trace = run_root / "traces" / f"{cell.name}.scene.jsonl"
    diagnostics = run_root / "diagnostics"
    marker = run_root / "status" / f"{cell.name}.done.json"
    config_sha = sha256(config_path)

    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("config_sha256") != config_sha:
            raise ValueError("existing done marker has a stale config SHA")
        return {
            "cell": cell.name,
            "status": "already_complete",
            "marker": str(marker),
        }

    validate_log(log, cell)
    audit_args = SimpleNamespace(repo_root=repo_root)
    video = audit_video(
        audit_args,
        cell=cell,
        output=output,
        report=diagnostics / f"{cell.name}.video.json",
        log=diagnostics / f"{cell.name}.video.log",
    )
    policy = audit_policy_trace(
        policy_trace,
        cell=cell,
        head_map=legacy_map,
        report_path=diagnostics / f"{cell.name}.policy.json",
    )
    motion = None
    if cell.uses_motion:
        motion = audit_motion_trace(
            motion_trace,
            cell=cell,
            legacy_map=legacy_map,
            report_path=diagnostics / f"{cell.name}.motion.json",
        )
    role_event = None
    if cell.uses_role_event:
        role_event = audit_role_event_trace(
            role_event_trace,
            cell=cell,
            head_map=legacy_map,
            report_path=diagnostics / f"{cell.name}.role_event.json",
        )
    scene = None
    if cell.scene_cache:
        scene = audit_scene_trace(
            scene_trace,
            cell=cell,
            report_path=diagnostics / f"{cell.name}.scene.json",
        )
    marker_payload = {
        "version": 1,
        "cell": cell.name,
        "config_sha256": config_sha,
        "log_sha256": sha256(log),
        "video_fingerprint": video["input_fingerprint"],
        "policy_records": policy["records"],
        "motion_records": None if motion is None else motion["records"],
        "role_event_records": (
            None if role_event is None else role_event["records"]
        ),
        "scene_events": None if scene is None else scene["events"],
        "recovered_without_inference": True,
    }
    if not dry_run:
        write_frozen(marker, marker_payload)
    return {
        "cell": cell.name,
        "status": "validated_dry_run" if dry_run else "recovered",
        "marker": str(marker),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "runs" / EXPERIMENT,
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--legacy-map", type=Path)
    parser.add_argument(
        "--cells",
        help="optional comma-separated cell names; defaults to every config",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    legacy_map = (
        args.legacy_map
        or repo_root
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv"
    ).resolve()
    selected = (
        {value.strip() for value in args.cells.split(",") if value.strip()}
        if args.cells
        else None
    )
    configs = [
        path
        for path in sorted((run_root / "configs").glob("*.json"))
        if selected is None or path.stem in selected
    ]
    if not configs:
        raise SystemExit("no matching frozen v115 cell configs")
    if selected is not None:
        missing = selected - {path.stem for path in configs}
        if missing:
            raise SystemExit(f"requested cells have no config: {sorted(missing)}")

    rows = []
    failures = []
    for config_path in configs:
        try:
            row = recover_cell(
                repo_root=repo_root,
                run_root=run_root,
                config_path=config_path,
                legacy_map=legacy_map,
                dry_run=args.dry_run,
            )
            rows.append(row)
            print(f"[v115-recovery] {row['status']} {row['cell']}")
        except Exception as error:
            failures.append(
                {"cell": config_path.stem, "error": str(error)}
            )
            print(f"[v115-recovery-failed] {config_path.stem}: {error}")
    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "run_root": str(run_root),
        "dry_run": bool(args.dry_run),
        "results": rows,
        "failures": failures,
        "ok": not failures,
    }
    write_runtime_json(
        run_root / "status" / "v115_recovery_summary.json",
        summary,
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} v115 cells could not be recovered; "
            "inspect v115_recovery_summary.json"
        )


if __name__ == "__main__":
    main()
