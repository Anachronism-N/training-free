#!/usr/bin/env python3
"""Run the v100 one-video mechanism selection before broad ablations.

The fast screen separates three decisions:
1. Responsive-head middle-cache policy.
2. Small, previously motivated add-ons.
3. A-B-A role-aware scene archive/recall.

Every cell generates exactly one 30-second video and writes auditable policy,
motion-event, transition, and scene-switch traces as applicable.
"""

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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRACE_LAYERS = (0, 7, 15, 23, 29)
EXPECTED_VIDEO_FRAMES = 477
EXPECTED_VIDEO_FPS = 16.0
EXPECTED_VIDEO_WIDTH = 832
EXPECTED_VIDEO_HEIGHT = 480
LEGACY_COUNTS = {10: 304, 11: 56}
LEGACY_PF_CROSS_TAB = {
    "wave": {10: 133, 11: 23},
    "anchor": {10: 169, 11: 3},
    "veil": {10: 2, 11: 30},
}


@dataclass(frozen=True)
class Cell:
    name: str
    stage: str
    prompt_kind: str
    support_policy: str = "stride"
    suppress_policy: str | None = "motion_cyclic"
    transition: bool = False
    scene_cache: bool = False
    scene_bridge: int = 1
    scene_manual: bool = False
    motion_top_k: int = 1
    variance_refresh: bool = False
    map_key: str = "legacy"

    @property
    def native(self) -> bool:
        return self.suppress_policy is None

    @property
    def uses_motion(self) -> bool:
        return self.suppress_policy in {
            "motion",
            "motion_cyclic",
            "cyclic_motion1",
        }


CELLS = (
    # 1. Responsive cache selection.
    Cell("single_pf_native", "responsive", "single", suppress_policy=None),
    Cell(
        "legacy_v98_stride_cyclic_sink1",
        "responsive",
        "single",
        suppress_policy="cyclic",
    ),
    Cell(
        "legacy_v98_stride_cyclic_sink3",
        "responsive",
        "single",
        suppress_policy="cyclic_sink3",
    ),
    Cell(
        "legacy_v98_stride_motion4",
        "responsive",
        "single",
        suppress_policy="motion",
    ),
    Cell(
        "legacy_v98_stride_motion2_cyclic2",
        "responsive",
        "single",
        suppress_policy="motion_cyclic",
    ),
    Cell(
        "legacy_v98_stride_recent8",
        "responsive",
        "single",
        suppress_policy="recent8",
    ),
    # 2. Small add-ons. Only v78 is already validated; the others are screens.
    Cell(
        "legacy_v98_motion2_cyclic2_v78",
        "tricks",
        "single",
        transition=True,
    ),
    Cell(
        "legacy_v98_hybrid_motion2_cyclic2",
        "tricks",
        "single",
        support_policy="hybrid",
    ),
    Cell(
        "legacy_v98_hybrid_motion2_cyclic2_v78",
        "tricks",
        "single",
        support_policy="hybrid",
        transition=True,
    ),
    Cell(
        "legacy_v98_motion2_cyclic2_variance_refresh",
        "tricks",
        "single",
        variance_refresh=True,
    ),
    # 3. A-B-A switch/return. Manual matching is diagnostic, not a final claim.
    Cell("aba_pf_native", "aba", "aba", suppress_policy=None),
    Cell(
        "aba_motion_no_episode",
        "aba",
        "aba",
    ),
    Cell(
        "aba_motion_episode_bridge1",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=1,
    ),
    Cell(
        "aba_motion_episode_hard",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=0,
    ),
    Cell(
        "aba_motion_episode_manual_bridge1",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=1,
        scene_manual=True,
    ),
    Cell(
        "aba_cyclic_sink3_episode_bridge1",
        "aba",
        "aba",
        suppress_policy="cyclic_sink3",
        scene_cache=True,
        scene_bridge=1,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_frozen(path: Path, payload: Any) -> str:
    content = canonical_json(payload)
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise RuntimeError(f"refusing to overwrite mixed contract: {path}")
    else:
        temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temp.write_bytes(content)
        try:
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()
    return digest


def wait_for_frozen(path: Path, payload: Any, timeout_seconds: int) -> str:
    expected = hashlib.sha256(canonical_json(payload)).hexdigest()
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(2)
    if not path.is_file():
        raise TimeoutError(f"timed out waiting for {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"mixed frozen artifact: {path} expected={expected} actual={actual}"
        )
    return actual


def read_matrix(path: Path, allowed: set[int]) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"{path}: expected a complete 30x12 matrix")
    observed = {value for row in rows for value in row}
    if not observed.issubset(allowed):
        raise ValueError(
            f"{path}: labels {sorted(observed)} are not within {sorted(allowed)}"
        )
    return rows


def validate_legacy_map(
    legacy_path: Path,
    pf_path: Path,
) -> dict[str, Any]:
    legacy = read_matrix(legacy_path, {10, 11})
    pf = read_matrix(pf_path, {-1, 1, 2})
    counts = Counter(value for row in legacy for value in row)
    if dict(counts) != LEGACY_COUNTS:
        raise ValueError(
            f"legacy v98 counts changed: {dict(counts)} != {LEGACY_COUNTS}"
        )
    pf_names = {-1: "wave", 1: "anchor", 2: "veil"}
    cross_tab: dict[str, dict[int, int]] = {}
    for pf_label, name in pf_names.items():
        role_counts = Counter(
            legacy[layer][head]
            for layer in range(30)
            for head in range(12)
            if pf[layer][head] == pf_label
        )
        cross_tab[name] = {
            10: int(role_counts[10]),
            11: int(role_counts[11]),
        }
    if cross_tab != LEGACY_PF_CROSS_TAB:
        raise ValueError(
            f"legacy v98/PF cross-tab changed: {cross_tab} "
            f"!= {LEGACY_PF_CROSS_TAB}"
        )
    return {
        "path": str(legacy_path),
        "sha256": sha256(legacy_path),
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "pf_cross_tab": {
            name: {str(key): value for key, value in sorted(values.items())}
            for name, values in cross_tab.items()
        },
    }


def selected_cells(
    mode: str,
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[Cell, ...]:
    candidates = CELLS if mode == "all" else tuple(
        cell for cell in CELLS if cell.stage == mode
    )
    return tuple(candidates[node_rank::num_nodes])


def resolve_head_map(args: argparse.Namespace, cell: Cell) -> Path:
    if cell.native:
        return args.pf_labels
    head_maps = getattr(args, "head_maps", None)
    if head_maps is not None:
        try:
            return Path(head_maps[cell.map_key])
        except KeyError as error:
            raise ValueError(
                f"{cell.name}: unknown head-map key {cell.map_key!r}"
            ) from error
    if cell.map_key != "legacy":
        raise ValueError(
            f"{cell.name}: map {cell.map_key!r} requires args.head_maps"
        )
    return args.legacy_map


def run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    if log_path is None:
        completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(
                "[command] "
                + subprocess.list2cmdline([str(value) for value in command])
                + "\n"
            )
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{subprocess.list2cmdline(command)}"
        )


def audit_video(
    args: argparse.Namespace,
    *,
    cell: Cell,
    output: Path,
    report: Path,
    log: Path,
) -> dict[str, Any]:
    prompt_index = (
        args.single_prompt_index
        if cell.prompt_kind == "single"
        else args.aba_prompt_index
    )
    command = [
        sys.executable,
        str(args.repo_root / "scripts" / "audit_indexed_videos.py"),
        "--video-dir",
        str(output),
        "--start-idx",
        str(prompt_index),
        "--end-idx",
        str(prompt_index + 1),
        "--expected-frames",
        str(EXPECTED_VIDEO_FRAMES),
        "--expected-fps",
        str(EXPECTED_VIDEO_FPS),
        "--expected-width",
        str(EXPECTED_VIDEO_WIDTH),
        "--expected-height",
        str(EXPECTED_VIDEO_HEIGHT),
        "--fps-tolerance",
        "0.05",
        "--allow-outside-interval",
        "--output-json",
        str(report),
    ]
    run_checked(command, log_path=log)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"video audit failed: {report}")
    return payload


def expected_policy(
    cell: Cell,
    label: int,
) -> tuple[tuple[str, ...], int, int, str]:
    if label == 10:
        if cell.support_policy == "cyclic":
            return (("CyclicStrategy",), 1, 4, "osc")
        if cell.support_policy == "hybrid":
            return (
                ("CyclicStrategy", "StrideStrategy"),
                3,
                4,
                "stride",
            )
        return (("StrideStrategy",), 3, 4, "stride")
    if label != 11 or cell.suppress_policy is None:
        raise ValueError(f"no binary policy for label={label} cell={cell.name}")
    return {
        "merge": (("MergeStrategy",), 3, 4, "merge"),
        "cyclic": (("CyclicStrategy",), 1, 4, "osc"),
        "cyclic_sink3": (("CyclicStrategy",), 3, 4, "osc"),
        "motion": (("MotionEventStrategy",), 3, 4, "motion_event"),
        "motion_cyclic": (
            ("CyclicStrategy", "MotionEventStrategy"),
            3,
            4,
            "motion_cyclic",
        ),
        "cyclic_motion1": (
            ("CyclicStrategy", "MotionEventStrategy"),
            1,
            4,
            "motion_cyclic",
        ),
        "recent5": ((), 3, 5, "stride"),
        "recent8": ((), 3, 8, "stride"),
        "recent8_sink1": ((), 1, 8, "stride"),
    }[cell.suppress_policy]


def audit_policy_trace(
    path: Path,
    *,
    cell: Cell,
    head_map: Path,
    report_path: Path,
) -> dict[str, Any]:
    labels = read_matrix(
        head_map,
        {-1, 1, 2} if cell.native else {10, 11},
    )
    failures: list[str] = []
    records = 0
    observed: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: {error}")
                continue
            if event.get("event") != "middle_selection":
                failures.append(
                    f"line {line_number}: unexpected event {event.get('event')}"
                )
                continue
            records += 1
            try:
                layer = int(event["layer"])
                head = int(event["head"])
                label = int(event["label"])
                strategies = tuple(
                    str(item["name"]) for item in event["strategies"]
                )
                observed.add((layer, head))
                if label != labels[layer][head]:
                    failures.append(
                        f"line {line_number}: label {label} != map "
                        f"{labels[layer][head]}"
                    )
                if cell.native:
                    continue
                actual = (
                    strategies,
                    int(event["sink_frames"]),
                    int(event["recent_frames"]),
                    str(event["policy_type"]),
                )
                expected = expected_policy(cell, label)
                if actual != expected:
                    failures.append(
                        f"line {line_number}: policy {actual} != {expected}"
                    )
                if not bool(event["explicit_composition_owns_dynamic"]):
                    failures.append(
                        f"line {line_number}: dynamic owner is not exclusive"
                    )
                if not bool(event["cache_contract_pass"]):
                    failures.append(
                        f"line {line_number}: cache contract failed "
                        f"{event['cache_contract_violations']}"
                    )
            except (IndexError, KeyError, TypeError, ValueError) as error:
                failures.append(f"line {line_number}: malformed event: {error}")
    if records == 0:
        failures.append("trace has no middle_selection events")
    expected_layers = set(TRACE_LAYERS)
    observed_layers = {layer for layer, _ in observed}
    if not expected_layers.issubset(observed_layers):
        failures.append(
            f"missing trace layers {sorted(expected_layers - observed_layers)}"
        )
    payload = {
        "version": 1,
        "cell": cell.name,
        "records": records,
        "observed_pairs": len(observed),
        "failures": failures[:100],
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"policy trace audit failed for {cell.name}: {failures[:5]}"
        )
    return payload


def audit_motion_trace(
    path: Path,
    *,
    cell: Cell,
    legacy_map: Path,
    report_path: Path,
) -> dict[str, Any]:
    labels = read_matrix(legacy_map, {10, 11})
    expected_layers = {
        layer
        for layer in TRACE_LAYERS
        if 11 in labels[layer]
    }
    observed_layers: set[int] = set()
    failures: list[str] = []
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
                if event["event"] != "motion_event_update":
                    raise ValueError(f"unexpected event {event['event']!r}")
                layer = int(event["layer"])
                num_frames = int(event["num_frames"])
                offsets = [int(value) for value in event["selected_offsets"]]
                selected_ids = [
                    int(value) for value in event["selected_frame_ids"]
                ]
                scores = [float(value) for value in event["all_scores"]]
                expected_head_count = labels[layer].count(11)
                if int(event["responsive_head_count"]) != expected_head_count:
                    raise ValueError("responsive head count differs from map")
                if len(scores) != num_frames:
                    raise ValueError("score count differs from block frames")
                if len(offsets) != min(cell.motion_top_k, num_frames):
                    raise ValueError("selected offset count differs from top-k")
                if any(
                    offset < 0 or offset >= num_frames for offset in offsets
                ):
                    raise ValueError("selected offset outside block")
                if selected_ids != [
                    int(event["frame_start_t"]) + offset
                    for offset in offsets
                ]:
                    raise ValueError("selected frame ids do not match offsets")
                if any(not math.isfinite(value) or value < 0 for value in scores):
                    raise ValueError("motion scores must be finite and non-negative")
                records += 1
                observed_layers.add(layer)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"line {line_number}: {error}")
    if records == 0:
        failures.append("motion trace has no records")
    if not expected_layers.issubset(observed_layers):
        failures.append(
            "motion trace is missing responsive sampled layers: "
            f"{sorted(expected_layers - observed_layers)}"
        )
    payload = {
        "version": 1,
        "cell": cell.name,
        "records": records,
        "observed_layers": sorted(observed_layers),
        "expected_layers": sorted(expected_layers),
        "failures": failures[:100],
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"motion trace audit failed for {cell.name}: {failures[:5]}"
        )
    return payload


def audit_scene_trace(
    path: Path,
    *,
    cell: Cell,
    report_path: Path,
) -> dict[str, Any]:
    events = [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    failures: list[str] = []
    if len(events) != 3:
        failures.append(f"expected 3 scene events, found {len(events)}")
    canonical = [
        int(event.get("canonical_scene_id", -1)) for event in events
    ]
    if canonical != [0, 1, 0]:
        failures.append(f"expected canonical A-B-A ids [0,1,0], got {canonical}")
    if events:
        if not bool(events[0].get("initial")):
            failures.append("first scene event is not initial")
        if any(
            event.get("event") != "scene_cache_switch"
            for event in events
        ):
            failures.append("unexpected scene trace event")
    if len(events) >= 3:
        if cell.scene_manual:
            if events[2].get("decision") != "manual_recall":
                failures.append("manual A2 event is not manual_recall")
        elif events[2].get("decision") != "recall":
            failures.append("automatic A2 event is not recall")
        if (
            int(events[2].get("action_counts", {}).get("restore_scene", 0))
            <= 0
        ):
            failures.append("A2 did not restore any stride scene bank")
        if float(events[2].get("similarity", -1.0)) < float(
            events[2].get("threshold", 0.0)
        ):
            failures.append("A2 recall similarity is below threshold")
    payload = {
        "version": 1,
        "cell": cell.name,
        "events": len(events),
        "canonical_scene_ids": canonical,
        "decisions": [event.get("decision") for event in events],
        "similarities": [event.get("similarity") for event in events],
        "failures": failures,
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"scene trace audit failed for {cell.name}: {failures}"
        )
    return payload


def inference_command(
    args: argparse.Namespace,
    *,
    cell: Cell,
    output: Path,
    transition_trace: Path,
    scene_trace: Path,
) -> tuple[list[str], Path, Path, int]:
    prompts = (
        args.single_prompts
        if cell.prompt_kind == "single"
        else args.aba_prompts
    )
    prompt_index = (
        args.single_prompt_index
        if cell.prompt_kind == "single"
        else args.aba_prompt_index
    )
    head_map = resolve_head_map(args, cell)
    command = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(args.pf_config),
        "--checkpoint_path",
        str(args.pf_checkpoint),
        "--data_path",
        str(prompts),
        "--output_folder",
        str(output),
        "--num_output_frames",
        "120",
        "--seed",
        str(args.seed),
        "--num_samples",
        "1",
        "--use_ema",
        "--save_with_index",
        "--start_idx",
        str(prompt_index),
        "--end_idx",
        str(prompt_index + 1),
        "--reseed_per_prompt",
        "--pyramidkv_head_config_path",
        str(head_map),
    ]
    if not cell.native:
        command.extend(
            [
                "--pyramidkv_history_polarity",
                "--pyramidkv_history_support_policy",
                cell.support_policy,
                "--pyramidkv_history_suppress_policy",
                str(cell.suppress_policy),
                "--pyramidkv_motion_event_top_k",
                str(cell.motion_top_k),
                "--pyramidkv_motion_event_sample_tokens",
                "64",
            ]
        )
    if cell.transition:
        command.extend(
            [
                "--pyramidkv_cache_transition",
                "--pyramidkv_cache_transition_mode",
                "full",
                "--pyramidkv_cache_transition_min_reliability",
                ".55",
                "--pyramidkv_cache_transition_min_novelty",
                ".01",
                "--pyramidkv_cache_transition_max_commit_fraction",
                ".75",
                "--pyramidkv_cache_transition_stagger_period",
                "1",
                "--pyramidkv_cache_transition_max_age_blocks",
                "6",
                "--pyramidkv_cache_transition_branches",
                "cond",
                "--pyramidkv_cache_transition_denoise_weight",
                "2",
                "--pyramidkv_cache_transition_trace_path",
                str(transition_trace),
                "--pyramidkv_cache_transition_debug",
            ]
        )
    if cell.variance_refresh:
        command.extend(
            [
                "--pyramidkv_history_value_renorm_strength",
                ".5",
                "--pyramidkv_history_value_recent_frames",
                "4",
                "--pyramidkv_history_value_gate_lambda",
                "3",
                "--pyramidkv_history_value_labels",
                "10,11",
                "--pyramidkv_history_value_layer_start",
                "10",
                "--pyramidkv_history_value_layer_end",
                "20",
                "--pyramidkv_history_value_moment_mode",
                "variance_only",
                "--pyramidkv_history_value_target_frames",
                "8",
                "--pyramidkv_history_value_max_std_ratio",
                "1.5",
            ]
        )
    if cell.scene_cache:
        command.extend(
            [
                "--pyramidkv_scene_cache",
                "--pyramidkv_scene_cache_match_mode",
                "idf",
                "--pyramidkv_scene_cache_similarity_threshold",
                ".20",
                "--pyramidkv_scene_cache_max_scenes",
                "8",
                "--pyramidkv_scene_cache_bridge_recent_frames",
                str(cell.scene_bridge),
                "--pyramidkv_scene_cache_trace_path",
                str(scene_trace),
                "--pyramidkv_scene_cache_debug",
            ]
        )
        if cell.scene_manual:
            command.extend(
                ["--pyramidkv_scene_cache_manual_ids", "0,1,0"]
            )
    return command, args.pf_repo, head_map, prompt_index


def run_cell(
    args: argparse.Namespace,
    *,
    cell: Cell,
    gpu: str,
    experiment_contract_sha256: str,
) -> dict[str, str]:
    output = args.out_root / "videos" / cell.name
    log = args.out_root / "logs" / f"{cell.name}.log"
    policy_trace = args.out_root / "traces" / f"{cell.name}.policy.jsonl"
    motion_trace = args.out_root / "traces" / f"{cell.name}.motion.jsonl"
    transition_trace = (
        args.out_root / "traces" / f"{cell.name}.transition.jsonl"
    )
    scene_trace = args.out_root / "traces" / f"{cell.name}.scene.jsonl"
    config_path = args.out_root / "configs" / f"{cell.name}.json"
    marker = args.out_root / "status" / f"{cell.name}.done.json"
    video_report = args.out_root / "diagnostics" / f"{cell.name}.video.json"
    policy_report = (
        args.out_root / "diagnostics" / f"{cell.name}.policy.json"
    )
    motion_report = (
        args.out_root / "diagnostics" / f"{cell.name}.motion.json"
    )
    scene_report = (
        args.out_root / "diagnostics" / f"{cell.name}.scene.json"
    )
    command, cwd, head_map, prompt_index = inference_command(
        args,
        cell=cell,
        output=output,
        transition_trace=transition_trace,
        scene_trace=scene_trace,
    )
    head_map_audit = getattr(args, "head_map_audits", {}).get(cell.map_key)
    cell_config = {
        "version": 1,
        "experiment": getattr(
            args, "experiment_name", "v100_fast_selection_1video"
        ),
        "experiment_contract_sha256": experiment_contract_sha256,
        "cell": asdict(cell),
        "gpu": str(gpu),
        "prompt_index": int(prompt_index),
        "prompt_path": str(
            args.single_prompts
            if cell.prompt_kind == "single"
            else args.aba_prompts
        ),
        "head_map": str(head_map),
        "head_map_sha256": sha256(head_map),
        "head_map_audit": head_map_audit,
        "command": command,
    }
    config_sha = write_frozen(config_path, cell_config)

    if marker.is_file():
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if completed.get("config_sha256") != config_sha:
            raise RuntimeError(f"stale completion marker: {marker}")
        audit_video(
            args,
            cell=cell,
            output=output,
            report=video_report,
            log=video_report.with_suffix(".log"),
        )
        audit_policy_trace(
            policy_trace,
            cell=cell,
            head_map=head_map,
            report_path=policy_report,
        )
        if cell.uses_motion:
            audit_motion_trace(
                motion_trace,
                cell=cell,
                legacy_map=head_map,
                report_path=motion_report,
            )
        if cell.scene_cache:
            audit_scene_trace(
                scene_trace,
                cell=cell,
                report_path=scene_report,
            )
        return {"name": cell.name, "status": "resumed"}

    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.mp4")):
        raise RuntimeError(
            f"{cell.name}: videos exist without a matching completion marker"
        )
    for stale in (
        policy_trace,
        motion_trace,
        transition_trace,
        scene_trace,
        policy_report,
        motion_report,
        scene_report,
        video_report,
    ):
        if stale.exists():
            raise RuntimeError(
                f"{cell.name}: stale artifact exists without marker: {stale}"
            )

    env = os.environ.copy()
    root_python = str(args.repo_root / "src")
    env["PYTHONPATH"] = (
        root_python
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "COMMIT_FORCING_ENABLE": "0",
            "PYRAMIDKV_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_PACK": "0",
            "PYRAMIDKV_USE_CPP_PACK_OUTPUT": "0",
            "PYRAMIDKV_USE_MEGA_CACHE": "0",
            "PYRAMIDKV_USE_MEGA_ATTN": "0",
            "PYRAMIDKV_CONTIG_ANCHOR_STORE": "0",
            "PYRAMIDKV_HEAD_MAP_DEBUG": "1",
            "PYRAMIDKV_POLICY_TRACE_PATH": str(policy_trace),
            "PYRAMIDKV_POLICY_TRACE_LAYERS": ",".join(
                str(value) for value in TRACE_LAYERS
            ),
            "PYRAMIDKV_POLICY_TRACE_STRIDE": "3",
            "PYRAMIDKV_POLICY_TRACE_MAX_RECORDS": "30000",
        }
    )
    if cell.uses_motion:
        env.update(
            {
                "PYRAMIDKV_MOTION_TRACE_PATH": str(motion_trace),
                "PYRAMIDKV_MOTION_TRACE_LAYERS": ",".join(
                    str(value) for value in TRACE_LAYERS
                ),
                "PYRAMIDKV_MOTION_DEBUG": "1",
            }
        )
    else:
        env.pop("PYRAMIDKV_MOTION_TRACE_PATH", None)
        env.pop("PYRAMIDKV_MOTION_DEBUG", None)

    print(
        f"[run] stage={cell.stage} cell={cell.name} gpu={gpu} "
        f"prompt={prompt_index} map={cell.map_key} "
        f"map_sha256={sha256(head_map)}",
        flush=True,
    )
    run_checked(command, cwd=cwd, env=env, log_path=log)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "PyramidKVPolicyTraceError",
        "PyramidKVMotionTraceError",
    )
    hits = [
        signature for signature in failure_signatures
        if signature in log_text
    ]
    if hits:
        raise RuntimeError(f"{cell.name}: failure signatures in log: {hits}")
    if "[PyramidKVRuntimePolicy]" not in log_text:
        raise RuntimeError(f"{cell.name}: runtime policy marker is missing")
    if not cell.native:
        required = (
            "[HistoryPolarityPolicy]",
            "legacy_pf_labels=false",
            "exclusive_owner=true",
        )
        if any(value not in log_text for value in required):
            raise RuntimeError(
                f"{cell.name}: neutral exclusive-policy marker is incomplete"
            )
    if cell.scene_cache and "[SceneCacheSwitch]" not in log_text:
        raise RuntimeError(f"{cell.name}: scene-cache switch marker is missing")
    if cell.uses_motion and "[PyramidKVMotionEvent]" not in log_text:
        raise RuntimeError(f"{cell.name}: motion-event marker is missing")

    video = audit_video(
        args,
        cell=cell,
        output=output,
        report=video_report,
        log=video_report.with_suffix(".log"),
    )
    policy = audit_policy_trace(
        policy_trace,
        cell=cell,
        head_map=head_map,
        report_path=policy_report,
    )
    motion = None
    if cell.uses_motion:
        motion = audit_motion_trace(
            motion_trace,
            cell=cell,
            legacy_map=head_map,
            report_path=motion_report,
        )
    scene = None
    if cell.scene_cache:
        scene = audit_scene_trace(
            scene_trace,
            cell=cell,
            report_path=scene_report,
        )
    if cell.transition and not transition_trace.is_file():
        raise RuntimeError(f"{cell.name}: transition trace is missing")

    marker_payload = {
        "version": 1,
        "cell": cell.name,
        "config_sha256": config_sha,
        "log_sha256": sha256(log),
        "video_fingerprint": video["input_fingerprint"],
        "policy_records": policy["records"],
        "motion_records": None if motion is None else motion["records"],
        "scene_events": None if scene is None else scene["events"],
        "completed_at_unix": int(time.time()),
    }
    write_frozen(marker, marker_payload)
    return {"name": cell.name, "status": "completed"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("responsive", "tricks", "aba", "all"),
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", "0")),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.environ.get("NUM_NODES", "1")),
    )
    parser.add_argument(
        "--gpu-list",
        default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", root)),
    )
    parser.add_argument("--out-root", type=Path, default=env_path("OUT_ROOT"))
    parser.add_argument("--pf-repo", type=Path, default=env_path("PF_REPO"))
    parser.add_argument("--pf-config", type=Path, default=env_path("PF_CONFIG"))
    parser.add_argument(
        "--pf-checkpoint", type=Path, default=env_path("PF_CHECKPOINT")
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--legacy-map", type=Path, default=env_path("LEGACY_MAP")
    )
    parser.add_argument(
        "--single-prompts", type=Path, default=env_path("SINGLE_PROMPTS")
    )
    parser.add_argument(
        "--aba-prompts", type=Path, default=env_path("ABA_PROMPTS")
    )
    parser.add_argument(
        "--single-prompt-index",
        type=int,
        default=int(os.environ.get("SINGLE_PROMPT_INDEX", "0")),
    )
    parser.add_argument(
        "--aba-prompt-index",
        type=int,
        default=int(os.environ.get("ABA_PROMPT_INDEX", "0")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--reproduce-known-broken-map",
        action="store_true",
        help=(
            "Explicitly reproduce the historical v100 304/56-map screen. "
            "The non-native cells are known to produce polygon noise."
        ),
    )
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.pf_repo = (
        args.pf_repo or args.repo_root / "third_party" / "Pyramid-Forcing"
    ).resolve()
    args.pf_config = (
        args.pf_config or args.pf_repo / "configs" / "pyramid-forcing.yaml"
    ).resolve()
    args.pf_checkpoint = (
        args.pf_checkpoint
        or args.pf_repo / "checkpoints" / "self_forcing_dmd.pt"
    ).resolve()
    args.pf_labels = (
        args.pf_labels
        or args.pf_repo / "configs" / "head_configs" / "best_labels.csv"
    ).resolve()
    args.legacy_map = (
        args.legacy_map
        or args.repo_root
        / "runs"
        / "v98_history_polarity"
        / "maps"
        / "history_polarity_zero.csv"
    ).resolve()
    args.single_prompts = (
        args.single_prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num32.txt"
    ).resolve()
    args.aba_prompts = (
        args.aba_prompts
        or args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.out_root = (
        args.out_root
        or args.repo_root / "runs" / "v100_fast_selection_1video"
    ).resolve()
    return args


def main() -> None:
    args = parse_args()
    if not args.reproduce_known_broken_map:
        raise SystemExit(
            "v100 is frozen as a historical 304/56-map reproduction and is "
            "disabled by default after docs/106. Use "
            "scripts/run_v107_polygon_rootcause_1video.py for recovery, or "
            "pass --reproduce-known-broken-map only to reproduce v100."
        )
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v100 fast screen requires seed 0")
    paths = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.single_prompts,
        args.aba_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    single_lines = [
        value for value in args.single_prompts.read_text(
            encoding="utf-8"
        ).splitlines() if value.strip()
    ]
    aba_lines = [
        value for value in args.aba_prompts.read_text(
            encoding="utf-8"
        ).splitlines() if value.strip()
    ]
    if not 0 <= args.single_prompt_index < len(single_lines):
        raise SystemExit("--single-prompt-index is outside the prompt file")
    if not 0 <= args.aba_prompt_index < len(aba_lines):
        raise SystemExit("--aba-prompt-index is outside the ABA prompt file")
    if len(
        [
            part for part in aba_lines[args.aba_prompt_index].split("||")
            if part.strip()
        ]
    ) != 3:
        raise SystemExit("selected ABA prompt must contain exactly 3 segments")

    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "traces", "configs", "status", "diagnostics"):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    legacy = validate_legacy_map(args.legacy_map, args.pf_labels)
    all_mode_cells = (
        CELLS if args.mode == "all"
        else tuple(cell for cell in CELLS if cell.stage == args.mode)
    )
    contract = {
        "version": 1,
        "experiment": "v100_fast_selection_1video",
        "mode": args.mode,
        "seed": args.seed,
        "num_output_frames": 120,
        "expected_decoded_frames": EXPECTED_VIDEO_FRAMES,
        "single_prompt": {
            "path": str(args.single_prompts),
            "sha256": sha256(args.single_prompts),
            "index": args.single_prompt_index,
            "text": single_lines[args.single_prompt_index],
        },
        "aba_prompt": {
            "path": str(args.aba_prompts),
            "sha256": sha256(args.aba_prompts),
            "index": args.aba_prompt_index,
            "text": aba_lines[args.aba_prompt_index],
        },
        "legacy_v98_map": legacy,
        "pf_labels": {
            "path": str(args.pf_labels),
            "sha256": sha256(args.pf_labels),
        },
        "config": {
            "path": str(args.pf_config),
            "sha256": sha256(args.pf_config),
        },
        "checkpoint": {
            "path": str(args.pf_checkpoint),
            "size": args.pf_checkpoint.stat().st_size,
        },
        "cells": [asdict(cell) for cell in all_mode_cells],
        "implementation_hashes": {
            str(path.relative_to(args.repo_root)): sha256(path)
            for path in (
                args.pf_repo / "inference.py",
                args.pf_repo / "pipeline" / "causal_inference.py",
                args.pf_repo / "pipeline" / "pyramidkv_config.py",
                args.pf_repo / "pyramidkv" / "adaptive_cache.py",
                args.pf_repo / "pyramidkv" / "base.py",
                args.pf_repo / "pyramidkv" / "factory.py",
                args.pf_repo / "pyramidkv" / "motion_event.py",
                args.pf_repo / "pyramidkv" / "policy_overrides.py",
                args.pf_repo / "pyramidkv" / "stride.py",
            )
        },
    }
    contract_path = (
        args.out_root / "contracts" / f"{args.mode}.contract.json"
    )
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            timeout_seconds=args.contract_wait_seconds,
        )

    cells = selected_cells(
        args.mode,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
    )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if len(gpus) < len(cells) or len(gpus) != len(set(gpus)):
        raise SystemExit(
            f"node {args.node_rank} needs {len(cells)} unique GPUs; "
            f"received {gpus}"
        )
    print(
        "[v100] "
        f"mode={args.mode} node={args.node_rank}/{args.num_nodes} "
        f"cells={[cell.name for cell in cells]} out={args.out_root}",
        flush=True,
    )
    failures: list[str] = []
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(cells))) as executor:
        futures = {
            executor.submit(
                run_cell,
                args,
                cell=cell,
                gpu=gpus[index],
                experiment_contract_sha256=contract_sha,
            ): cell
            for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[done] {result}", flush=True)
            except Exception as error:
                failures.append(f"{cell.name}: {error}")
                print(f"[failed] {cell.name}: {error}", flush=True)
    node_summary = {
        "version": 1,
        "mode": args.mode,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "results": sorted(results, key=lambda item: item["name"]),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root
        / "status"
        / f"{args.mode}.node{args.node_rank}.summary.json"
    )
    write_frozen(summary_path, node_summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
