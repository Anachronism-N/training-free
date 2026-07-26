#!/usr/bin/env python3
"""Run cache-ownership recovery, from one-prompt smoke test to 128 prompts.

Each ``smoke1`` invocation generates one selected parity/classifier video for
one prompt; existing PF and binary videos are audited and reused. Larger modes
evaluate all recovery cells on disjoint four-node prompt shards. Every mode
freezes source/model/map hashes and rejects traces that violate exclusive
[sink + middle + recent] ownership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRACE_LAYERS = (0, 7, 15, 23, 29)
TRACE_STRIDE = 3
TRACE_MAX_RECORDS = 60000
EXPECTED_VIDEO_FRAMES = 477
EXPECTED_VIDEO_FPS = 16.0
EXPECTED_VIDEO_WIDTH = 832
EXPECTED_VIDEO_HEIGHT = 480


@dataclass(frozen=True)
class Cell:
    name: str
    engine: str
    map_key: str | None
    route: str
    transition: bool = False


CELLS = (
    Cell(
        "pf_ar_neutral_stride_cyclic",
        "pf",
        "pf_ar_binary_control",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_stride_cyclic",
        "pf",
        "history_polarity_zero",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_stride_cyclic_v78",
        "pf",
        "history_polarity_zero",
        "history_stride_cyclic",
        transition=True,
    ),
    Cell(
        "history_polarity_random_stride_cyclic",
        "pf",
        "history_polarity_zero_random",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_inverted_stride_cyclic",
        "pf",
        "history_polarity_zero_inverted",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_tau_m0p1_stride_cyclic",
        "pf",
        "history_polarity_m0p1",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_tau_p0p1_stride_cyclic",
        "pf",
        "history_polarity_0p1",
        "history_stride_cyclic",
    ),
    Cell(
        "history_polarity_stride_merge_fixed",
        "pf",
        "history_polarity_zero",
        "history_stride_merge",
    ),
)

CAUSAL_ONLY_CELLS = (
    Cell(
        "pf_aw_neutral_stride_merge",
        "pf",
        "pf_aw_binary_control",
        "history_stride_merge",
    ),
    Cell(
        "history_polarity_random_stride_merge",
        "pf",
        "history_polarity_zero_random",
        "history_stride_merge",
    ),
)
ALL_CELLS = CELLS + CAUSAL_ONLY_CELLS
CAUSAL_CELL_NAMES = (
    "pf_aw_neutral_stride_merge",
    "history_polarity_stride_merge_fixed",
    "history_polarity_stride_cyclic",
    "history_polarity_random_stride_merge",
)

SMOKE_CELL_NAMES = {
    "pf-ar": "pf_ar_neutral_stride_cyclic",
    "pf-aw-stride-merge": "pf_aw_neutral_stride_merge",
    "history-polarity": "history_polarity_stride_cyclic",
    "history-polarity-stride-merge": (
        "history_polarity_stride_merge_fixed"
    ),
    "history-polarity-random-stride-merge": (
        "history_polarity_random_stride_merge"
    ),
}


def cells_for_mode(
    mode: str,
    smoke_cell: str = "pf-ar",
) -> tuple[Cell, ...]:
    if mode == "smoke1":
        selected = SMOKE_CELL_NAMES[smoke_cell]
        return tuple(cell for cell in ALL_CELLS if cell.name == selected)
    if mode == "causal32":
        by_name = {cell.name: cell for cell in ALL_CELLS}
        return tuple(by_name[name] for name in CAUSAL_CELL_NAMES)
    return CELLS


STRIDE_CYCLIC_POLICIES = {
    10: (("StrideStrategy",), 3, 4, "stride"),
    11: (("CyclicStrategy",), 1, 4, "osc"),
}
STRIDE_MERGE_POLICIES = {
    10: (("StrideStrategy",), 3, 4, "stride"),
    11: (("MergeStrategy",), 3, 4, "merge"),
}
EXCLUSIVE_CACHE_CONTRACTS = {
    "history_stride_cyclic": {
        "owner": "HeadComposition",
        "segments": ("static_sink", "explicit_middle", "dynamic_recent"),
        "supportive_label_10": {
            "sink_frames": 3,
            "middle": "stride",
            "middle_capacity_frames": 4,
            "stride_interval": 6,
            "recent_frames": 4,
        },
        "responsive_label_11": {
            "sink_frames": 1,
            "middle": "cyclic",
            "middle_capacity_frames": 4,
            "cyclic_period": 6,
            "recent_frames": 4,
        },
    },
    "history_stride_merge": {
        "owner": "HeadComposition",
        "segments": ("static_sink", "explicit_middle", "dynamic_recent"),
        "supportive_label_10": {
            "sink_frames": 3,
            "middle": "stride",
            "middle_capacity_frames": 4,
            "stride_interval": 6,
            "recent_frames": 4,
        },
        "suppressive_label_11": {
            "sink_frames": 3,
            "middle": "merge",
            "middle_capacity_frames": 4,
            "merge_patch_size": 2,
            "merge_block_frames": 4,
            "recent_frames": 4,
        },
    },
}
# Backward-compatible name for callers that inspect the default recovery
# route. Per-cell manifests use the route-specific contract below.
EXCLUSIVE_CACHE_CONTRACT = EXCLUSIVE_CACHE_CONTRACTS[
    "history_stride_cyclic"
]


def exclusive_cache_contract(route: str) -> dict[str, Any]:
    try:
        return EXCLUSIVE_CACHE_CONTRACTS[route]
    except KeyError as error:
        raise ValueError(
            f"no exclusive cache contract for route {route!r}"
        ) from error


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
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing mixed frozen artifact: {path}")
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    return digest


def wait_for_frozen(
    path: Path,
    payload: Any,
    *,
    timeout_seconds: int,
) -> str:
    content = canonical_json(payload)
    digest = hashlib.sha256(content).hexdigest()
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    while time.monotonic() < deadline:
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            existing = b""
        if existing == content:
            return digest
        if existing:
            raise RuntimeError(f"refusing mixed frozen artifact: {path}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for frozen artifact: {path}")


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
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
    if completed.returncode:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {rendered}"
        )


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
    ).strip()


def load_matrix(
    path: Path,
    *,
    labels: set[int],
    require_all_labels: bool = True,
) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"{path}: expected a 30x12 head map")
    observed = {value for row in rows for value in row}
    invalid = observed - labels
    missing = labels - observed
    if invalid or (require_all_labels and missing):
        raise ValueError(
            f"{path}: expected "
            f"{'all ' if require_all_labels else 'only '}"
            f"labels {sorted(labels)}, "
            f"found {sorted(observed)}"
        )
    return rows


def validate_map_manifest(
    manifest_path: Path,
    *,
    pf_labels: Path,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "history_polarity_zero",
        "history_polarity_zero_random",
        "history_polarity_zero_inverted",
        "history_polarity_m0p1",
        "history_polarity_0p1",
        "pf_ar_binary_control",
        "pf_aw_binary_control",
    }
    maps = payload.get("maps")
    if not isinstance(maps, dict) or not required.issubset(maps):
        missing = sorted(required - set(maps or {}))
        raise ValueError(f"map manifest is missing recovery maps: {missing}")
    resolved: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        item = maps[name]
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        actual_hash = sha256(path)
        if actual_hash != item.get("sha256"):
            raise ValueError(f"{name}: map SHA256 mismatch")
        load_matrix(
            path,
            labels={10, 11},
            require_all_labels=not name.startswith(
                "history_polarity_"
            )
            or name
            in {
                "history_polarity_zero",
                "history_polarity_zero_random",
                "history_polarity_zero_inverted",
            },
        )
        resolved[name] = {
            "path": str(path),
            "sha256": actual_hash,
        }
    pf_labels = pf_labels.resolve()
    pf_hash = sha256(pf_labels)
    if (
        str(Path(str(payload.get("pf_labels", ""))).resolve())
        != str(pf_labels)
        or payload.get("pf_labels_sha256") != pf_hash
    ):
        raise ValueError("PF label binding differs from the map manifest")
    load_matrix(pf_labels, labels={-1, 1, 2})
    resolved["pf_labels"] = {
        "path": str(pf_labels),
        "sha256": pf_hash,
    }
    return resolved


def ensure_maps(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    map_dir = args.out_root / "maps"
    manifest = map_dir / "history_polarity_manifest.json"
    builder = args.repo_root / "scripts" / "build_v98_history_polarity_maps.py"
    score_csv = args.score_root / "scores" / "qk_head_scores.csv"
    score_artifact = (
        args.score_root / "scores" / "qk_head_score_artifact.json"
    )
    for path in (builder, score_csv, score_artifact, args.pf_labels):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not manifest.is_file():
        lock = args.out_root / ".map_build_lock"
        if args.node_rank == 0:
            try:
                lock.mkdir(parents=False)
            except FileExistsError as error:
                raise RuntimeError(f"map build lock already exists: {lock}") from error
            try:
                run_checked(
                    [
                        sys.executable,
                        str(builder),
                        "--scores",
                        str(score_csv),
                        "--score-artifact",
                        str(score_artifact),
                        "--pf-labels",
                        str(args.pf_labels),
                        "--output-dir",
                        str(map_dir),
                    ]
                )
            finally:
                lock.rmdir()
        else:
            deadline = time.monotonic() + args.map_wait_seconds
            while not manifest.is_file() and time.monotonic() < deadline:
                time.sleep(5)
            if not manifest.is_file():
                raise TimeoutError(
                    f"timed out waiting for node 0 to build {manifest}"
                )

    run_checked(
        [
            sys.executable,
            str(builder),
            "--scores",
            str(score_csv),
            "--score-artifact",
            str(score_artifact),
            "--pf-labels",
            str(args.pf_labels),
            "--output-dir",
            str(map_dir),
            "--validate-only",
        ]
    )
    return manifest, validate_map_manifest(
        manifest,
        pf_labels=args.pf_labels,
    )


def trace_policies(route: str) -> dict[int, tuple[Any, int, int, str]]:
    if route == "history_stride_cyclic":
        return STRIDE_CYCLIC_POLICIES
    if route == "history_stride_merge":
        return STRIDE_MERGE_POLICIES
    raise ValueError(f"no PF trace policy for route {route!r}")


def audit_trace(
    trace_path: Path,
    *,
    map_path: Path,
    route: str,
    prompt_count: int,
    output_path: Path,
) -> dict[str, Any]:
    labels = load_matrix(
        map_path,
        labels={-1, 1, 2} if route == "native" else {10, 11},
        require_all_labels=route == "native",
    )
    policies = trace_policies(route)
    failures: list[str] = []
    event_count = 0
    prompt_ids: set[int] = set()
    observed_pairs: set[tuple[int, int]] = set()
    cache_failures = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
                if event["event"] != "middle_selection":
                    raise ValueError(f"unexpected event {event['event']!r}")
                layer = int(event["layer"])
                head = int(event["head"])
                prompt_id = int(event["prompt_id"])
                label = int(event["label"])
                branch = str(event["branch"])
                strategy_names = tuple(
                    str(item["name"]) for item in event["strategies"]
                )
                sink_frames = int(event["sink_frames"])
                recent_frames = int(event["recent_frames"])
                policy_type = str(event["policy_type"])
                frame_seqlen = int(event["frame_seqlen"])
                sink_count = int(event["sink_frame_count"])
                sink_tokens = int(event["sink_token_count"])
                recent_count = int(event["recent_frame_count"])
                recent_tokens = int(event["recent_token_count"])
                middle_tokens = int(event["union_token_count"])
                cache_pass = bool(event["cache_contract_pass"])
                violations = list(event["cache_contract_violations"])
                sink_overlap = list(event["middle_sink_overlap"])
                recent_overlap = list(event["middle_recent_overlap"])
                explicit_owner = bool(
                    event["explicit_composition_owns_dynamic"]
                )
                composition_present = bool(event["composition_present"])
                dynamic_owner = str(event["dynamic_policy_owner"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"line {line_number}: malformed trace: {error}")
                continue
            event_count += 1
            if layer not in TRACE_LAYERS or not 0 <= head < 12:
                failures.append(
                    f"line {line_number}: invalid layer/head {layer}/{head}"
                )
                continue
            expected_label = labels[layer][head]
            if label != expected_label:
                failures.append(
                    f"line {line_number}: label {label} != map {expected_label}"
                )
                continue
            expected = policies[label]
            if (
                strategy_names,
                sink_frames,
                recent_frames,
                policy_type,
            ) != expected:
                failures.append(
                    f"line {line_number}: policy "
                    f"{(strategy_names, sink_frames, recent_frames, policy_type)} "
                    f"!= {expected}"
                )
            if branch != "cond":
                failures.append(
                    f"line {line_number}: unexpected CFG branch {branch!r}"
                )
            if recent_count > recent_frames:
                failures.append(
                    f"line {line_number}: dynamic cache has "
                    f"{recent_count}>{recent_frames} frames"
                )
            if frame_seqlen <= 0:
                failures.append(
                    f"line {line_number}: invalid frame_seqlen={frame_seqlen}"
                )
            else:
                if sink_count > sink_frames or (
                    sink_tokens > sink_frames * frame_seqlen
                ):
                    failures.append(
                        f"line {line_number}: sink budget exceeded "
                        f"frames={sink_count}/{sink_frames} "
                        f"tokens={sink_tokens}/{sink_frames * frame_seqlen}"
                    )
                if recent_tokens > recent_frames * frame_seqlen:
                    failures.append(
                        f"line {line_number}: dynamic token budget exceeded "
                        f"{recent_tokens}>{recent_frames * frame_seqlen}"
                    )
                if middle_tokens > 4 * frame_seqlen:
                    failures.append(
                        f"line {line_number}: middle token budget exceeded "
                        f"{middle_tokens}>{4 * frame_seqlen}"
                    )
            if (
                not cache_pass
                or violations
                or sink_overlap
                or recent_overlap
                or not explicit_owner
                or not composition_present
                or dynamic_owner != "composition_recent"
            ):
                cache_failures += 1
                failures.append(
                    f"line {line_number}: cache ownership failed "
                    f"pass={cache_pass} violations={violations} "
                    f"sink_overlap={sink_overlap} "
                    f"recent_overlap={recent_overlap} "
                    f"explicit_owner={explicit_owner} "
                    f"composition_present={composition_present} "
                    f"dynamic_owner={dynamic_owner}"
                )
            prompt_ids.add(prompt_id)
            observed_pairs.add((layer, head))

    expected_pairs = {
        (layer, head)
        for layer in TRACE_LAYERS
        for head in range(12)
    }
    if observed_pairs != expected_pairs:
        failures.append(
            "trace layer/head coverage mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)[:12]}"
        )
    if len(prompt_ids) != prompt_count:
        failures.append(
            f"trace prompt epochs {len(prompt_ids)} != {prompt_count}"
        )
    if event_count == 0:
        failures.append("trace is empty")
    payload = {
        "version": 1,
        "method": "v99_cache_ownership_trace_audit",
        "trace": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "route": route,
        "events": event_count,
        "prompt_epochs": len(prompt_ids),
        "observed_layer_head_pairs": len(observed_pairs),
        "cache_contract_failures": cache_failures,
        "failures": failures,
        "pass": not failures,
    }
    output_path.write_bytes(canonical_json(payload))
    if failures:
        raise RuntimeError(
            f"trace audit failed for {trace_path}: {failures[:4]}"
        )
    return payload


def indexed_output_exists(output: Path, start: int, end: int) -> bool:
    pattern = re.compile(r"^(\d+)")
    for path in output.glob("*.mp4"):
        match = pattern.match(path.name)
        if match and start <= int(match.group(1)) < end:
            return True
    return False


def audit_videos(
    args: argparse.Namespace,
    *,
    output: Path,
    start: int,
    end: int,
    report: Path,
    log: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(args.repo_root / "scripts" / "audit_indexed_videos.py"),
        "--video-dir",
        str(output),
        "--start-idx",
        str(start),
        "--end-idx",
        str(end),
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
    if not payload.get("pass"):
        raise RuntimeError(f"video audit did not pass: {report}")
    return payload


def ensure_reused_baselines(
    args: argparse.Namespace,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    manifest = args.out_root / "reused_baselines.json"
    sources = {
        "pf_native": args.reuse_pf_dir,
        "pf_binary_read_reference": args.reuse_pf_binary_dir,
    }
    if args.reuse_sf_dir is not None:
        sources["sf_native"] = args.reuse_sf_dir
    if args.node_rank == 0:
        payload_sources = {}
        for name, source in sources.items():
            report = (
                args.out_root
                / "diagnostics"
                / f"reuse_{name}.video.json"
            )
            result = audit_videos(
                args,
                output=source,
                start=start,
                end=end,
                report=report,
                log=report.with_suffix(".log"),
            )
            payload_sources[name] = {
                "path": str(source.resolve()),
                "input_fingerprint": result["input_fingerprint"],
                "video_count": end - start,
                "audit_path": str(report.resolve()),
                "audit_sha256": sha256(report),
            }
        write_frozen(
            manifest,
            {
                "version": 1,
                "method": "v99_reused_video_baselines",
                "mode": args.mode,
                "prompt_sha256": sha256(args.prompts),
                "start_idx": start,
                "end_idx": end,
                "sources": payload_sources,
            },
        )
    else:
        deadline = time.monotonic() + args.map_wait_seconds
        while not manifest.is_file() and time.monotonic() < deadline:
            time.sleep(5)
        if not manifest.is_file():
            raise TimeoutError(
                f"timed out waiting for reused baseline audit: {manifest}"
            )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        payload.get("mode") != args.mode
        or int(payload.get("start_idx", -1)) != start
        or int(payload.get("end_idx", -1)) != end
        or payload.get("prompt_sha256") != sha256(args.prompts)
    ):
        raise ValueError(
            "reused baseline manifest has a different prompt range or mode"
        )
    for name, source in sources.items():
        item = payload.get("sources", {}).get(name, {})
        if item.get("path") != str(source.resolve()):
            raise ValueError(
                f"reused baseline path changed for {name}: "
                f"{item.get('path')} != {source.resolve()}"
            )
    return {
        "path": str(manifest.resolve()),
        "sha256": sha256(manifest),
        "sources": payload["sources"],
    }


def inference_command(
    args: argparse.Namespace,
    cell: Cell,
    *,
    output: Path,
    start: int,
    end: int,
    map_path: Path | None,
    transition_trace: Path,
) -> tuple[list[str], Path]:
    if cell.engine != "pf":
        raise ValueError(f"{cell.name}: v99 only generates PF-based cells")
    command = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(args.pf_config),
        "--checkpoint_path",
        str(args.pf_checkpoint),
    ]
    cwd = args.pf_repo
    command.extend(
        [
            "--data_path",
            str(args.prompts),
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
            str(start),
            "--end_idx",
            str(end),
            "--reseed_per_prompt",
        ]
    )
    if map_path is None:
        raise ValueError(f"{cell.name}: PF cell has no head map")
    command.extend(
        ["--pyramidkv_head_config_path", str(map_path)]
    )
    if cell.route.startswith("history_"):
        command.extend(
            [
                "--pyramidkv_history_polarity",
                "--pyramidkv_history_support_policy",
                "stride",
                "--pyramidkv_history_suppress_policy",
                (
                    "cyclic"
                    if cell.route == "history_stride_cyclic"
                    else "merge"
                ),
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
    return command, cwd


def run_cell(
    args: argparse.Namespace,
    *,
    cell: Cell,
    gpu: str,
    start: int,
    end: int,
    contract_sha256: str,
    maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = args.out_root / cell.name
    log = args.out_root / "logs" / f"{cell.name}.shard{args.node_rank}.log"
    trace = (
        args.out_root
        / "traces"
        / f"{cell.name}.shard{args.node_rank}.policy.jsonl"
    )
    transition_trace = (
        args.out_root
        / "traces"
        / f"{cell.name}.shard{args.node_rank}.transition.jsonl"
    )
    config_path = (
        args.out_root
        / "configs"
        / f"{cell.name}.shard{args.node_rank}.json"
    )
    marker = (
        args.out_root
        / "status"
        / f"{cell.name}.shard{args.node_rank}.done.json"
    )
    video_report = (
        args.out_root
        / "diagnostics"
        / f"{cell.name}.shard{args.node_rank}.video.json"
    )
    video_log = video_report.with_suffix(".log")
    trace_report = (
        args.out_root
        / "diagnostics"
        / f"{cell.name}.shard{args.node_rank}.trace.json"
    )
    map_item = maps.get(cell.map_key) if cell.map_key else None
    map_path = Path(map_item["path"]) if map_item else None
    config = {
        "version": 1,
        "experiment": "v99_binary_cache_recovery",
        "contract_sha256": contract_sha256,
        "node_rank": args.node_rank,
        "shard": args.node_rank,
        "start_idx": start,
        "end_idx": end,
        "gpu": gpu,
        "cell": asdict(cell),
        "map_path": str(map_path) if map_path else None,
        "map_sha256": map_item["sha256"] if map_item else None,
        "trace_layers": list(TRACE_LAYERS),
        "trace_stride": TRACE_STRIDE,
        "trace_max_records": TRACE_MAX_RECORDS,
        "exclusive_cache_contract": exclusive_cache_contract(cell.route),
    }
    config_sha256 = write_frozen(config_path, config)

    if marker.is_file():
        frozen = json.loads(marker.read_text(encoding="utf-8"))
        if (
            frozen.get("config_sha256") != config_sha256
            or not log.is_file()
        ):
            raise RuntimeError(f"stale completion marker: {marker}")
        video = audit_videos(
            args,
            output=output,
            start=start,
            end=end,
            report=video_report,
            log=video_log,
        )
        if cell.engine == "pf":
            if not trace.is_file():
                raise RuntimeError(f"missing trace for completed cell: {trace}")
            audit_trace(
                trace,
                map_path=map_path,
                route=cell.route,
                prompt_count=end - start,
                output_path=trace_report,
            )
        if video.get("input_fingerprint") != frozen.get(
            "video_input_fingerprint"
        ):
            raise RuntimeError(f"video fingerprint changed: {cell.name}")
        return {"name": cell.name, "status": "resumed"}

    if indexed_output_exists(output, start, end):
        raise RuntimeError(
            f"{cell.name}: partial videos exist without a marker; "
            "use a fresh OUT_ROOT"
        )
    output.mkdir(parents=True, exist_ok=True)
    for stale in (trace, transition_trace, trace_report, video_report):
        if stale.exists():
            raise RuntimeError(
                f"{cell.name}: stale artifact exists without marker: {stale}"
            )

    command, cwd = inference_command(
        args,
        cell,
        output=output,
        start=start,
        end=end,
        map_path=map_path,
        transition_trace=transition_trace,
    )
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "COMMIT_FORCING_ENABLE": "0",
            "PYRAMIDKV_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_PACK": "0",
            "PYRAMIDKV_USE_CPP_PACK_OUTPUT": "0",
            "PYRAMIDKV_USE_MEGA_CACHE": "0",
            "PYRAMIDKV_USE_MEGA_ATTN": "0",
            "PYRAMIDKV_HEAD_MAP_DEBUG": "1",
            "PYRAMIDKV_POLICY_TRACE_LAYERS": ",".join(
                str(value) for value in TRACE_LAYERS
            ),
            "PYRAMIDKV_POLICY_TRACE_STRIDE": str(TRACE_STRIDE),
            "PYRAMIDKV_POLICY_TRACE_MAX_RECORDS": str(TRACE_MAX_RECORDS),
        }
    )
    if cell.engine == "pf":
        env["PYRAMIDKV_POLICY_TRACE_PATH"] = str(trace)
    run_checked(command, cwd=cwd, env=env, log_path=log)

    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "PyramidKVPolicyTraceError",
    )
    if any(signature in log_text for signature in failure_signatures):
        raise RuntimeError(f"failure signature found in {log}")
    if cell.engine == "pf":
        if "[PyramidKVRuntimePolicy]" not in log_text:
            raise RuntimeError(f"missing runtime policy marker in {log}")
        if cell.route.startswith("history_") and (
            "[HistoryPolarityPolicy]" not in log_text
            or "legacy_pf_labels=false" not in log_text
            or "exclusive_owner=true" not in log_text
            or "exclusive_dynamic=True" not in log_text
        ):
            raise RuntimeError(
                f"missing exclusive neutral-history policy marker in {log}"
            )
        if cell.transition and not transition_trace.is_file():
            raise RuntimeError(
                f"missing transition trace for {cell.name}"
            )
        if not trace.is_file():
            raise RuntimeError(f"missing policy trace for {cell.name}")
        audit_trace(
            trace,
            map_path=map_path,
            route=cell.route,
            prompt_count=end - start,
            output_path=trace_report,
        )

    video = audit_videos(
        args,
        output=output,
        start=start,
        end=end,
        report=video_report,
        log=video_log,
    )
    marker_payload = {
        "version": 1,
        "name": cell.name,
        "node_rank": args.node_rank,
        "config_sha256": config_sha256,
        "log_sha256": sha256(log),
        "policy_trace_sha256": (
            sha256(trace) if cell.engine == "pf" else None
        ),
        "transition_trace_sha256": (
            sha256(transition_trace) if cell.transition else None
        ),
        "video_input_fingerprint": video["input_fingerprint"],
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
        choices=("smoke1", "causal32", "screen32", "main128"),
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", "-1")),
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
    parser.add_argument(
        "--score-root", type=Path, default=env_path("SCORE_ROOT")
    )
    parser.add_argument("--prompts", type=Path, default=env_path("PROMPTS"))
    parser.add_argument("--pf-repo", type=Path, default=env_path("PF_REPO"))
    parser.add_argument(
        "--pf-config", type=Path, default=env_path("PF_CONFIG")
    )
    parser.add_argument(
        "--pf-checkpoint", type=Path, default=env_path("PF_CHECKPOINT")
    )
    parser.add_argument(
        "--pf-labels", type=Path, default=env_path("PF_LABELS")
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=int(os.environ.get("PROMPT_INDEX", "0")),
    )
    parser.add_argument(
        "--smoke-cell",
        choices=tuple(SMOKE_CELL_NAMES),
        default=os.environ.get("SMOKE_CELL", "pf-ar"),
    )
    parser.add_argument(
        "--reuse-sf-dir",
        type=Path,
        default=(
            Path(os.environ["REUSE_SF_DIR"])
            if os.environ.get("REUSE_SF_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--reuse-pf-dir",
        type=Path,
        default=(
            Path(os.environ["REUSE_PF_DIR"])
            if os.environ.get("REUSE_PF_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--reuse-pf-binary-dir",
        type=Path,
        default=(
            Path(os.environ["REUSE_PF_BINARY_DIR"])
            if os.environ.get("REUSE_PF_BINARY_DIR")
            else None
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-wait-seconds", type=int, default=1800)
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
    args.score_root = (
        args.score_root
        or args.repo_root / "runs" / "v98_middle_relative_scores"
    ).resolve()
    prompt_name = (
        "MovieGenVideoBench_num32.txt"
        if args.mode in {"smoke1", "causal32", "screen32"}
        else "MovieGenVideoBench_num128.txt"
    )
    args.prompts = (
        args.prompts or args.pf_repo / "prompts" / prompt_name
    ).resolve()
    default_out = (
        f"v99_binary_cache_recovery_{args.mode}"
    )
    args.out_root = (
        args.out_root or args.repo_root / "runs" / default_out
    ).resolve()
    for name in (
        "reuse_sf_dir",
        "reuse_pf_dir",
        "reuse_pf_binary_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def main() -> None:
    args = parse_args()
    smoke_mode = args.mode == "smoke1"
    if smoke_mode and args.smoke_cell not in SMOKE_CELL_NAMES:
        choices = ", ".join(SMOKE_CELL_NAMES)
        raise SystemExit(
            f"--smoke-cell/SMOKE_CELL must be one of: {choices}"
        )
    cells = cells_for_mode(args.mode, args.smoke_cell)
    valid_ranks = {0} if smoke_mode else set(range(4))
    if args.node_rank not in valid_ranks:
        expected = "0" if smoke_mode else "0, 1, 2, or 3"
        raise SystemExit(
            f"--node-rank/NODE_RANK must be {expected} for {args.mode}"
        )
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    required_gpus = len(cells) if smoke_mode else min(8, len(cells))
    if len(gpus) < required_gpus or len(set(gpus)) != len(gpus):
        raise SystemExit(
            "--gpu-list/GPU_LIST must contain at least "
            f"{required_gpus} unique ids"
        )
    if args.seed != 0:
        raise SystemExit("the frozen recovery screen requires seed 0")
    if args.reuse_pf_dir is None or args.reuse_pf_binary_dir is None:
        raise SystemExit(
            "provide --reuse-pf-dir and --reuse-pf-binary-dir "
            "(or matching REUSE_* env vars); SF reuse is optional"
        )
    required = (
        args.repo_root,
        args.pf_repo,
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.prompts,
        args.reuse_pf_dir,
        args.reuse_pf_binary_dir,
    )
    if args.reuse_sf_dir is not None:
        required = (*required, args.reuse_sf_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required paths: {missing}")
    prompt_lines = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_count = len(prompt_lines)
    expected_prompt_count = (
        32
        if args.mode in {"smoke1", "causal32", "screen32"}
        else 128
    )
    if prompt_count != expected_prompt_count:
        raise SystemExit(
            f"expected {expected_prompt_count} prompts, found {prompt_count}"
        )
    if not 0 <= args.prompt_index < prompt_count:
        raise SystemExit(
            f"--prompt-index must be in [0, {prompt_count}), "
            f"got {args.prompt_index}"
        )
    dirty = git_output(
        args.repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if dirty:
        raise SystemExit(
            "v99 requires a clean committed checkout; dirty entries:\n"
            + dirty
        )
    run_commit = git_output(args.repo_root, "rev-parse", "HEAD")

    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "status", "configs", "traces", "diagnostics"):
        (args.out_root / name).mkdir(exist_ok=True)
    node_lock = args.out_root / f".node{args.node_rank}.lock"
    try:
        node_lock.mkdir()
    except FileExistsError as error:
        raise SystemExit(f"node lock already exists: {node_lock}") from error
    (node_lock / "owner.json").write_bytes(
        canonical_json(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at_unix": int(time.time()),
            }
        )
    )

    try:
        manifest_path, maps = ensure_maps(args)
        if smoke_mode:
            start = args.prompt_index
            end = start + 1
        else:
            shard_size = prompt_count // 4
            start = args.node_rank * shard_size
            end = start + shard_size
        reused_baselines = ensure_reused_baselines(
            args,
            start=start if smoke_mode else 0,
            end=end if smoke_mode else prompt_count,
        )
        implementation_paths = (
            Path(__file__).resolve(),
            args.repo_root
            / "third_party"
            / "Pyramid-Forcing"
            / "inference.py",
            args.repo_root
            / "third_party"
            / "Pyramid-Forcing"
            / "pyramidkv"
            / "adaptive_cache.py",
            args.repo_root
            / "third_party"
            / "Pyramid-Forcing"
            / "pyramidkv"
            / "policy_overrides.py",
            args.repo_root
            / "third_party"
            / "Pyramid-Forcing"
            / "pipeline"
            / "pyramidkv_config.py",
            args.repo_root
            / "third_party"
            / "Pyramid-Forcing"
            / "pipeline"
            / "causal_inference.py",
        )
        contract = {
            "version": 1,
            "experiment": "v99_binary_cache_recovery",
            "mode": args.mode,
            "run_commit": run_commit,
            "prompt": {
                "path": str(args.prompts),
                "sha256": sha256(args.prompts),
                "count": prompt_count,
                "selected_text": (
                    prompt_lines[args.prompt_index] if smoke_mode else None
                ),
            },
            "frames": 120,
            "seed": 0,
            "shards": 1 if smoke_mode else 4,
            "prompt_index": args.prompt_index if smoke_mode else None,
            "cells": [asdict(cell) for cell in cells],
            "maps": maps,
            "map_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "reused_baselines": reused_baselines,
            "models": {
                "pf_checkpoint": {
                    "path": str(args.pf_checkpoint),
                    "sha256": sha256(args.pf_checkpoint),
                },
            },
            "configs": {
                "pf": {
                    "path": str(args.pf_config),
                    "sha256": sha256(args.pf_config),
                },
            },
            "implementation": {
                str(path.relative_to(args.repo_root)): sha256(path)
                for path in implementation_paths
            },
            "trace_contract": {
                "layers": list(TRACE_LAYERS),
                "stride": TRACE_STRIDE,
                "max_records": TRACE_MAX_RECORDS,
                "cache_contract_pass_required": True,
                "exclusive_cache_contracts": EXCLUSIVE_CACHE_CONTRACTS,
            },
            "video_contract": {
                "decoded_frames": EXPECTED_VIDEO_FRAMES,
                "fps": EXPECTED_VIDEO_FPS,
                "width": EXPECTED_VIDEO_WIDTH,
                "height": EXPECTED_VIDEO_HEIGHT,
            },
        }
        contract_path = args.out_root / "experiment_contract.json"
        if args.node_rank == 0:
            contract_sha256 = write_frozen(contract_path, contract)
        else:
            contract_sha256 = wait_for_frozen(
                contract_path,
                contract,
                timeout_seconds=args.map_wait_seconds,
            )

        offsets = (0, 2, 5, 7)
        futures = {}
        max_workers = (
            len(cells) if smoke_mode else min(8, len(cells))
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, cell in enumerate(cells):
                gpu_offset = 0 if smoke_mode else offsets[args.node_rank]
                gpu = gpus[(index + gpu_offset) % len(gpus)]
                future = executor.submit(
                    run_cell,
                    args,
                    cell=cell,
                    gpu=gpu,
                    start=start,
                    end=end,
                    contract_sha256=contract_sha256,
                    maps=maps,
                )
                futures[future] = cell.name
            failures = []
            results = []
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures.append(f"{name}: {error}")
                    print(f"[v99-cache-recovery:error] {name}: {error}", flush=True)
                else:
                    results.append(result)
                    print(
                        f"[v99-cache-recovery] {name}: {result['status']}",
                        flush=True,
                    )
        if failures:
            raise RuntimeError("; ".join(failures))
        node_done = {
            "version": 1,
            "node_rank": args.node_rank,
            "contract_sha256": contract_sha256,
            "start_idx": start,
            "end_idx": end,
            "results": sorted(results, key=lambda item: item["name"]),
            "completed_at_unix": int(time.time()),
        }
        write_frozen(
            args.out_root / "status" / f"node{args.node_rank}.done.json",
            node_done,
        )
        print(
            "[v99-cache-recovery:done] "
            f"node={args.node_rank} prompts={start}:{end} "
            f"contract={contract_sha256}",
            flush=True,
        )
    finally:
        owner = node_lock / "owner.json"
        if owner.exists():
            owner.unlink()
        node_lock.rmdir()


if __name__ == "__main__":
    main()
