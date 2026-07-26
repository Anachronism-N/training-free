#!/usr/bin/env python3
"""Run the post-screen 128-prompt paper ablation on four 8-GPU nodes.

The v100 one-video screen decides the candidate cache. This runner then
freezes that decision and evaluates eight method/ablation cells on all 128
MovieGenVideoBench prompts. Each node owns one disjoint 32-prompt shard and
runs one method per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_v100_fast_selection_1video import (
    EXPECTED_VIDEO_FPS,
    EXPECTED_VIDEO_FRAMES,
    EXPECTED_VIDEO_HEIGHT,
    EXPECTED_VIDEO_WIDTH,
    TRACE_LAYERS,
    Cell as PolicyCell,
    audit_motion_trace,
    audit_policy_trace,
    read_matrix,
    run_checked,
    sha256,
    validate_legacy_map,
    wait_for_frozen,
    write_frozen,
)


ALLOWED_SUPPORT_POLICIES = ("stride", "hybrid")
ALLOWED_SUPPRESS_POLICIES = (
    "cyclic_sink3",
    "motion",
    "motion_cyclic",
    "recent8",
)
ROUTE_ABLATION_PRIORITY = (
    "cyclic_sink3",
    "recent8",
    "motion_cyclic",
    "motion",
)


@dataclass(frozen=True)
class Method:
    name: str
    map_key: str
    support_policy: str
    suppress_policy: str
    transition: bool
    purpose: str

    @property
    def uses_motion(self) -> bool:
        return self.suppress_policy in {"motion", "motion_cyclic"}

    def policy_cell(self) -> PolicyCell:
        return PolicyCell(
            name=self.name,
            stage="paper_ablation",
            prompt_kind="single",
            support_policy=self.support_policy,
            suppress_policy=self.suppress_policy,
            transition=self.transition,
        )


def build_methods(args: argparse.Namespace) -> tuple[Method, ...]:
    """Build eight non-duplicate cells around the selected candidate."""

    support_toggle = (
        "hybrid" if args.candidate_support == "stride" else "stride"
    )
    route_alternatives = [
        route
        for route in ROUTE_ABLATION_PRIORITY
        if route != args.candidate_suppress
    ][:2]
    transition_toggle = not args.candidate_transition
    methods = (
        Method(
            "ours_full",
            "legacy",
            args.candidate_support,
            args.candidate_suppress,
            args.candidate_transition,
            "selected full method",
        ),
        Method(
            "ablate_transition_toggle",
            "legacy",
            args.candidate_support,
            args.candidate_suppress,
            transition_toggle,
            "toggle the validated v78 transition gate",
        ),
        Method(
            "ablate_support_route_toggle",
            "legacy",
            support_toggle,
            args.candidate_suppress,
            args.candidate_transition,
            "toggle Supportive stride versus stride-cyclic hybrid",
        ),
        Method(
            f"ablate_responsive_{route_alternatives[0]}",
            "legacy",
            args.candidate_support,
            route_alternatives[0],
            args.candidate_transition,
            "replace the selected Responsive cache at fixed membership",
        ),
        Method(
            f"ablate_responsive_{route_alternatives[1]}",
            "legacy",
            args.candidate_support,
            route_alternatives[1],
            args.candidate_transition,
            "second Responsive cache replacement at fixed membership",
        ),
        Method(
            "control_random_membership",
            "random",
            args.candidate_support,
            args.candidate_suppress,
            args.candidate_transition,
            "count-matched random role assignment",
        ),
        Method(
            "control_pf_aw_membership",
            "pf_aw",
            args.candidate_support,
            args.candidate_suppress,
            args.candidate_transition,
            "PF Anchor+Wave versus Veil membership with our two routes",
        ),
        Method(
            f"control_threshold_{args.threshold_control}",
            f"threshold_{args.threshold_control}",
            args.candidate_support,
            args.candidate_suppress,
            args.candidate_transition,
            "history-polarity threshold stability control",
        ),
    )
    signatures = {
        (
            method.map_key,
            method.support_policy,
            method.suppress_policy,
            method.transition,
        )
        for method in methods
    }
    if len(methods) != 8 or len(signatures) != len(methods):
        raise ValueError(f"paper ablation contains duplicate cells: {methods}")
    return methods


def audit_video_interval(
    args: argparse.Namespace,
    *,
    method: Method,
    start: int,
    end: int,
    output: Path,
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
    if not payload.get("ok"):
        raise RuntimeError(
            f"{method.name}: video audit did not return ok=true"
        )
    return payload


def inference_command(
    args: argparse.Namespace,
    *,
    method: Method,
    head_map: Path,
    output: Path,
    start: int,
    end: int,
    transition_trace: Path,
) -> list[str]:
    command = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(args.pf_config),
        "--checkpoint_path",
        str(args.pf_checkpoint),
        "--data_path",
        str(args.prompts),
        "--output_folder",
        str(output),
        "--num_output_frames",
        "120",
        "--seed",
        "0",
        "--num_samples",
        "1",
        "--use_ema",
        "--save_with_index",
        "--start_idx",
        str(start),
        "--end_idx",
        str(end),
        "--reseed_per_prompt",
        "--pyramidkv_head_config_path",
        str(head_map),
        "--pyramidkv_history_polarity",
        "--pyramidkv_history_support_policy",
        method.support_policy,
        "--pyramidkv_history_suppress_policy",
        method.suppress_policy,
        "--pyramidkv_motion_event_top_k",
        "1",
        "--pyramidkv_motion_event_sample_tokens",
        "64",
    ]
    if method.transition:
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
    return command


def interval_has_videos(output: Path, start: int, end: int) -> bool:
    for path in output.glob("*.mp4"):
        try:
            prompt_index = int(path.name.split("-", 1)[0])
        except (IndexError, ValueError):
            continue
        if start <= prompt_index < end:
            return True
    return False


def run_method_shard(
    args: argparse.Namespace,
    *,
    method: Method,
    head_map: Path,
    gpu: str,
    start: int,
    end: int,
    contract_sha256: str,
) -> dict[str, object]:
    shard = args.node_rank
    output = args.out_root / "videos" / method.name
    log = args.out_root / "logs" / f"{method.name}.shard{shard}.log"
    policy_trace = (
        args.out_root
        / "traces"
        / f"{method.name}.shard{shard}.policy.jsonl"
    )
    motion_trace = (
        args.out_root
        / "traces"
        / f"{method.name}.shard{shard}.motion.jsonl"
    )
    transition_trace = (
        args.out_root
        / "traces"
        / f"{method.name}.shard{shard}.transition.jsonl"
    )
    config_path = (
        args.out_root / "configs" / f"{method.name}.shard{shard}.json"
    )
    marker = (
        args.out_root / "status" / f"{method.name}.shard{shard}.done.json"
    )
    video_report = (
        args.out_root
        / "diagnostics"
        / f"{method.name}.shard{shard}.video.json"
    )
    policy_report = (
        args.out_root
        / "diagnostics"
        / f"{method.name}.shard{shard}.policy.json"
    )
    motion_report = (
        args.out_root
        / "diagnostics"
        / f"{method.name}.shard{shard}.motion.json"
    )
    command = inference_command(
        args,
        method=method,
        head_map=head_map,
        output=output,
        start=start,
        end=end,
        transition_trace=transition_trace,
    )
    config = {
        "version": 1,
        "experiment": "v101_paper_ablation_128",
        "contract_sha256": contract_sha256,
        "node_rank": int(args.node_rank),
        "gpu": str(gpu),
        "interval": [int(start), int(end)],
        "method": asdict(method),
        "head_map": {
            "path": str(head_map),
            "sha256": sha256(head_map),
        },
        "command": command,
    }
    config_sha256 = write_frozen(config_path, config)
    policy_cell = method.policy_cell()

    if marker.is_file():
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if completed.get("config_sha256") != config_sha256:
            raise RuntimeError(f"stale completion marker: {marker}")
        audit_video_interval(
            args,
            method=method,
            start=start,
            end=end,
            output=output,
            report=video_report,
            log=video_report.with_suffix(".log"),
        )
        audit_policy_trace(
            policy_trace,
            cell=policy_cell,
            head_map=head_map,
            report_path=policy_report,
        )
        if method.uses_motion:
            audit_motion_trace(
                motion_trace,
                cell=policy_cell,
                legacy_map=head_map,
                report_path=motion_report,
            )
        return {
            "method": method.name,
            "status": "resumed",
            "interval": [start, end],
        }

    output.mkdir(parents=True, exist_ok=True)
    if interval_has_videos(output, start, end):
        raise RuntimeError(
            f"{method.name}: interval videos exist without marker "
            f"[{start}, {end})"
        )
    for stale in (
        log,
        policy_trace,
        motion_trace,
        transition_trace,
        video_report,
        policy_report,
        motion_report,
    ):
        if stale.exists():
            raise RuntimeError(
                f"{method.name}: stale shard artifact without marker: {stale}"
            )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(args.repo_root / "src")
        + (
            os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else ""
        )
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
            "PYRAMIDKV_POLICY_TRACE_STRIDE": "6",
            "PYRAMIDKV_POLICY_TRACE_MAX_RECORDS": "120000",
        }
    )
    if method.uses_motion:
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
        f"[run] method={method.name} gpu={gpu} shard={shard} "
        f"interval=[{start},{end})",
        flush=True,
    )
    run_checked(command, cwd=args.pf_repo, env=env, log_path=log)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "PyramidKVPolicyTraceError",
        "PyramidKVMotionTraceError",
    )
    hits = [
        signature
        for signature in failure_signatures
        if signature in log_text
    ]
    if hits:
        raise RuntimeError(
            f"{method.name}: failure signatures in log: {hits}"
        )
    required_markers = (
        "[PyramidKVRuntimePolicy]",
        "[HistoryPolarityPolicy]",
        "legacy_pf_labels=false",
        "exclusive_owner=true",
    )
    missing_markers = [
        marker_text
        for marker_text in required_markers
        if marker_text not in log_text
    ]
    if missing_markers:
        raise RuntimeError(
            f"{method.name}: missing runtime markers {missing_markers}"
        )
    if method.uses_motion and "[PyramidKVMotionEvent]" not in log_text:
        raise RuntimeError(
            f"{method.name}: motion-event runtime marker is missing"
        )

    video = audit_video_interval(
        args,
        method=method,
        start=start,
        end=end,
        output=output,
        report=video_report,
        log=video_report.with_suffix(".log"),
    )
    policy = audit_policy_trace(
        policy_trace,
        cell=policy_cell,
        head_map=head_map,
        report_path=policy_report,
    )
    motion = None
    if method.uses_motion:
        motion = audit_motion_trace(
            motion_trace,
            cell=policy_cell,
            legacy_map=head_map,
            report_path=motion_report,
        )
    if method.transition and not transition_trace.is_file():
        raise RuntimeError(
            f"{method.name}: transition trace was not produced"
        )

    marker_payload = {
        "version": 1,
        "method": method.name,
        "node_rank": int(args.node_rank),
        "interval": [int(start), int(end)],
        "config_sha256": config_sha256,
        "log_sha256": sha256(log),
        "video_fingerprint": video["input_fingerprint"],
        "policy_records": policy["records"],
        "motion_records": (
            None if motion is None else motion["records"]
        ),
        "completed_at_unix": int(time.time()),
    }
    write_frozen(marker, marker_payload)
    return {
        "method": method.name,
        "status": "completed",
        "interval": [start, end],
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", "0")),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.environ.get("NUM_NODES", "4")),
    )
    parser.add_argument(
        "--gpu-list",
        default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument(
        "--candidate-support",
        choices=ALLOWED_SUPPORT_POLICIES,
        default=os.environ.get("CANDIDATE_SUPPORT", "stride"),
    )
    parser.add_argument(
        "--candidate-suppress",
        choices=ALLOWED_SUPPRESS_POLICIES,
        default=os.environ.get("CANDIDATE_SUPPRESS", "motion_cyclic"),
    )
    parser.add_argument(
        "--candidate-transition",
        choices=("on", "off"),
        default=os.environ.get("CANDIDATE_TRANSITION", "on"),
    )
    parser.add_argument(
        "--threshold-control",
        choices=("m0p1", "p0p1"),
        default=os.environ.get(
            "THRESHOLD_CONTROL", "m0p1"
        ),
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
        "--pf-checkpoint",
        type=Path,
        default=env_path("PF_CHECKPOINT"),
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument("--prompts", type=Path, default=env_path("PROMPTS"))
    parser.add_argument("--legacy-map", type=Path, default=env_path("LEGACY_MAP"))
    parser.add_argument("--random-map", type=Path, default=env_path("RANDOM_MAP"))
    parser.add_argument("--pf-aw-map", type=Path, default=env_path("PF_AW_MAP"))
    parser.add_argument(
        "--threshold-m0p1-map",
        type=Path,
        default=env_path("THRESHOLD_M0P1_MAP"),
    )
    parser.add_argument(
        "--threshold-p0p1-map",
        type=Path,
        default=env_path("THRESHOLD_P0P1_MAP"),
    )
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    args = parser.parse_args()

    args.candidate_transition = args.candidate_transition == "on"
    args.repo_root = args.repo_root.resolve()
    args.pf_repo = (
        args.pf_repo
        or args.repo_root / "third_party" / "Pyramid-Forcing"
    ).resolve()
    args.pf_config = (
        args.pf_config
        or args.pf_repo / "configs" / "pyramid-forcing.yaml"
    ).resolve()
    args.pf_checkpoint = (
        args.pf_checkpoint
        or args.pf_repo / "checkpoints" / "self_forcing_dmd.pt"
    ).resolve()
    args.pf_labels = (
        args.pf_labels
        or args.pf_repo / "configs" / "head_configs" / "best_labels.csv"
    ).resolve()
    args.prompts = (
        args.prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num128.txt"
    ).resolve()
    map_root = (
        args.repo_root / "runs" / "v98_history_polarity" / "maps"
    )
    args.legacy_map = (
        args.legacy_map or map_root / "history_polarity_zero.csv"
    ).resolve()
    args.random_map = (
        args.random_map or map_root / "history_polarity_zero_random.csv"
    ).resolve()
    args.pf_aw_map = (
        args.pf_aw_map or map_root / "pf_aw_binary_control.csv"
    ).resolve()
    args.threshold_m0p1_map = (
        args.threshold_m0p1_map
        or map_root / "history_polarity_m0p1.csv"
    ).resolve()
    args.threshold_p0p1_map = (
        args.threshold_p0p1_map
        or map_root / "history_polarity_0p1.csv"
    ).resolve()
    args.out_root = (
        args.out_root
        or args.repo_root / "runs" / "v101_paper_ablation_128"
    ).resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.num_nodes != 4:
        raise SystemExit("v101 paper ablation is frozen to exactly 4 nodes")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, 4)")
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise SystemExit("each v101 node requires exactly 8 unique GPU ids")

    map_paths = {
        "legacy": args.legacy_map,
        "random": args.random_map,
        "pf_aw": args.pf_aw_map,
        "threshold_m0p1": args.threshold_m0p1_map,
        "threshold_p0p1": args.threshold_p0p1_map,
    }
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
        *map_paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    prompt_lines = [
        line
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompt_lines) != 128:
        raise SystemExit(
            f"v101 requires exactly 128 prompts, found {len(prompt_lines)}"
        )

    legacy_audit = validate_legacy_map(
        args.legacy_map, args.pf_labels
    )
    map_audits: dict[str, dict[str, object]] = {
        "legacy": legacy_audit
    }
    for key, path in map_paths.items():
        if key == "legacy":
            continue
        rows = read_matrix(path, {10, 11})
        counts = {
            str(label): sum(row.count(label) for row in rows)
            for label in (10, 11)
        }
        if any(value <= 0 for value in counts.values()):
            raise SystemExit(f"{key}: both roles must be represented")
        map_audits[key] = {
            "path": str(path),
            "sha256": sha256(path),
            "counts": counts,
        }

    methods = build_methods(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "videos",
        "logs",
        "traces",
        "configs",
        "status",
        "diagnostics",
        "contracts",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)

    interval_size = len(prompt_lines) // args.num_nodes
    start = args.node_rank * interval_size
    end = (
        len(prompt_lines)
        if args.node_rank == args.num_nodes - 1
        else start + interval_size
    )
    runner_path = Path(__file__).resolve()
    implementation_paths = (
        runner_path,
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "motion_event.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "stride.py",
    )
    contract = {
        "version": 1,
        "experiment": "v101_paper_ablation_128",
        "seed": 0,
        "num_nodes": 4,
        "shards": [[rank * 32, (rank + 1) * 32] for rank in range(4)],
        "video_contract": {
            "latent_frames": 120,
            "decoded_frames": EXPECTED_VIDEO_FRAMES,
            "fps": EXPECTED_VIDEO_FPS,
            "width": EXPECTED_VIDEO_WIDTH,
            "height": EXPECTED_VIDEO_HEIGHT,
        },
        "candidate": {
            "support_policy": args.candidate_support,
            "suppress_policy": args.candidate_suppress,
            "transition": args.candidate_transition,
            "threshold_control": args.threshold_control,
        },
        "methods": [asdict(method) for method in methods],
        "prompts": {
            "path": str(args.prompts),
            "sha256": sha256(args.prompts),
            "count": len(prompt_lines),
        },
        "maps": map_audits,
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
        "implementation_hashes": {
            str(path.relative_to(args.repo_root)): sha256(path)
            for path in implementation_paths
        },
    }
    contract_path = args.out_root / "contracts" / "experiment.json"
    if args.node_rank == 0:
        contract_sha256 = write_frozen(contract_path, contract)
    else:
        contract_sha256 = wait_for_frozen(
            contract_path,
            contract,
            timeout_seconds=args.contract_wait_seconds,
        )

    print(
        "[v101] "
        f"node={args.node_rank}/4 interval=[{start},{end}) "
        f"candidate={args.candidate_support}+{args.candidate_suppress}"
        f"+transition:{args.candidate_transition} "
        f"methods={[method.name for method in methods]}",
        flush=True,
    )
    failures: list[str] = []
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_method_shard,
                args,
                method=method,
                head_map=map_paths[method.map_key],
                gpu=gpus[index],
                start=start,
                end=end,
                contract_sha256=contract_sha256,
            ): method
            for index, method in enumerate(methods)
        }
        for future in as_completed(futures):
            method = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[done] {result}", flush=True)
            except Exception as error:
                message = f"{method.name}: {error}"
                failures.append(message)
                print(f"[failed] {message}", flush=True)

    summary = {
        "version": 1,
        "experiment": "v101_paper_ablation_128",
        "node_rank": int(args.node_rank),
        "interval": [int(start), int(end)],
        "contract_sha256": contract_sha256,
        "results": sorted(results, key=lambda item: str(item["method"])),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root / "status" / f"node{args.node_rank}.summary.json"
    )
    write_frozen(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
