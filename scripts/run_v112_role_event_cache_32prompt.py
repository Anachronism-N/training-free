#!/usr/bin/env python3
"""Promote one v111 cache candidate to an auditable 32-prompt screen."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    run_cell,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)
from run_v111_role_event_cache_1video import CELLS as V111_CELLS


EXPERIMENT = "v112_role_event_cache_32prompt"
PROMPT_COUNT = 32


@dataclass(frozen=True)
class Method:
    key: str
    source_cell: Cell
    role: str


_V111_BY_NAME = {cell.name: cell for cell in V111_CELLS}
_CANDIDATES = {
    "support_landmark_suppress_recent": (
        "legacy_v98_support_landmark4_suppress_recent8"
    ),
    "support_hybrid_suppress_recent": (
        "legacy_v98_support_landmark2_motion1_suppress_recent8"
    ),
    "support_recent_suppress_motion": (
        "legacy_v98_support_recent8_suppress_motion_pair2"
    ),
    "support_landmark_suppress_motion": (
        "legacy_v98_support_landmark4_suppress_motion_pair2"
    ),
    "support_hybrid_suppress_motion": (
        "legacy_v98_support_landmark2_motion1_suppress_motion_pair2"
    ),
}
_CONTROLS = (
    (
        "control_all_recent8",
        "legacy_v98_all_recent8_control",
    ),
    (
        "control_all_landmark4",
        "legacy_v98_all_landmark4_control",
    ),
    (
        "control_all_motion_pair2",
        "legacy_v98_all_motion_pair2_control",
    ),
)


def methods_for(candidate: str, suite: str) -> tuple[Method, ...]:
    candidate_method = Method(
        key=f"candidate_{candidate}",
        source_cell=_V111_BY_NAME[_CANDIDATES[candidate]],
        role="promoted_candidate",
    )
    controls = tuple(
        Method(
            key=key,
            source_cell=_V111_BY_NAME[source],
            role="role_neutral_control",
        )
        for key, source in _CONTROLS
    )
    if suite == "minimal":
        return (candidate_method, controls[0])
    return (candidate_method, *controls)


def task_cell(method: Method, prompt_index: int) -> Cell:
    return replace(
        method.source_cell,
        name=f"{method.key}__p{prompt_index:03d}",
        stage="screen32",
    )


def all_tasks(
    candidate: str,
    suite: str,
) -> tuple[tuple[Method, int, Cell], ...]:
    return tuple(
        (method, prompt_index, task_cell(method, prompt_index))
        for method in methods_for(candidate, suite)
        for prompt_index in range(PROMPT_COUNT)
    )


def selected_tasks(
    candidate: str,
    suite: str,
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[tuple[Method, int, Cell], ...]:
    return all_tasks(candidate, suite)[node_rank::num_nodes]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("generate", "audit"),
        default="generate",
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(_CANDIDATES),
        required=True,
    )
    parser.add_argument(
        "--suite",
        choices=("minimal", "full"),
        default="full",
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
        "--pf-checkpoint",
        type=Path,
        default=env_path("PF_CHECKPOINT"),
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--legacy-map",
        type=Path,
        default=env_path("LEGACY_V98_MAP"),
    )
    parser.add_argument(
        "--single-prompts",
        type=Path,
        default=env_path("SINGLE_PROMPTS"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--promotion-approved",
        action="store_true",
        default=os.environ.get("V111_PROMOTION_APPROVED", "0") == "1",
        help="confirm that the v111 one-video blind review selected this candidate",
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
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv"
    ).resolve()
    args.single_prompts = (
        args.single_prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num32.txt"
    ).resolve()
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.out_root = (
        args.out_root
        or args.repo_root
        / "runs"
        / EXPERIMENT
        / args.candidate
    ).resolve()
    args.experiment_name = EXPERIMENT
    return args


def validate_inputs(args: argparse.Namespace) -> list[str]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v112 screen requires seed 0")
    if args.mode == "generate" and not args.promotion_approved:
        raise SystemExit(
            "v112 is gated by the v111 one-video review; set "
            "V111_PROMOTION_APPROVED=1 after recording the decision"
        )
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.single_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    prompts = [
        line.strip()
        for line in args.single_prompts.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(
            f"v112 requires exactly {PROMPT_COUNT} prompts, found "
            f"{len(prompts)} in {args.single_prompts}"
        )
    return prompts


def experiment_contract(
    args: argparse.Namespace,
    prompts: list[str],
    map_audit: dict,
) -> dict:
    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v111_role_event_cache_1video.py",
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.legacy_map,
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "base.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "role_event.py",
    )
    methods = methods_for(args.candidate, args.suite)
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "candidate": args.candidate,
        "suite": args.suite,
        "seed": 0,
        "prompt_count": PROMPT_COUNT,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "head_membership": {
            "supportive": 304,
            "suppressive": 56,
            "map_sha256": EXPECTED_MAP_SHA256,
        },
        "cache_contract": {
            "forbidden_candidate_middle": ["stride", "cyclic", "merge"],
            "max_full_frame_equivalents": 9,
            "clean_kv_only": True,
            "exclusive_dynamic_owner": True,
        },
        "methods": [
            {
                "key": method.key,
                "role": method.role,
                "source_cell": asdict(method.source_cell),
            }
            for method in methods
        ],
        "prompts": {
            "path": str(args.single_prompts),
            "sha256": sha256(args.single_prompts),
            "items": prompts,
        },
        "map_audit": map_audit,
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


def publish_video(
    args: argparse.Namespace,
    *,
    method: Method,
    prompt_index: int,
    cell: Cell,
    contract_sha256: str,
) -> dict[str, object]:
    source_dir = args.out_root / "videos" / cell.name
    videos = sorted(source_dir.rglob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(
            f"{cell.name}: expected one source video, found {len(videos)}"
        )
    source = videos[0]
    target = (
        args.out_root
        / "published"
        / method.key
        / f"{prompt_index:06d}.mp4"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(
                f"refusing mixed published video {target}; source={source}"
            )
        link_mode = "existing"
    else:
        try:
            os.link(source, target)
            link_mode = "hardlink"
        except OSError:
            target.symlink_to(source.resolve())
            link_mode = "symlink"
    marker = (
        args.out_root
        / "status"
        / "published"
        / f"{method.key}.p{prompt_index:03d}.json"
    )
    payload = {
        "version": 1,
        "experiment_contract_sha256": contract_sha256,
        "method": method.key,
        "prompt_index": prompt_index,
        "task_cell": cell.name,
        "source": str(source),
        "target": str(target),
        "size": source.stat().st_size,
    }
    write_frozen(marker, payload)
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
    }


def run_task(
    args: argparse.Namespace,
    *,
    method: Method,
    prompt_index: int,
    cell: Cell,
    gpu: str,
    contract_sha256: str,
) -> dict[str, object]:
    task_args = argparse.Namespace(**vars(args))
    task_args.single_prompt_index = int(prompt_index)
    run_result = run_cell(
        task_args,
        cell=cell,
        gpu=gpu,
        experiment_contract_sha256=contract_sha256,
    )
    published = publish_video(
        args,
        method=method,
        prompt_index=prompt_index,
        cell=cell,
        contract_sha256=contract_sha256,
    )
    return {
        **published,
        "generation_status": run_result["status"],
        "gpu": str(gpu),
    }


def run_worker(
    args: argparse.Namespace,
    *,
    gpu: str,
    tasks: list[tuple[Method, int, Cell]],
    contract_sha256: str,
) -> list[dict[str, object]]:
    results = []
    for method, prompt_index, cell in tasks:
        print(
            f"[v112-task] gpu={gpu} method={method.key} "
            f"prompt={prompt_index}",
            flush=True,
        )
        results.append(
            run_task(
                args,
                method=method,
                prompt_index=prompt_index,
                cell=cell,
                gpu=gpu,
                contract_sha256=contract_sha256,
            )
        )
    return results


def audit_published(
    args: argparse.Namespace,
    *,
    contract_sha256: str,
) -> dict[str, object]:
    failures: list[str] = []
    method_rows: list[dict[str, object]] = []
    for method in methods_for(args.candidate, args.suite):
        method_dir = args.out_root / "published" / method.key
        observed = sorted(method_dir.glob("*.mp4"))
        expected_names = {
            f"{prompt_index:06d}.mp4"
            for prompt_index in range(PROMPT_COUNT)
        }
        actual_names = {path.name for path in observed}
        if actual_names != expected_names:
            failures.append(
                f"{method.key}: missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        sizes = []
        for prompt_index in range(PROMPT_COUNT):
            target = method_dir / f"{prompt_index:06d}.mp4"
            marker = (
                args.out_root
                / "status"
                / "published"
                / f"{method.key}.p{prompt_index:03d}.json"
            )
            if not target.is_file():
                failures.append(
                    f"{method.key}:{prompt_index}: missing published video"
                )
                continue
            if not marker.is_file():
                failures.append(
                    f"{method.key}:{prompt_index}: missing publication marker"
                )
                continue
            try:
                marker_payload = json.loads(
                    marker.read_text(encoding="utf-8")
                )
                source = Path(marker_payload["source"])
                if (
                    marker_payload["experiment_contract_sha256"]
                    != contract_sha256
                    or marker_payload["method"] != method.key
                    or int(marker_payload["prompt_index"]) != prompt_index
                    or Path(marker_payload["target"]) != target
                    or not source.is_file()
                    or not target.samefile(source)
                    or int(marker_payload["size"]) != target.stat().st_size
                ):
                    failures.append(
                        f"{method.key}:{prompt_index}: mixed publication marker"
                    )
                    continue
            except (KeyError, TypeError, ValueError, OSError) as error:
                failures.append(
                    f"{method.key}:{prompt_index}: invalid publication "
                    f"marker: {error}"
                )
                continue
            sizes.append(target.stat().st_size)
        if sizes and min(sizes) <= 0:
            failures.append(f"{method.key}: zero-byte published video")
        method_rows.append(
            {
                "key": method.key,
                "role": method.role,
                "video_dir": str(method_dir),
                "video_count": len(observed),
                "total_bytes": sum(sizes),
            }
        )
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "candidate": args.candidate,
        "suite": args.suite,
        "experiment_contract_sha256": contract_sha256,
        "methods": method_rows,
        "failures": failures,
        "ok": not failures,
    }
    if failures:
        write_runtime_json(
            args.out_root / "status" / "published.audit.failed.json",
            payload,
        )
        raise RuntimeError(f"published audit failed: {failures[:5]}")
    write_frozen(args.out_root / "published_manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    prompts = validate_inputs(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "diagnostics",
        "contracts",
        "published",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)

    map_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    if map_audit["sha256"] != EXPECTED_MAP_SHA256:
        raise SystemExit("legacy v98 map changed after validation")
    args.head_map_audits = {"legacy": map_audit}
    contract = experiment_contract(args, prompts, map_audit)
    contract_path = (
        args.out_root
        / "contracts"
        / f"{args.suite}.{args.candidate}.json"
    )
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            args.contract_wait_seconds,
        )
    print(
        "[V112Contract] "
        + canonical_json(
            {
                "candidate": args.candidate,
                "suite": args.suite,
                "contract_sha256": contract_sha,
                "map_sha256": map_audit["sha256"],
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )

    if args.mode == "audit":
        payload = audit_published(args, contract_sha256=contract_sha)
        print(
            f"[complete] published methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    tasks = list(
        selected_tasks(
            args.candidate,
            args.suite,
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
        )
    )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    worker_tasks = [tasks[index::len(gpus)] for index in range(len(gpus))]
    worker_tasks = [items for items in worker_tasks if items]
    print(
        f"[v112] node={args.node_rank}/{args.num_nodes} "
        f"tasks={len(tasks)} workers={len(worker_tasks)} "
        f"out={args.out_root}",
        flush=True,
    )

    failures: list[str] = []
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(worker_tasks))) as executor:
        futures = {
            executor.submit(
                run_worker,
                args,
                gpu=gpus[index],
                tasks=items,
                contract_sha256=contract_sha,
            ): gpus[index]
            for index, items in enumerate(worker_tasks)
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as error:
                failures.append(f"gpu={gpu}: {error}")
                print(f"[failed] gpu={gpu}: {error}", flush=True)

    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "candidate": args.candidate,
        "suite": args.suite,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": len(tasks),
        "results": sorted(
            results,
            key=lambda item: (str(item["method"]), int(item["prompt_index"])),
        ),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root
        / "status"
        / f"node{args.node_rank}.summary.json"
    )
    write_runtime_json(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
