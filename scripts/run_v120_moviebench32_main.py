#!/usr/bin/env python3
"""Run SF, PF, and promoted binary-cache candidates on MovieBench-32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from run_v100_fast_selection_1video import (
    Cell,
    audit_video,
    canonical_json,
    run_cell,
    run_checked,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)
from run_v115_role_memory_cache_1video import CELLS as V115_CELLS
from run_v119_candidate_refinement_1video import CELLS as V119_CELLS


EXPERIMENT = "v120_moviebench32_main"
PROMPT_COUNT = 32
DEFAULT_CANDIDATES = ("landmark_motion1",)


@dataclass(frozen=True)
class Method:
    key: str
    engine: str
    source_cell: Cell | None
    role: str


_CELLS_BY_NAME = {
    cell.name: cell for cell in (*V115_CELLS, *V119_CELLS)
}
_CANDIDATE_SPECS = {
    "landmark_motion1": (
        "legacy_v98_support_landmark4_suppress_motion_pair1",
        "v116_balanced_candidate",
    ),
    "prototype_motion1": (
        "legacy_v98_support_prototype4_suppress_motion_pair1",
        "v116_story_candidate",
    ),
    "landmark_retrieval2": (
        "legacy_v98_support_landmark4_suppress_retrieval2",
        "v116_retrieval_candidate",
    ),
    "landmark_retrieval1": (
        "legacy_v98_landmark4_retrieval1",
        "v119_retrieval_top1",
    ),
    "landmark_retrieval1_age24": (
        "legacy_v98_landmark4_retrieval1_age24",
        "v119_bounded_retrieval",
    ),
    "landmark_retrieval_motion": (
        "legacy_v98_landmark4_retrieval1_motion1_age24",
        "v119_bounded_retrieval_motion",
    ),
    "landmark_motion1_sink3_budget9": (
        "legacy_v98_landmark2_motion1_sink3_budget9",
        "v119_budget_matched_sink3",
    ),
}


def parse_candidate_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not keys:
        raise ValueError("at least one ours candidate is required")
    if len(keys) > 2:
        raise ValueError("MovieBench-32 main allows at most two ours candidates")
    if len(keys) != len(set(keys)):
        raise ValueError("candidate list contains duplicates")
    unknown = sorted(set(keys) - set(_CANDIDATE_SPECS))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    return keys


def methods_for(candidate_keys: tuple[str, ...]) -> tuple[Method, ...]:
    methods = [
        Method("sf_native", "sf", None, "baseline"),
        Method(
            "pf_native",
            "pf",
            Cell(
                "pf_native_source",
                "baseline",
                "single",
                suppress_policy=None,
                map_key="pf",
            ),
            "baseline",
        ),
    ]
    methods.extend(
        Method(
            f"ours_{key}",
            "pf",
            _CELLS_BY_NAME[_CANDIDATE_SPECS[key][0]],
            _CANDIDATE_SPECS[key][1],
        )
        for key in candidate_keys
    )
    return tuple(methods)


def task_name(method: Method, prompt_index: int) -> str:
    return f"{method.key}__p{int(prompt_index):03d}"


def task_cell(method: Method, prompt_index: int) -> Cell:
    if method.source_cell is None:
        return Cell(
            task_name(method, prompt_index),
            "moviebench32",
            "single",
            suppress_policy=None,
            map_key="sf",
        )
    return replace(
        method.source_cell,
        name=task_name(method, prompt_index),
        stage="moviebench32",
    )


def all_tasks(
    methods: tuple[Method, ...],
) -> tuple[tuple[Method, int, Cell], ...]:
    return tuple(
        (method, prompt_index, task_cell(method, prompt_index))
        for method in methods
        for prompt_index in range(PROMPT_COUNT)
    )


def selected_tasks(
    methods: tuple[Method, ...],
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[tuple[Method, int, Cell], ...]:
    return all_tasks(methods)[node_rank::num_nodes]


def published_name(prompt_index: int, *, indexed: bool = False) -> str:
    if indexed:
        return f"{int(prompt_index):06d}-0_v120.mp4"
    return f"{int(prompt_index):06d}.mp4"


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(
                f"refusing mixed published video {target}; source={source}"
            )
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


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
        "--candidates",
        default=os.environ.get(
            "V120_CANDIDATES",
            ",".join(DEFAULT_CANDIDATES),
        ),
    )
    parser.add_argument("--list-candidates", action="store_true")
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
    parser.add_argument("--sf-repo", type=Path, default=env_path("SF_REPO"))
    parser.add_argument("--sf-config", type=Path, default=env_path("SF_CONFIG"))
    parser.add_argument(
        "--sf-checkpoint",
        type=Path,
        default=env_path("SF_CHECKPOINT"),
    )
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
        "--prompts",
        type=Path,
        default=env_path("PROMPTS"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--promotion-approved",
        action="store_true",
        default=os.environ.get("V119_PROMOTION_APPROVED", "0") == "1",
        help="confirm that a v119 one-video candidate was manually approved",
    )
    args = parser.parse_args()

    if args.list_candidates:
        for key, (cell_name, role) in _CANDIDATE_SPECS.items():
            print(f"{key}\t{role}\t{cell_name}")
        raise SystemExit(0)
    try:
        args.candidate_keys = parse_candidate_keys(args.candidates)
    except ValueError as error:
        parser.error(str(error))

    args.repo_root = args.repo_root.resolve()
    args.sf_repo = (
        args.sf_repo or args.repo_root / "third_party" / "Self-Forcing"
    ).resolve()
    args.sf_config = (
        args.sf_config or args.sf_repo / "configs" / "self_forcing_dmd.yaml"
    ).resolve()
    args.sf_checkpoint = (
        args.sf_checkpoint
        or args.sf_repo / "checkpoints" / "self_forcing_dmd.pt"
    ).resolve()
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
    args.prompts = (
        args.prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num32.txt"
    ).resolve()
    args.single_prompts = args.prompts
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.aba_prompt_index = 0
    candidate_digest = hashlib.sha256(
        ",".join(args.candidate_keys).encode("utf-8")
    ).hexdigest()[:12]
    args.method_set_id = (
        f"ours{len(args.candidate_keys)}_{candidate_digest}"
    )
    args.out_root = (
        args.out_root
        or args.repo_root / "runs" / EXPERIMENT / args.method_set_id
    ).resolve()
    args.experiment_name = EXPERIMENT
    return args


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v120 main table requires seed 0")
    if args.mode == "generate" and not args.promotion_approved:
        raise SystemExit(
            "v120 is gated by v119 one-video review; set "
            "V119_PROMOTION_APPROVED=1 after recording the decision"
        )
    required = (
        args.sf_repo / "inference.py",
        args.sf_config,
        args.sf_checkpoint,
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(
            f"v120 requires exactly {PROMPT_COUNT} prompts, found {len(prompts)}"
        )
    return prompts


def experiment_contract(
    args: argparse.Namespace,
    *,
    methods: tuple[Method, ...],
    prompts: list[str],
    map_audit: dict[str, object],
) -> dict[str, object]:
    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.repo_root / "scripts" / "run_v115_role_memory_cache_1video.py",
        args.repo_root / "scripts" / "run_v119_candidate_refinement_1video.py",
        args.legacy_map,
        args.sf_repo / "inference.py",
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "base.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "role_event.py",
        args.pf_repo / "pyramidkv" / "role_memory.py",
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "candidate_keys": list(args.candidate_keys),
        "seed": 0,
        "prompt_count": PROMPT_COUNT,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "methods": [
            {
                "key": method.key,
                "engine": method.engine,
                "role": method.role,
                "source_cell": (
                    None
                    if method.source_cell is None
                    else asdict(method.source_cell)
                ),
            }
            for method in methods
        ],
        "prompts": {
            "path": str(args.prompts),
            "sha256": sha256(args.prompts),
            "items": [
                {"index": index, "text": text}
                for index, text in enumerate(prompts)
            ],
        },
        "head_membership": {
            "supportive": 304,
            "suppressive": 56,
            "map_sha256": EXPECTED_MAP_SHA256,
        },
        "map_audit": map_audit,
        "vbench_long_dimensions": [
            "subject_consistency",
            "background_consistency",
            "aesthetic_quality",
            "imaging_quality",
            "motion_smoothness",
            "dynamic_degree",
        ],
        "sf": {
            "repo": str(args.sf_repo),
            "config": str(args.sf_config),
            "config_sha256": sha256(args.sf_config),
            "checkpoint": str(args.sf_checkpoint),
            "checkpoint_size": args.sf_checkpoint.stat().st_size,
        },
        "pf": {
            "repo": str(args.pf_repo),
            "config": str(args.pf_config),
            "config_sha256": sha256(args.pf_config),
            "checkpoint": str(args.pf_checkpoint),
            "checkpoint_size": args.pf_checkpoint.stat().st_size,
            "labels": str(args.pf_labels),
            "labels_sha256": sha256(args.pf_labels),
        },
        "implementation_hashes": {
            str(path.relative_to(args.repo_root)): sha256(path)
            for path in implementation_paths
        },
    }


def native_sf_environment(args: argparse.Namespace, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    prefixes = (
        "LIFECACHE_",
        "HEAD_ROLE_",
        "STRUCTURED_MEMORY_",
        "COMMIT_FORCING_",
        "PYRAMIDKV_",
        "CEMR_",
        "PROBECACHE_",
    )
    for key in list(env):
        if key.startswith(prefixes):
            env.pop(key, None)
    root_python = str(args.repo_root / "src")
    env["PYTHONPATH"] = (
        root_python
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["COMMIT_FORCING_ENABLE"] = "0"
    return env


def run_sf_task(
    args: argparse.Namespace,
    *,
    method: Method,
    prompt_index: int,
    cell: Cell,
    gpu: str,
    contract_sha256: str,
) -> dict[str, str]:
    output = args.out_root / "videos" / cell.name
    log = args.out_root / "logs" / f"{cell.name}.log"
    config_path = args.out_root / "configs" / f"{cell.name}.json"
    marker = args.out_root / "status" / f"{cell.name}.done.json"
    video_report = args.out_root / "diagnostics" / f"{cell.name}.video.json"
    video_log = video_report.with_suffix(".log")
    config = {
        "version": 1,
        "experiment": EXPERIMENT,
        "experiment_contract_sha256": contract_sha256,
        "method": method.key,
        "engine": "sf",
        "cell": asdict(cell),
        "gpu": str(gpu),
        "prompt_index": int(prompt_index),
        "prompt_path": str(args.prompts),
        "prompt_sha256": sha256(args.prompts),
        "config_path": str(args.sf_config),
        "config_sha256": sha256(args.sf_config),
        "checkpoint_path": str(args.sf_checkpoint),
        "checkpoint_size": args.sf_checkpoint.stat().st_size,
        "num_output_frames": 120,
        "seed": 0,
        "reseed_per_prompt": True,
        "native_environment_scrubbed": True,
    }
    config_sha = write_frozen(config_path, config)
    task_args = argparse.Namespace(**vars(args))
    task_args.single_prompt_index = int(prompt_index)

    if marker.is_file():
        frozen = json.loads(marker.read_text(encoding="utf-8"))
        if (
            frozen.get("config_sha256") != config_sha
            or frozen.get("experiment_contract_sha256") != contract_sha256
            or not log.is_file()
        ):
            raise RuntimeError(f"stale SF completion marker: {marker}")
        video = audit_video(
            task_args,
            cell=cell,
            output=output,
            report=video_report,
            log=video_log,
        )
        if video.get("input_fingerprint") != frozen.get(
            "video_input_fingerprint"
        ):
            raise RuntimeError(f"SF video fingerprint changed: {cell.name}")
        return {"name": cell.name, "status": "resumed"}

    if output.exists() and any(output.rglob("*.mp4")):
        raise RuntimeError(
            f"{cell.name}: partial SF videos exist without a marker; "
            "use a fresh OUT_ROOT"
        )
    output.mkdir(parents=True, exist_ok=True)
    for stale in (log, video_report, video_log):
        if stale.exists():
            raise RuntimeError(
                f"{cell.name}: stale SF artifact exists without marker: {stale}"
            )
    command = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(args.sf_config),
        "--checkpoint_path",
        str(args.sf_checkpoint),
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
        str(prompt_index),
        "--end_idx",
        str(prompt_index + 1),
        "--reseed_per_prompt",
    ]
    run_checked(
        command,
        cwd=args.sf_repo,
        env=native_sf_environment(args, gpu),
        log_path=log,
    )
    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
    )
    if any(signature in log_text for signature in failure_signatures):
        raise RuntimeError(f"failure signature found in {log}")
    video = audit_video(
        task_args,
        cell=cell,
        output=output,
        report=video_report,
        log=video_log,
    )
    write_frozen(
        marker,
        {
            "version": 1,
            "experiment_contract_sha256": contract_sha256,
            "config_sha256": config_sha,
            "video_input_fingerprint": video["input_fingerprint"],
        },
    )
    return {"name": cell.name, "status": "generated"}


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
        / published_name(prompt_index)
    )
    indexed_target = (
        args.out_root
        / "published_indexed"
        / method.key
        / published_name(prompt_index, indexed=True)
    )
    link_mode = link_or_validate(source, target)
    indexed_link_mode = link_or_validate(source, indexed_target)
    marker = (
        args.out_root
        / "status"
        / "published"
        / f"{method.key}.p{prompt_index:03d}.json"
    )
    write_frozen(
        marker,
        {
            "version": 1,
            "experiment_contract_sha256": contract_sha256,
            "method": method.key,
            "engine": method.engine,
            "prompt_index": int(prompt_index),
            "task_cell": cell.name,
            "source": str(source),
            "target": str(target),
            "indexed_target": str(indexed_target),
            "size": source.stat().st_size,
        },
    )
    return {
        "method": method.key,
        "prompt_index": int(prompt_index),
        "status": link_mode,
        "indexed_status": indexed_link_mode,
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
    if method.engine == "sf":
        generated = run_sf_task(
            task_args,
            method=method,
            prompt_index=prompt_index,
            cell=cell,
            gpu=gpu,
            contract_sha256=contract_sha256,
        )
    else:
        generated = run_cell(
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
        "generation_status": generated["status"],
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
            f"[v120-task] gpu={gpu} method={method.key} "
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
    methods: tuple[Method, ...],
    contract_sha256: str,
) -> dict[str, object]:
    failures: list[str] = []
    method_rows: list[dict[str, object]] = []
    expected_names = {
        published_name(index) for index in range(PROMPT_COUNT)
    }
    expected_indexed_names = {
        published_name(index, indexed=True) for index in range(PROMPT_COUNT)
    }
    for method in methods:
        method_dir = args.out_root / "published" / method.key
        indexed_dir = args.out_root / "published_indexed" / method.key
        observed = sorted(method_dir.glob("*.mp4"))
        observed_indexed = sorted(indexed_dir.glob("*.mp4"))
        actual_names = {path.name for path in observed}
        actual_indexed_names = {path.name for path in observed_indexed}
        if actual_names != expected_names:
            failures.append(
                f"{method.key}: missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        if actual_indexed_names != expected_indexed_names:
            failures.append(
                f"{method.key}: indexed_missing="
                f"{sorted(expected_indexed_names - actual_indexed_names)} "
                f"indexed_extra="
                f"{sorted(actual_indexed_names - expected_indexed_names)}"
            )
        sizes = []
        for prompt_index in range(PROMPT_COUNT):
            target = method_dir / published_name(prompt_index)
            indexed_target = indexed_dir / published_name(
                prompt_index,
                indexed=True,
            )
            marker = (
                args.out_root
                / "status"
                / "published"
                / f"{method.key}.p{prompt_index:03d}.json"
            )
            if (
                not target.is_file()
                or not indexed_target.is_file()
                or not marker.is_file()
            ):
                failures.append(
                    f"{method.key}:{prompt_index}: missing artifact"
                )
                continue
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                source = Path(payload["source"])
                if (
                    payload["experiment_contract_sha256"] != contract_sha256
                    or payload["method"] != method.key
                    or payload["engine"] != method.engine
                    or int(payload["prompt_index"]) != prompt_index
                    or Path(payload["target"]) != target
                    or Path(payload["indexed_target"]) != indexed_target
                    or not source.is_file()
                    or not target.samefile(source)
                    or not indexed_target.samefile(source)
                    or int(payload["size"]) != target.stat().st_size
                    or indexed_target.stat().st_size
                    != target.stat().st_size
                ):
                    failures.append(
                        f"{method.key}:{prompt_index}: mixed marker"
                    )
                    continue
            except (KeyError, TypeError, ValueError, OSError) as error:
                failures.append(
                    f"{method.key}:{prompt_index}: invalid marker: {error}"
                )
                continue
            sizes.append(target.stat().st_size)
        method_rows.append(
            {
                "key": method.key,
                "engine": method.engine,
                "role": method.role,
                "video_dir": str(method_dir),
                "indexed_video_dir": str(indexed_dir),
                "video_count": len(observed),
                "indexed_video_count": len(observed_indexed),
                "total_bytes": sum(sizes),
            }
        )
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "experiment_contract_sha256": contract_sha256,
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
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
    prompts = load_prompts(args)
    methods = methods_for(args.candidate_keys)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "status/published",
        "diagnostics",
        "contracts",
        "videos",
        "published",
        "published_indexed",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)

    map_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    if map_audit["sha256"] != EXPECTED_MAP_SHA256:
        raise SystemExit("legacy v98 map changed after validation")
    args.head_map_audits = {
        "legacy": map_audit,
        "pf": {
            "path": str(args.pf_labels),
            "sha256": sha256(args.pf_labels),
            "kind": "pf_native_labels",
        },
    }
    contract = experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audit,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            args.contract_wait_seconds,
        )
    print(
        "[V120Contract] "
        + canonical_json(
            {
                "method_set_id": args.method_set_id,
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "map_sha256": map_audit["sha256"],
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )

    if args.mode == "audit":
        payload = audit_published(
            args,
            methods=methods,
            contract_sha256=contract_sha,
        )
        print(
            f"[complete] published methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    tasks = list(
        selected_tasks(
            methods,
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
        f"[v120] node={args.node_rank}/{args.num_nodes} "
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
        "method_set_id": args.method_set_id,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": len(tasks),
        "results": sorted(
            results,
            key=lambda item: (
                str(item["method"]),
                int(item["prompt_index"]),
            ),
        ),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root / "status" / f"node{args.node_rank}.summary.json"
    )
    write_runtime_json(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
