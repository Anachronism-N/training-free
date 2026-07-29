#!/usr/bin/env python3
"""Run PF-table-style external baselines with deterministic prompt shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from audit_indexed_videos import _probe_video
from run_v100_fast_selection_1video import (
    run_checked,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v120_moviebench32_main import link_or_validate


PROMPT_COUNT = 128
DEFAULT_PROMPTS = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
    "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
METHOD_ORDER = ("deep_forcing", "rolling_forcing", "longlive")
COMPARISON_CLASSES = {
    "deep_forcing": "same_checkpoint_external_method",
    "rolling_forcing": "external_trained_system",
    "longlive": "external_trained_system",
}


@dataclass(frozen=True)
class Method:
    key: str
    repo: Path
    config: Path
    checkpoints: tuple[Path, ...]
    wan_model_dir: Path


def parse_method_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not keys:
        raise ValueError("at least one external method is required")
    if len(keys) != len(set(keys)):
        raise ValueError("external method list contains duplicates")
    unknown = sorted(set(keys) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"unknown external methods: {unknown}")
    return tuple(key for key in METHOD_ORDER if key in keys)


def interval_for_worker(
    worker_rank: int,
    worker_count: int,
    item_count: int = PROMPT_COUNT,
) -> tuple[int, int]:
    if worker_count <= 0 or not 0 <= worker_rank < worker_count:
        raise ValueError("invalid worker rank/count")
    return (
        item_count * worker_rank // worker_count,
        item_count * (worker_rank + 1) // worker_count,
    )


def write_frozen_text(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    digest = __import__("hashlib").sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen text artifact differs: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return digest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("preflight", "generate", "audit"),
        default="generate",
    )
    parser.add_argument(
        "--duration",
        type=int,
        choices=(30, 60),
        default=int(os.environ.get("V129_DURATION_SECONDS", "30")),
    )
    parser.add_argument(
        "--methods",
        default=os.environ.get(
            "V129_EXTERNAL_METHODS",
            ",".join(METHOD_ORDER),
        ),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(os.environ.get("V129_PROMPTS", DEFAULT_PROMPTS)),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", root)),
    )
    parser.add_argument("--out-root", type=Path, default=env_path("OUT_ROOT"))
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument("--deep-repo", type=Path, default=env_path("DEEP_REPO"))
    parser.add_argument(
        "--deep-config",
        type=Path,
        default=env_path("DEEP_CONFIG"),
    )
    parser.add_argument(
        "--deep-checkpoint",
        type=Path,
        default=env_path("DEEP_CHECKPOINT"),
    )
    parser.add_argument(
        "--rolling-repo",
        type=Path,
        default=env_path("ROLLING_REPO"),
    )
    parser.add_argument(
        "--rolling-config",
        type=Path,
        default=env_path("ROLLING_CONFIG"),
    )
    parser.add_argument(
        "--rolling-checkpoint",
        type=Path,
        default=env_path("ROLLING_CHECKPOINT"),
    )
    parser.add_argument(
        "--longlive-repo",
        type=Path,
        default=env_path("LONGLIVE_REPO"),
    )
    parser.add_argument(
        "--longlive-config",
        type=Path,
        default=env_path("LONGLIVE_CONFIG"),
    )
    parser.add_argument(
        "--longlive-generator",
        type=Path,
        default=env_path("LONGLIVE_GENERATOR"),
    )
    parser.add_argument(
        "--longlive-lora",
        type=Path,
        default=env_path("LONGLIVE_LORA"),
    )
    args = parser.parse_args()
    try:
        args.method_keys = parse_method_keys(args.methods)
    except ValueError as error:
        parser.error(str(error))

    args.repo_root = args.repo_root.resolve()
    args.prompts = args.prompts.resolve()
    args.deep_repo = (
        args.deep_repo or args.repo_root / "third_party" / "DeepForcing"
    ).resolve()
    args.deep_config = (
        args.deep_config
        or args.deep_repo
        / "configs"
        / "self_forcing_dmd"
        / "self_forcing_dmd_sink10.yaml"
    ).resolve()
    args.deep_checkpoint = (
        args.deep_checkpoint
        or args.deep_repo / "checkpoints" / "self_forcing_dmd.pt"
    ).resolve()
    args.rolling_repo = (
        args.rolling_repo
        or args.repo_root / "third_party" / "RollingForcing"
    ).resolve()
    args.rolling_config = (
        args.rolling_config
        or args.rolling_repo / "configs" / "rolling_forcing_dmd.yaml"
    ).resolve()
    args.rolling_checkpoint = (
        args.rolling_checkpoint
        or args.rolling_repo / "checkpoints" / "rolling_forcing_dmd.pt"
    ).resolve()
    args.longlive_repo = (
        args.longlive_repo
        or args.repo_root / "third_party" / "LongLive-RAG"
    ).resolve()
    args.longlive_config = (
        args.longlive_config
        or args.longlive_repo / "configs" / "longlive_native.yaml"
    ).resolve()
    args.longlive_generator = (
        args.longlive_generator
        or args.longlive_repo / "checkpoints" / "longlive_base.pt"
    ).resolve()
    args.longlive_lora = (
        args.longlive_lora
        or args.longlive_repo / "checkpoints" / "longlive_lora.pt"
    ).resolve()
    args.num_output_frames = 120 if args.duration == 30 else 240
    args.expected_video_frames = 4 * args.num_output_frames - 3
    args.out_root = (
        args.out_root
        or args.repo_root
        / "runs"
        / f"v129_moviebench128_{args.duration}s_external"
    ).resolve()
    return args


def methods_for(args: argparse.Namespace) -> tuple[Method, ...]:
    methods = {
        "deep_forcing": Method(
            "deep_forcing",
            args.deep_repo,
            args.deep_config,
            (args.deep_checkpoint,),
            args.deep_repo / "wan_models" / "Wan2.1-T2V-1.3B",
        ),
        "rolling_forcing": Method(
            "rolling_forcing",
            args.rolling_repo,
            args.rolling_config,
            (args.rolling_checkpoint,),
            args.rolling_repo / "wan_models" / "Wan2.1-T2V-1.3B",
        ),
        "longlive": Method(
            "longlive",
            args.longlive_repo,
            args.longlive_config,
            (args.longlive_generator, args.longlive_lora),
            args.longlive_repo / "wan_models" / "Wan2.1-T2V-1.3B",
        ),
    }
    return tuple(methods[key] for key in args.method_keys)


def validate_inputs(
    args: argparse.Namespace,
    methods: tuple[Method, ...],
) -> list[str]:
    if args.seed != 0:
        raise SystemExit("v129 comparison requires seed 0")
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("require 0 <= NODE_RANK < NUM_NODES")
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    required = [args.prompts]
    for method in methods:
        required.extend(
            [
                method.repo / "inference.py",
                method.config,
                *method.checkpoints,
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(
        str(method.wan_model_dir)
        for method in methods
        if not method.wan_model_dir.is_dir()
    )
    if missing:
        raise SystemExit(
            "missing required external-baseline assets:\n  "
            + "\n  ".join(missing)
        )
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(
            f"expected {PROMPT_COUNT} prompts, found {len(prompts)}"
        )
    return prompts


def experiment_contract(
    args: argparse.Namespace,
    methods: tuple[Method, ...],
    prompts: list[str],
) -> dict[str, Any]:
    return {
        "version": 1,
        "experiment": f"v129_moviebench128_{args.duration}s_external",
        "duration_seconds_target": args.duration,
        "num_output_frames": args.num_output_frames,
        "decoded_video_contract": {
            "frames": args.expected_video_frames,
            "fps": 16,
            "duration_seconds": args.expected_video_frames / 16.0,
            "width": 832,
            "height": 480,
        },
        "seed": args.seed,
        "reseed_per_prompt": True,
        "prompt_count": PROMPT_COUNT,
        "prompts": {
            "path": str(args.prompts),
            "sha256": sha256(args.prompts),
            "items": [
                {"index": index, "text": prompt}
                for index, prompt in enumerate(prompts)
            ],
        },
        "methods": [
            {
                "key": method.key,
                "repo": str(method.repo),
                "entrypoint": str(method.repo / "inference.py"),
                "entrypoint_sha256": sha256(method.repo / "inference.py"),
                "config": str(method.config),
                "config_sha256": sha256(method.config),
                "checkpoints": [
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                    }
                    for path in method.checkpoints
                ],
                "wan_model_dir": str(method.wan_model_dir),
                "comparison_class": COMPARISON_CLASSES[method.key],
            }
            for method in methods
        ],
        "assignment": {
            "num_nodes": args.num_nodes,
            "gpus_per_node": len(
                [
                    value
                    for value in args.gpu_list.split(",")
                    if value.strip()
                ]
            ),
            "partition": "contiguous_equal_intervals",
            "single_process_per_gpu": True,
            "distributed_launcher_environment_scrubbed": True,
        },
    }


def worker_paths(
    args: argparse.Namespace,
    method: Method,
    worker_rank: int,
) -> dict[str, Path]:
    prefix = f"worker{worker_rank:03d}"
    return {
        "raw": args.out_root / "raw" / method.key / prefix,
        "log": args.out_root / "logs" / method.key / f"{prefix}.log",
        "config": (
            args.out_root / "configs" / method.key / f"{prefix}.json"
        ),
        "done": args.out_root / "status" / method.key / f"{prefix}.done.json",
        "prompt_shard": (
            args.out_root / "prompt_shards" / f"{prefix}.txt"
        ),
        "longlive_config": (
            args.out_root
            / "configs"
            / method.key
            / f"{prefix}.yaml"
        ),
    }


def source_video(
    method: Method,
    raw_dir: Path,
    prompt_index: int,
) -> Path:
    candidates = source_candidates(method, raw_dir, prompt_index)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{method.key}:{prompt_index}: expected one raw video under "
            f"{raw_dir}, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def source_candidates(
    method: Method,
    raw_dir: Path,
    prompt_index: int,
) -> list[Path]:
    if method.key == "longlive":
        return sorted(
            raw_dir.glob(f"rank*-{prompt_index}-0_*.mp4")
        )
    return sorted(raw_dir.glob(f"{prompt_index}-0_*.mp4"))


def expected_frames_for(method_key: str, default: int) -> int:
    # LongLive decodes N latent frames to 4*N-6 pixel frames, while the
    # SF-family contract is 4*N-3. Keep the exception duration-agnostic.
    return default - 3 if method_key == "longlive" else default


def validate_video(
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, object]:
    metadata = _probe_video(path, decode=True)
    checks = {
        "frames": (metadata.get("frames"), expected_frames),
        "fps": (round(float(metadata.get("fps", 0.0)), 3), 16.0),
        "width": (metadata.get("width"), 832),
        "height": (metadata.get("height"), 480),
        "fully_decoded": (metadata.get("fully_decoded"), True),
    }
    failures = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if failures:
        raise RuntimeError(f"invalid video {path}: {failures}")
    return metadata


def repair_partial_raw_videos(
    args: argparse.Namespace,
    method: Method,
    raw_dir: Path,
    *,
    start: int,
    end: int,
    contract_sha: str,
    worker_config_sha: str,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for prompt_index in range(start, end):
        candidates = source_candidates(method, raw_dir, prompt_index)
        if len(candidates) > 1:
            raise RuntimeError(
                f"{method.key}:{prompt_index}: multiple resume candidates "
                f"{[path.name for path in candidates]}"
            )
        if not candidates:
            publication_removed = clear_incomplete_publication(
                args,
                method,
                prompt_index=prompt_index,
                raw_dir=raw_dir,
                expected_source=None,
                contract_sha=contract_sha,
                worker_config_sha=worker_config_sha,
            )
            if publication_removed:
                repairs.append(
                    {
                        "prompt_index": prompt_index,
                        "path": None,
                        "size": 0,
                        "reason": "published artifact had no raw source",
                        "publication_removed": publication_removed,
                    }
                )
            continue
        candidate = candidates[0]
        if candidate.resolve().parent != raw_dir.resolve():
            raise RuntimeError(f"unsafe raw resume path: {candidate}")
        try:
            validate_video(
                candidate,
                expected_frames=expected_frames_for(
                    method.key, args.expected_video_frames
                ),
            )
        except Exception as error:
            size = candidate.stat().st_size
            publication_removed = clear_incomplete_publication(
                args,
                method,
                prompt_index=prompt_index,
                raw_dir=raw_dir,
                expected_source=candidate,
                contract_sha=contract_sha,
                worker_config_sha=worker_config_sha,
            )
            candidate.unlink()
            repairs.append(
                {
                    "prompt_index": prompt_index,
                    "path": str(candidate),
                    "size": size,
                    "reason": str(error),
                    "publication_removed": publication_removed,
                }
            )
            print(
                f"[v129-resume-repair] method={method.key} "
                f"prompt={prompt_index} removed={candidate.name} "
                f"published_removed={publication_removed} reason={error}",
                flush=True,
            )
    return repairs


def prompt_marker(
    args: argparse.Namespace,
    method: Method,
    prompt_index: int,
) -> Path:
    return (
        args.out_root
        / "status"
        / "published"
        / f"{method.key}.p{prompt_index:03d}.json"
    )


def publication_paths(
    args: argparse.Namespace,
    method: Method,
    prompt_index: int,
) -> tuple[Path, Path]:
    return (
        args.out_root
        / "published"
        / method.key
        / f"{prompt_index:06d}.mp4",
        args.out_root
        / "published_indexed"
        / method.key
        / f"{prompt_index:06d}-0_v129_{args.duration}s.mp4",
    )


def clear_incomplete_publication(
    args: argparse.Namespace,
    method: Method,
    *,
    prompt_index: int,
    raw_dir: Path,
    expected_source: Path | None,
    contract_sha: str,
    worker_config_sha: str,
) -> list[str]:
    target, indexed = publication_paths(args, method, prompt_index)
    marker_path = prompt_marker(args, method, prompt_index)
    present = [
        path
        for path in (target, indexed)
        if path.exists() or path.is_symlink()
    ]
    if not present and not marker_path.is_file():
        return []
    if not marker_path.is_file():
        raise RuntimeError(
            f"{method.key}:{prompt_index}: unowned published artifact "
            f"cannot be repaired: {[str(path) for path in present]}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_source = Path(str(marker.get("source", ""))).resolve()
    expected_marker = {
        "experiment_contract_sha256": contract_sha,
        "worker_config_sha256": worker_config_sha,
        "method": method.key,
        "prompt_index": prompt_index,
        "target": str(target),
        "indexed_target": str(indexed),
    }
    failures = {
        key: {"actual": marker.get(key), "expected": value}
        for key, value in expected_marker.items()
        if marker.get(key) != value
    }
    if marker_source.parent != raw_dir.resolve():
        failures["source_parent"] = {
            "actual": str(marker_source.parent),
            "expected": str(raw_dir.resolve()),
        }
    if (
        expected_source is not None
        and marker_source != expected_source.resolve()
    ):
        failures["source"] = {
            "actual": str(marker_source),
            "expected": str(expected_source.resolve()),
        }
    if failures:
        raise RuntimeError(
            f"{method.key}:{prompt_index}: refusing stale publication "
            f"cleanup: {failures}"
        )
    removed = []
    for path in (target, indexed):
        if path.exists() or path.is_symlink():
            if path.parent.resolve() not in {
                (args.out_root / "published" / method.key).resolve(),
                (
                    args.out_root
                    / "published_indexed"
                    / method.key
                ).resolve(),
            }:
                raise RuntimeError(f"unsafe publication path: {path}")
            path.unlink()
            removed.append(str(path))
    marker_path.unlink()
    removed.append(str(marker_path))
    return removed


def publish_prompt(
    args: argparse.Namespace,
    method: Method,
    *,
    prompt_index: int,
    source: Path,
    contract_sha: str,
    worker_config_sha: str,
) -> dict[str, Any]:
    metadata = validate_video(
        source,
        expected_frames=expected_frames_for(
            method.key, args.expected_video_frames
        ),
    )
    target, indexed = publication_paths(args, method, prompt_index)
    link_mode = link_or_validate(source, target)
    indexed_mode = link_or_validate(source, indexed)
    marker = {
        "version": 1,
        "experiment_contract_sha256": contract_sha,
        "worker_config_sha256": worker_config_sha,
        "method": method.key,
        "prompt_index": prompt_index,
        "source": str(source),
        "target": str(target),
        "indexed_target": str(indexed),
        "size": source.stat().st_size,
        "media": metadata,
    }
    write_frozen(prompt_marker(args, method, prompt_index), marker)
    return {
        "prompt_index": prompt_index,
        "link_mode": link_mode,
        "indexed_link_mode": indexed_mode,
    }


def build_longlive_config(
    args: argparse.Namespace,
    method: Method,
    *,
    shard: Path,
    output: Path,
    start: int,
) -> str:
    payload = yaml.safe_load(method.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid LongLive config: {method.config}")
    payload.update(
        {
            "generator_ckpt": str(args.longlive_generator),
            "lora_ckpt": str(args.longlive_lora),
            "data_path": str(shard),
            "output_folder": str(output),
            "idx_offset": int(start),
            "inference_iter": -1,
            "skip_existing": True,
            "num_output_frames": int(args.num_output_frames),
            "use_ema": False,
            "seed": int(args.seed),
            "num_samples": 1,
            "save_with_index": True,
            "reseed_per_prompt": True,
        }
    )
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)


def command_for(
    args: argparse.Namespace,
    method: Method,
    *,
    paths: dict[str, Path],
    start: int,
    end: int,
) -> list[str]:
    common = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(method.config),
        "--data_path",
        str(args.prompts),
        "--output_folder",
        str(paths["raw"]),
        "--num_output_frames",
        str(args.num_output_frames),
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
        "--skip_existing",
    ]
    if method.key == "deep_forcing":
        return [
            *common,
            "--checkpoint_path",
            str(args.deep_checkpoint),
            "--Budget",
            "16",
            "--Recent",
            "4",
        ]
    if method.key == "rolling_forcing":
        return [
            *common,
            "--checkpoint_path",
            str(args.rolling_checkpoint),
        ]
    if method.key == "longlive":
        return [
            sys.executable,
            "inference.py",
            "--config_path",
            str(paths["longlive_config"]),
        ]
    raise ValueError(f"unsupported external method: {method.key}")


def run_worker_method(
    args: argparse.Namespace,
    method: Method,
    *,
    gpu: str,
    worker_rank: int,
    worker_count: int,
    prompts: list[str],
    contract_sha: str,
) -> dict[str, Any]:
    start, end = interval_for_worker(worker_rank, worker_count)
    if start == end:
        return {
            "method": method.key,
            "worker_rank": worker_rank,
            "status": "empty",
        }
    paths = worker_paths(args, method, worker_rank)
    for key in ("raw",):
        paths[key].mkdir(parents=True, exist_ok=True)
    shard_text = "\n".join(prompts[start:end]) + "\n"
    shard_sha = write_frozen_text(paths["prompt_shard"], shard_text)
    if method.key == "longlive":
        config_text = build_longlive_config(
            args,
            method,
            shard=paths["prompt_shard"],
            output=paths["raw"],
            start=start,
        )
        longlive_config_sha = write_frozen_text(
            paths["longlive_config"],
            config_text,
        )
    else:
        longlive_config_sha = None
    command = command_for(
        args,
        method,
        paths=paths,
        start=start,
        end=end,
    )
    config = {
        "version": 1,
        "experiment_contract_sha256": contract_sha,
        "method": method.key,
        "worker_rank": worker_rank,
        "worker_count": worker_count,
        "gpu": str(gpu),
        "prompt_interval": [start, end],
        "prompt_shard": str(paths["prompt_shard"]),
        "prompt_shard_sha256": shard_sha,
        "longlive_config_sha256": longlive_config_sha,
        "num_output_frames": args.num_output_frames,
        "expected_video_frames": args.expected_video_frames,
        "seed": args.seed,
        "reseed_per_prompt": True,
        "distributed_launcher_environment_scrubbed": True,
        "command": command,
    }
    worker_config_sha = write_frozen(paths["config"], config)
    if paths["done"].is_file():
        done = json.loads(paths["done"].read_text(encoding="utf-8"))
        if (
            done.get("experiment_contract_sha256") != contract_sha
            or done.get("worker_config_sha256") != worker_config_sha
        ):
            raise RuntimeError(f"stale worker marker: {paths['done']}")
        for prompt_index in range(start, end):
            marker = prompt_marker(args, method, prompt_index)
            if not marker.is_file():
                raise RuntimeError(f"missing prompt marker: {marker}")
        return {
            "method": method.key,
            "worker_rank": worker_rank,
            "status": "resumed",
            "prompt_interval": [start, end],
        }

    resume_repairs = repair_partial_raw_videos(
        args,
        method,
        paths["raw"],
        start=start,
        end=end,
        contract_sha=contract_sha,
        worker_config_sha=worker_config_sha,
    )
    env = os.environ.copy()
    for key in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONHASHSEED"] = str(args.seed)
    run_checked(
        command,
        cwd=method.repo,
        env=env,
        log_path=paths["log"],
    )
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
    )
    hits = [value for value in failure_signatures if value in log_text]
    if hits:
        raise RuntimeError(
            f"{method.key} worker {worker_rank} log failures: {hits}"
        )
    publications = []
    for prompt_index in range(start, end):
        publications.append(
            publish_prompt(
                args,
                method,
                prompt_index=prompt_index,
                source=source_video(method, paths["raw"], prompt_index),
                contract_sha=contract_sha,
                worker_config_sha=worker_config_sha,
            )
        )
    write_frozen(
        paths["done"],
        {
            "version": 1,
            "experiment_contract_sha256": contract_sha,
            "worker_config_sha256": worker_config_sha,
            "method": method.key,
            "worker_rank": worker_rank,
            "prompt_interval": [start, end],
            "publication_count": len(publications),
            "resume_repairs": resume_repairs,
            "log": str(paths["log"]),
            "log_sha256": sha256(paths["log"]),
        },
    )
    return {
        "method": method.key,
        "worker_rank": worker_rank,
        "status": "generated",
        "prompt_interval": [start, end],
    }


def run_gpu_worker(
    args: argparse.Namespace,
    methods: tuple[Method, ...],
    *,
    gpu: str,
    worker_rank: int,
    worker_count: int,
    prompts: list[str],
    contract_sha: str,
) -> list[dict[str, Any]]:
    results = []
    for method in methods:
        print(
            f"[v129-external] gpu={gpu} worker={worker_rank}/"
            f"{worker_count} method={method.key}",
            flush=True,
        )
        results.append(
            run_worker_method(
                args,
                method,
                gpu=gpu,
                worker_rank=worker_rank,
                worker_count=worker_count,
                prompts=prompts,
                contract_sha=contract_sha,
            )
        )
    return results


def audit(
    args: argparse.Namespace,
    methods: tuple[Method, ...],
    contract_sha: str,
) -> dict[str, Any]:
    failures: list[str] = []
    rows = []
    for method in methods:
        video_dir = args.out_root / "published" / method.key
        indexed_dir = args.out_root / "published_indexed" / method.key
        expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
        expected_indexed = {
            f"{index:06d}-0_v129_{args.duration}s.mp4"
            for index in range(PROMPT_COUNT)
        }
        actual = {path.name for path in video_dir.glob("*.mp4")}
        actual_indexed = {
            path.name for path in indexed_dir.glob("*.mp4")
        }
        if actual != expected:
            failures.append(
                f"{method.key}: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        if actual_indexed != expected_indexed:
            failures.append(
                f"{method.key}: indexed_missing="
                f"{sorted(expected_indexed - actual_indexed)} "
                f"indexed_extra="
                f"{sorted(actual_indexed - expected_indexed)}"
            )
        total_bytes = 0
        for prompt_index in range(PROMPT_COUNT):
            target = video_dir / f"{prompt_index:06d}.mp4"
            indexed = (
                indexed_dir
                / f"{prompt_index:06d}-0_v129_{args.duration}s.mp4"
            )
            marker_path = prompt_marker(args, method, prompt_index)
            try:
                marker = json.loads(
                    marker_path.read_text(encoding="utf-8")
                )
                source = Path(str(marker["source"]))
                if (
                    marker["experiment_contract_sha256"] != contract_sha
                    or marker["method"] != method.key
                    or int(marker["prompt_index"]) != prompt_index
                    or Path(str(marker["target"])) != target
                    or Path(str(marker["indexed_target"])) != indexed
                    or not target.is_file()
                    or not indexed.is_file()
                    or not source.is_file()
                    or not target.samefile(source)
                    or not indexed.samefile(source)
                    or int(marker["size"]) != target.stat().st_size
                    or indexed.stat().st_size != target.stat().st_size
                ):
                    raise ValueError("marker/source mismatch")
                total_bytes += target.stat().st_size
            except Exception as error:
                failures.append(
                    f"{method.key}:{prompt_index}: {error}"
                )
        rows.append(
            {
                "key": method.key,
                "role": COMPARISON_CLASSES[method.key],
                "video_dir": str(video_dir),
                "indexed_video_dir": str(indexed_dir),
                "video_count": len(actual),
                "indexed_video_count": len(actual_indexed),
                "total_bytes": total_bytes,
            }
        )
    payload = {
        "version": 1,
        "experiment": f"v129_moviebench128_{args.duration}s_external",
        "experiment_contract_sha256": contract_sha,
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
        "num_output_frames": args.num_output_frames,
        "methods": rows,
        "failures": failures,
        "ok": not failures,
    }
    if failures:
        write_runtime_json(
            args.out_root / "status" / "published.audit.failed.json",
            payload,
        )
        raise RuntimeError(f"external baseline audit failed: {failures[:5]}")
    write_frozen(args.out_root / "published_manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    methods = methods_for(args)
    prompts = validate_inputs(args, methods)
    for name in (
        "contracts",
        "configs",
        "logs",
        "prompt_shards",
        "published",
        "published_indexed",
        "raw",
        "status",
        "status/published",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    contract = experiment_contract(args, methods, prompts)
    contract_path = args.out_root / "contracts" / "experiment.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            args.contract_wait_seconds,
        )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    worker_count = args.num_nodes * len(gpus)
    assignments = [
        (
            gpu,
            args.node_rank * len(gpus) + local_rank,
        )
        for local_rank, gpu in enumerate(gpus)
    ]
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "contract_sha256": contract_sha,
                    "duration": args.duration,
                    "methods": [method.key for method in methods],
                    "node_rank": args.node_rank,
                    "worker_count": worker_count,
                    "assignments": [
                        {
                            "gpu": gpu,
                            "worker_rank": rank,
                            "interval": interval_for_worker(
                                rank,
                                worker_count,
                            ),
                        }
                        for gpu, rank in assignments
                    ],
                },
                indent=2,
            )
        )
        return
    if args.mode == "audit":
        if args.node_rank != 0:
            raise SystemExit("audit must run only on NODE_RANK=0")
        payload = audit(args, methods, contract_sha)
        print(
            f"[v129-external-audit] methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    failures: list[str] = []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = {
            executor.submit(
                run_gpu_worker,
                args,
                methods,
                gpu=gpu,
                worker_rank=worker_rank,
                worker_count=worker_count,
                prompts=prompts,
                contract_sha=contract_sha,
            ): (gpu, worker_rank)
            for gpu, worker_rank in assignments
        }
        for future in as_completed(futures):
            gpu, worker_rank = futures[future]
            try:
                rows = future.result()
                results.extend(rows)
                print(
                    f"[v129-external-worker-complete] gpu={gpu} "
                    f"worker={worker_rank} methods={len(rows)}",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"gpu={gpu} worker={worker_rank}: {error}")
                print(
                    f"[v129-external-worker-failed] gpu={gpu} "
                    f"worker={worker_rank}: {error}",
                    flush=True,
                )
    report = {
        "version": 1,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "results": results,
        "failures": failures,
        "ok": not failures,
    }
    write_runtime_json(
        args.out_root / "status" / f"node{args.node_rank}.json",
        report,
    )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
