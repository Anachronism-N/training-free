#!/usr/bin/env python3
"""Audit v195 Causal-checkpoint profile shards and runtime logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from analyze_v189_structured_head_phase import aggregate_operator
from prepare_v195_cross_checkpoint_head_phase_profile import (
    CALLS,
    HEADS,
    LAYERS,
    PROFILE_CONTRACT,
    PROFILE_METHOD,
    PROFILE_VERSION,
    PROMPT_COUNT,
    sha256,
    verify,
)

FATAL_LOG_MARKERS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Failed to save profile",
    "candidate-superset verification",
    "[error]",
)


def _same_path(left: str, right: str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _rank(path: Path) -> int:
    match = re.fullmatch(r"shard(\d{2})\.pt", path.name)
    if match is None:
        raise ValueError(f"unexpected v195 profile filename: {path.name}")
    return int(match.group(1))


def _log_for_rank(log_root: Path, rank: int) -> Path:
    matches = sorted(log_root.glob(f"*_rank{rank:02d}.log"))
    if len(matches) != 1:
        raise ValueError(f"v195 rank {rank} requires exactly one log, found {matches}")
    return matches[0]


def audit(
    manifest_path: Path,
    profile_root: Path,
    log_root: Path,
    output_path: Path,
) -> dict:
    manifest = verify(manifest_path)
    operator = str(manifest["operator"])
    world = int(manifest["execution_contract"]["world_shards"])
    paths = sorted(profile_root.glob("shard*.pt"))
    ranks = [_rank(path) for path in paths]
    if len(paths) != world or ranks != list(range(world)):
        raise ValueError(f"v195 requires ranks 0..{world - 1}, observed={ranks}")

    checkpoint = manifest["checkpoint"]
    expected_kind = f"moviegen128_v195_cf_{operator}_head_phase"
    expected_per_prompt = CALLS * LAYERS * HEADS
    shard_reports = []
    all_prompts: set[int] = set()
    for path, rank in zip(paths, ranks):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        records = list(payload.get("records") or [])
        expected_prompts = set(range(rank, PROMPT_COUNT, world))
        observed_prompts = {int(row["prompt_id"]) for row in records}
        counts = Counter(int(row["prompt_id"]) for row in records)
        if (
            payload.get("version") != PROFILE_VERSION
            or payload.get("contract") != PROFILE_CONTRACT
            or payload.get("method") != PROFILE_METHOD
            or tuple(payload.get("policies") or ()) != ("recent", "coverage")
            or payload.get("reference_policy") != "union"
            or metadata.get("kind") != expected_kind
            or metadata.get("profile_contract") != PROFILE_CONTRACT
            or metadata.get("coverage_operator") != operator
            or int(metadata.get("seed", -1)) != int(manifest["seed"])
            or int(metadata.get("num_output_frames", -1))
            != int(manifest["num_output_frames"])
            or metadata.get("call_indices") != "0,1,2,3"
            or int(metadata.get("ar_stride", -1)) != 3
            or int(metadata.get("query_stride", -1)) != 8
            or int(metadata.get("min_frame", -1)) != 12
            or metadata.get("chunk_offsets") != "0"
            or metadata.get("checkpoint_state_key") != "generator"
            or metadata.get("use_ema") is not False
            or int(metadata.get("model_local_attn_size", -1)) != 21
            or metadata.get("skip_video_decode") is not True
            or not _same_path(metadata.get("data_path", ""), manifest["prompt_file"])
            or not _same_path(
                metadata.get("head_config_path", ""), manifest["profile_map"]
            )
            or not _same_path(metadata.get("checkpoint_path", ""), checkpoint["path"])
            or set(metadata.get("completed_prompt_ids") or ()) != expected_prompts
            or observed_prompts != expected_prompts
            or any(counts[prompt] != expected_per_prompt for prompt in expected_prompts)
            or len(records) != len(expected_prompts) * expected_per_prompt
            or any(row.get("profile_contract") != PROFILE_CONTRACT for row in records)
        ):
            raise ValueError(f"v195 profile shard contract failed: {path}")
        if all_prompts & observed_prompts:
            raise ValueError(f"v195 prompt assignment overlaps at shard {rank}")
        all_prompts |= observed_prompts

        log_path = _log_for_rank(log_root, rank)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        required_markers = (
            "[ModelAttentionContract] local_attn_size=21 source=cli_override",
            "[CheckpointLoad] state_key=generator use_ema=False strict=true",
            f"coverage_operator={operator}",
            "contract=v189",
            "reference=representation_complete_union",
            "skip_video_decode=True",
        )
        missing = [marker for marker in required_markers if marker not in log]
        fatal = [marker for marker in FATAL_LOG_MARKERS if marker in log]
        if missing or fatal:
            raise ValueError(
                f"v195 runtime log failed: {log_path}; missing={missing} fatal={fatal}"
            )
        shard_reports.append(
            {
                "rank": rank,
                "profile": str(path.resolve()),
                "profile_sha256": sha256(path),
                "log": str(log_path.resolve()),
                "log_sha256": sha256(log_path),
                "prompt_ids": sorted(observed_prompts),
                "record_count": len(records),
            }
        )

    if all_prompts != set(range(PROMPT_COUNT)):
        raise ValueError("v195 profile shards do not cover prompts 0..127 exactly once")
    _, aggregate_audit = aggregate_operator(profile_root, operator)
    if int(aggregate_audit.get("record_count", -1)) != int(
        manifest["expected_record_count"]
    ):
        raise ValueError("v195 aggregate record count drifted")
    report = {
        "version": 1,
        "experiment": "v195_cross_checkpoint_head_phase_profile_audit",
        "ok": True,
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "operator": operator,
        "checkpoint_sha256": checkpoint["sha256"],
        "world_shards": world,
        "prompt_count": PROMPT_COUNT,
        "record_count": aggregate_audit["record_count"],
        "aggregate_audit": aggregate_audit,
        "shards": shard_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "[v195-audit] PASS "
        f"operator={operator} shards={world} records={report['record_count']}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.profile_root, args.log_root, args.output)


if __name__ == "__main__":
    main()
