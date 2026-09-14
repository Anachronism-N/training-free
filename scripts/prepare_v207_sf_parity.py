#!/usr/bin/env python3
"""Freeze and verify the short-trajectory SF runtime-parity contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPERIMENT = "v207_sf_runtime_parity"
RUN_ORDER = ("sf_native_a", "sf_native_b", "pf_plain_sf21", "adaptive_recent21")
SOURCE_PROMPT_INDEX = 3
NUM_OUTPUT_FRAMES = 9
SEED = 20703
NUM_BLOCKS = 3
DENOISE_CALLS = 4
LAYERS = 30
HEADS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen parity input differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def prepare(
    *,
    repo_root: Path,
    source_prompts: Path,
    checkpoint: Path,
    sf_config: Path,
    pf_plain_config: Path,
    pf_adaptive_config: Path,
    output_root: Path,
) -> dict:
    prompts = source_prompts.read_text(encoding="utf-8").splitlines()
    if len(prompts) != 128 or any(not item.strip() for item in prompts):
        raise ValueError("parity source must contain exactly 128 non-empty prompts")
    selected = prompts[SOURCE_PROMPT_INDEX].strip()
    prompt_path = output_root / "prompt.txt"
    prompt_sha = write_frozen(
        prompt_path, (selected + "\n").encode("utf-8")
    )
    map_path = output_root / "all_profile_banks.csv"
    map_payload = (
        "\n".join(",".join("10" for _ in range(HEADS)) for _ in range(LAYERS))
        + "\n"
    ).encode("ascii")
    map_sha = write_frozen(map_path, map_payload)
    runtime_paths = (
        "src/lifecycle_kv/parity_trace.py",
        "third_party/Self-Forcing/inference.py",
        "third_party/Self-Forcing/pipeline/causal_inference.py",
        "third_party/Self-Forcing/wan/modules/causal_model.py",
        "third_party/Pyramid-Forcing/inference.py",
        "third_party/Pyramid-Forcing/pipeline/causal_inference.py",
        "third_party/Pyramid-Forcing/wan/modules/causal_model.py",
        "third_party/Pyramid-Forcing/wan/modules/attention/core.py",
        "third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py",
        "third_party/Pyramid-Forcing/pyramidkv/denoise_schedule.py",
        "third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py",
        "scripts/prepare_v207_sf_parity.py",
        "scripts/compare_v207_sf_parity.py",
        "scripts/run_v207_sf_parity.sh",
    )
    for relative in runtime_paths:
        if not (repo_root / relative).is_file():
            raise FileNotFoundError(repo_root / relative)
    artifacts = {
        "source_prompts": {
            "path": str(source_prompts.resolve()),
            "sha256": sha256(source_prompts),
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "sf_config": {
            "path": str(sf_config.resolve()),
            "sha256": sha256(sf_config),
        },
        "pf_plain_config": {
            "path": str(pf_plain_config.resolve()),
            "sha256": sha256(pf_plain_config),
        },
        "pf_adaptive_config": {
            "path": str(pf_adaptive_config.resolve()),
            "sha256": sha256(pf_adaptive_config),
        },
        "prompt": {"path": str(prompt_path.resolve()), "sha256": prompt_sha},
        "head_map": {"path": str(map_path.resolve()), "sha256": map_sha},
    }
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "git_commit": _git_commit(repo_root),
        "run_order": list(RUN_ORDER),
        "source_prompt_index": SOURCE_PROMPT_INDEX,
        "prompt_text": selected,
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "num_ar_blocks": NUM_BLOCKS,
        "num_denoise_calls": DENOISE_CALLS,
        "seed": SEED,
        "dtype": "torch.bfloat16",
        "local_attention_frames": 21,
        "runtime_paths": {
            relative: sha256(repo_root / relative) for relative in runtime_paths
        },
        "artifacts": artifacts,
        "trace_contract": {
            "layers": list(range(LAYERS)),
            "heads": list(range(HEADS)),
            "full_pipeline_tensors": True,
            "attention_sample_values": 4096,
            "expected_attention_events": NUM_BLOCKS * (DENOISE_CALLS + 1) * LAYERS,
            "comparison_chain": [
                ["sf_native_a", "sf_native_b", "native_repeat_floor"],
                ["sf_native_a", "pf_plain_sf21", "vendored_runtime_parity"],
                ["pf_plain_sf21", "adaptive_recent21", "adaptive_cache_parity"],
                ["sf_native_a", "adaptive_recent21", "end_to_end_parity"],
            ],
        },
        "adaptive_recent21_contract": {
            "schedule": "recent",
            "sink_frames": 1,
            "middle_frames": 0,
            "recent_frames": 20,
            "read_frame_equivalents": 21,
            "clean_read": "recent",
        },
        "decision_rule": (
            "Do not interpret candidate-vs-SF generation deltas until vendored "
            "plain SF21 and Adaptive recent21 differences are within the native "
            "repeat floor, or the first divergence is explicitly explained."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def verify(manifest_path: Path, repo_root: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("version") != 1
        or payload.get("experiment") != EXPERIMENT
        or payload.get("run_order") != list(RUN_ORDER)
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or int(payload.get("num_ar_blocks", -1)) != NUM_BLOCKS
        or int(payload.get("num_denoise_calls", -1)) != DENOISE_CALLS
    ):
        raise ValueError("invalid SF parity manifest contract")
    for row in payload["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"parity artifact hash drift: {path}")
    for relative, expected_hash in payload["runtime_paths"].items():
        path = repo_root / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"parity runtime hash drift: {relative}")
    if _git_commit(repo_root) != payload["git_commit"]:
        raise ValueError("parity git commit drift")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, required=True)
    prepare_parser.add_argument("--source-prompts", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--sf-config", type=Path, required=True)
    prepare_parser.add_argument("--pf-plain-config", type=Path, required=True)
    prepare_parser.add_argument("--pf-adaptive-config", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            repo_root=args.repo_root.resolve(),
            source_prompts=args.source_prompts.resolve(),
            checkpoint=args.checkpoint.resolve(),
            sf_config=args.sf_config.resolve(),
            pf_plain_config=args.pf_plain_config.resolve(),
            pf_adaptive_config=args.pf_adaptive_config.resolve(),
            output_root=args.output_root.resolve(),
        )
    else:
        payload = verify(args.manifest.resolve(), args.repo_root.resolve())
    print(
        "[v207-parity-inputs] PASS "
        f"runs={len(payload['run_order'])} blocks={payload['num_ar_blocks']} "
        f"frames={payload['num_output_frames']} seed={payload['seed']}"
    )


if __name__ == "__main__":
    main()
