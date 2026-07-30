#!/usr/bin/env python3
"""Build natural and controlled A-B-A suites for v142 head profiling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


V141_PATH = Path(__file__).with_name("build_v141_full_prompt_switch_suite.py")
V141_SPEC = importlib.util.spec_from_file_location("v141_suite", V141_PATH)
V141 = importlib.util.module_from_spec(V141_SPEC)
assert V141_SPEC.loader is not None
V141_SPEC.loader.exec_module(V141)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 128:
        raise ValueError(
            f"v142 natural suite requires 128 prompts, found {len(prompts)}"
        )
    if len(set(prompts)) != len(prompts):
        raise ValueError("v142 natural prompts must be unique")
    return prompts


def _write_jobs(
    output_dir: Path,
    *,
    stem: str,
    jobs: list[dict],
) -> tuple[Path, Path]:
    prompts_path = output_dir / f"{stem}.txt"
    manifest_path = output_dir / f"{stem}.jsonl"
    prompts_path.write_text(
        "\n".join(str(job["base_prompt"]) for job in jobs) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    return prompts_path, manifest_path


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(natural_prompts)
    natural_jobs = [
        {
            "dataset_index": index,
            "job_id": f"natural_{index:03d}",
            "kind": "output_causal_natural",
            "seed": int(seed),
            "base_prompt": prompt,
            "source_prompt_index": index,
        }
        for index, prompt in enumerate(prompts)
    ]
    aba_jobs = []
    for source in V141.build_jobs(seed):
        job = dict(source)
        job["job_id"] = str(job["job_id"]).replace("aba_", "v142_aba_", 1)
        job["kind"] = "output_causal_persistent_aba"
        job["persistent_episode"] = "A1"
        job["persistent_capture_frames"] = [0, 18, 36]
        aba_jobs.append(job)

    natural_paths = _write_jobs(
        output_dir,
        stem="v142_natural_128",
        jobs=natural_jobs,
    )
    aba_paths = _write_jobs(
        output_dir,
        stem="v142_persistent_aba_32",
        jobs=aba_jobs,
    )
    metadata = {
        "version": 1,
        "seed": int(seed),
        "natural_count": len(natural_jobs),
        "aba_count": len(aba_jobs),
        "natural_source": str(natural_prompts),
        "natural_prompts_sha256": _sha256(natural_paths[0]),
        "natural_manifest_sha256": _sha256(natural_paths[1]),
        "aba_prompts_sha256": _sha256(aba_paths[0]),
        "aba_manifest_sha256": _sha256(aba_paths[1]),
        "switch_frames": list(V141.SWITCH_FRAMES),
        "persistent_capture_frames": [0, 18, 36],
        "policy_candidates": [
            "recent_budget",
            "boundary_recent",
            "uniform_recent",
        ],
        "policy_budget_frames": 8,
    }
    (output_dir / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--natural-prompts", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata = write_suite(
        args.output_dir,
        natural_prompts=args.natural_prompts,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
