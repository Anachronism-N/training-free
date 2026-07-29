#!/usr/bin/env python3
"""Build the frozen MovieBench-128 v138 history-intervention suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PROMPTS = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_jobs(prompts: list[str], seed: int) -> list[dict]:
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError(
            f"expected {EXPECTED_PROMPTS} prompts, found {len(prompts)}"
        )
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompt suite contains an empty prompt")
    return [
        {
            "dataset_index": index,
            "job_id": f"moviebench_history_intervention_{index:03d}",
            "kind": "history_intervention",
            "base_prompt": prompt.strip(),
            "seed": int(seed),
        }
        for index, prompt in enumerate(prompts)
    ]


def write_suite(source: Path, output_dir: Path, seed: int) -> dict:
    prompts = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs = build_jobs(prompts, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "moviebench128_history_intervention.txt"
    manifest_path = output_dir / "moviebench128_history_intervention.jsonl"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    manifest_path.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    metadata = {
        "version": 1,
        "kind": "history_intervention",
        "count": len(jobs),
        "seed": int(seed),
        "source": str(source),
        "source_sha256": _sha256(source),
        "prompt_sha256": _sha256(prompt_path),
        "manifest_sha256": _sha256(manifest_path),
        "interventions": [
            "reverse",
            "phase_shift_1",
            "freeze_latest",
            "value_mismatch",
            "cross_video_wrong_history_retrieval",
        ],
    }
    (output_dir / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moviebench-qwen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata = write_suite(
        args.moviebench_qwen, args.output_dir, args.seed
    )
    print(
        "[v138-suite] "
        f"count={metadata['count']} seed={metadata['seed']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
