#!/usr/bin/env python3
"""Build natural and controlled A-B suites for v143 head profiling."""

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

AB_SWITCH_FRAME = 57
AB_CAPTURE_FRAMES = [0, 18, 36, 54]


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
        raise ValueError(f"v143 requires 128 natural prompts, found {len(prompts)}")
    if len(set(prompts)) != 128:
        raise ValueError("v143 natural prompts must be unique")
    return prompts


def _write_jobs(
    output_dir: Path,
    *,
    stem: str,
    jobs: list[dict],
) -> tuple[Path, Path]:
    prompt_path = output_dir / f"{stem}.txt"
    manifest_path = output_dir / f"{stem}.jsonl"
    prompt_path.write_text(
        "\n".join(str(job["base_prompt"]) for job in jobs) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    return prompt_path, manifest_path


def build_ab_jobs(seed: int = 0) -> list[dict]:
    jobs = []
    for source in V141.build_jobs(seed):
        prompt_a, prompt_b = source["schedule_prompts"][:2]
        job = dict(source)
        job.update(
            {
                "dataset_index": len(jobs),
                "job_id": str(source["job_id"]).replace("aba_", "v143_ab_", 1),
                "kind": "multiaxis_full_prompt_ab",
                "base_prompt": f"{prompt_a} || {prompt_b}",
                "schedule_prompts": [prompt_a, prompt_b],
                "switch_frames": [AB_SWITCH_FRAME],
                "segment_labels": ["A", "B"],
                "persistent_episode": "A",
                "persistent_capture_frames": list(AB_CAPTURE_FRAMES),
            }
        )
        jobs.append(job)
    return jobs


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    seed: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    natural = [
        {
            "dataset_index": index,
            "job_id": f"v143_natural_{index:03d}",
            "kind": "multiaxis_region_natural",
            "seed": int(seed),
            "base_prompt": prompt,
            "source_prompt_index": index,
        }
        for index, prompt in enumerate(_read_prompts(natural_prompts))
    ]
    ab = build_ab_jobs(seed)
    natural_paths = _write_jobs(
        output_dir, stem="v143_natural_128", jobs=natural
    )
    ab_paths = _write_jobs(output_dir, stem="v143_ab_32", jobs=ab)
    metadata = {
        "version": 1,
        "seed": int(seed),
        "natural_count": len(natural),
        "ab_count": len(ab),
        "natural_source": str(natural_prompts),
        "natural_prompts_sha256": _sha256(natural_paths[0]),
        "natural_manifest_sha256": _sha256(natural_paths[1]),
        "ab_prompts_sha256": _sha256(ab_paths[0]),
        "ab_manifest_sha256": _sha256(ab_paths[1]),
        "ab_switch_frames": [AB_SWITCH_FRAME],
        "ab_capture_frames": list(AB_CAPTURE_FRAMES),
        "region_calibration_frame": 9,
        "region_long_frames": [21, 63, 117],
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
