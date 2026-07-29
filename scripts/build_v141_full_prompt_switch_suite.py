#!/usr/bin/env python3
"""Build controlled full-prompt A-B-A schedules for causal head profiling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


V134_PATH = Path(__file__).with_name("build_v134_head_discovery_suite.py")
V134_SPEC = importlib.util.spec_from_file_location("v134_suite", V134_PATH)
V134 = importlib.util.module_from_spec(V134_SPEC)
assert V134_SPEC.loader is not None
V134_SPEC.loader.exec_module(V134)

SWITCH_FRAMES = [39, 78]
SEGMENT_LABELS = ["A1", "B", "A2"]
SWITCH_TYPES = {
    "scene_action": (
        "action",
        "scene",
        "object",
        "camera",
        "atmosphere",
    ),
    "identity_scene": (
        "identity",
        "appearance",
        "action",
        "scene",
        "object",
        "camera",
        "atmosphere",
    ),
}


def _fields(subject_index: int, variant: int) -> dict[str, str]:
    return {
        factor: V134.COMPONENTS[factor][subject_index][variant]
        for factor in V134.FACTORS
    }


def build_jobs(seed: int = 0) -> list[dict]:
    jobs = []
    family_count = len(V134.COMPONENTS["identity"])
    for switch_type, changed_factors in SWITCH_TYPES.items():
        for family_index in range(family_count):
            fields_a = _fields(family_index, 0)
            fields_b = dict(fields_a)
            variant_b = _fields(family_index, 1)
            for factor in changed_factors:
                fields_b[factor] = variant_b[factor]
            prompt_a = V134._render_prompt(fields_a, paraphrase=False)
            prompt_b = V134._render_prompt(fields_b, paraphrase=False)
            paraphrase_a = V134._render_prompt(fields_a, paraphrase=True)
            paraphrase_b = V134._render_prompt(fields_b, paraphrase=True)
            schedule = " || ".join((prompt_a, prompt_b, prompt_a))
            jobs.append(
                {
                    "dataset_index": len(jobs),
                    "job_id": f"aba_{switch_type}_{family_index:02d}",
                    "kind": "full_prompt_switch",
                    "family_id": f"family_{family_index:02d}",
                    "family_index": family_index,
                    "switch_type": switch_type,
                    "changed_factors": list(changed_factors),
                    "preserved_factors": [
                        factor
                        for factor in V134.FACTORS
                        if factor not in changed_factors
                    ],
                    "seed": int(seed),
                    "base_prompt": schedule,
                    "schedule_prompts": [prompt_a, prompt_b, prompt_a],
                    "switch_frames": list(SWITCH_FRAMES),
                    "segment_labels": list(SEGMENT_LABELS),
                    "shadow_prompts": {
                        "exact_a": prompt_a,
                        "exact_b": prompt_b,
                        "paraphrase_a": paraphrase_a,
                        "paraphrase_b": paraphrase_b,
                    },
                }
            )
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_suite(output_dir: Path, seed: int = 0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(seed)
    prompts_path = output_dir / "v141_full_prompt_switch_32.txt"
    manifest_path = output_dir / "v141_full_prompt_switch_32.jsonl"
    prompts_path.write_text(
        "\n".join(str(job["base_prompt"]) for job in jobs) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    metadata = {
        "version": 1,
        "job_count": len(jobs),
        "seed": int(seed),
        "switch_frames": list(SWITCH_FRAMES),
        "segment_labels": list(SEGMENT_LABELS),
        "switch_type_counts": {
            switch_type: sum(
                job["switch_type"] == switch_type for job in jobs
            )
            for switch_type in SWITCH_TYPES
        },
        "prompts_sha256": _sha256(prompts_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    (output_dir / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metadata = write_suite(args.output_dir, args.seed)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
