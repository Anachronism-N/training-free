#!/usr/bin/env python3
"""Build a crossed prompt-factor x seed-replicate head-profile suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


V144_PATH = Path(__file__).with_name(
    "build_v144_factorized_mechanism_suite.py"
)
V144_SPEC = importlib.util.spec_from_file_location(
    "v144_suite_for_v145", V144_PATH
)
V144 = importlib.util.module_from_spec(V144_SPEC)
assert V144_SPEC.loader is not None
V144_SPEC.loader.exec_module(V144)

V134 = V144.V134
VARIANTS = (
    "base",
    "paraphrase",
    "identity",
    "scene",
    "full_semantic",
)
SEED_REPLICATES = (0, 1)
FAMILY_COUNT = 16
JOBS_PER_FAMILY = len(VARIANTS) * len(SEED_REPLICATES)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_jobs(seed_base: int = 145000) -> list[dict]:
    jobs = []
    if len(V134.COMPONENTS["identity"]) != FAMILY_COUNT:
        raise RuntimeError("v145 requires the 16-family controlled prompt grid")
    for family_index in range(FAMILY_COUNT):
        base_fields = V144._fields(family_index, 0)
        alternate_fields = V144._fields(family_index, 1)
        reference_prompt = V134._render_prompt(
            base_fields, paraphrase=False
        )
        for seed_replicate in SEED_REPLICATES:
            seed = (
                int(seed_base)
                + family_index
                + seed_replicate * 10000
            )
            for variant in VARIANTS:
                fields = dict(base_fields)
                for factor in V144.CHANGED_FACTORS[variant]:
                    fields[factor] = alternate_fields[factor]
                prompt = V134._render_prompt(
                    fields, paraphrase=(variant == "paraphrase")
                )
                jobs.append(
                    {
                        "dataset_index": len(jobs),
                        "job_id": (
                            f"v145_f{family_index:02d}_"
                            f"s{seed_replicate}_{variant}"
                        ),
                        "kind": "crossed_seed_head_mechanism",
                        "family_id": f"family_{family_index:02d}",
                        "family_index": family_index,
                        "family_split": (
                            "discovery"
                            if family_index % 2 == 0
                            else "validation"
                        ),
                        "seed_replicate": seed_replicate,
                        "variant": variant,
                        "reference_variant": "base",
                        "seed": seed,
                        "reference_seed": seed,
                        "changed_factors": list(
                            V144.CHANGED_FACTORS[variant]
                        ),
                        "surface_rewrite": variant == "paraphrase",
                        "base_prompt": prompt,
                        "reference_prompt": reference_prompt,
                        "token_jaccard_to_base": V144._token_jaccard(
                            reference_prompt, prompt
                        ),
                        "normalized_token_edit_distance": (
                            V144._normalized_levenshtein(
                                reference_prompt, prompt
                            )
                        ),
                    }
                )
    expected = FAMILY_COUNT * JOBS_PER_FAMILY
    if len(jobs) != expected:
        raise AssertionError(
            f"expected {expected} v145 jobs, got {len(jobs)}"
        )
    return jobs


def write_suite(output_dir: Path, seed_base: int = 145000) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(seed_base)
    prompts = "\n".join(str(job["base_prompt"]) for job in jobs) + "\n"
    manifest = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n"
        for job in jobs
    )
    prompt_path = output_dir / "v145_crossed_seed_head_160.txt"
    manifest_path = output_dir / "v145_crossed_seed_head_160.jsonl"
    prompt_path.write_text(prompts, encoding="utf-8")
    manifest_path.write_text(manifest, encoding="utf-8")
    metadata = {
        "version": 1,
        "job_count": len(jobs),
        "family_count": FAMILY_COUNT,
        "seed_replicates": list(SEED_REPLICATES),
        "variants": list(VARIANTS),
        "jobs_per_family": JOBS_PER_FAMILY,
        "seed_base": int(seed_base),
        "prompts_sha256": _sha256(prompts),
        "manifest_sha256": _sha256(manifest),
        "purpose": (
            "test whether paired prompt-factor effects reproduce across "
            "independent generation seeds and held-out prompt families"
        ),
    }
    (output_dir / "suite_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=145000)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(args.output_dir, args.seed_base),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
