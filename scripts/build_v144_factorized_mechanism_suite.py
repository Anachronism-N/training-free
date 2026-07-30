#!/usr/bin/env python3
"""Build matched identity/scene/action/camera/seed head-mechanism profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


V134_PATH = Path(__file__).with_name("build_v134_head_discovery_suite.py")
V134_SPEC = importlib.util.spec_from_file_location("v134_suite_v144", V134_PATH)
V134 = importlib.util.module_from_spec(V134_SPEC)
assert V134_SPEC.loader is not None
V134_SPEC.loader.exec_module(V134)

VARIANTS = (
    "base",
    "seed_control",
    "paraphrase",
    "identity",
    "scene",
    "action",
    "camera",
    "full_semantic",
)
CHANGED_FACTORS = {
    "base": (),
    "seed_control": (),
    "paraphrase": (),
    "identity": ("identity",),
    "scene": ("scene",),
    "action": ("action",),
    "camera": ("camera",),
    "full_semantic": tuple(V134.FACTORS),
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(_tokenize(left)), set(_tokenize(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _normalized_levenshtein(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokenize(left), _tokenize(right)
    previous = list(range(len(right_tokens) + 1))
    for left_index, left_token in enumerate(left_tokens, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right_tokens, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1] / max(1, len(left_tokens), len(right_tokens))


def _fields(family_index: int, variant: int) -> dict[str, str]:
    return {
        factor: V134.COMPONENTS[factor][family_index][variant]
        for factor in V134.FACTORS
    }


def build_jobs(seed_base: int = 144000) -> list[dict]:
    jobs = []
    family_count = len(V134.COMPONENTS["identity"])
    for family_index in range(family_count):
        base_fields = _fields(family_index, 0)
        alternate_fields = _fields(family_index, 1)
        base_prompt = V134._render_prompt(base_fields, paraphrase=False)
        family_seed = int(seed_base + family_index)
        for variant in VARIANTS:
            fields = dict(base_fields)
            for factor in CHANGED_FACTORS[variant]:
                fields[factor] = alternate_fields[factor]
            prompt = V134._render_prompt(
                fields, paraphrase=(variant == "paraphrase")
            )
            seed = (
                family_seed + 10000
                if variant == "seed_control"
                else family_seed
            )
            jobs.append(
                {
                    "dataset_index": len(jobs),
                    "job_id": f"v144_f{family_index:02d}_{variant}",
                    "kind": "factorized_head_mechanism",
                    "family_id": f"family_{family_index:02d}",
                    "family_index": family_index,
                    "variant": variant,
                    "reference_variant": "base",
                    "seed": seed,
                    "reference_seed": family_seed,
                    "same_seed_as_base": variant != "seed_control",
                    "changed_factors": list(CHANGED_FACTORS[variant]),
                    "surface_rewrite": variant == "paraphrase",
                    "base_prompt": prompt,
                    "reference_prompt": base_prompt,
                    "token_jaccard_to_base": _token_jaccard(
                        base_prompt, prompt
                    ),
                    "normalized_token_edit_distance": (
                        _normalized_levenshtein(base_prompt, prompt)
                    ),
                }
            )
    if len(jobs) != 128:
        raise AssertionError(f"expected 128 v144 jobs, got {len(jobs)}")
    return jobs


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_suite(output_dir: Path, seed_base: int = 144000) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(seed_base)
    prompts = "\n".join(str(job["base_prompt"]) for job in jobs) + "\n"
    manifest = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n"
        for job in jobs
    )
    prompt_path = output_dir / "v144_factorized_mechanism_128.txt"
    manifest_path = output_dir / "v144_factorized_mechanism_128.jsonl"
    prompt_path.write_text(prompts, encoding="utf-8")
    manifest_path.write_text(manifest, encoding="utf-8")
    metadata = {
        "version": 1,
        "job_count": len(jobs),
        "family_count": 16,
        "variants_per_family": len(VARIANTS),
        "variants": list(VARIANTS),
        "changed_factors": {
            key: list(value) for key, value in CHANGED_FACTORS.items()
        },
        "seed_base": int(seed_base),
        "matched_seed_variants": [
            variant for variant in VARIANTS if variant != "seed_control"
        ],
        "prompts_sha256": _sha256(prompts),
        "manifest_sha256": _sha256(manifest),
        "purpose": (
            "separate prompt surface form, semantic factor, and random "
            "trajectory effects on Q/K/V geometry and cache-policy demand"
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
    parser.add_argument("--seed-base", type=int, default=144000)
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
