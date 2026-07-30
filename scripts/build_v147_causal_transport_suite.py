#!/usr/bin/env python3
"""Build the v147 paired-seed causal transport suite and probe plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path


LAYERS = 30
HEADS = 12
PROMPT_COUNT = 32
SEED_REPLICATES = (0, 1)
AXIS_PRIORITY = {
    "k_shift": 5,
    "policy_shift": 4,
    "q_shift": 3,
    "v_shift": 2,
    "value_scale_shift": 1,
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str | float | int | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return result if math.isfinite(result) else float("-inf")


def _json_finite(value: str | float | int | None) -> float | None:
    result = _finite(value)
    return result if math.isfinite(result) else None


def select_source_axis(analysis_dir: Path) -> dict:
    audit_path = analysis_dir / "feature_reproducibility_audit.csv"
    rows = _read_csv(audit_path)
    eligible = [
        row
        for row in rows
        if row.get("variant") in {"identity", "scene", "full_semantic"}
        and row.get("axis") in AXIS_PRIORITY
    ]
    if not eligible:
        raise RuntimeError("v145 audit has no eligible cache-related axes")

    def score(row: dict) -> tuple:
        family = _finite(row["layer_residual_family_split_spearman"])
        seed = _finite(row["layer_residual_seed_replicate_spearman"])
        reproducible = int(row["reproducible_factor_axis_candidate"])
        direction = _finite(row["median_seed_delta_direction_cosine"])
        specificity = _finite(
            row["median_cross_factor_specificity_margin"]
        )
        descriptor_bonus = min(direction, specificity)
        if row["axis"] not in {"q_shift", "k_shift", "v_shift"}:
            descriptor_bonus = 0.0
        return (
            reproducible,
            min(family, seed),
            descriptor_bonus,
            AXIS_PRIORITY[row["axis"]],
            row["variant"],
        )

    selected = max(eligible, key=score)
    return {
        "variant": selected["variant"],
        "axis": selected["axis"],
        "screening_pass": bool(
            int(selected["reproducible_factor_axis_candidate"])
        ),
        "layer_residual_family_split_spearman": _json_finite(
            selected["layer_residual_family_split_spearman"]
        ),
        "layer_residual_seed_replicate_spearman": _json_finite(
            selected["layer_residual_seed_replicate_spearman"]
        ),
        "median_seed_delta_direction_cosine": _json_finite(
            selected["median_seed_delta_direction_cosine"]
        ),
        "median_cross_factor_specificity_margin": _json_finite(
            selected["median_cross_factor_specificity_margin"]
        ),
        "audit_path": str(audit_path),
    }


def build_head_maps(
    analysis_dir: Path,
    source_axis: dict,
    *,
    per_layer_count: int,
    random_seed: int,
) -> dict[str, dict[int, list[int]]]:
    if not 1 <= per_layer_count <= HEADS // 3:
        raise ValueError("per-layer head count must be in [1, 4]")
    rows = _read_csv(
        analysis_dir / "head_factor_reproducibility.csv"
    )
    selected = [
        row
        for row in rows
        if row["variant"] == source_axis["variant"]
    ]
    if len(selected) != LAYERS * HEADS:
        raise RuntimeError(
            f"expected {LAYERS * HEADS} v145 head rows, got {len(selected)}"
        )
    field = f"all_{source_axis['axis']}_mean"
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for row in selected:
        layer = int(row["layer"])
        head = int(row["head"])
        value = _finite(row[field])
        if not math.isfinite(value):
            raise RuntimeError(
                f"v145 head ranking is non-finite at layer={layer}, head={head}"
            )
        if not 0 <= layer < LAYERS or not 0 <= head < HEADS:
            raise RuntimeError("v145 head coordinate is out of range")
        by_layer.setdefault(layer, []).append((head, value))
    if set(by_layer) != set(range(LAYERS)):
        raise RuntimeError("v145 head rows do not cover all layers")
    rng = random.Random(random_seed)
    maps = {"top": {}, "bottom": {}, "random": {}, "all": {}}
    for layer in range(LAYERS):
        values = by_layer[layer]
        if len(values) != HEADS:
            raise RuntimeError(f"layer {layer} does not contain 12 heads")
        median = sorted(value for _, value in values)[HEADS // 2 - 1:HEADS // 2 + 1]
        median = sum(median) / len(median)
        ranked = sorted(
            ((head, value - median) for head, value in values),
            key=lambda item: (item[1], item[0]),
        )
        maps["bottom"][layer] = sorted(
            head for head, _ in ranked[:per_layer_count]
        )
        maps["top"][layer] = sorted(
            head for head, _ in ranked[-per_layer_count:]
        )
        control_pool = sorted(
            set(range(HEADS))
            - set(maps["bottom"][layer])
            - set(maps["top"][layer])
        )
        maps["random"][layer] = sorted(
            rng.sample(control_pool, per_layer_count)
        )
        maps["all"][layer] = list(range(HEADS))
    return maps


def build_probe_plan(
    analysis_dir: Path,
    *,
    per_layer_count: int = 3,
    random_seed: int = 20260731,
) -> dict:
    source_axis = select_source_axis(analysis_dir)
    maps = build_head_maps(
        analysis_dir,
        source_axis,
        per_layer_count=per_layer_count,
        random_seed=random_seed,
    )
    probes = []

    def add(group: str, policy: str) -> None:
        probes.append(
            {
                "name": f"{group}_{policy}",
                "group": group,
                "policy": policy,
                "head_map": {
                    str(layer): heads
                    for layer, heads in maps[group].items()
                },
            }
        )

    for group in ("top", "bottom", "random"):
        add(group, "recent4")
    add("all", "recent4")
    for group in ("top", "random"):
        add(group, "uniform8")
    for group in ("top", "bottom", "random"):
        add(group, "q_retrieval8")
    for group in ("top", "bottom", "random"):
        add(group, "value_shift")
    for band, layers in (
        ("early", range(0, 10)),
        ("middle", range(10, 20)),
        ("late", range(20, 30)),
    ):
        probes.append(
            {
                "name": f"top_recent4_{band}",
                "group": f"top_{band}",
                "policy": "recent4",
                "head_map": {
                    str(layer): maps["top"][layer] for layer in layers
                },
            }
        )
    overlap = {}
    for layer in range(LAYERS):
        for left, right in (("top", "random"), ("bottom", "random")):
            key = f"{left}_{right}"
            overlap.setdefault(key, 0)
            overlap[key] += len(
                set(maps[left][layer]) & set(maps[right][layer])
            )
    return {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "source": {
            "experiment": "v145_crossed_seed_head_profile",
            "analysis_dir": str(analysis_dir),
            "selected_axis": source_axis,
            "selection_rule": (
                "prefer passed identity/scene/full-semantic axes, then "
                "maximize the minimum held-out-family and seed-replicate "
                "Spearman; break ties toward K/policy/Q/V/value-scale"
            ),
            "per_layer_count": per_layer_count,
            "random_seed": random_seed,
            "random_overlap_with_extremes": overlap,
        },
        "probes": probes,
        "claim_boundary": (
            "The v145 axis only defines a preregistered ranking. Functional "
            "head roles require top-vs-bottom and top-vs-random downstream "
            "effects in v147."
        ),
    }


def _read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 128 or len(set(prompts)) != 128:
        raise ValueError("v147 requires 128 unique rewritten MovieBench prompts")
    return prompts


def _prompt_terms(prompt: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", prompt.lower())
        if len(token) > 2
    }


def _select_prompt_indices(
    diverse_index: Path, prompts: list[str]
) -> list[int]:
    payload = json.loads(diverse_index.read_text(encoding="utf-8"))
    selected = {
        int(item["source_index"]) for item in payload.get("items") or ()
    }
    if len(selected) != 16 or min(selected) < 0 or max(selected) >= 128:
        raise ValueError("the frozen diverse subset must contain 16 indices")
    terms = [_prompt_terms(prompt) for prompt in prompts]

    def lexical_distance(left: int, right: int) -> float:
        union = terms[left] | terms[right]
        if not union:
            return 0.0
        return 1.0 - len(terms[left] & terms[right]) / len(union)

    while len(selected) < PROMPT_COUNT:
        candidates = [index for index in range(128) if index not in selected]
        chosen = max(
            candidates,
            key=lambda index: (
                min(
                    lexical_distance(index, existing)
                    for existing in selected
                ),
                min(abs(index - existing) for existing in selected),
                -index,
            ),
        )
        selected.add(chosen)
    return sorted(selected)


def build_jobs(
    prompts: list[str],
    source_indices: list[int],
    *,
    seed_base: int,
) -> list[dict]:
    jobs = []
    for prompt_slot, source_index in enumerate(source_indices):
        for replicate in SEED_REPLICATES:
            seed = seed_base + source_index + replicate * 10000
            jobs.append(
                {
                    "dataset_index": len(jobs),
                    "job_id": (
                        f"v147_p{prompt_slot:02d}_"
                        f"src{source_index:03d}_s{replicate}"
                    ),
                    "kind": "causal_transport_profile",
                    "prompt_slot": prompt_slot,
                    "source_prompt_index": source_index,
                    "seed_replicate": replicate,
                    "seed": seed,
                    "reference_seed": seed,
                    "base_prompt": prompts[source_index],
                }
            )
    if len(jobs) != PROMPT_COUNT * len(SEED_REPLICATES):
        raise AssertionError("v147 job count differs")
    return jobs


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    diverse_index: Path,
    v145_analysis_dir: Path,
    seed_base: int,
    per_layer_count: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(natural_prompts)
    source_indices = _select_prompt_indices(diverse_index, prompts)
    jobs = build_jobs(prompts, source_indices, seed_base=seed_base)
    plan = build_probe_plan(
        v145_analysis_dir,
        per_layer_count=per_layer_count,
    )
    prompt_text = "\n".join(job["base_prompt"] for job in jobs) + "\n"
    manifest_text = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n"
        for job in jobs
    )
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    (output_dir / "v147_causal_transport_64.txt").write_text(
        prompt_text, encoding="utf-8"
    )
    (output_dir / "v147_causal_transport_64.jsonl").write_text(
        manifest_text, encoding="utf-8"
    )
    (output_dir / "v147_downstream_probe_plan.json").write_text(
        plan_text, encoding="utf-8"
    )
    metadata = {
        "version": 1,
        "job_count": len(jobs),
        "unique_prompt_count": PROMPT_COUNT,
        "seed_replicates": list(SEED_REPLICATES),
        "source_prompt_indices": source_indices,
        "seed_base": seed_base,
        "probe_count_excluding_native_replay": len(plan["probes"]),
        "downstream_context_count": 3,
        "expected_downstream_records_per_profile": (
            len(plan["probes"]) + 1
        )
        * 3,
        "prompts_sha256": _sha256(prompt_text),
        "manifest_sha256": _sha256(manifest_text),
        "probe_plan_sha256": _sha256(plan_text),
        "natural_prompt_source": str(natural_prompts),
        "diverse_index_source": str(diverse_index),
        "v145_analysis_source": str(v145_analysis_dir),
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
    parser.add_argument("--diverse-index", type=Path, required=True)
    parser.add_argument("--v145-analysis-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=147000)
    parser.add_argument("--per-layer-count", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(
                args.output_dir,
                natural_prompts=args.natural_prompts,
                diverse_index=args.diverse_index,
                v145_analysis_dir=args.v145_analysis_dir,
                seed_base=args.seed_base,
                per_layer_count=args.per_layer_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
