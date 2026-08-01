#!/usr/bin/env python3
"""Build the independent v151 signed-policy and low-tail causal suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

try:
    from scripts.build_v148_axis_causal_suite import (
        HEADS,
        LAYERS,
        _load_axis_sources,
        _read_prompts,
    )
    from scripts.build_v150_policy_group_suite import _balanced_random_sets
except ModuleNotFoundError:
    from build_v148_axis_causal_suite import (
        HEADS,
        LAYERS,
        _load_axis_sources,
        _read_prompts,
    )
    from build_v150_policy_group_suite import _balanced_random_sets


SUITE = "v151_signed_policy_low_tail_core"
PER_LAYER_COUNT = 4
RANDOM_MAP_COUNT = 8
PROMPT_COUNT = 32
SEED_REPLICATES = (0, 1)
TARGET = 0.02
CALIBRATION_MIN_SCALE = 0.001
CALIBRATION_MAX_SCALE = 50.0
CALIBRATION_REFINEMENT_STEPS = 4
RANDOM_SEED = 20260803
CONTEXT_TIMESTEPS = (1000, 750, 500, 250)
INTERVENTIONS = {
    "uniform": (
        "policy_contrast",
        {"left": "uniform8", "right": "recent8"},
    ),
    "boundary": (
        "policy_contrast",
        {"left": "boundary8", "right": "recent8"},
    ),
    "key_shift": ("key_shift", {}),
    "value_shift": ("value_shift", {}),
}
FIXED_GROUPS = (
    "scalar_low4",
    "scalar_middle4",
    "scalar_high4",
    "signed_low4",
    "signed_middle4",
    "signed_high4",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_terms(prompt: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", prompt.lower())
        if len(token) > 2
    }


def select_holdout_indices(
    prompts: list[str], excluded: set[int], count: int = PROMPT_COUNT
) -> list[int]:
    if len(prompts) != 128:
        raise ValueError("v151 requires exactly 128 MovieBench prompts")
    if not excluded or min(excluded) < 0 or max(excluded) >= len(prompts):
        raise ValueError("v151 received invalid discovery prompt indices")
    candidates = [index for index in range(len(prompts)) if index not in excluded]
    if len(candidates) < count:
        raise ValueError("v151 does not have enough holdout prompts")
    terms = [_prompt_terms(prompt) for prompt in prompts]

    def distance(left: int, right: int) -> float:
        union = terms[left] | terms[right]
        return 0.0 if not union else 1.0 - len(terms[left] & terms[right]) / len(union)

    selected: list[int] = []
    while len(selected) < count:
        chosen = max(
            (index for index in candidates if index not in selected),
            key=lambda index: (
                min(
                    distance(index, existing)
                    for existing in (selected if selected else sorted(excluded))
                ),
                sum(distance(index, existing) for existing in excluded)
                / len(excluded),
                -index,
            ),
        )
        selected.append(chosen)
    return sorted(selected)


def _load_excluded_indices(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {int(value) for value in payload.get("source_prompt_indices", [])}
    if len(values) != 32:
        raise ValueError(
            f"v151 expected 32 v150 discovery indices, found {len(values)}"
        )
    return values


def _rank_partition(scores: list[list[float | None]]) -> dict[str, list]:
    maps = {"low4": {}, "middle4": {}, "high4": {}}
    for layer in range(LAYERS):
        ordered = sorted(
            range(HEADS),
            key=lambda head: (float(scores[layer][head]), head),
        )
        maps["low4"][str(layer)] = sorted(ordered[:PER_LAYER_COUNT])
        maps["middle4"][str(layer)] = sorted(
            ordered[PER_LAYER_COUNT : 2 * PER_LAYER_COUNT]
        )
        maps["high4"][str(layer)] = sorted(ordered[-PER_LAYER_COUNT:])
    return maps


def _load_signed_maps(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != 1:
        raise ValueError("v151 signed map has an unsupported version")
    maps = payload.get("maps") or {}
    if set(maps) != {"low4", "middle4", "high4"}:
        raise ValueError("v151 signed map is incomplete")
    for group, head_map in maps.items():
        if set(head_map) != {str(layer) for layer in range(LAYERS)}:
            raise ValueError(f"v151 signed {group} map has missing layers")
        for layer, heads in head_map.items():
            normalized = sorted({int(head) for head in heads})
            if len(normalized) != PER_LAYER_COUNT or not all(
                0 <= head < HEADS for head in normalized
            ):
                raise ValueError(
                    f"v151 signed map has invalid heads at {group}/{layer}"
                )
            head_map[layer] = normalized
    return maps, payload


def _coordinates(head_map: dict[str, list[int]]) -> set[tuple[int, int]]:
    return {
        (int(layer), int(head))
        for layer, heads in head_map.items()
        for head in heads
    }


def build_head_maps(
    scalar_scores: list[list[float | None]],
    signed_maps: dict,
    *,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict, dict]:
    scalar = _rank_partition(scalar_scores)
    maps = {
        "scalar_low4": scalar["low4"],
        "scalar_middle4": scalar["middle4"],
        "scalar_high4": scalar["high4"],
        "signed_low4": signed_maps["low4"],
        "signed_middle4": signed_maps["middle4"],
        "signed_high4": signed_maps["high4"],
        **{f"random{index}": {} for index in range(RANDOM_MAP_COUNT)},
    }
    random_usage = {}
    for layer in range(LAYERS):
        forbidden = {
            frozenset(maps[group][str(layer)]) for group in FIXED_GROUPS
        }
        random_sets, usage = _balanced_random_sets(
            layer=layer,
            forbidden=forbidden,
            random_seed=random_seed,
        )
        for index, heads in enumerate(random_sets):
            maps[f"random{index}"][str(layer)] = sorted(heads)
        random_usage[str(layer)] = usage

    scalar_high = _coordinates(maps["scalar_high4"])
    signed_high = _coordinates(maps["signed_high4"])
    scalar_low = _coordinates(maps["scalar_low4"])
    signed_low = _coordinates(maps["signed_low4"])
    diagnostics = {
        "scalar_signed_high_overlap": {
            "intersection": len(scalar_high & signed_high),
            "jaccard": len(scalar_high & signed_high)
            / len(scalar_high | signed_high),
        },
        "scalar_signed_low_overlap": {
            "intersection": len(scalar_low & signed_low),
            "jaccard": len(scalar_low & signed_low)
            / len(scalar_low | signed_low),
        },
        "random_usage_min": min(min(values) for values in random_usage.values()),
        "random_usage_max": max(max(values) for values in random_usage.values()),
        "random_head_usage_by_layer": random_usage,
    }
    for group, head_map in maps.items():
        if len(_coordinates(head_map)) != LAYERS * PER_LAYER_COUNT:
            raise AssertionError(f"v151 {group} map has the wrong size")
    return maps, diagnostics


def _probe(
    *, group: str, intervention: str, head_map: dict[str, list[int]]
) -> dict:
    policy, policy_args = INTERVENTIONS[intervention]
    return {
        "name": f"{group}_{intervention}_t020",
        "group": group,
        "rank_group": group,
        "policy": policy,
        "policy_args": policy_args,
        "intervention": intervention,
        "head_map": head_map,
        "calibration": {
            "mode": "projected_relative_rms",
            "target": TARGET,
            "min_scale": CALIBRATION_MIN_SCALE,
            "max_scale": CALIBRATION_MAX_SCALE,
            "refinement_steps": CALIBRATION_REFINEMENT_STEPS,
        },
        "target": TARGET,
        "selected_heads_per_layer": PER_LAYER_COUNT,
        "control_family": (
            "balanced_random" if group.startswith("random") else "rank_partition"
        ),
    }


def build_probe_plan(
    maps: dict,
    *,
    source: dict,
) -> dict:
    probes = []
    probe_lookup = {}
    for group in FIXED_GROUPS:
        for intervention in INTERVENTIONS:
            probe = _probe(
                group=group,
                intervention=intervention,
                head_map=maps[group],
            )
            probes.append(probe)
            probe_lookup[(group, intervention)] = probe["name"]
    random_probes = []
    for index in range(RANDOM_MAP_COUNT):
        group = f"random{index}"
        probe = _probe(
            group=group,
            intervention="uniform",
            head_map=maps[group],
        )
        probes.append(probe)
        random_probes.append(probe["name"])
    if len(probes) != 32 or len({probe["name"] for probe in probes}) != 32:
        raise AssertionError("v151 probe grid must contain 32 unique probes")

    families = {}
    for family in ("scalar", "signed"):
        groups = {
            role: f"{family}_{role}4" for role in ("low", "middle", "high")
        }
        families[family] = {
            "groups": groups,
            "probes": {
                intervention: {
                    role: probe_lookup[(group, intervention)]
                    for role, group in groups.items()
                }
                for intervention in INTERVENTIONS
            },
            "random_uniform_probes": random_probes,
        }
    return {
        "version": 1,
        "suite": SUITE,
        "layers": LAYERS,
        "heads": HEADS,
        "contexts": [
            {
                "mode": "noisy",
                "current_frame": 117,
                "nominal_timestep": timestep,
            }
            for timestep in CONTEXT_TIMESTEPS
        ],
        "probes": probes,
        "families": families,
        "random_control_count": RANDOM_MAP_COUNT,
        "calibration_target": TARGET,
        "source": source,
        "claim_boundary": (
            "The scalar branch is an independent confirmation of the v150 "
            "low-tail observation. The signed branch tests a v145-discovered "
            "scene-specific policy-preference map. Neither branch measures "
            "trajectory-level video quality."
        ),
    }


def _build_jobs(
    prompts: list[str], indices: list[int], *, seed_base: int
) -> list[dict]:
    jobs = []
    for prompt_slot, source_index in enumerate(indices):
        for replicate in SEED_REPLICATES:
            seed = int(seed_base) + source_index + replicate * 10000
            jobs.append(
                {
                    "dataset_index": len(jobs),
                    "job_id": (
                        f"v151_p{prompt_slot:02d}_src{source_index:03d}_"
                        f"s{replicate}"
                    ),
                    "kind": SUITE,
                    "prompt_slot": prompt_slot,
                    "source_prompt_index": source_index,
                    "seed_replicate": replicate,
                    "seed": seed,
                    "reference_seed": seed,
                    "base_prompt": prompts[source_index],
                    "reference_prompt": prompts[source_index],
                }
            )
    return jobs


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    v145_analysis_dir: Path,
    signed_map_path: Path,
    v150_suite_metadata: Path,
    seed_base: int = 151000,
    random_seed: int = RANDOM_SEED,
) -> dict:
    prompts = _read_prompts(natural_prompts)
    excluded = _load_excluded_indices(v150_suite_metadata)
    selected = select_holdout_indices(prompts, excluded)
    if excluded.intersection(selected):
        raise AssertionError("v151 holdout prompt selection leaked v150 prompts")

    sources = _load_axis_sources(v145_analysis_dir)
    signed_maps, signed_payload = _load_signed_maps(signed_map_path)
    maps, map_diagnostics = build_head_maps(
        sources["scores"]["policy"],
        signed_maps,
        random_seed=random_seed,
    )
    source = {
        "scalar_axis": sources["metadata"]["policy"],
        "scalar_score": "v145 full_semantic/all_policy_shift_mean",
        "scalar_hypothesis_origin": "post-hoc v150 low-tail observation",
        "signed_axis": signed_payload["axis"],
        "signed_source_screen_pass": bool(
            signed_payload["source_screen_pass"]
        ),
        "signed_map_sha256": _sha256_file(signed_map_path),
        "v145_analysis_dir": str(v145_analysis_dir),
        "v150_suite_metadata": str(v150_suite_metadata),
        "v150_excluded_prompt_indices": sorted(excluded),
        "map_diagnostics": map_diagnostics,
        "selection_rule": (
            "all fixed maps are frozen before holdout generation; random "
            "maps are balanced and forbidden from matching either partition"
        ),
    }
    plan = build_probe_plan(maps, source=source)
    jobs = _build_jobs(prompts, selected, seed_base=seed_base)
    prompt_text = "\n".join(job["base_prompt"] for job in jobs) + "\n"
    manifest_text = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n" for job in jobs
    )
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v151_core_64.txt").write_text(prompt_text, encoding="utf-8")
    (output_dir / "v151_core_64.jsonl").write_text(
        manifest_text, encoding="utf-8"
    )
    (output_dir / "v151_probe_plan.json").write_text(
        plan_text, encoding="utf-8"
    )
    metadata = {
        "version": 1,
        "suite": SUITE,
        "job_count": len(jobs),
        "unique_prompt_count": len(selected),
        "seed_replicates": list(SEED_REPLICATES),
        "seed_base": int(seed_base),
        "source_prompt_indices": selected,
        "excluded_v150_prompt_indices": sorted(excluded),
        "probe_count_excluding_native_replay": len(plan["probes"]),
        "downstream_context_count": len(CONTEXT_TIMESTEPS),
        "expected_downstream_records_per_profile": (
            len(plan["probes"]) + 1
        )
        * len(CONTEXT_TIMESTEPS),
        "prompts_sha256": _sha256_text(prompt_text),
        "manifest_sha256": _sha256_text(manifest_text),
        "plan_sha256": _sha256_text(plan_text),
        "natural_prompt_source": str(natural_prompts),
        "signed_source_screen_pass": source["signed_source_screen_pass"],
        "map_diagnostics": map_diagnostics,
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
    parser.add_argument("--v145-analysis-dir", type=Path, required=True)
    parser.add_argument("--signed-map", type=Path, required=True)
    parser.add_argument("--v150-suite-metadata", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=151000)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(
                args.output_dir,
                natural_prompts=args.natural_prompts,
                v145_analysis_dir=args.v145_analysis_dir,
                signed_map_path=args.signed_map,
                v150_suite_metadata=args.v150_suite_metadata,
                seed_base=args.seed_base,
                random_seed=args.random_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
