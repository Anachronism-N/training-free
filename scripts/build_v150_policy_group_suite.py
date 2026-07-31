#!/usr/bin/env python3
"""Build the v150 policy-group confirmation profiling suites."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

try:
    from scripts.build_v148_axis_causal_suite import (
        CORE_PROMPTS,
        DOSE_PROMPTS,
        HEADS,
        LAYERS,
        SEED_REPLICATES,
        _build_jobs,
        _load_axis_sources,
        _read_prompts,
        _select_prompt_indices,
        _sha256,
        _write_jobs,
    )
except ModuleNotFoundError:
    from build_v148_axis_causal_suite import (
        CORE_PROMPTS,
        DOSE_PROMPTS,
        HEADS,
        LAYERS,
        SEED_REPLICATES,
        _build_jobs,
        _load_axis_sources,
        _read_prompts,
        _select_prompt_indices,
        _sha256,
        _write_jobs,
    )


AXIS = "policy"
INTERVENTIONS = ("key_shift", "value_shift", "policy_contrast")
PRIMARY_INTERVENTION = "policy_contrast"
POLICY_ARGS = {
    "policy_contrast": {"left": "uniform8", "right": "recent8"}
}
PER_LAYER_COUNT = 4
RANDOM_MAP_COUNT = 8
CORE_TARGET = 0.02
STRENGTH_TARGETS = (0.01, 0.02, 0.05)
CALIBRATION_MIN_SCALE = 0.001
CALIBRATION_MAX_SCALE = 50.0
RANDOM_SEED = 20260802


def _target_tag(target: float) -> str:
    scaled = int(round(float(target) * 1000))
    if not 0 < scaled < 1000:
        raise ValueError(f"unsupported calibration target: {target}")
    return f"t{scaled:03d}"


def _calibration(target: float) -> dict:
    return {
        "mode": "projected_relative_rms",
        "target": float(target),
        "min_scale": CALIBRATION_MIN_SCALE,
        "max_scale": CALIBRATION_MAX_SCALE,
    }


def _map_coordinates(head_map: dict[str, list[int]]) -> set[tuple[int, int]]:
    return {
        (int(layer), int(head))
        for layer, heads in head_map.items()
        for head in heads
    }


def _mean_map_score(
    head_map: dict[str, list[int]], scores: list[list[float | None]]
) -> float:
    values = [
        float(scores[layer][head])
        for layer, head in sorted(_map_coordinates(head_map))
    ]
    return sum(values) / len(values)


def _balanced_random_sets(
    *,
    layer: int,
    forbidden: set[frozenset[int]],
    random_seed: int,
) -> tuple[list[list[int]], list[int]]:
    """Select eight unique four-head controls with near-uniform usage."""

    rng = random.Random(random_seed + 104729 * layer)
    candidates = [
        tuple(values)
        for values in itertools.combinations(range(HEADS), PER_LAYER_COUNT)
        if frozenset(values) not in forbidden
    ]
    usage = [0] * HEADS
    selected: list[list[int]] = []
    seen = set(forbidden)
    for _ in range(RANDOM_MAP_COUNT):
        available = [
            values for values in candidates if frozenset(values) not in seen
        ]
        rng.shuffle(available)

        def objective(values: tuple[int, ...]) -> tuple[int, int, int]:
            updated = [
                count + int(head in values)
                for head, count in enumerate(usage)
            ]
            return (
                max(updated) - min(updated),
                sum(count * count for count in updated),
                max(updated),
            )

        chosen = min(available, key=objective)
        selected.append(list(chosen))
        seen.add(frozenset(chosen))
        for head in chosen:
            usage[head] += 1
    if max(usage) - min(usage) > 1:
        raise AssertionError(
            f"layer {layer} random-map usage is unbalanced: {usage}"
        )
    return selected, usage


def build_policy_maps(
    scores: list[list[float | None]],
    *,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[str, dict[str, list[int]]], dict]:
    maps: dict[str, dict[str, list[int]]] = {
        "top4": {},
        "bottom4": {},
        "middle4": {},
        **{f"random{index}": {} for index in range(RANDOM_MAP_COUNT)},
    }
    usage_by_layer = {}
    for layer in range(LAYERS):
        ordered = sorted(
            range(HEADS),
            key=lambda head: (float(scores[layer][head]), head),
        )
        bottom = sorted(ordered[:PER_LAYER_COUNT])
        middle = sorted(ordered[PER_LAYER_COUNT : 2 * PER_LAYER_COUNT])
        top = sorted(ordered[-PER_LAYER_COUNT:])
        partition = {frozenset(bottom), frozenset(middle), frozenset(top)}
        if len(partition) != 3 or set().union(*partition) != set(range(HEADS)):
            raise AssertionError(f"layer {layer} rank partition is invalid")
        random_sets, usage = _balanced_random_sets(
            layer=layer,
            forbidden=partition,
            random_seed=random_seed,
        )
        layer_key = str(layer)
        maps["bottom4"][layer_key] = bottom
        maps["middle4"][layer_key] = middle
        maps["top4"][layer_key] = top
        for index, values in enumerate(random_sets):
            maps[f"random{index}"][layer_key] = sorted(values)
        usage_by_layer[layer_key] = usage

    signatures = {
        group: tuple(
            tuple(head_map[str(layer)]) for layer in range(LAYERS)
        )
        for group, head_map in maps.items()
    }
    if len(set(signatures.values())) != len(signatures):
        raise AssertionError("v150 contains duplicate global head maps")
    for group, head_map in maps.items():
        if len(_map_coordinates(head_map)) != LAYERS * PER_LAYER_COUNT:
            raise AssertionError(f"{group} has an invalid selected-head count")

    top_coordinates = _map_coordinates(maps["top4"])
    diagnostics = {
        "group_score_means": {
            group: _mean_map_score(head_map, scores)
            for group, head_map in maps.items()
        },
        "top_overlap": {
            group: {
                "intersection": len(top_coordinates & _map_coordinates(head_map)),
                "jaccard": len(top_coordinates & _map_coordinates(head_map))
                / len(top_coordinates | _map_coordinates(head_map)),
            }
            for group, head_map in maps.items()
            if group != "top4"
        },
        "random_head_usage_by_layer": usage_by_layer,
        "random_usage_min": min(
            min(values) for values in usage_by_layer.values()
        ),
        "random_usage_max": max(
            max(values) for values in usage_by_layer.values()
        ),
    }
    return maps, diagnostics


def _probe(
    *,
    group: str,
    policy: str,
    target: float,
    head_map: dict[str, list[int]],
) -> dict:
    tag = _target_tag(target)
    return {
        "name": f"policy_{group}_{policy}_{tag}",
        "group": f"policy_{group}",
        "policy": policy,
        "policy_args": dict(POLICY_ARGS.get(policy, {})),
        "calibration": _calibration(target),
        "head_map": head_map,
        "axis": AXIS,
        "rank_group": group,
        "control_family": (
            "balanced_random" if group.startswith("random") else "rank_group"
        ),
        "dose": PER_LAYER_COUNT,
        "target": float(target),
    }


def _comparison(
    *,
    policy: str,
    target: float,
    groups: dict[str, str],
) -> dict:
    return {
        "axis": AXIS,
        "policy": policy,
        "target": float(target),
        "top_probe": groups["top4"],
        "bottom_probe": groups["bottom4"],
        "middle_probe": groups["middle4"],
        "random_probes": [
            groups[f"random{index}"] for index in range(RANDOM_MAP_COUNT)
        ],
    }


def _build_plan(
    *,
    suite: str,
    maps: dict[str, dict[str, list[int]]],
    policies: tuple[str, ...],
    targets: tuple[float, ...],
    source_metadata: dict,
) -> dict:
    probes = []
    comparisons = []
    for target in targets:
        for policy in policies:
            names = {}
            for group, head_map in maps.items():
                probe = _probe(
                    group=group,
                    policy=policy,
                    target=target,
                    head_map=head_map,
                )
                probes.append(probe)
                names[group] = probe["name"]
            comparisons.append(
                _comparison(policy=policy, target=target, groups=names)
            )
    expected = len(maps) * len(policies) * len(targets)
    if len(probes) != expected or len({row["name"] for row in probes}) != expected:
        raise AssertionError("v150 probe grid is incomplete or duplicated")
    return {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "suite": suite,
        "source": source_metadata,
        "probes": probes,
        "comparisons": comparisons,
        "primary_intervention": PRIMARY_INTERVENTION,
        "random_control_count": RANDOM_MAP_COUNT,
        "calibration_targets": list(targets),
    }


def build_probe_plans(
    analysis_dir: Path,
    *,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict, dict]:
    source = _load_axis_sources(analysis_dir)
    scores = source["scores"][AXIS]
    maps, map_diagnostics = build_policy_maps(
        scores, random_seed=random_seed
    )
    source_metadata = {
        "experiment": "v145_crossed_seed_head_profile",
        "analysis_dir": str(analysis_dir),
        "feature_audit_sha256": _sha256(
            (analysis_dir / "feature_reproducibility_audit.csv").read_text(
                encoding="utf-8"
            )
        ),
        "head_scores_sha256": _sha256(
            (analysis_dir / "head_factor_reproducibility.csv").read_text(
                encoding="utf-8"
            )
        ),
        "factor": "full_semantic",
        "axis": source["metadata"][AXIS],
        "random_seed": int(random_seed),
        "per_layer_count": PER_LAYER_COUNT,
        "random_map_count": RANDOM_MAP_COUNT,
        "map_diagnostics": map_diagnostics,
        "policy_contrast": POLICY_ARGS[PRIMARY_INTERVENTION],
        "calibration": {
            "min_scale": CALIBRATION_MIN_SCALE,
            "max_scale": CALIBRATION_MAX_SCALE,
            "core_target": CORE_TARGET,
            "strength_targets": list(STRENGTH_TARGETS),
        },
        "claim_boundary": (
            "v150 tests only the v145 full-semantic policy ranking. PF labels "
            "do not define or filter any head map. Eight deterministic, "
            "balanced, count-matched random maps are independent controls."
        ),
    }
    core = _build_plan(
        suite="v150_policy_group_core",
        maps=maps,
        policies=INTERVENTIONS,
        targets=(CORE_TARGET,),
        source_metadata=source_metadata,
    )
    strength = _build_plan(
        suite="v150_policy_group_strength",
        maps=maps,
        policies=(PRIMARY_INTERVENTION,),
        targets=STRENGTH_TARGETS,
        source_metadata=source_metadata,
    )
    if len(core["probes"]) != 33 or len(strength["probes"]) != 33:
        raise AssertionError("v150 requires 33 probes in each suite")
    return core, strength


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    diverse_index: Path,
    v145_analysis_dir: Path,
    seed_base: int,
    random_seed: int = RANDOM_SEED,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(natural_prompts)
    source_indices = _select_prompt_indices(diverse_index, prompts)
    core_plan, strength_plan = build_probe_plans(
        v145_analysis_dir, random_seed=random_seed
    )
    core_jobs = _build_jobs(
        prompts,
        source_indices,
        seed_base=seed_base,
        kind=core_plan["suite"],
    )
    strength_jobs = _build_jobs(
        prompts,
        source_indices[:DOSE_PROMPTS],
        seed_base=seed_base,
        kind=strength_plan["suite"],
    )
    core_prompt_text, core_manifest_text = _write_jobs(
        output_dir, prefix="v150_policy_core_64", jobs=core_jobs
    )
    strength_prompt_text, strength_manifest_text = _write_jobs(
        output_dir, prefix="v150_policy_strength_32", jobs=strength_jobs
    )
    core_plan_text = json.dumps(core_plan, indent=2, sort_keys=True) + "\n"
    strength_plan_text = (
        json.dumps(strength_plan, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "v150_policy_core_plan.json").write_text(
        core_plan_text, encoding="utf-8"
    )
    (output_dir / "v150_policy_strength_plan.json").write_text(
        strength_plan_text, encoding="utf-8"
    )
    metadata = {
        "version": 1,
        "seed_base": int(seed_base),
        "seed_replicates": list(SEED_REPLICATES),
        "source_prompt_indices": source_indices,
        "natural_prompt_source": str(natural_prompts),
        "diverse_index_source": str(diverse_index),
        "v145_analysis_source": str(v145_analysis_dir),
        "random_seed": int(random_seed),
        "core": {
            "job_count": len(core_jobs),
            "unique_prompt_count": CORE_PROMPTS,
            "probe_count_excluding_native_replay": len(core_plan["probes"]),
            "downstream_context_count": 2,
            "expected_downstream_records_per_profile": 2
            * (len(core_plan["probes"]) + 1),
            "prompts_sha256": _sha256(core_prompt_text),
            "manifest_sha256": _sha256(core_manifest_text),
            "plan_sha256": _sha256(core_plan_text),
        },
        "strength": {
            "job_count": len(strength_jobs),
            "unique_prompt_count": DOSE_PROMPTS,
            "probe_count_excluding_native_replay": len(
                strength_plan["probes"]
            ),
            "downstream_context_count": 2,
            "expected_downstream_records_per_profile": 2
            * (len(strength_plan["probes"]) + 1),
            "prompts_sha256": _sha256(strength_prompt_text),
            "manifest_sha256": _sha256(strength_manifest_text),
            "plan_sha256": _sha256(strength_plan_text),
        },
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
    parser.add_argument("--seed-base", type=int, default=148000)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(
                args.output_dir,
                natural_prompts=args.natural_prompts,
                diverse_index=args.diverse_index,
                v145_analysis_dir=args.v145_analysis_dir,
                seed_base=args.seed_base,
                random_seed=args.random_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
