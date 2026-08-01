#!/usr/bin/env python3
"""Build the v152 native-state online policy-selection profiling suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.build_v148_axis_causal_suite import HEADS, LAYERS, _read_prompts
    from scripts.build_v150_policy_group_suite import _balanced_random_sets
except ModuleNotFoundError:
    from build_v148_axis_causal_suite import HEADS, LAYERS, _read_prompts
    from build_v150_policy_group_suite import _balanced_random_sets


SUITE = "v152_online_policy_core"
PROMPT_COUNT = 64
SEED_REPLICATES = (0, 1)
PER_LAYER_COUNT = 4
CONTEXT_TIMESTEPS = (1000, 750, 500, 250)
POLICIES = ("uniform8", "recent8")
RANDOM_MAP_COUNT = 4
RANDOM_SEED = 20260804

DYNAMIC_GROUPS = {
    "oracle_uniform4": {
        "type": "policy_error_margin",
        "direction": "high",
        "expected_policy": "uniform8",
    },
    "oracle_recent4": {
        "type": "policy_error_margin",
        "direction": "low",
        "expected_policy": "recent8",
    },
    "qk_uniform4": {
        "type": "qk_policy_margin",
        "direction": "high",
        "expected_policy": "uniform8",
    },
    "qk_recent4": {
        "type": "qk_policy_margin",
        "direction": "low",
        "expected_policy": "recent8",
    },
    "mass_uniform4": {
        "type": "old_history_mass",
        "direction": "high",
        "expected_policy": "uniform8",
    },
    "mass_recent4": {
        "type": "old_history_mass",
        "direction": "low",
        "expected_policy": "recent8",
    },
}
STATIC_GROUPS = (
    "signed_high4",
    "signed_low4",
    "random0",
    "random1",
    "random2",
    "random3",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_indices(path: Path, *, expected: int) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {int(value) for value in payload.get("source_prompt_indices", [])}
    if len(values) != expected or any(value < 0 or value >= 128 for value in values):
        raise ValueError(f"{path} does not contain {expected} valid prompt indices")
    return values


def select_holdout_indices(
    *, v150_indices: set[int], v151_indices: set[int]
) -> list[int]:
    if v150_indices & v151_indices:
        raise ValueError("v150 and v151 prompt sets unexpectedly overlap")
    excluded = v150_indices | v151_indices
    selected = sorted(set(range(128)) - excluded)
    if len(selected) != PROMPT_COUNT:
        raise ValueError(
            f"v152 requires the 64 prompts unused by v150/v151, found {len(selected)}"
        )
    return selected


def _load_signed_maps(path: Path) -> dict[str, dict[str, list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    maps = payload.get("maps") or {}
    if set(maps) != {"low4", "middle4", "high4"}:
        raise ValueError("v151 signed source map is incomplete")
    normalized = {}
    for role in ("low4", "high4"):
        head_map = maps[role]
        if set(head_map) != {str(layer) for layer in range(LAYERS)}:
            raise ValueError(f"v151 signed {role} has incomplete layers")
        normalized[role] = {}
        for layer in range(LAYERS):
            heads = sorted({int(value) for value in head_map[str(layer)]})
            if len(heads) != PER_LAYER_COUNT or any(
                head < 0 or head >= HEADS for head in heads
            ):
                raise ValueError(f"v151 signed {role}/{layer} is invalid")
            normalized[role][str(layer)] = heads
    return normalized


def build_static_maps(
    signed_maps: dict[str, dict[str, list[int]]],
    *,
    random_seed: int = RANDOM_SEED,
) -> dict[str, dict[str, list[int]]]:
    maps = {
        "signed_high4": signed_maps["high4"],
        "signed_low4": signed_maps["low4"],
        **{f"random{index}": {} for index in range(RANDOM_MAP_COUNT)},
    }
    for layer in range(LAYERS):
        forbidden = {
            frozenset(maps["signed_high4"][str(layer)]),
            frozenset(maps["signed_low4"][str(layer)]),
        }
        random_sets, _ = _balanced_random_sets(
            layer=layer,
            forbidden=forbidden,
            random_seed=random_seed,
        )
        for index in range(RANDOM_MAP_COUNT):
            maps[f"random{index}"][str(layer)] = sorted(random_sets[index])
    return maps


def _dynamic_selector(spec: dict) -> dict:
    return {
        "type": spec["type"],
        "direction": spec["direction"],
        "heads_per_layer": PER_LAYER_COUNT,
        "budget_frames": 8,
        "recent_frames": 4,
        "spatial_samples": 8,
    }


def build_probe_plan(
    static_maps: dict[str, dict[str, list[int]]],
    *,
    source: dict,
) -> dict:
    probes = []
    pair_probes = {}
    group_metadata = {}
    for group, spec in DYNAMIC_GROUPS.items():
        group_metadata[group] = {
            "kind": "dynamic",
            "selector": _dynamic_selector(spec),
            "expected_policy": spec["expected_policy"],
        }
        pair_probes[group] = {}
        for policy in POLICIES:
            name = f"{group}_{policy}"
            probes.append(
                {
                    "name": name,
                    "group": group,
                    "policy": policy,
                    "head_selector": _dynamic_selector(spec),
                }
            )
            pair_probes[group][policy] = name
    for group in STATIC_GROUPS:
        group_metadata[group] = {
            "kind": "static_control",
            "expected_policy": None,
        }
        pair_probes[group] = {}
        for policy in POLICIES:
            name = f"{group}_{policy}"
            probes.append(
                {
                    "name": name,
                    "group": group,
                    "policy": policy,
                    "head_map": static_maps[group],
                }
            )
            pair_probes[group][policy] = name
    if len(probes) != 24 or len({probe["name"] for probe in probes}) != 24:
        raise AssertionError("v152 requires 24 unique downstream probes")
    return {
        "version": 2,
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
        "pair_probes": pair_probes,
        "groups": group_metadata,
        "policies": list(POLICIES),
        "source": source,
        "claim_boundary": (
            "Selectors are frozen from native replay and tested with equal-budget "
            "one-step policy replacements. The oracle selector is an upper bound; "
            "only qk_policy_margin is online-computable from a shared candidate "
            "bank. This experiment does not measure trajectory-level video quality."
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
                        f"v152_p{prompt_slot:02d}_src{source_index:03d}_"
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
    v150_suite_metadata: Path,
    v151_suite_metadata: Path,
    signed_map_path: Path,
    seed_base: int = 152000,
    random_seed: int = RANDOM_SEED,
) -> dict:
    prompts = _read_prompts(natural_prompts)
    if len(prompts) != 128:
        raise ValueError("v152 requires exactly 128 MovieBench prompts")
    v150_indices = _load_indices(v150_suite_metadata, expected=32)
    v151_indices = _load_indices(v151_suite_metadata, expected=32)
    selected = select_holdout_indices(
        v150_indices=v150_indices, v151_indices=v151_indices
    )
    signed_maps = _load_signed_maps(signed_map_path)
    static_maps = build_static_maps(signed_maps, random_seed=random_seed)
    source = {
        "natural_prompt_source": str(natural_prompts),
        "v150_suite_metadata": str(v150_suite_metadata),
        "v151_suite_metadata": str(v151_suite_metadata),
        "signed_map_path": str(signed_map_path),
        "signed_map_sha256": _sha256_file(signed_map_path),
        "excluded_prompt_indices": sorted(v150_indices | v151_indices),
        "selection_rule": "all MovieBench prompts unused by v150 and v151",
    }
    plan = build_probe_plan(static_maps, source=source)
    jobs = _build_jobs(prompts, selected, seed_base=seed_base)
    prompt_text = "\n".join(job["base_prompt"] for job in jobs) + "\n"
    manifest_text = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n" for job in jobs
    )
    plan_text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v152_core_128.txt").write_text(prompt_text, encoding="utf-8")
    (output_dir / "v152_core_128.jsonl").write_text(
        manifest_text, encoding="utf-8"
    )
    (output_dir / "v152_probe_plan.json").write_text(
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
        "excluded_prompt_indices": sorted(v150_indices | v151_indices),
        "probe_count_excluding_native_replay": len(plan["probes"]),
        "downstream_context_count": len(CONTEXT_TIMESTEPS),
        "expected_downstream_records_per_profile": (
            len(plan["probes"]) + 1
        )
        * len(CONTEXT_TIMESTEPS),
        "prompts_sha256": _sha256_text(prompt_text),
        "manifest_sha256": _sha256_text(manifest_text),
        "plan_sha256": _sha256_text(plan_text),
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
    parser.add_argument("--v150-suite-metadata", type=Path, required=True)
    parser.add_argument("--v151-suite-metadata", type=Path, required=True)
    parser.add_argument("--signed-map", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=152000)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(
                args.output_dir,
                natural_prompts=args.natural_prompts,
                v150_suite_metadata=args.v150_suite_metadata,
                v151_suite_metadata=args.v151_suite_metadata,
                signed_map_path=args.signed_map,
                seed_base=args.seed_base,
                random_seed=args.random_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
