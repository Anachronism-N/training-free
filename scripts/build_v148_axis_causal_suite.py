#!/usr/bin/env python3
"""Build the v148 axis-matched causal head profiling suites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


LAYERS = 30
HEADS = 12
CORE_PROMPTS = 32
DOSE_PROMPTS = 16
SEED_REPLICATES = (0, 1)
FACTOR = "full_semantic"
AXES = ("k", "v", "policy")
INTERVENTIONS = ("key_shift", "value_shift", "recent4")
MATCHED_INTERVENTION = {
    "k": "key_shift",
    "v": "value_shift",
    "policy": "recent4",
}
AXIS_AUDIT_NAME = {
    "k": "k_shift",
    "v": "v_shift",
    "policy": "policy_shift",
}
AXIS_SCORE_COLUMN = {
    axis: f"all_{audit_name}_mean"
    for axis, audit_name in AXIS_AUDIT_NAME.items()
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str | float | int | None, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def _read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 128:
        raise ValueError(f"expected 128 prompts in {path}, found {len(prompts)}")
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"prompt file contains duplicates: {path}")
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

    while len(selected) < CORE_PROMPTS:
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


def _load_axis_sources(analysis_dir: Path) -> dict[str, dict]:
    audit_rows = _read_csv(
        analysis_dir / "feature_reproducibility_audit.csv"
    )
    score_rows = _read_csv(
        analysis_dir / "head_factor_reproducibility.csv"
    )
    sources: dict[str, dict] = {}
    for axis in AXES:
        audit_name = AXIS_AUDIT_NAME[axis]
        matches = [
            row
            for row in audit_rows
            if row.get("variant") == FACTOR
            and row.get("axis") == audit_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {FACTOR}/{audit_name} audit row, "
                f"found {len(matches)}"
            )
        row = matches[0]
        if int(row["reproducible_factor_axis_candidate"]) != 1:
            raise ValueError(
                f"v148 requires a reproducible {FACTOR}/{audit_name} axis"
            )
        sources[axis] = {
            "factor": FACTOR,
            "axis": audit_name,
            "score_column": AXIS_SCORE_COLUMN[axis],
            "layer_residual_family_split_spearman": _finite(
                row["layer_residual_family_split_spearman"],
                name=f"{axis}.family_spearman",
            ),
            "layer_residual_seed_replicate_spearman": _finite(
                row["layer_residual_seed_replicate_spearman"],
                name=f"{axis}.seed_spearman",
            ),
            "reproducibility_screen_pass": True,
        }

    scores: dict[str, list[list[float | None]]] = {
        axis: [[None for _ in range(HEADS)] for _ in range(LAYERS)]
        for axis in AXES
    }
    for row in score_rows:
        if row.get("variant") != FACTOR:
            continue
        layer = int(row["layer"])
        head = int(row["head"])
        if not (0 <= layer < LAYERS and 0 <= head < HEADS):
            raise ValueError(f"invalid v145 head coordinate {(layer, head)}")
        for axis in AXES:
            column = AXIS_SCORE_COLUMN[axis]
            if scores[axis][layer][head] is not None:
                raise ValueError(
                    f"duplicate v145 score for {axis}/{layer}/{head}"
                )
            scores[axis][layer][head] = _finite(
                row[column], name=f"{axis}.{layer}.{head}"
            )
    for axis in AXES:
        missing = [
            (layer, head)
            for layer in range(LAYERS)
            for head in range(HEADS)
            if scores[axis][layer][head] is None
        ]
        if missing:
            raise ValueError(
                f"v145 score grid for {axis} is incomplete: {missing[:8]}"
            )
    return {"metadata": sources, "scores": scores}


def _load_pf_labels(path: Path) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [[int(value) for value in row] for row in csv.reader(handle)]
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(
            f"PF label map must be {LAYERS}x{HEADS}, found "
            f"{len(rows)}x{len(rows[0]) if rows else 0}"
        )
    labels = {-1, 1, 2}
    if any(value not in labels for row in rows for value in row):
        raise ValueError("PF label map contains an unknown label")
    return rows


def _rank_maps(
    scores: list[list[float | None]],
    *,
    axis: str,
    random_seed: int,
    per_layer_count: int,
) -> dict:
    if per_layer_count != 3:
        raise ValueError("v148 is preregistered for three heads per layer")
    top: dict[str, list[int]] = {}
    bottom: dict[str, list[int]] = {}
    random_maps = ({}, {})
    top_dose = {count: {} for count in range(1, 5)}
    bottom_dose = {count: {} for count in range(1, 5)}
    for layer in range(LAYERS):
        ordered = sorted(
            range(HEADS),
            key=lambda head: (float(scores[layer][head]), head),
        )
        bottom[str(layer)] = sorted(ordered[:per_layer_count])
        top[str(layer)] = sorted(ordered[-per_layer_count:])
        middle = list(ordered[per_layer_count:-per_layer_count])
        if len(middle) != 2 * per_layer_count:
            raise AssertionError("v148 middle-rank partition has wrong size")
        rng = random.Random(
            random_seed + 1009 * AXES.index(axis) + 104729 * layer
        )
        rng.shuffle(middle)
        random_maps[0][str(layer)] = sorted(middle[:per_layer_count])
        random_maps[1][str(layer)] = sorted(middle[per_layer_count:])
        for count in range(1, 5):
            top_dose[count][str(layer)] = sorted(ordered[-count:])
            bottom_dose[count][str(layer)] = sorted(ordered[:count])
    return {
        "top": top,
        "bottom": bottom,
        "random0": random_maps[0],
        "random1": random_maps[1],
        "top_dose": top_dose,
        "bottom_dose": bottom_dose,
    }


def _pf_stratified_pair(
    scores: list[list[float | None]], labels: list[list[int]]
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict]:
    high: dict[str, list[int]] = {}
    low: dict[str, list[int]] = {}
    chosen_labels: dict[str, int] = {}
    spreads: dict[str, float] = {}
    for layer in range(LAYERS):
        candidates = []
        for label in (-1, 1, 2):
            members = [
                head for head in range(HEADS) if labels[layer][head] == label
            ]
            if len(members) < 2:
                continue
            ordered = sorted(
                members,
                key=lambda head: (float(scores[layer][head]), head),
            )
            spread = float(scores[layer][ordered[-1]]) - float(
                scores[layer][ordered[0]]
            )
            candidates.append(
                (spread, len(members), -abs(label), label, ordered)
            )
        if not candidates:
            raise ValueError(f"layer {layer} has no repeated PF class")
        spread, _, _, label, ordered = max(candidates)
        low[str(layer)] = [ordered[0]]
        high[str(layer)] = [ordered[-1]]
        chosen_labels[str(layer)] = int(label)
        spreads[str(layer)] = float(spread)
    return high, low, {
        "selection": (
            "within each layer, choose the PF class with the largest score "
            "range, then compare its highest- and lowest-score heads"
        ),
        "label_counts": dict(sorted(Counter(chosen_labels.values()).items())),
        "labels_by_layer": chosen_labels,
        "score_spread_by_layer": spreads,
    }


def _probe(
    *,
    name: str,
    group: str,
    policy: str,
    head_map: dict[str, list[int]],
    axis: str,
    rank_group: str,
    control_family: str,
    dose: int,
) -> dict:
    return {
        "name": name,
        "group": group,
        "policy": policy,
        "head_map": head_map,
        "axis": axis,
        "rank_group": rank_group,
        "control_family": control_family,
        "dose": int(dose),
    }


def _map_coordinates(head_map: dict[str, list[int]]) -> set[tuple[int, int]]:
    return {
        (int(layer), int(head))
        for layer, heads in head_map.items()
        for head in heads
    }


def build_probe_plans(
    analysis_dir: Path,
    *,
    pf_labels_path: Path,
    random_seed: int = 20260801,
    per_layer_count: int = 3,
) -> tuple[dict, dict]:
    source = _load_axis_sources(analysis_dir)
    labels = _load_pf_labels(pf_labels_path)
    maps = {
        axis: _rank_maps(
            source["scores"][axis],
            axis=axis,
            random_seed=random_seed,
            per_layer_count=per_layer_count,
        )
        for axis in AXES
    }
    pf_pairs = {}
    for axis in AXES:
        high, low, metadata = _pf_stratified_pair(
            source["scores"][axis], labels
        )
        pf_pairs[axis] = {"top": high, "bottom": low, **metadata}
    axis_map_diagnostics = {}
    for axis in AXES:
        axis_map_diagnostics[axis] = {}
        for group in ("top", "bottom"):
            coordinates = _map_coordinates(maps[axis][group])
            axis_map_diagnostics[axis][f"{group}_pf_label_counts"] = dict(
                sorted(
                    Counter(
                        labels[layer][head] for layer, head in coordinates
                    ).items()
                )
            )
    top_map_overlaps = {}
    for left_index, left in enumerate(AXES):
        left_map = _map_coordinates(maps[left]["top"])
        for right in AXES[left_index + 1 :]:
            right_map = _map_coordinates(maps[right]["top"])
            intersection = len(left_map & right_map)
            union = len(left_map | right_map)
            top_map_overlaps[f"{left}__{right}"] = {
                "intersection": intersection,
                "jaccard": intersection / union,
            }

    core_probes = []
    core_hypotheses = []
    for axis in AXES:
        diagonal = MATCHED_INTERVENTION[axis]
        for policy in INTERVENTIONS:
            top_name = f"{axis}_top_{policy}"
            bottom_name = f"{axis}_bottom_{policy}"
            core_probes.extend(
                [
                    _probe(
                        name=top_name,
                        group=f"{axis}_top",
                        policy=policy,
                        head_map=maps[axis]["top"],
                        axis=axis,
                        rank_group="top",
                        control_family="rank_extreme",
                        dose=per_layer_count,
                    ),
                    _probe(
                        name=bottom_name,
                        group=f"{axis}_bottom",
                        policy=policy,
                        head_map=maps[axis]["bottom"],
                        axis=axis,
                        rank_group="bottom",
                        control_family="rank_extreme",
                        dose=per_layer_count,
                    ),
                ]
            )
            hypothesis = {
                "axis": axis,
                "policy": policy,
                "matched": policy == diagonal,
                "top_probe": top_name,
                "bottom_probe": bottom_name,
                "random_probes": [],
            }
            if policy == diagonal:
                for random_index in range(2):
                    random_name = (
                        f"{axis}_random{random_index}_{policy}"
                    )
                    core_probes.append(
                        _probe(
                            name=random_name,
                            group=f"{axis}_random{random_index}",
                            policy=policy,
                            head_map=maps[axis][
                                f"random{random_index}"
                            ],
                            axis=axis,
                            rank_group=f"random{random_index}",
                            control_family="middle_rank_partition",
                            dose=per_layer_count,
                        )
                    )
                    hypothesis["random_probes"].append(random_name)
            core_hypotheses.append(hypothesis)

    pf_hypotheses = []
    for axis in AXES:
        policy = MATCHED_INTERVENTION[axis]
        top_name = f"{axis}_pfmatched_top_{policy}"
        bottom_name = f"{axis}_pfmatched_bottom_{policy}"
        core_probes.extend(
            [
                _probe(
                    name=top_name,
                    group=f"{axis}_pfmatched_top",
                    policy=policy,
                    head_map=pf_pairs[axis]["top"],
                    axis=axis,
                    rank_group="top",
                    control_family="pf_label_matched",
                    dose=1,
                ),
                _probe(
                    name=bottom_name,
                    group=f"{axis}_pfmatched_bottom",
                    policy=policy,
                    head_map=pf_pairs[axis]["bottom"],
                    axis=axis,
                    rank_group="bottom",
                    control_family="pf_label_matched",
                    dose=1,
                ),
            ]
        )
        pf_hypotheses.append(
            {
                "axis": axis,
                "policy": policy,
                "top_probe": top_name,
                "bottom_probe": bottom_name,
                "metadata": {
                    key: value
                    for key, value in pf_pairs[axis].items()
                    if key not in {"top", "bottom"}
                },
            }
        )

    dose_probes = []
    dose_hypotheses = []
    for axis in AXES:
        policy = MATCHED_INTERVENTION[axis]
        pairs = []
        for count in range(1, 5):
            top_name = f"{axis}_top{count}_{policy}"
            bottom_name = f"{axis}_bottom{count}_{policy}"
            dose_probes.extend(
                [
                    _probe(
                        name=top_name,
                        group=f"{axis}_top{count}",
                        policy=policy,
                        head_map=maps[axis]["top_dose"][count],
                        axis=axis,
                        rank_group="top",
                        control_family="dose",
                        dose=count,
                    ),
                    _probe(
                        name=bottom_name,
                        group=f"{axis}_bottom{count}",
                        policy=policy,
                        head_map=maps[axis]["bottom_dose"][count],
                        axis=axis,
                        rank_group="bottom",
                        control_family="dose",
                        dose=count,
                    ),
                ]
            )
            pairs.append(
                {
                    "dose": count,
                    "top_probe": top_name,
                    "bottom_probe": bottom_name,
                }
            )
        dose_hypotheses.append(
            {"axis": axis, "policy": policy, "pairs": pairs}
        )

    common_source = {
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
        "factor": FACTOR,
        "axes": source["metadata"],
        "random_seed": random_seed,
        "per_layer_count": per_layer_count,
        "pf_labels_path": str(pf_labels_path),
        "pf_labels_sha256": _sha256(
            pf_labels_path.read_text(encoding="utf-8")
        ),
        "pf_label_counts": dict(
            sorted(Counter(value for row in labels for value in row).items())
        ),
        "axis_map_diagnostics": axis_map_diagnostics,
        "top_map_overlaps": top_map_overlaps,
        "claim_boundary": (
            "PF labels are used only for a post-hoc within-class control. "
            "They do not define any v148 ranking."
        ),
    }
    core = {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "suite": "v148_axis_core",
        "source": common_source,
        "probes": core_probes,
        "hypotheses": core_hypotheses,
        "pf_matched_hypotheses": pf_hypotheses,
    }
    dose = {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "suite": "v148_axis_dose",
        "source": common_source,
        "probes": dose_probes,
        "dose_hypotheses": dose_hypotheses,
    }
    if len(core_probes) != 30 or len(dose_probes) != 24:
        raise AssertionError(
            f"unexpected v148 probe counts: {len(core_probes)}, "
            f"{len(dose_probes)}"
        )
    return core, dose


def _build_jobs(
    prompts: list[str],
    source_indices: list[int],
    *,
    seed_base: int,
    kind: str,
) -> list[dict]:
    jobs = []
    for prompt_slot, source_index in enumerate(source_indices):
        for replicate in SEED_REPLICATES:
            seed = seed_base + source_index + replicate * 10000
            jobs.append(
                {
                    "dataset_index": len(jobs),
                    "job_id": (
                        f"{kind}_p{prompt_slot:02d}_"
                        f"src{source_index:03d}_s{replicate}"
                    ),
                    "kind": kind,
                    "prompt_slot": prompt_slot,
                    "source_prompt_index": source_index,
                    "seed_replicate": replicate,
                    "seed": seed,
                    "reference_seed": seed,
                    "base_prompt": prompts[source_index],
                }
            )
    return jobs


def _write_jobs(
    output_dir: Path,
    *,
    prefix: str,
    jobs: list[dict],
) -> tuple[str, str]:
    prompt_text = "\n".join(job["base_prompt"] for job in jobs) + "\n"
    manifest_text = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n"
        for job in jobs
    )
    (output_dir / f"{prefix}.txt").write_text(
        prompt_text, encoding="utf-8"
    )
    (output_dir / f"{prefix}.jsonl").write_text(
        manifest_text, encoding="utf-8"
    )
    return prompt_text, manifest_text


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    diverse_index: Path,
    v145_analysis_dir: Path,
    pf_labels_path: Path,
    seed_base: int,
    per_layer_count: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(natural_prompts)
    source_indices = _select_prompt_indices(diverse_index, prompts)
    core_plan, dose_plan = build_probe_plans(
        v145_analysis_dir,
        pf_labels_path=pf_labels_path,
        per_layer_count=per_layer_count,
    )
    core_jobs = _build_jobs(
        prompts,
        source_indices,
        seed_base=seed_base,
        kind="v148_axis_core",
    )
    dose_jobs = _build_jobs(
        prompts,
        source_indices[:DOSE_PROMPTS],
        seed_base=seed_base,
        kind="v148_axis_dose",
    )
    core_prompt_text, core_manifest_text = _write_jobs(
        output_dir, prefix="v148_axis_core_64", jobs=core_jobs
    )
    dose_prompt_text, dose_manifest_text = _write_jobs(
        output_dir, prefix="v148_axis_dose_32", jobs=dose_jobs
    )
    core_plan_text = json.dumps(core_plan, indent=2, sort_keys=True) + "\n"
    dose_plan_text = json.dumps(dose_plan, indent=2, sort_keys=True) + "\n"
    (output_dir / "v148_axis_core_plan.json").write_text(
        core_plan_text, encoding="utf-8"
    )
    (output_dir / "v148_axis_dose_plan.json").write_text(
        dose_plan_text, encoding="utf-8"
    )
    metadata = {
        "version": 1,
        "seed_base": seed_base,
        "seed_replicates": list(SEED_REPLICATES),
        "source_prompt_indices": source_indices,
        "core": {
            "job_count": len(core_jobs),
            "unique_prompt_count": CORE_PROMPTS,
            "probe_count_excluding_native_replay": len(
                core_plan["probes"]
            ),
            "downstream_context_count": 2,
            "expected_downstream_records_per_profile": (
                len(core_plan["probes"]) + 1
            )
            * 2,
            "prompts_sha256": _sha256(core_prompt_text),
            "manifest_sha256": _sha256(core_manifest_text),
            "plan_sha256": _sha256(core_plan_text),
        },
        "dose": {
            "job_count": len(dose_jobs),
            "unique_prompt_count": DOSE_PROMPTS,
            "probe_count_excluding_native_replay": len(
                dose_plan["probes"]
            ),
            "downstream_context_count": 2,
            "expected_downstream_records_per_profile": (
                len(dose_plan["probes"]) + 1
            )
            * 2,
            "prompts_sha256": _sha256(dose_prompt_text),
            "manifest_sha256": _sha256(dose_manifest_text),
            "plan_sha256": _sha256(dose_plan_text),
        },
        "natural_prompt_source": str(natural_prompts),
        "diverse_index_source": str(diverse_index),
        "v145_analysis_source": str(v145_analysis_dir),
        "pf_labels_source": str(pf_labels_path),
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
    parser.add_argument("--pf-labels", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=148000)
    parser.add_argument("--per-layer-count", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            write_suite(
                args.output_dir,
                natural_prompts=args.natural_prompts,
                diverse_index=args.diverse_index,
                v145_analysis_dir=args.v145_analysis_dir,
                pf_labels_path=args.pf_labels,
                seed_base=args.seed_base,
                per_layer_count=args.per_layer_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
