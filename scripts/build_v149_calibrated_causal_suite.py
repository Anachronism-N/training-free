#!/usr/bin/env python3
"""Build calibrated causal-leverage profiling suites for v149."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.build_v148_axis_causal_suite import (
        AXES,
        AXIS_AUDIT_NAME,
        CORE_PROMPTS,
        DOSE_PROMPTS,
        FACTOR,
        HEADS,
        LAYERS,
        SEED_REPLICATES,
        _build_jobs,
        _load_axis_sources,
        _load_pf_labels,
        _map_coordinates,
        _pf_stratified_pair,
        _rank_maps,
        _read_prompts,
        _select_prompt_indices,
        _sha256,
        _write_jobs,
    )
except ModuleNotFoundError:
    from build_v148_axis_causal_suite import (
        AXES,
        AXIS_AUDIT_NAME,
        CORE_PROMPTS,
        DOSE_PROMPTS,
        FACTOR,
        HEADS,
        LAYERS,
        SEED_REPLICATES,
        _build_jobs,
        _load_axis_sources,
        _load_pf_labels,
        _map_coordinates,
        _pf_stratified_pair,
        _rank_maps,
        _read_prompts,
        _select_prompt_indices,
        _sha256,
        _write_jobs,
    )


INTERVENTIONS = ("key_shift", "value_shift", "policy_contrast")
MATCHED_INTERVENTION = {
    "k": "key_shift",
    "v": "value_shift",
    "policy": "policy_contrast",
}
POLICY_ARGS = {
    "policy_contrast": {"left": "uniform8", "right": "recent8"}
}
DEFAULT_CALIBRATION_TARGET = 0.05
DEFAULT_CALIBRATION_MIN_SCALE = 0.01
DEFAULT_CALIBRATION_MAX_SCALE = 500.0


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
    calibration: dict,
) -> dict:
    return {
        "name": name,
        "group": group,
        "policy": policy,
        "policy_args": dict(POLICY_ARGS.get(policy, {})),
        "calibration": dict(calibration),
        "head_map": head_map,
        "axis": axis,
        "rank_group": rank_group,
        "control_family": control_family,
        "dose": int(dose),
    }


def _source_metadata(
    analysis_dir: Path,
    *,
    pf_labels_path: Path,
    source: dict,
    labels: list[list[int]],
    maps: dict,
    pf_pairs: dict,
    random_seed: int,
    per_layer_count: int,
    calibration: dict,
) -> dict:
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
    return {
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
        "axis_audit_names": AXIS_AUDIT_NAME,
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
        "pf_matched_selection": {
            axis: {
                key: value
                for key, value in pf_pairs[axis].items()
                if key not in {"top", "bottom"}
            }
            for axis in AXES
        },
        "calibration": calibration,
        "policy_contrast": POLICY_ARGS["policy_contrast"],
        "claim_boundary": (
            "v149 rankings come only from v145 full-semantic scores. PF "
            "labels define post-hoc within-class controls. Every intervention "
            "is calibrated independently at each layer after the native "
            "attention output projection."
        ),
    }


def build_probe_plans(
    analysis_dir: Path,
    *,
    pf_labels_path: Path,
    random_seed: int = 20260801,
    per_layer_count: int = 3,
    calibration_target: float = DEFAULT_CALIBRATION_TARGET,
) -> tuple[dict, dict]:
    if not 0 < calibration_target <= 0.5:
        raise ValueError("calibration_target must be in (0, 0.5]")
    calibration = {
        "mode": "projected_relative_rms",
        "target": float(calibration_target),
        "min_scale": DEFAULT_CALIBRATION_MIN_SCALE,
        "max_scale": DEFAULT_CALIBRATION_MAX_SCALE,
    }
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

    core_probes = []
    core_hypotheses = []
    for axis in AXES:
        diagonal = MATCHED_INTERVENTION[axis]
        for policy in INTERVENTIONS:
            top_name = f"{axis}_top_{policy}_cal"
            bottom_name = f"{axis}_bottom_{policy}_cal"
            for name, rank_group in (
                (top_name, "top"),
                (bottom_name, "bottom"),
            ):
                core_probes.append(
                    _probe(
                        name=name,
                        group=f"{axis}_{rank_group}",
                        policy=policy,
                        head_map=maps[axis][rank_group],
                        axis=axis,
                        rank_group=rank_group,
                        control_family="rank_extreme",
                        dose=per_layer_count,
                        calibration=calibration,
                    )
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
                        f"{axis}_random{random_index}_{policy}_cal"
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
                            calibration=calibration,
                        )
                    )
                    hypothesis["random_probes"].append(random_name)
            core_hypotheses.append(hypothesis)

    pf_hypotheses = []
    for axis in AXES:
        policy = MATCHED_INTERVENTION[axis]
        top_name = f"{axis}_pfmatched_top_{policy}_cal"
        bottom_name = f"{axis}_pfmatched_bottom_{policy}_cal"
        for name, rank_group in (
            (top_name, "top"),
            (bottom_name, "bottom"),
        ):
            core_probes.append(
                _probe(
                    name=name,
                    group=f"{axis}_pfmatched_{rank_group}",
                    policy=policy,
                    head_map=pf_pairs[axis][rank_group],
                    axis=axis,
                    rank_group=rank_group,
                    control_family="pf_label_matched",
                    dose=1,
                    calibration=calibration,
                )
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
            top_name = f"{axis}_top{count}_{policy}_cal"
            bottom_name = f"{axis}_bottom{count}_{policy}_cal"
            for name, rank_group in (
                (top_name, "top"),
                (bottom_name, "bottom"),
            ):
                dose_probes.append(
                    _probe(
                        name=name,
                        group=f"{axis}_{rank_group}{count}",
                        policy=policy,
                        head_map=maps[axis][
                            f"{rank_group}_dose"
                        ][count],
                        axis=axis,
                        rank_group=rank_group,
                        control_family="calibrated_dose",
                        dose=count,
                        calibration=calibration,
                    )
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

    common_source = _source_metadata(
        analysis_dir,
        pf_labels_path=pf_labels_path,
        source=source,
        labels=labels,
        maps=maps,
        pf_pairs=pf_pairs,
        random_seed=random_seed,
        per_layer_count=per_layer_count,
        calibration=calibration,
    )
    core = {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "suite": "v149_calibrated_core",
        "source": common_source,
        "probes": core_probes,
        "hypotheses": core_hypotheses,
        "pf_matched_hypotheses": pf_hypotheses,
    }
    dose = {
        "version": 1,
        "layers": LAYERS,
        "heads": HEADS,
        "suite": "v149_calibrated_dose",
        "source": common_source,
        "probes": dose_probes,
        "dose_hypotheses": dose_hypotheses,
    }
    if len(core_probes) != 30 or len(dose_probes) != 24:
        raise AssertionError(
            f"unexpected v149 probe counts: {len(core_probes)}, "
            f"{len(dose_probes)}"
        )
    return core, dose


def write_suite(
    output_dir: Path,
    *,
    natural_prompts: Path,
    diverse_index: Path,
    v145_analysis_dir: Path,
    pf_labels_path: Path,
    seed_base: int,
    per_layer_count: int,
    calibration_target: float,
    v148_raw_observations: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(natural_prompts)
    source_indices = _select_prompt_indices(diverse_index, prompts)
    core_plan, dose_plan = build_probe_plans(
        v145_analysis_dir,
        pf_labels_path=pf_labels_path,
        per_layer_count=per_layer_count,
        calibration_target=calibration_target,
    )
    core_jobs = _build_jobs(
        prompts,
        source_indices,
        seed_base=seed_base,
        kind="v149_calibrated_core",
    )
    dose_jobs = _build_jobs(
        prompts,
        source_indices[:DOSE_PROMPTS],
        seed_base=seed_base,
        kind="v149_calibrated_dose",
    )
    core_prompt_text, core_manifest_text = _write_jobs(
        output_dir, prefix="v149_calibrated_core_64", jobs=core_jobs
    )
    dose_prompt_text, dose_manifest_text = _write_jobs(
        output_dir, prefix="v149_calibrated_dose_32", jobs=dose_jobs
    )
    core_plan_text = json.dumps(core_plan, indent=2, sort_keys=True) + "\n"
    dose_plan_text = json.dumps(dose_plan, indent=2, sort_keys=True) + "\n"
    (output_dir / "v149_calibrated_core_plan.json").write_text(
        core_plan_text, encoding="utf-8"
    )
    (output_dir / "v149_calibrated_dose_plan.json").write_text(
        dose_plan_text, encoding="utf-8"
    )
    raw_reference = None
    if v148_raw_observations is not None:
        if not v148_raw_observations.is_file():
            raise FileNotFoundError(v148_raw_observations)
        raw_reference = {
            "path": str(v148_raw_observations),
            "sha256": hashlib.sha256(
                v148_raw_observations.read_bytes()
            ).hexdigest(),
        }
    metadata = {
        "version": 1,
        "seed_base": seed_base,
        "seed_replicates": list(SEED_REPLICATES),
        "source_prompt_indices": source_indices,
        "calibration_target": calibration_target,
        "paired_v148_raw_reference": raw_reference,
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
    parser.add_argument(
        "--calibration-target",
        type=float,
        default=DEFAULT_CALIBRATION_TARGET,
    )
    parser.add_argument("--v148-raw-observations", type=Path)
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
                calibration_target=args.calibration_target,
                v148_raw_observations=args.v148_raw_observations,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
