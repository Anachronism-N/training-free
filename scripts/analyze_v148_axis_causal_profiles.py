#!/usr/bin/env python3
"""Analyze v148 axis-matched downstream-causal profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


LAYERS = 30
HEADS = 12
REPLICATES = (0, 1)
PROFILE_VERSION = 8
EPSILON = 1e-10
MIN_LOG_EFFECT = math.log(1.05)
CONTEXTS = ("noisy_t1000", "noisy_t500")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x_rank = _rankdata(x[finite])
    y_rank = _rankdata(y[finite])
    if x_rank.std() <= 1e-12 or y_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _bootstrap_ci(
    values: dict[int, float],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    keys = sorted(values)
    array = np.asarray([values[key] for key in keys], dtype=np.float64)
    if array.size < 2:
        value = float(array.mean()) if array.size else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    boot = array[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(
        np.quantile(boot, 0.975)
    )


def _context_key(record: dict) -> str:
    mode = str(record["mode"])
    timestep = int(record["nominal_timestep"])
    return "clean" if mode == "clean" else f"noisy_t{timestep}"


def _canonical_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_plan(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("version", -1)) != 1
        or int(payload.get("layers", -1)) != LAYERS
        or int(payload.get("heads", -1)) != HEADS
        or payload.get("suite") not in {
            "v148_axis_core",
            "v148_axis_dose",
        }
    ):
        raise RuntimeError("v148 probe plan has an invalid model contract")
    probes = payload.get("probes") or []
    names = [str(probe.get("name")) for probe in probes]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("v148 probe plan is empty or repeats a probe")
    expected = 30 if payload["suite"] == "v148_axis_core" else 24
    if len(probes) != expected:
        raise RuntimeError(
            f"{payload['suite']} requires {expected} probes, found "
            f"{len(probes)}"
        )
    return payload, _canonical_digest(payload)


def _expected_prompt_count(plan: dict) -> int:
    return 32 if plan["suite"] == "v148_axis_core" else 16


def _load_profiles(
    directory: Path,
    *,
    plan: dict,
    plan_sha256: str,
    expected_count: int,
) -> tuple[list[dict], list[dict]]:
    paths = sorted(directory.glob("*.pt"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} v148 profiles, found {len(paths)}"
        )
    prompt_count = _expected_prompt_count(plan)
    expected_kind = plan["suite"]
    expected_records = 2 * LAYERS
    expected_downstream = 2 * (len(plan["probes"]) + 1)
    profiles = []
    audits = []
    seen = set()
    prompt_seeds: dict[tuple[int, int], int] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        key = (int(job["prompt_slot"]), int(job["seed_replicate"]))
        if key in seen:
            raise RuntimeError(f"duplicate v148 profile coordinate: {key}")
        seen.add(key)
        if (
            int(payload.get("version", -1)) != PROFILE_VERSION
            or str(job.get("kind")) != expected_kind
        ):
            raise RuntimeError(f"{path} is not a {expected_kind} profile")
        if not (
            int(metadata["seed"])
            == int(job["seed"])
            == int(job["reference_seed"])
        ):
            raise RuntimeError(f"{path} violates the runtime seed contract")
        if metadata.get("incomplete_calls"):
            raise RuntimeError(f"{path} contains incomplete captures")
        if (
            int(metadata.get("captured_calls", -1)) != 2
            or int(metadata.get("record_count", -1)) != expected_records
        ):
            raise RuntimeError(f"{path} has an invalid 2x30 capture grid")
        plan_meta = metadata.get("downstream_probe_plan") or {}
        if str(plan_meta.get("sha256")) != plan_sha256:
            raise RuntimeError(f"{path} uses a different probe plan")
        records = payload.get("records") or []
        downstream = payload.get("downstream_probe_records") or []
        if (
            len(records) != expected_records
            or len(downstream) != expected_downstream
            or int(payload.get("downstream_probe_expected_count", -1))
            != expected_downstream
        ):
            raise RuntimeError(f"{path} has an invalid record count")
        state_layers = Counter(
            (
                str(row["mode"]),
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in records
        )
        if len(state_layers) != 2 or set(state_layers.values()) != {LAYERS}:
            raise RuntimeError(f"{path} has an invalid state/layer grid")
        probe_grid = Counter(
            (_context_key(row), str(row["probe_name"]))
            for row in downstream
        )
        if (
            set(context for context, _ in probe_grid) != set(CONTEXTS)
            or len(probe_grid) != expected_downstream
            or set(probe_grid.values()) != {1}
        ):
            raise RuntimeError(f"{path} has an invalid probe grid")
        prompt_seeds[key] = int(metadata["seed"])
        profiles.append(payload)
        audits.append(
            {
                "dataset_index": int(job["dataset_index"]),
                "prompt_slot": key[0],
                "source_prompt_index": int(job["source_prompt_index"]),
                "seed_replicate": key[1],
                "seed": int(metadata["seed"]),
                "captured_calls": int(metadata["captured_calls"]),
                "record_count": int(metadata["record_count"]),
                "downstream_record_count": len(downstream),
                "path": str(path),
            }
        )
    expected_grid = {
        (prompt, replicate)
        for prompt in range(prompt_count)
        for replicate in REPLICATES
    }
    if seen != expected_grid:
        raise RuntimeError(
            "incomplete v148 prompt/seed grid: "
            f"missing={sorted(expected_grid - seen)}"
        )
    for prompt in range(prompt_count):
        if prompt_seeds[(prompt, 0)] == prompt_seeds[(prompt, 1)]:
            raise RuntimeError(f"prompt {prompt} repeats its seed")
    return profiles, audits


def _downstream_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    for payload in profiles:
        job = payload["job"]
        for record in payload["downstream_probe_records"]:
            layer_metadata = record.get("layer_metadata") or {}
            replacement = [
                float(value["replacement_relative_rms"])
                for value in layer_metadata.values()
                if "replacement_relative_rms" in value
            ]
            shifted = [
                int(value.get("shifted_old_frames", 0))
                for value in layer_metadata.values()
                if "shifted_old_frames" in value
            ]
            rows.append(
                {
                    "prompt_slot": int(job["prompt_slot"]),
                    "source_prompt_index": int(
                        job["source_prompt_index"]
                    ),
                    "seed_replicate": int(job["seed_replicate"]),
                    "seed": int(job["seed"]),
                    "context": _context_key(record),
                    "mode": str(record["mode"]),
                    "nominal_timestep": int(
                        record["nominal_timestep"]
                    ),
                    "probe_name": str(record["probe_name"]),
                    "policy": str(record["policy"]),
                    "group": str(record["group"]),
                    "selected_head_count": int(
                        record["selected_head_count"]
                    ),
                    "flow_relative_rms": float(
                        record["flow_metrics"]["relative_rms"]
                    ),
                    "flow_cosine": float(
                        record["flow_metrics"]["cosine"]
                    ),
                    "x0_relative_rms": float(
                        record["x0_metrics"]["relative_rms"]
                    ),
                    "x0_cosine": float(
                        record["x0_metrics"]["cosine"]
                    ),
                    "mean_local_replacement_relative_rms": (
                        float(np.mean(replacement))
                        if replacement
                        else 0.0
                    ),
                    "min_shifted_old_frames": (
                        min(shifted) if shifted else 0
                    ),
                }
            )
    return rows


def _lookup(rows: list[dict], metric: str) -> dict[tuple, float]:
    result = {}
    for row in rows:
        key = (
            int(row["prompt_slot"]),
            int(row["seed_replicate"]),
            str(row["context"]),
            str(row["probe_name"]),
        )
        if key in result:
            raise RuntimeError(f"duplicate downstream observation: {key}")
        result[key] = float(row[metric])
    return result


def _summarize_effects(
    effects: dict[tuple[int, int, str], float],
    *,
    label: str,
    metric: str,
    metadata: dict,
    bootstrap_seed: int,
    effect_definition: str = "paired_log_ratio",
) -> list[dict]:
    rows = []
    contexts = sorted({key[2] for key in effects})
    prompt_ids = sorted({key[0] for key in effects})
    for context in [*contexts, "pooled"]:
        selected = {
            key: value
            for key, value in effects.items()
            if context == "pooled" or key[2] == context
        }
        prompt_values = {
            prompt: float(
                np.mean(
                    [
                        value
                        for (current_prompt, _, _), value in selected.items()
                        if current_prompt == prompt
                    ]
                )
            )
            for prompt in prompt_ids
        }
        low, high = _bootstrap_ci(
            prompt_values, seed=bootstrap_seed + len(rows) * 17
        )
        values = np.asarray(list(selected.values()), dtype=np.float64)
        replicate_values = {
            replicate: [
                float(
                    np.mean(
                        [
                            value
                            for (
                                current_prompt,
                                current_replicate,
                                _,
                            ), value in selected.items()
                            if current_prompt == prompt
                            and current_replicate == replicate
                        ]
                    )
                )
                for prompt in prompt_ids
            ]
            for replicate in REPLICATES
        }
        rows.append(
            {
                "comparison": label,
                **metadata,
                "metric": metric,
                "effect_definition": effect_definition,
                "context": context,
                "unit_count": int(values.size),
                "prompt_count": len(prompt_values),
                "mean_effect": float(values.mean()),
                "median_effect": float(np.median(values)),
                "positive_fraction": float((values > 0).mean()),
                "prompt_bootstrap_mean_ci_low": low,
                "prompt_bootstrap_mean_ci_high": high,
                "seed_replicate_spearman": _spearman(
                    replicate_values[0], replicate_values[1]
                ),
            }
        )
    return rows


def _paired_log_effects(
    lookup: dict[tuple, float],
    *,
    left_probe: str,
    right_probe: str,
) -> dict[tuple[int, int, str], float]:
    keys = {
        (prompt, replicate, context)
        for prompt, replicate, context, probe in lookup
        if probe == left_probe
    }
    effects = {}
    for key in keys:
        left = lookup[(*key, left_probe)]
        right = lookup[(*key, right_probe)]
        effects[key] = math.log((left + EPSILON) / (right + EPSILON))
    return effects


def _random_ensemble_effects(
    lookup: dict[tuple, float],
    *,
    top_probe: str,
    random_probes: list[str],
) -> dict[tuple[int, int, str], float]:
    keys = {
        (prompt, replicate, context)
        for prompt, replicate, context, probe in lookup
        if probe == top_probe
    }
    effects = {}
    for key in keys:
        top = lookup[(*key, top_probe)]
        random_log = float(
            np.mean(
                [
                    math.log(lookup[(*key, probe)] + EPSILON)
                    for probe in random_probes
                ]
            )
        )
        effects[key] = math.log(top + EPSILON) - random_log
    return effects


def _probe_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["probe_name"], row["context"])].append(row)
    summaries = []
    for (probe, context), values in sorted(grouped.items()):
        summaries.append(
            {
                "probe_name": probe,
                "context": context,
                "unit_count": len(values),
                "selected_head_count": int(
                    values[0]["selected_head_count"]
                ),
                "mean_x0_relative_rms": float(
                    np.mean([row["x0_relative_rms"] for row in values])
                ),
                "median_x0_relative_rms": float(
                    np.median([row["x0_relative_rms"] for row in values])
                ),
                "mean_flow_relative_rms": float(
                    np.mean([row["flow_relative_rms"] for row in values])
                ),
                "mean_local_replacement_relative_rms": float(
                    np.mean(
                        [
                            row["mean_local_replacement_relative_rms"]
                            for row in values
                        ]
                    )
                ),
            }
        )
    return summaries


def _qualifies(row: dict) -> bool:
    return bool(
        row["context"] != "pooled"
        and row["median_effect"] >= MIN_LOG_EFFECT
        and (
            row["prompt_bootstrap_mean_ci_low"] > 0
            or row["positive_fraction"] >= 0.65
        )
        and row["seed_replicate_spearman"] >= 0.30
    )


def _analyze_core(
    rows: list[dict], plan: dict
) -> tuple[list[dict], list[dict], dict]:
    lookup = _lookup(rows, "x0_relative_rms")
    comparisons = []
    unit_effects = {}
    seed = 14800
    for hypothesis in plan["hypotheses"]:
        axis = hypothesis["axis"]
        policy = hypothesis["policy"]
        metadata = {
            "axis": axis,
            "policy": policy,
            "control": "bottom",
            "matched_intervention": int(hypothesis["matched"]),
        }
        effects = _paired_log_effects(
            lookup,
            left_probe=hypothesis["top_probe"],
            right_probe=hypothesis["bottom_probe"],
        )
        unit_effects[(axis, policy, "bottom")] = effects
        comparisons.extend(
            _summarize_effects(
                effects,
                label=f"{axis}:{policy}:top>bottom",
                metric="x0_relative_rms",
                metadata=metadata,
                bootstrap_seed=seed,
            )
        )
        seed += 101
        if hypothesis["random_probes"]:
            random_effects = _random_ensemble_effects(
                lookup,
                top_probe=hypothesis["top_probe"],
                random_probes=hypothesis["random_probes"],
            )
            unit_effects[(axis, policy, "random_ensemble")] = random_effects
            comparisons.extend(
                _summarize_effects(
                    random_effects,
                    label=f"{axis}:{policy}:top>random_ensemble",
                    metric="x0_relative_rms",
                    metadata={
                        **metadata,
                        "control": "random_ensemble",
                    },
                    bootstrap_seed=seed,
                )
            )
            seed += 101

    pf_comparisons = []
    for hypothesis in plan["pf_matched_hypotheses"]:
        effects = _paired_log_effects(
            lookup,
            left_probe=hypothesis["top_probe"],
            right_probe=hypothesis["bottom_probe"],
        )
        pf_comparisons.extend(
            _summarize_effects(
                effects,
                label=(
                    f"{hypothesis['axis']}:{hypothesis['policy']}:"
                    "pfmatched_top>bottom"
                ),
                metric="x0_relative_rms",
                metadata={
                    "axis": hypothesis["axis"],
                    "policy": hypothesis["policy"],
                    "control": "pf_label_matched",
                    "matched_intervention": 1,
                },
                bootstrap_seed=seed,
            )
        )
        seed += 101

    specificity = []
    for axis in ("k", "v", "policy"):
        diagonal = next(
            row["policy"]
            for row in plan["hypotheses"]
            if row["axis"] == axis and row["matched"]
        )
        diagonal_effects = unit_effects[(axis, diagonal, "bottom")]
        for off_policy in (
            policy
            for policy in ("key_shift", "value_shift", "recent4")
            if policy != diagonal
        ):
            off_effects = unit_effects[(axis, off_policy, "bottom")]
            differences = {
                key: diagonal_effects[key] - off_effects[key]
                for key in diagonal_effects
            }
            specificity.extend(
                _summarize_effects(
                    differences,
                    label=f"{axis}:matched>{off_policy}",
                    metric="top_bottom_log_effect",
                    metadata={
                        "axis": axis,
                        "policy": diagonal,
                        "control": off_policy,
                        "matched_intervention": 1,
                    },
                    bootstrap_seed=seed,
                    effect_definition="difference_of_log_ratios",
                )
            )
            seed += 101

    mechanism_contexts = {}
    pf_contexts = {}
    specificity_contexts = {}
    for axis in ("k", "v", "policy"):
        matched_policy = next(
            row["policy"]
            for row in plan["hypotheses"]
            if row["axis"] == axis and row["matched"]
        )
        bottom_pass = {
            row["context"]
            for row in comparisons
            if row["axis"] == axis
            and row["policy"] == matched_policy
            and row["control"] == "bottom"
            and _qualifies(row)
        }
        random_pass = {
            row["context"]
            for row in comparisons
            if row["axis"] == axis
            and row["policy"] == matched_policy
            and row["control"] == "random_ensemble"
            and _qualifies(row)
        }
        mechanism_contexts[axis] = sorted(bottom_pass & random_pass)
        pf_contexts[axis] = sorted(
            {
                row["context"]
                for row in pf_comparisons
                if row["axis"] == axis and _qualifies(row)
            }
        )
        off_policies = {
            row["control"]
            for row in specificity
            if row["axis"] == axis and row["context"] != "pooled"
        }
        passing_by_policy = {
            policy: {
                row["context"]
                for row in specificity
                if row["axis"] == axis
                and row["control"] == policy
                and _qualifies(row)
            }
            for policy in off_policies
        }
        if passing_by_policy:
            specificity_contexts[axis] = sorted(
                set.intersection(*passing_by_policy.values())
            )
        else:
            specificity_contexts[axis] = []
    report = {
        "suite": plan["suite"],
        "gates": {
            "g0_native_replay_parity": True,
            "g1_axis_matched_causal_effect": {
                axis: bool(contexts)
                for axis, contexts in mechanism_contexts.items()
            },
            "g2_pf_independent_effect": {
                axis: bool(contexts)
                for axis, contexts in pf_contexts.items()
            },
            "g3_intervention_specificity": {
                axis: bool(contexts)
                for axis, contexts in specificity_contexts.items()
            },
        },
        "qualifying_axis_contexts": mechanism_contexts,
        "qualifying_pf_matched_contexts": pf_contexts,
        "qualifying_specificity_contexts": specificity_contexts,
        "claim_boundary": (
            "A functional axis requires the same denoising context to pass "
            "top-vs-bottom and top-vs-two-random controls. Independence from "
            "PF additionally requires the within-layer, same-PF-label pair."
        ),
        "minimum_qualifying_median_log_effect": MIN_LOG_EFFECT,
    }
    return comparisons + pf_comparisons, specificity, report


def _analyze_dose(rows: list[dict], plan: dict) -> tuple[list[dict], dict]:
    lookup = _lookup(rows, "x0_relative_rms")
    comparisons = []
    unit_by_axis_dose = {}
    seed = 15800
    for hypothesis in plan["dose_hypotheses"]:
        axis = hypothesis["axis"]
        policy = hypothesis["policy"]
        for pair in hypothesis["pairs"]:
            effects = _paired_log_effects(
                lookup,
                left_probe=pair["top_probe"],
                right_probe=pair["bottom_probe"],
            )
            unit_by_axis_dose[(axis, int(pair["dose"]))] = effects
            comparisons.extend(
                _summarize_effects(
                    effects,
                    label=f"{axis}:{policy}:top>bottom:dose{pair['dose']}",
                    metric="x0_relative_rms",
                    metadata={
                        "axis": axis,
                        "policy": policy,
                        "control": "bottom",
                        "matched_intervention": 1,
                        "dose": int(pair["dose"]),
                    },
                    bootstrap_seed=seed,
                )
            )
            seed += 101
    dose_growth = []
    growth_contexts = {}
    for axis in ("k", "v", "policy"):
        low = unit_by_axis_dose[(axis, 1)]
        high = unit_by_axis_dose[(axis, 4)]
        effects = {key: high[key] - low[key] for key in high}
        rows_for_axis = _summarize_effects(
            effects,
            label=f"{axis}:dose4>dose1:separation",
            metric="top_bottom_log_effect",
            metadata={
                "axis": axis,
                "policy": next(
                    row["policy"]
                    for row in plan["dose_hypotheses"]
                    if row["axis"] == axis
                ),
                "control": "dose1",
                "matched_intervention": 1,
                "dose": 4,
            },
            bootstrap_seed=seed,
            effect_definition="dose4_minus_dose1_log_separation",
        )
        seed += 101
        dose_growth.extend(rows_for_axis)
        growth_contexts[axis] = sorted(
            row["context"] for row in rows_for_axis if _qualifies(row)
        )
    positive_doses = {
        axis: sorted(
            {
                int(row["dose"])
                for row in comparisons
                if row["axis"] == axis and _qualifies(row)
            }
        )
        for axis in ("k", "v", "policy")
    }
    report = {
        "suite": plan["suite"],
        "gates": {
            "g0_native_replay_parity": True,
            "g1_positive_rank_separation_at_multiple_doses": {
                axis: len(doses) >= 2 for axis, doses in positive_doses.items()
            },
            "g2_dose4_exceeds_dose1": {
                axis: bool(contexts)
                for axis, contexts in growth_contexts.items()
            },
        },
        "qualifying_doses": positive_doses,
        "qualifying_growth_contexts": growth_contexts,
        "claim_boundary": (
            "Dose effects compare top-k and bottom-k at equal head count. "
            "Absolute perturbation growth with head count is not evidence."
        ),
        "minimum_qualifying_median_log_effect": MIN_LOG_EFFECT,
    }
    return comparisons + dose_growth, report


def _integrity_report(rows: list[dict]) -> dict:
    native = [row for row in rows if row["probe_name"] == "native_replay"]
    replay_max = max(
        max(row["flow_relative_rms"], row["x0_relative_rms"])
        for row in native
    )
    shifted = [
        row["min_shifted_old_frames"]
        for row in rows
        if row["policy"] in {"key_shift", "value_shift"}
    ]
    return {
        "native_replay_max_relative_rms": replay_max,
        "native_replay_pass": replay_max <= 1e-4,
        "shift_interventions_non_degenerate": bool(shifted)
        and min(shifted) > 1,
    }


def analyze(
    *,
    profile_dir: Path,
    probe_plan_path: Path,
    output_dir: Path,
    expected_count: int,
) -> dict:
    plan, plan_sha256 = _load_plan(probe_plan_path)
    profiles, audits = _load_profiles(
        profile_dir,
        plan=plan,
        plan_sha256=plan_sha256,
        expected_count=expected_count,
    )
    rows = _downstream_rows(profiles)
    integrity = _integrity_report(rows)
    if not all(
        (
            integrity["native_replay_pass"],
            integrity["shift_interventions_non_degenerate"],
        )
    ):
        raise RuntimeError(f"v148 integrity gate failed: {integrity}")
    if plan["suite"] == "v148_axis_core":
        comparisons, specificity, report = _analyze_core(rows, plan)
    else:
        comparisons, report = _analyze_dose(rows, plan)
        specificity = []
    report.update(
        {
            "profile_count": len(profiles),
            "prompt_count": len(
                {int(row["prompt_slot"]) for row in audits}
            ),
            "seed_replicates": list(REPLICATES),
            "probe_plan": str(probe_plan_path),
            "probe_plan_sha256": plan_sha256,
            "downstream_observation_count": len(rows),
            **integrity,
            "source": plan["source"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_audit.csv", audits)
    _write_csv(output_dir / "downstream_observations.csv.gz", rows)
    _write_csv(output_dir / "probe_effect_summary.csv", _probe_summaries(rows))
    _write_csv(output_dir / "axis_comparisons.csv", comparisons)
    _write_csv(output_dir / "axis_specificity.csv", specificity)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = f"""# v148 Axis-Matched Causal Profiling Results

## Integrity

- Suite: `{plan['suite']}`
- Profiles: `{report['profile_count']}`
- Prompts: `{report['prompt_count']}`
- Native replay maximum relative RMS: `{report['native_replay_max_relative_rms']:.6g}`
- Shift interventions non-degenerate: `{report['shift_interventions_non_degenerate']}`

## Gates

```json
{json.dumps(report['gates'], indent=2, sort_keys=True)}
```

## Claim boundary

{report['claim_boundary']}

The analyzer reports perturbation sensitivity, not video quality. A passing
axis must still be tested in a trajectory-level method experiment.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                profile_dir=args.profile_dir,
                probe_plan_path=args.probe_plan,
                output_dir=args.output_dir,
                expected_count=args.expected_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
