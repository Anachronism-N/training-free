#!/usr/bin/env python3
"""Analyze v173 residual-space cache compatibility profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


POLICIES = ("recent", "coverage", "episode")
LABELS = {"recent": 20, "coverage": 21, "episode": 22}
LAYERS = 30
HEADS = 12
PROMPTS = 128
EXPECTED_BUDGET = {"recent": 9, "coverage": 9, "episode": 9, "union": 15}
SPLIT_SEED = 1732026


def finite(value, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.pt"))
    if not paths:
        paths = sorted(root.rglob("*.pt"))
    if not paths:
        raise ValueError(f"no profile shards found below {root}")
    return paths


def load_records(root: Path, *, strict: bool = False) -> tuple[list[dict], dict]:
    import torch

    records: list[dict] = []
    shard_rows = []
    seen_locations = set()
    for path in profile_paths(root):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") != 1:
            raise ValueError(f"{path}: unsupported profile version")
        if payload.get("method") != "residual_space_equal_budget_cache_compatibility":
            raise ValueError(f"{path}: unexpected profiling method")
        if tuple(payload.get("policies") or ()) != POLICIES:
            raise ValueError(f"{path}: policy contract drift")
        shard_records = payload.get("records") or []
        if not shard_records:
            raise ValueError(f"{path}: profile contains no records")
        prompt_ids = set()
        for record_index, record in enumerate(shard_records):
            prompt_id = int(record["prompt_id"])
            layer = int(record["layer"])
            heads = int(record["heads"])
            if not 0 <= prompt_id < PROMPTS:
                raise ValueError(f"{path}: invalid prompt id {prompt_id}")
            if not 0 <= layer < LAYERS or heads != HEADS:
                raise ValueError(f"{path}: invalid layer/head shape")
            prompt_ids.add(prompt_id)
            location = (
                prompt_id,
                layer,
                int(record["current_frame"]),
                int(record["call_index"]),
                str(record["cache_update_mode"]),
                str(record["cfg_branch"]),
            )
            if location in seen_locations:
                raise ValueError(f"duplicate profile location: {location}")
            seen_locations.add(location)
            policies = record.get("policies") or {}
            budgets = record.get("budgets") or {}
            if set(policies) != set(POLICIES):
                raise ValueError(f"{path}:{record_index}: incomplete metrics")
            if set(budgets) != set(POLICIES) | {"union"}:
                raise ValueError(f"{path}:{record_index}: incomplete budgets")
            for policy in POLICIES:
                metrics = policies[policy]
                for metric in (
                    "residual_relative_mse",
                    "residual_cosine",
                    "raw_relative_mse",
                    "raw_cosine",
                    "output_rms",
                ):
                    values = metrics.get(metric) or []
                    if len(values) != HEADS:
                        raise ValueError(
                            f"{path}:{record_index}:{policy}:{metric} shape drift"
                        )
                    for head, value in enumerate(values):
                        score = finite(
                            value,
                            name=f"{path.name}:{record_index}:{policy}:{metric}:{head}",
                        )
                        if metric.endswith("mse") and score < 0:
                            raise ValueError("relative MSE must be non-negative")
            for policy, expected in EXPECTED_BUDGET.items():
                budget = budgets[policy]
                observed = int(budget["max_frame_equivalents"])
                if observed > expected:
                    raise ValueError(
                        f"{path}:{record_index}:{policy} budget {observed}>{expected}"
                    )
                per_sequence = budget.get("per_sequence_frame_equivalents") or []
                if len(per_sequence) != HEADS:
                    raise ValueError(
                        f"{path}:{record_index}:{policy} sequence budget shape drift"
                    )
                selected = budget.get("selected_physical_frames_per_sequence")
                if layer in {0, 10, 20, 29}:
                    if not isinstance(selected, list) or len(selected) != HEADS:
                        raise ValueError(
                            f"{path}:{record_index}:{policy} missing frame-id trace"
                        )
                    codebook = budget.get("selected_source_codebook") or []
                    for head, (items, frame_budget) in enumerate(
                        zip(selected, per_sequence)
                    ):
                        frame_ids = [int(item[0]) for item in items]
                        if len(frame_ids) != int(frame_budget):
                            raise ValueError(
                                f"{path}:{record_index}:{policy}:H{head} "
                                "frame-id/budget mismatch"
                            )
                        if len(frame_ids) != len(set(frame_ids)):
                            raise ValueError(
                                f"{path}:{record_index}:{policy}:H{head} "
                                "duplicate physical frame"
                            )
                        try:
                            sources = {
                                str(codebook[int(item[1])]) for item in items
                            }
                        except (IndexError, TypeError, ValueError) as error:
                            raise ValueError(
                                f"{path}:{record_index}:{policy}:H{head} "
                                "invalid source code"
                            ) from error
                        if policy == "recent" and not sources.issubset(
                            {"static", "dynamic"}
                        ):
                            raise ValueError("Recent readout contains middle history")
                        if policy == "coverage" and "anchor:coherent_motion" in sources:
                            raise ValueError("Coverage readout contains a motion episode")
                elif selected is not None:
                    raise ValueError(
                        f"{path}:{record_index}:{policy} unexpected frame-id trace"
                    )
            records.append(record)
        shard_rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "record_count": len(shard_records),
                "prompt_ids": sorted(prompt_ids),
            }
        )

    prompt_ids = sorted({int(row["prompt_id"]) for row in records})
    calls = sorted({int(row["call_index"]) for row in records})
    modes = sorted({str(row["cache_update_mode"]) for row in records})
    branches = sorted({str(row["cfg_branch"]) for row in records})
    coverage = Counter((int(row["prompt_id"]), int(row["layer"])) for row in records)
    if strict:
        if prompt_ids != list(range(PROMPTS)):
            raise ValueError("strict v173 audit requires prompt ids 0..127")
        if calls != [0, 2] or modes != ["noisy"] or branches != ["cond"]:
            raise ValueError(
                "strict v173 call contract drift: "
                f"calls={calls} modes={modes} branches={branches}"
            )
        counts = set(coverage.values())
        if len(coverage) != PROMPTS * LAYERS or len(counts) != 1:
            raise ValueError("strict v173 profile coverage is incomplete or ragged")
    audit = {
        "strict": bool(strict),
        "shard_count": len(shard_rows),
        "record_count": len(records),
        "prompt_ids": prompt_ids,
        "call_indices": calls,
        "update_modes": modes,
        "branches": branches,
        "records_per_prompt_layer": sorted(set(coverage.values())),
        "shards": shard_rows,
    }
    return records, audit


def bootstrap_ci(values: Iterable[float], *, seed: int, samples: int) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def one_sided_sign_p(values: Iterable[float]) -> float:
    nonzero = [float(value) for value in values if float(value) != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = sum(value > 0 for value in nonzero)
    return min(
        1.0,
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n),
    )


def benjamini_hochberg(rows: list[dict], *, p_key: str = "p_value") -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index][p_key])
    total = len(order)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = total - reverse_rank + 1
        adjusted = min(1.0, rows[index][p_key] * total / rank)
        running = min(running, adjusted)
        rows[index]["q_value"] = running


def _prompt_means(samples: list[dict], value_key: str) -> dict[int, float]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[int(sample["prompt_id"])].append(float(sample[value_key]))
    return {
        prompt: float(np.mean(values))
        for prompt, values in grouped.items()
    }


def flatten_head_samples(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        for head in range(HEADS):
            row = {
                "prompt_id": int(record["prompt_id"]),
                "layer": int(record["layer"]),
                "head": head,
                "current_frame": int(record["current_frame"]),
                "call_index": int(record["call_index"]),
            }
            for policy in POLICIES:
                error = finite(
                    record["policies"][policy]["residual_relative_mse"][head],
                    name=f"{policy}:residual_relative_mse",
                )
                row[f"log_error_{policy}"] = math.log(max(error, 1e-12))
                row[f"budget_{policy}"] = int(
                    record["budgets"][policy][
                        "per_sequence_frame_equivalents"
                    ][head]
                )
            grouped[(row["layer"], head)].append(row)
    return grouped


def analyze_heads(
    records: list[dict],
    *,
    calibration_prompts: int = 64,
    bootstrap_samples: int = 2000,
) -> tuple[list[dict], list[list[int]]]:
    if not 1 <= calibration_prompts < PROMPTS:
        raise ValueError("calibration prompt count must split the 128 prompts")
    split_rng = np.random.default_rng(SPLIT_SEED)
    permutation = [int(value) for value in split_rng.permutation(PROMPTS)]
    calibration_ids = set(permutation[:calibration_prompts])
    grouped = flatten_head_samples(records)
    if set(grouped) != {
        (layer, head) for layer in range(LAYERS) for head in range(HEADS)
    }:
        raise ValueError("head sample grid is incomplete")

    head_rows = []
    pair_tests = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            samples = grouped[(layer, head)]
            calibration = [
                row for row in samples if row["prompt_id"] in calibration_ids
            ]
            validation = [
                row for row in samples if row["prompt_id"] not in calibration_ids
            ]
            if not calibration or not validation:
                raise ValueError(f"L{layer}H{head}: empty calibration/validation split")
            calibration_errors = {
                policy: float(
                    np.mean([row[f"log_error_{policy}"] for row in calibration])
                )
                for policy in POLICIES
            }
            ranked = sorted(POLICIES, key=calibration_errors.__getitem__)
            best = ranked[0]
            calibration_margin = (
                calibration_errors[ranked[1]] - calibration_errors[best]
            )
            comparison_rows = []
            for alternative in POLICIES:
                if alternative == best:
                    continue
                values = []
                for row in validation:
                    copied = dict(row)
                    copied["advantage"] = (
                        row[f"log_error_{alternative}"]
                        - row[f"log_error_{best}"]
                    )
                    values.append(copied)
                prompt_advantages = _prompt_means(values, "advantage")
                ordered = [
                    prompt_advantages[prompt]
                    for prompt in sorted(prompt_advantages)
                ]
                ci = bootstrap_ci(
                    ordered,
                    seed=1732026 + layer * 101 + head * 7 + POLICIES.index(alternative),
                    samples=bootstrap_samples,
                )
                call_means = {}
                for call_index in sorted({row["call_index"] for row in values}):
                    subset = [row for row in values if row["call_index"] == call_index]
                    call_means[str(call_index)] = float(
                        np.mean(list(_prompt_means(subset, "advantage").values()))
                    )
                frame_values = sorted({row["current_frame"] for row in values})
                frame_midpoint = float(np.median(frame_values))
                early = [row for row in values if row["current_frame"] <= frame_midpoint]
                late = [row for row in values if row["current_frame"] > frame_midpoint]
                split_means = {
                    "early": float(
                        np.mean(list(_prompt_means(early, "advantage").values()))
                    ),
                    "late": float(
                        np.mean(list(_prompt_means(late, "advantage").values()))
                    ),
                }
                test = {
                    "layer": layer,
                    "head": head,
                    "best": best,
                    "alternative": alternative,
                    "prompt_count": len(ordered),
                    "mean_log_advantage": float(np.mean(ordered)),
                    "median_log_advantage": float(np.median(ordered)),
                    "bootstrap_ci95": ci,
                    "win_fraction": float(np.mean(np.asarray(ordered) > 0.0)),
                    "p_value": one_sided_sign_p(ordered),
                    "call_means": call_means,
                    "ar_split_means": split_means,
                }
                pair_tests.append(test)
                comparison_rows.append(test)
            availability = float(
                np.mean(
                    [
                        row[f"budget_{best}"] == EXPECTED_BUDGET[best]
                        for row in validation
                    ]
                )
            )
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "calibration_best": best,
                    "calibration_log_errors": calibration_errors,
                    "calibration_margin": float(calibration_margin),
                    "validation_budget_full_fraction": availability,
                    "comparisons": comparison_rows,
                }
            )

    benjamini_hochberg(pair_tests)
    labels = [[LABELS["recent"] for _ in range(HEADS)] for _ in range(LAYERS)]
    for row in head_rows:
        comparisons = row["comparisons"]
        gates = {
            "calibration_margin_ge_0p01": row["calibration_margin"] >= 0.01,
            "budget_full_ge_0p80": row["validation_budget_full_fraction"] >= 0.80,
            "validation_mean_ge_log1p02": all(
                item["mean_log_advantage"] >= math.log(1.02)
                for item in comparisons
            ),
            "bootstrap_lower_gt_0": all(
                item["bootstrap_ci95"][0] > 0.0 for item in comparisons
            ),
            "prompt_win_fraction_ge_0p60": all(
                item["win_fraction"] >= 0.60 for item in comparisons
            ),
            "bh_q_le_0p10": all(
                item["q_value"] <= 0.10 for item in comparisons
            ),
            "call_stable": all(
                all(value > 0.0 for value in item["call_means"].values())
                for item in comparisons
            ),
            "ar_stable": all(
                all(value > 0.0 for value in item["ar_split_means"].values())
                for item in comparisons
            ),
        }
        row["gates"] = gates
        row["supported"] = all(gates.values())
        row["assigned_policy"] = (
            row["calibration_best"] if row["supported"] else "recent"
        )
        row["assigned_label"] = LABELS[row["assigned_policy"]]
        labels[row["layer"]][row["head"]] = row["assigned_label"]
    return head_rows, labels


def write_map(path: Path, rows: list[list[int]]) -> dict:
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError("head map must be 30x12")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    counts = Counter(value for row in rows for value in row)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "counts": {str(key): counts.get(key, 0) for key in sorted(LABELS.values())},
        "per_layer": [
            {str(key): row.count(key) for key in sorted(LABELS.values())}
            for row in rows
        ],
    }


def build_control_maps(labels: list[list[int]]) -> dict[str, list[list[int]]]:
    swapped = [
        [
            LABELS["episode"]
            if value == LABELS["coverage"]
            else LABELS["coverage"]
            if value == LABELS["episode"]
            else value
            for value in row
        ]
        for row in labels
    ]
    maps = {
        "matched": [list(row) for row in labels],
        "swapped": swapped,
        "all_recent": [[LABELS["recent"]] * HEADS for _ in range(LAYERS)],
        "all_coverage": [[LABELS["coverage"]] * HEADS for _ in range(LAYERS)],
        "all_episode": [[LABELS["episode"]] * HEADS for _ in range(LAYERS)],
    }
    for replica in range(4):
        random_rows = []
        for layer, row in enumerate(labels):
            shuffled = np.asarray(row, dtype=np.int64).copy()
            rng = np.random.default_rng(1732026 + replica * 1009 + layer)
            rng.shuffle(shuffled)
            random_rows.append([int(value) for value in shuffled.tolist()])
        maps[f"random_count_matched_{replica}"] = random_rows
    return maps


def write_analysis(
    profile_root: Path,
    output_dir: Path,
    *,
    calibration_prompts: int,
    bootstrap_samples: int,
    strict: bool,
) -> dict:
    records, audit = load_records(profile_root, strict=strict)
    head_rows, labels = analyze_heads(
        records,
        calibration_prompts=calibration_prompts,
        bootstrap_samples=bootstrap_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = build_control_maps(labels)
    map_manifest = {
        name: write_map(output_dir / "maps" / f"{name}.csv", rows)
        for name, rows in maps.items()
    }
    scores_path = output_dir / "head_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "layer",
            "head",
            "calibration_best",
            "calibration_margin",
            "validation_budget_full_fraction",
            "supported",
            "assigned_policy",
            "assigned_label",
            "min_validation_advantage",
            "min_bootstrap_lower",
            "max_q_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in head_rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in fieldnames
                    if key
                    not in {
                        "min_validation_advantage",
                        "min_bootstrap_lower",
                        "max_q_value",
                    }
                }
                | {
                    "min_validation_advantage": min(
                        item["mean_log_advantage"] for item in row["comparisons"]
                    ),
                    "min_bootstrap_lower": min(
                        item["bootstrap_ci95"][0] for item in row["comparisons"]
                    ),
                    "max_q_value": max(
                        item["q_value"] for item in row["comparisons"]
                    ),
                }
            )
    assigned = Counter(row["assigned_policy"] for row in head_rows)
    supported = Counter(
        row["calibration_best"] for row in head_rows if row["supported"]
    )
    payload = {
        "version": 1,
        "experiment": "v173_residual_cache_compatibility",
        "method": "RCCP",
        "claim_boundary": (
            "operator-aligned head assignments are hypotheses until matched "
            "generation beats swapped and count-matched random controls"
        ),
        "profile_audit": audit,
        "split": {
            "seed": SPLIT_SEED,
            "calibration_prompt_ids": sorted(
                int(value)
                for value in np.random.default_rng(SPLIT_SEED)
                .permutation(PROMPTS)[:calibration_prompts]
                .tolist()
            ),
            "validation_prompt_ids": sorted(
                set(range(PROMPTS))
                - set(
                    int(value)
                    for value in np.random.default_rng(SPLIT_SEED)
                    .permutation(PROMPTS)[:calibration_prompts]
                    .tolist()
                )
            ),
        },
        "gates": {
            "calibration_margin": 0.01,
            "mean_log_advantage": math.log(1.02),
            "bootstrap_lower": 0.0,
            "prompt_win_fraction": 0.60,
            "bh_q": 0.10,
            "full_budget_fraction": 0.80,
            "call_and_ar_sign_stability": True,
        },
        "supported_best_policy_counts": dict(sorted(supported.items())),
        "assigned_policy_counts": dict(sorted(assigned.items())),
        "nonlocal_supported_head_count": sum(
            row["supported"] and row["calibration_best"] != "recent"
            for row in head_rows
        ),
        "head_rows": head_rows,
        "maps": map_manifest,
        "head_scores_csv": {
            "path": str(scores_path.resolve()),
            "sha256": sha256(scores_path),
        },
        "generation_ready": bool(
            any(
                row["supported"] and row["calibration_best"] != "recent"
                for row in head_rows
            )
        ),
    }
    output_path = output_dir / "analysis.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[v173-analysis] "
        f"records={len(records)} supported={dict(supported)} "
        f"assigned={dict(assigned)} generation_ready={payload['generation_ready']} "
        f"output={output_path}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v173_cache_compatibility"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=run_root / "profiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "analysis",
    )
    parser.add_argument("--calibration-prompts", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_analysis(
        args.profile_root,
        args.output_dir,
        calibration_prompts=args.calibration_prompts,
        bootstrap_samples=args.bootstrap_samples,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
