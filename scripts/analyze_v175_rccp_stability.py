#!/usr/bin/env python3
"""Audit RCCP head-map stability before generation-side intervention."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import analyze_v173_cache_compatibility as v173


SPLIT_SEEDS = tuple(1752026 + index * 1009 for index in range(12))
DISCOVERY_SEED = 1751301


def hard_negative_maps(
    stable_labels: list[list[int]],
    records: list[dict],
    *,
    discovery_prompt_ids: set[int],
) -> dict[str, list[list[int]]]:
    """Build layer/count-matched controls from the least compatible heads."""

    samples = v173.flatten_head_samples(records)
    maps = {}
    selected_by_replica = {}
    for replica in range(4):
        rows = [[v173.LABELS["recent"]] * v173.HEADS for _ in range(v173.LAYERS)]
        for layer in range(v173.LAYERS):
            for policy in ("coverage", "episode"):
                label = v173.LABELS[policy]
                count = stable_labels[layer].count(label)
                if count == 0:
                    continue
                candidates = []
                for head in range(v173.HEADS):
                    if (
                        stable_labels[layer][head] != v173.LABELS["recent"]
                        or rows[layer][head] != v173.LABELS["recent"]
                    ):
                        continue
                    head_samples = [
                        row
                        for row in samples[(layer, head)]
                        if row["prompt_id"] in discovery_prompt_ids
                    ]
                    advantage = float(
                        np.mean(
                            [
                                row["log_error_recent"] - row[f"log_error_{policy}"]
                                for row in head_samples
                            ]
                        )
                    )
                    # Tiny deterministic jitter creates distinct tied replicas.
                    jitter = float(
                        np.random.default_rng(
                            1759000 + replica * 1009 + layer * 31 + head
                        ).uniform(-1e-9, 1e-9)
                    )
                    candidates.append((advantage + jitter, head))
                if len(candidates) < count:
                    raise ValueError(
                        f"L{layer}: cannot construct disjoint {policy} hard "
                        f"negative with count={count} candidates={len(candidates)}"
                    )
                # Sample from the most-compatible rejected pool so controls
                # are near-miss hard negatives, not trivially bad heads.
                pool_size = min(len(candidates), max(count, count * 4))
                pool = [
                    head
                    for _, head in sorted(candidates, reverse=True)[:pool_size]
                ]
                offset = replica % len(pool)
                chosen = [pool[(offset + index) % len(pool)] for index in range(count)]
                for head in chosen:
                    rows[layer][head] = label
        maps[f"hard_negative_{replica}"] = rows
        selected_by_replica[replica] = {
            (layer, head, value)
            for layer, row in enumerate(rows)
            for head, value in enumerate(row)
            if value != v173.LABELS["recent"]
        }
    if any(selected_by_replica.values()) and any(
        selected_by_replica[left] == selected_by_replica[right]
        for left in selected_by_replica
        for right in selected_by_replica
        if left < right
    ):
        raise ValueError(
            "hard-negative replicas collapsed to duplicate maps; reduce the "
            "replica count or enlarge the within-layer candidate pool"
        )
    return maps


def write_csv(path: Path, rows: list[list[int]]) -> dict:
    return v173.write_map(path, rows)


def render(payload: dict) -> str:
    stability = payload["stability"]
    mean_jaccard = stability["mean_pairwise_jaccard"]
    mean_jaccard_text = "N/A" if mean_jaccard is None else f"{mean_jaccard:.4f}"
    lines = [
        "# v175 RCCP Stability Audit",
        "",
        f"Profile complete: {payload['profile_complete']}",
        f"Observed prompts: {payload['observed_prompt_count']}/128",
        f"Mean non-empty split Jaccard: {mean_jaccard_text}",
        f"Stable nonlocal heads: {stability['stable_nonlocal_head_count']}",
        f"Generation ready: {payload['generation_ready']}",
        "",
        "| Policy | Stable heads |",
        "|---|---:|",
    ]
    for policy in ("coverage", "episode"):
        lines.append(
            f"| {policy} | {stability['stable_nonlocal_counts'].get(policy, 0)} |"
        )
    lines.extend(["", payload["claim_boundary"], ""])
    return "\n".join(lines)


def analyze(
    profile_root: Path,
    output_dir: Path,
    *,
    require_complete: bool,
    bootstrap_samples: int,
) -> dict:
    records, audit = v173.load_records(profile_root, strict=require_complete)
    observed = list(audit["prompt_ids"])
    discovery_count = len(observed) // 2
    discovery_prompts, transfer_prompts = v173.split_prompt_ids(
        observed,
        calibration_prompts=discovery_count,
        split_seed=DISCOVERY_SEED,
    )
    stability = v173.analyze_split_stability(
        records,
        split_seeds=SPLIT_SEEDS,
        calibration_prompts=len(discovery_prompts) // 2,
        bootstrap_samples=bootstrap_samples,
        prompt_ids=discovery_prompts,
    )
    labels = stability.pop("stable_labels")
    controls = {
        "stable_matched": labels,
        "stable_all_recent": [
            [v173.LABELS["recent"]] * v173.HEADS
            for _ in range(v173.LAYERS)
        ],
        **hard_negative_maps(
            labels,
            records,
            discovery_prompt_ids=set(discovery_prompts),
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        name: write_csv(output_dir / "maps" / f"{name}.csv", rows)
        for name, rows in controls.items()
    }
    stable_count = int(stability["stable_nonlocal_head_count"])
    payload = {
        "version": 1,
        "experiment": "v175_rccp_stability",
        "profile_complete": bool(audit["complete_profile"]),
        "observed_prompt_count": len(observed),
        "profile_audit": audit,
        "discovery_transfer_split": {
            "seed": DISCOVERY_SEED,
            "discovery_prompt_ids": discovery_prompts,
            "transfer_prompt_ids": transfer_prompts,
            "membership_uses_transfer_prompts": False,
        },
        "stability": stability,
        "maps": maps,
        "generation_ready": bool(audit["complete_profile"] and stable_count > 0),
        "claim_boundary": (
            "Stable RCCP membership remains a profiling hypothesis until "
            "stable_matched beats layer/count-matched hard negatives in paired "
            "generation. A zero-head result rejects the current classifier."
        ),
    }
    output_path = output_dir / "stability.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".md").write_text(render(payload), encoding="utf-8")
    print(
        "[v175-stability] "
        f"prompts={len(observed)} complete={audit['complete_profile']} "
        f"stable_heads={stable_count} generation_ready={payload['generation_ready']} "
        f"output={output_path}"
    )
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=root / "runs" / "v173_cache_compatibility" / "profiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "v175_rccp_stability" / "analysis",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analyze(
        args.profile_root,
        args.output_dir,
        require_complete=not args.allow_partial,
        bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
