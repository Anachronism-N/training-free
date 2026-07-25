#!/usr/bin/env python3
"""Extract immutable per-head QK scores without assigning class labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import build_v96_qk_head_thresholds as qk


TEMPORAL_METRICS = (
    "positive_rate",
    "mean_logit",
    "mean_abs_logit",
    "signed_logit_mass",
    "sign_switch_rate",
    "dominant_period",
    "spectral_peak_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--run-manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def validate_profiles(
    profiles: list[dict],
    num_layers: int,
    num_heads: int,
) -> dict:
    expected_layers = list(range(num_layers))
    per_profile = []
    pair_sides: dict[tuple[str, int], list[str]] = {}
    for profile in profiles:
        if int(profile["version"]) < 2:
            raise ValueError(
                f"{profile['path']}: profile version {profile['version']} "
                "predates explicit kv_cache.layer_idx capture and cannot be reused"
            )
        records = profile["records"]
        layers = sorted({int(record["layer"]) for record in records})
        branches = sorted({str(record["cfg_branch"]) for record in records})
        sources = sorted(
            {str(record.get("layer_index_source", "")) for record in records}
        )
        update_modes = sorted(
            {str(record["cache_update_mode"]) for record in records}
        )
        audit = profile.get("audit") or {}
        if int(audit.get("expected_num_layers", -1)) != num_layers:
            raise ValueError(
                f"{profile['path']}: recorder expected layer count mismatch"
            )
        if int(audit.get("expected_num_heads", -1)) != num_heads:
            raise ValueError(
                f"{profile['path']}: recorder expected head count mismatch"
            )
        if layers != expected_layers:
            raise ValueError(
                f"{profile['path']}: expected layers {expected_layers}, "
                f"found {layers}"
            )
        if branches != ["cond", "uncond"]:
            raise ValueError(
                f"{profile['path']}: expected cond/uncond, found {branches}"
            )
        if sources != ["kv_cache.layer_idx"]:
            raise ValueError(
                f"{profile['path']}: non-explicit layer sources {sources}"
            )
        if update_modes != ["clean", "noisy"]:
            raise ValueError(
                f"{profile['path']}: expected clean/noisy records, "
                f"found {update_modes}"
            )
        for record_index, record in enumerate(records):
            logits = record.get("logits")
            key_frames = record.get("key_frames")
            if logits is None or tuple(logits.shape[:1]) != (num_heads,):
                raise ValueError(
                    f"{profile['path']}: record {record_index} has invalid "
                    f"head dimension {getattr(logits, 'shape', None)}"
                )
            if (
                key_frames is None
                or logits.ndim != 2
                or int(logits.shape[1]) != int(key_frames.numel())
            ):
                raise ValueError(
                    f"{profile['path']}: record {record_index} has "
                    "misaligned logits/key frames"
                )
        missing_layer_branches = [
            (layer, branch)
            for layer in expected_layers
            for branch in ("cond", "uncond")
            if not any(
                int(record["layer"]) == layer
                and str(record["cfg_branch"]) == branch
                for record in records
            )
        ]
        if missing_layer_branches:
            raise ValueError(
                f"{profile['path']}: missing layer/branch records "
                f"{missing_layer_branches[:12]}"
            )
        pair_id = str(profile["pair_id"])
        side = str(profile["side"])
        if not pair_id or side not in {"a", "b"}:
            raise ValueError(
                f"{profile['path']}: invalid pair metadata "
                f"pair_id={pair_id!r} side={side!r}"
            )
        pair_sides.setdefault((pair_id, int(profile["seed"])), []).append(side)
        per_profile.append(
            {
                "path": profile["path"],
                "sha256": sha256(Path(profile["path"])),
                "pair_id": profile["pair_id"],
                "side": profile["side"],
                "seed": profile["seed"],
                "record_count": len(records),
                "layers": layers,
                "branches": branches,
                "update_modes": update_modes,
                "layer_index_sources": sources,
            }
        )
    invalid_pairs = {
        f"{pair_id}:seed{seed}": sorted(sides)
        for (pair_id, seed), sides in pair_sides.items()
        if sorted(sides) != ["a", "b"]
    }
    if invalid_pairs:
        raise ValueError(
            "counterfactual profile pairs must contain exactly one a/b side: "
            f"{invalid_pairs}"
        )
    return {
        "profile_count": len(profiles),
        "expected_layers": expected_layers,
        "all_profiles_valid": True,
        "profiles": per_profile,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = qk.load_profiles(args.profiles)
    layer_audit = validate_profiles(
        profiles, args.num_layers, args.num_heads
    )
    keys = [
        (layer, head)
        for layer in range(args.num_layers)
        for head in range(args.num_heads)
    ]

    cfg_observations = qk.collect_cfg_observations(
        profiles, args.num_heads
    )
    semantic_observations = qk.collect_semantic_observations(
        profiles, args.num_heads
    )
    temporal = qk.collect_temporal_statistics(profiles, args.num_heads)
    cfg_raw = qk.summarize_observations(cfg_observations, keys)
    semantic_raw = qk.summarize_observations(semantic_observations, keys)
    cfg_scores = qk.layer_robust_z(
        cfg_raw, args.num_layers, args.num_heads
    )
    semantic_scores = qk.layer_robust_z(
        semantic_raw, args.num_layers, args.num_heads
    )
    consensus_scores = {
        key: 0.5 * (cfg_scores[key] + semantic_scores[key])
        for key in keys
    }

    entries = []
    observations = []
    for layer, head in keys:
        key = (layer, head)
        missing_temporal = [
            metric
            for metric in TEMPORAL_METRICS
            if not temporal.get(key, {}).get(metric)
        ]
        if missing_temporal:
            raise ValueError(
                f"L{layer}H{head}: missing temporal metrics {missing_temporal}"
            )
        entry = {
            "layer": layer,
            "head": head,
            "cfg_raw": cfg_raw[key],
            "semantic_raw": semantic_raw[key],
            "cfg_score": cfg_scores[key],
            "semantic_score": semantic_scores[key],
            "consensus_score": consensus_scores[key],
            "cfg_observation_count": len(cfg_observations[key]),
            "semantic_observation_count": len(semantic_observations[key]),
            "temporal_observation_count": len(
                temporal[key]["positive_rate"]
            ),
            **{
                metric: qk.median(temporal[key][metric])
                for metric in TEMPORAL_METRICS
            },
        }
        entries.append(entry)
        observations.append(
            {
                "layer": layer,
                "head": head,
                "cfg_nrms": cfg_observations[key],
                "semantic_nrms": semantic_observations[key],
                "temporal": dict(temporal[key]),
            }
        )

    score_csv = args.output_dir / "qk_head_scores.csv"
    with score_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)

    observation_path = args.output_dir / "qk_head_observations.json"
    observation_path.write_text(
        json.dumps(
            {
                "version": 1,
                "method": "unthresholded_qk_head_observations",
                "entries": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    score_sets = {
        "cfg_score": [entry["cfg_score"] for entry in entries],
        "semantic_score": [
            entry["semantic_score"] for entry in entries
        ],
        "consensus_score": [
            entry["consensus_score"] for entry in entries
        ],
        "positive_rate": [entry["positive_rate"] for entry in entries],
        "signed_logit_mass": [
            entry["signed_logit_mass"] for entry in entries
        ],
    }
    diagnostics = {}
    for name, values in score_sets.items():
        item = {"distribution": distribution(values)}
        if name.endswith("_score"):
            item["gmm"] = [
                qk.fit_gmm_1d(values, components)
                for components in (1, 2, 3)
            ]
            item["gmm2_intersection"] = qk.gmm_threshold(item["gmm"][1])
            item["otsu_threshold"] = qk.otsu_threshold(values)
        diagnostics[name] = item

    artifact = {
        "version": 1,
        "method": "qk_head_score_artifact_no_classification",
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_count": len(entries),
        "score_definition": {
            "cfg": "layer_robust_z(log1p(median_cfg_nrms))",
            "semantic": "layer_robust_z(log1p(median_semantic_nrms))",
            "consensus": "0.5 * (cfg_score + semantic_score)",
            "classification": None,
        },
        "layer_audit": layer_audit,
        "files": {
            "score_csv": str(score_csv.resolve()),
            "score_csv_sha256": sha256(score_csv),
            "observations": str(observation_path.resolve()),
            "observations_sha256": sha256(observation_path),
        },
        "diagnostics": diagnostics,
    }
    if args.run_manifest is not None:
        if not args.run_manifest.is_file():
            raise ValueError(f"missing run manifest: {args.run_manifest}")
        artifact["files"]["run_manifest"] = str(
            args.run_manifest.resolve()
        )
        artifact["files"]["run_manifest_sha256"] = sha256(
            args.run_manifest
        )
    artifact_path = args.output_dir / "qk_head_score_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "layer_capture_audit.json").write_text(
        json.dumps(layer_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# Unthresholded QK Head Scores",
        "",
        f"- profiles: {len(profiles)}",
        f"- heads: {len(entries)} ({args.num_layers} x {args.num_heads})",
        "- classification applied: no",
        "- layer source: `kv_cache.layer_idx`",
        f"- scores: `{score_csv.name}`",
        f"- score artifact: `{artifact_path.name}`",
        "",
        "All thresholds and alternative classifiers must consume this immutable "
        "score artifact. Profiling inference must not be rerun to change labels.",
    ]
    (args.output_dir / "qk_head_score_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )
    print(
        "[QKHeadScoreArtifact] "
        f"profiles={len(profiles)} heads={len(entries)} "
        f"layers={args.num_layers} classification=none "
        f"csv_sha256={artifact['files']['score_csv_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
