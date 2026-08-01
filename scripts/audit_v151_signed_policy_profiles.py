#!/usr/bin/env python3
"""Audit v151 profiles without discarding independently valid contexts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_PROFILES = 64
EXPECTED_LAYERS = 30
EXPECTED_HEADS_PER_PROBE = 120
EXPECTED_PROFILE_VERSION = 8
MAX_REPLAY_RELATIVE_RMS = 1e-4
MAX_CALIBRATION_RELATIVE_ERROR = 0.02
MIN_ACCEPTABLE_SCALE = 0.005
MAX_ACCEPTABLE_SCALE = 50.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_plan(path: Path) -> tuple[dict, list[str], list[str], int, float]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(plan.get("version", -1)) != 1
        or plan.get("suite") != "v151_signed_policy_low_tail_core"
        or int(plan.get("layers", -1)) != EXPECTED_LAYERS
        or int(plan.get("heads", -1)) != 12
    ):
        raise RuntimeError("invalid v151 probe plan contract")
    probes = plan.get("probes") or []
    probe_names = [str(probe.get("name")) for probe in probes]
    if len(probe_names) != 32 or len(set(probe_names)) != 32:
        raise RuntimeError("v151 plan must contain 32 unique probes")
    contexts = [
        f"{row['mode']}_t{int(row['nominal_timestep'])}"
        for row in plan.get("contexts") or []
    ]
    if contexts != ["noisy_t1000", "noisy_t750", "noisy_t500", "noisy_t250"]:
        raise RuntimeError("v151 plan has an unexpected context grid")
    steps = {
        int(probe.get("calibration", {}).get("refinement_steps", -1))
        for probe in probes
    }
    targets = {
        float(probe.get("calibration", {}).get("target", float("nan")))
        for probe in probes
    }
    if len(steps) != 1 or len(targets) != 1:
        raise RuntimeError("v151 probes do not share one calibration contract")
    refinement_steps = steps.pop()
    target = targets.pop()
    if refinement_steps <= 0 or not math.isfinite(target) or target <= 0:
        raise RuntimeError("v151 calibration contract is invalid")
    return plan, probe_names, contexts, refinement_steps, target


def audit_profiles(
    profile_dir: Path,
    video_dir: Path,
    plan_path: Path,
) -> tuple[dict, list[dict]]:
    import torch

    _, probe_names, contexts, refinement_steps, target = _load_plan(plan_path)
    profile_paths = sorted(profile_dir.glob("*.pt"))
    video_paths = sorted(video_dir.glob("*.mp4"))
    if len(profile_paths) != EXPECTED_PROFILES or len(video_paths) != EXPECTED_PROFILES:
        raise RuntimeError(
            "v151 requires exactly 64 profiles and 64 videos: "
            f"profiles={len(profile_paths)} videos={len(video_paths)}"
        )

    expected_probe_names = {*probe_names, "native_replay"}
    context_stats = {
        context: {
            "layer_count": 0,
            "calibration_error_failure_count": 0,
            "calibration_scale_failure_count": 0,
            "clipped_count": 0,
            "degenerate_count": 0,
            "refinement_bound_hit_count": 0,
            "max_calibration_relative_error": 0.0,
            "min_calibration_scale": float("inf"),
            "max_calibration_scale": 0.0,
        }
        for context in contexts
    }
    offenders: list[dict] = []
    seen_profiles = set()
    probe_grid = Counter()
    max_replay = 0.0

    for path in profile_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        job = payload.get("job") or {}
        metadata = payload.get("metadata") or {}
        coordinate = (int(job["prompt_slot"]), int(job["seed_replicate"]))
        if coordinate in seen_profiles:
            raise RuntimeError(f"duplicate v151 profile coordinate {coordinate}")
        seen_profiles.add(coordinate)
        if (
            int(payload.get("version", -1)) != EXPECTED_PROFILE_VERSION
            or job.get("kind") != "v151_signed_policy_low_tail_core"
            or not (
                int(metadata["seed"])
                == int(job["seed"])
                == int(job["reference_seed"])
            )
            or metadata.get("incomplete_calls")
            or int(metadata.get("captured_calls", -1)) != len(contexts)
            or int(metadata.get("record_count", -1)) != len(contexts) * EXPECTED_LAYERS
        ):
            raise RuntimeError(f"{path} violates the v151 profile contract")
        rows = payload.get("downstream_probe_records") or []
        if (
            len(rows) != len(contexts) * len(expected_probe_names)
            or int(payload.get("downstream_probe_expected_count", -1)) != len(rows)
        ):
            raise RuntimeError(f"{path} has an incomplete downstream grid")

        for row in rows:
            context = f"{row['mode']}_t{int(row['nominal_timestep'])}"
            probe_name = str(row["probe_name"])
            if context not in context_stats or probe_name not in expected_probe_names:
                raise RuntimeError(
                    f"{path} has unknown downstream cell {context}/{probe_name}"
                )
            probe_grid[(*coordinate, context, probe_name)] += 1
            if probe_name == "native_replay":
                max_replay = max(
                    max_replay,
                    float(row["flow_metrics"]["relative_rms"]),
                    float(row["x0_metrics"]["relative_rms"]),
                )
                continue
            if int(row.get("selected_head_count", -1)) != EXPECTED_HEADS_PER_PROBE:
                raise RuntimeError(f"{path}/{probe_name} selected the wrong head count")
            layer_metadata = row.get("layer_metadata") or {}
            if len(layer_metadata) != EXPECTED_LAYERS:
                raise RuntimeError(f"{path}/{probe_name} has incomplete layers")
            for layer_key, layer in layer_metadata.items():
                layer_index = int(layer_key)
                if (
                    int(layer.get("calibration_refinement_steps", -1))
                    != refinement_steps
                ):
                    raise RuntimeError(
                        f"{path}/{probe_name}/layer{layer_index} changed "
                        "refinement steps"
                    )
                observed_target = float(layer.get("calibration_target", float("nan")))
                error = float(layer.get("calibration_relative_error", float("nan")))
                scale = float(layer.get("calibration_scale", float("nan")))
                if (
                    not math.isfinite(observed_target)
                    or abs(observed_target - target) > 1e-8
                    or not math.isfinite(error)
                    or not math.isfinite(scale)
                ):
                    raise RuntimeError(
                        f"{path}/{probe_name}/layer{layer_index} has "
                        "non-finite calibration"
                    )
                clipped = bool(layer.get("calibration_clipped", True))
                degenerate = bool(layer.get("calibration_degenerate", True))
                bound_hit = bool(
                    layer.get("calibration_refinement_bound_hit", True)
                )
                reasons = []
                if error > MAX_CALIBRATION_RELATIVE_ERROR:
                    reasons.append("relative_error")
                if not MIN_ACCEPTABLE_SCALE <= scale <= MAX_ACCEPTABLE_SCALE:
                    reasons.append("scale")
                if clipped:
                    reasons.append("clipped")
                if degenerate:
                    reasons.append("degenerate")
                if bound_hit:
                    reasons.append("refinement_bound_hit")

                stats = context_stats[context]
                stats["layer_count"] += 1
                stats["calibration_error_failure_count"] += int(
                    error > MAX_CALIBRATION_RELATIVE_ERROR
                )
                stats["calibration_scale_failure_count"] += int(
                    not MIN_ACCEPTABLE_SCALE <= scale <= MAX_ACCEPTABLE_SCALE
                )
                stats["clipped_count"] += int(clipped)
                stats["degenerate_count"] += int(degenerate)
                stats["refinement_bound_hit_count"] += int(bound_hit)
                stats["max_calibration_relative_error"] = max(
                    stats["max_calibration_relative_error"], error
                )
                stats["min_calibration_scale"] = min(
                    stats["min_calibration_scale"], scale
                )
                stats["max_calibration_scale"] = max(
                    stats["max_calibration_scale"], scale
                )
                if reasons:
                    offenders.append(
                        {
                            "prompt_slot": coordinate[0],
                            "seed_replicate": coordinate[1],
                            "context": context,
                            "probe_name": probe_name,
                            "policy": str(row["policy"]),
                            "rank_group": str(row["group"]),
                            "layer": layer_index,
                            "calibration_relative_error": error,
                            "calibration_scale": scale,
                            "reasons": ";".join(reasons),
                            "profile": str(path),
                        }
                    )

    expected_coordinates = {
        (prompt, seed) for prompt in range(32) for seed in (0, 1)
    }
    if seen_profiles != expected_coordinates:
        raise RuntimeError("v151 prompt/seed grid is incomplete")
    expected_grid_size = (
        EXPECTED_PROFILES * len(contexts) * len(expected_probe_names)
    )
    if len(probe_grid) != expected_grid_size or set(probe_grid.values()) != {1}:
        raise RuntimeError("v151 downstream probe coordinates are incomplete")
    if max_replay > MAX_REPLAY_RELATIVE_RMS:
        raise RuntimeError(
            f"v151 native replay failed: {max_replay:.6g} > "
            f"{MAX_REPLAY_RELATIVE_RMS:.6g}"
        )

    expected_layers_per_context = EXPECTED_PROFILES * len(probe_names) * EXPECTED_LAYERS
    intact_contexts = []
    for context, stats in context_stats.items():
        if stats["layer_count"] != expected_layers_per_context:
            raise RuntimeError(
                f"{context} has {stats['layer_count']} calibrated layers, "
                f"expected {expected_layers_per_context}"
            )
        stats["integrity_pass"] = not any(
            stats[key]
            for key in (
                "calibration_error_failure_count",
                "calibration_scale_failure_count",
                "clipped_count",
                "degenerate_count",
                "refinement_bound_hit_count",
            )
        )
        if stats["integrity_pass"]:
            intact_contexts.append(context)

    report = {
        "version": 1,
        "suite": "v151_signed_policy_low_tail_core",
        "profile_count": len(profile_paths),
        "video_count": len(video_paths),
        "probe_count_excluding_native_replay": len(probe_names),
        "refinement_steps": refinement_steps,
        "calibration_target": target,
        "maximum_allowed_calibration_relative_error": (
            MAX_CALIBRATION_RELATIVE_ERROR
        ),
        "native_replay_max_relative_rms": max_replay,
        "intact_contexts": intact_contexts,
        "invalid_contexts": [
            context for context in contexts if context not in intact_contexts
        ],
        "offending_layer_count": len(offenders),
        "contexts": context_stats,
    }
    return report, offenders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict", "analysis"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--offenders-csv", type=Path, required=True)
    args = parser.parse_args()

    report, offenders = audit_profiles(
        args.profile_dir, args.video_dir, args.probe_plan
    )
    report["mode"] = args.mode
    report["accepted"] = bool(
        len(report["invalid_contexts"]) == 0
        if args.mode == "strict"
        else len(report["intact_contexts"]) > 0
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.offenders_csv, offenders)
    for context, stats in report["contexts"].items():
        print(
            "[v151-context-audit] "
            f"context={context} pass={int(stats['integrity_pass'])} "
            f"layers={stats['layer_count']} "
            f"error_fail={stats['calibration_error_failure_count']} "
            f"scale_fail={stats['calibration_scale_failure_count']} "
            f"max_error={stats['max_calibration_relative_error']:.6g}"
        )
    print(
        "[v151-audit-summary] "
        f"mode={args.mode} accepted={int(report['accepted'])} "
        f"intact={','.join(report['intact_contexts']) or 'none'} "
        f"invalid={','.join(report['invalid_contexts']) or 'none'} "
        f"offenders={len(offenders)} "
        f"replay={report['native_replay_max_relative_rms']:.6g}"
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
