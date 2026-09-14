#!/usr/bin/env python3
"""Compare native SF, vendored plain SF21, and Adaptive recent21 traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


RUNS = ("sf_native_a", "sf_native_b", "pf_plain_sf21", "adaptive_recent21")
PAIRS = (
    ("sf_native_a", "sf_native_b", "native_repeat_floor"),
    ("sf_native_a", "pf_plain_sf21", "vendored_runtime_parity"),
    ("pf_plain_sf21", "adaptive_recent21", "adaptive_cache_parity"),
    ("sf_native_a", "adaptive_recent21", "end_to_end_parity"),
)
GATED_PAIRS = {"vendored_runtime_parity", "adaptive_cache_parity"}
GATED_EVENTS = {
    "input_noise",
    "noisy_input",
    "attention_output",
    "flow_prediction",
    "denoised_prediction",
    "scheduler_noise",
    "scheduler_output",
    "committed_latent",
    "clean_refresh_input",
    "clean_flow_prediction",
    "final_latents",
}
EXPECTED_COUNTS = {
    "input_noise": 1,
    "noisy_input": 12,
    "attention_output": 450,
    "cache_readout": 450,
    "flow_prediction": 12,
    "denoised_prediction": 12,
    "scheduler_noise": 9,
    "scheduler_output": 9,
    "committed_latent": 3,
    "clean_refresh_input": 3,
    "clean_flow_prediction": 3,
    "final_latents": 1,
    "decoded_video": 1,
}
EVENT_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "input_noise",
            "noisy_input",
            "cache_readout",
            "attention_output",
            "flow_prediction",
            "denoised_prediction",
            "scheduler_noise",
            "scheduler_output",
            "committed_latent",
            "clean_refresh_input",
            "clean_flow_prediction",
            "final_latents",
            "decoded_video",
        )
    )
}


def _identity(row: dict) -> tuple:
    context = row.get("context") or {}
    metadata = row.get("metadata") or {}
    return (
        str(row["event"]),
        context.get("video_index"),
        context.get("ar_block"),
        context.get("current_start_frame"),
        context.get("call_kind"),
        context.get("call_index"),
        context.get("call_count"),
        context.get("timestep"),
        metadata.get("layer"),
        metadata.get("subcall"),
    )


def _sort_identity(identity: tuple) -> tuple:
    event, _, block, start, kind, call, _, timestep, layer, subcall = identity
    return (
        EVENT_ORDER.get(event, 999),
        -1 if block is None else int(block),
        -1 if start is None else int(start),
        {"noisy": 0, "block_commit": 1, "clean": 2, "final": 3}.get(kind, 9),
        -1 if call is None else int(call),
        -1.0 if timestep is None else float(timestep),
        -1 if layer is None else int(layer),
        "" if subcall is None else str(subcall),
    )


def _load_run(directory: Path) -> tuple[dict[tuple, dict], Counter]:
    manifest_path = directory / "events.jsonl"
    meta_path = directory / "trace_meta.json"
    if not manifest_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"incomplete trace directory: {directory}")
    rows: dict[tuple, dict] = {}
    counts: Counter = Counter()
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        identity = _identity(row)
        if identity in rows:
            raise ValueError(f"duplicate parity event identity: {directory} {identity}")
        payload = torch.load(
            directory / row["file"], map_location="cpu", weights_only=False
        )
        rows[identity] = payload
        counts[str(row["event"])] += 1
    return rows, counts


def _comparison_values(left: dict, right: dict) -> tuple[torch.Tensor, torch.Tensor, str]:
    if "full" in left and "full" in right:
        left_value, right_value = left["full"], right["full"]
        source = "full"
    else:
        left_value, right_value = left["sample"], right["sample"]
        source = "sample"
    if tuple(left_value.shape) != tuple(right_value.shape):
        raise ValueError(
            f"comparison shape mismatch: {tuple(left_value.shape)} vs {tuple(right_value.shape)}"
        )
    return left_value.float().reshape(-1), right_value.float().reshape(-1), source


def _metrics(left: dict, right: dict) -> dict[str, float | bool | str]:
    left_value, right_value, source = _comparison_values(left, right)
    if left_value.numel() == 0:
        return {
            "source": source,
            "exact": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "relative_l2": 0.0,
            "cosine": 1.0,
        }
    delta = left_value - right_value
    left_norm = float(torch.linalg.vector_norm(left_value).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            left_value.unsqueeze(0), right_value.unsqueeze(0), dim=1, eps=1e-12
        ).item()
    )
    return {
        "source": source,
        "exact": bool(torch.equal(left_value, right_value)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": delta_norm / max(left_norm, 1e-12),
        "cosine": cosine,
    }


def _row_key(identity: tuple, tensor_name: str) -> tuple:
    return identity + (tensor_name,)


def _within_floor(
    metrics: dict,
    floor: dict | None,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    floor_multiplier: float,
) -> tuple[bool, float, float]:
    relative_limit = max(
        relative_tolerance,
        0.0 if floor is None else float(floor["relative_l2"]) * floor_multiplier,
    )
    absolute_limit = max(
        absolute_tolerance,
        0.0 if floor is None else float(floor["max_abs"]) * floor_multiplier,
    )
    passed = bool(metrics["exact"]) or (
        float(metrics["relative_l2"]) <= relative_limit
        and float(metrics["max_abs"]) <= absolute_limit
    )
    return passed, relative_limit, absolute_limit


def _topology_summary(run_rows: dict[tuple, dict]) -> dict:
    issues: list[dict] = []
    event_count = 0
    max_ffe = 0
    for identity, payload in sorted(run_rows.items(), key=lambda item: _sort_identity(item[0])):
        if identity[0] != "cache_readout":
            continue
        event_count += 1
        metadata = payload.get("metadata") or {}
        rows = metadata.get("frame_ids_per_sequence")
        if not isinstance(rows, list):
            issues.append({"identity": list(identity), "reason": "missing_frame_ids"})
            continue
        for sequence, frame_ids in enumerate(rows):
            if not isinstance(frame_ids, list):
                issues.append(
                    {"identity": list(identity), "sequence": sequence, "reason": "invalid_frame_ids"}
                )
                continue
            max_ffe = max(max_ffe, len(frame_ids))
            if len(frame_ids) != len(set(frame_ids)):
                issues.append(
                    {"identity": list(identity), "sequence": sequence, "reason": "duplicate_frames", "frames": frame_ids}
                )
            if frame_ids != sorted(frame_ids):
                issues.append(
                    {"identity": list(identity), "sequence": sequence, "reason": "non_monotonic_frames", "frames": frame_ids}
                )
    return {
        "cache_readout_events": event_count,
        "max_unique_frame_equivalents": max_ffe,
        "issue_count": len(issues),
        "first_issues": issues[:20],
    }


def _cache_membership_comparison(
    left_rows: dict[tuple, dict], right_rows: dict[tuple, dict]
) -> dict:
    def grouped(rows: dict[tuple, dict]) -> dict[tuple, list[list[int]]]:
        result: dict[tuple, list[list[int]]] = defaultdict(list)
        for identity, payload in rows.items():
            if identity[0] != "cache_readout":
                continue
            # Drop the implementation-specific subcall field. A merged Adaptive
            # readout then aligns with the matching native block/call/layer.
            key = identity[:-1]
            frame_rows = (payload.get("metadata") or {}).get(
                "frame_ids_per_sequence"
            )
            if not isinstance(frame_rows, list):
                continue
            for frame_ids in frame_rows:
                if isinstance(frame_ids, list):
                    result[key].append([int(item) for item in frame_ids])
        return result

    left, right = grouped(left_rows), grouped(right_rows)
    shared = sorted(
        set(left) & set(right), key=lambda identity: _sort_identity(identity + (None,))
    )
    mismatches: list[dict] = []
    for identity in shared:
        left_patterns = sorted(set(tuple(row) for row in left[identity]))
        right_patterns = sorted(set(tuple(row) for row in right[identity]))
        if left_patterns != right_patterns:
            mismatches.append(
                {
                    "identity": list(identity),
                    "left_patterns": [list(row) for row in left_patterns],
                    "right_patterns": [list(row) for row in right_patterns],
                }
            )
    missing_left = sorted(
        set(right) - set(left), key=lambda identity: _sort_identity(identity + (None,))
    )
    missing_right = sorted(
        set(left) - set(right), key=lambda identity: _sort_identity(identity + (None,))
    )
    return {
        "shared_contexts": len(shared),
        "mismatch_count": len(mismatches),
        "missing_left_count": len(missing_left),
        "missing_right_count": len(missing_right),
        "pass": bool(shared) and not mismatches and not missing_left and not missing_right,
        "first_mismatches": mismatches[:20],
    }


def analyze(
    *,
    trace_root: Path,
    output_dir: Path,
    relative_tolerance: float,
    absolute_tolerance: float,
    floor_multiplier: float,
) -> dict:
    loaded = {run: _load_run(trace_root / run) for run in RUNS}
    coverage = {
        run: {
            "observed": dict(counts),
            "expected": dict(EXPECTED_COUNTS),
            "pass": all(counts[event] == count for event, count in EXPECTED_COUNTS.items()),
        }
        for run, (_, counts) in loaded.items()
    }
    repeat_rows: dict[tuple, dict] = {}
    details: list[dict] = []
    summaries: dict[str, dict] = {}
    for left_name, right_name, label in PAIRS:
        left_rows, right_rows = loaded[left_name][0], loaded[right_name][0]
        shared = sorted(set(left_rows) & set(right_rows), key=_sort_identity)
        missing_left = sorted(
            (
                identity
                for identity in set(right_rows) - set(left_rows)
                if identity[0] in GATED_EVENTS
            ),
            key=_sort_identity,
        )
        missing_right = sorted(
            (
                identity
                for identity in set(left_rows) - set(right_rows)
                if identity[0] in GATED_EVENTS
            ),
            key=_sort_identity,
        )
        pair_rows: list[dict] = []
        for identity in shared:
            if identity[0] == "cache_readout":
                continue
            left_tensors = left_rows[identity]["tensors"]
            right_tensors = right_rows[identity]["tensors"]
            for tensor_name in sorted(set(left_tensors) & set(right_tensors)):
                try:
                    metric = _metrics(left_tensors[tensor_name], right_tensors[tensor_name])
                    shape_match = True
                except ValueError:
                    metric = {
                        "source": "shape_mismatch",
                        "exact": False,
                        "max_abs": 1e30,
                        "mean_abs": 1e30,
                        "relative_l2": 1e30,
                        "cosine": -1.0,
                    }
                    shape_match = False
                floor = repeat_rows.get(_row_key(identity, tensor_name))
                passed, rel_limit, abs_limit = _within_floor(
                    metric,
                    floor,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    floor_multiplier=floor_multiplier,
                )
                row = {
                    "pair": label,
                    "left": left_name,
                    "right": right_name,
                    "event": identity[0],
                    "ar_block": identity[2],
                    "current_start_frame": identity[3],
                    "call_kind": identity[4],
                    "call_index": identity[5],
                    "timestep": identity[7],
                    "layer": identity[8],
                    "tensor": tensor_name,
                    "shape_match": shape_match,
                    **metric,
                    "relative_limit": rel_limit,
                    "absolute_limit": abs_limit,
                    "pass": passed,
                }
                pair_rows.append(row)
                details.append(row)
                if label == "native_repeat_floor":
                    repeat_rows[_row_key(identity, tensor_name)] = metric
        gated_rows = [row for row in pair_rows if row["event"] in GATED_EVENTS]
        first_divergence = next((row for row in gated_rows if not row["pass"]), None)
        pair_pass = (
            bool(gated_rows)
            and not missing_left
            and not missing_right
            and all(row["pass"] for row in gated_rows)
            and coverage[left_name]["pass"]
            and coverage[right_name]["pass"]
        )
        summaries[label] = {
            "left": left_name,
            "right": right_name,
            "shared_event_count": len(shared),
            "gated_tensor_count": len(gated_rows),
            "failed_gated_tensor_count": sum(not row["pass"] for row in gated_rows),
            "missing_left_count": len(missing_left),
            "missing_right_count": len(missing_right),
            "pass": pair_pass,
            "first_divergence": first_divergence,
            "max_relative_l2": max(
                (float(row["relative_l2"]) for row in gated_rows), default=1e30
            ),
            "max_abs": max(
                (float(row["max_abs"]) for row in gated_rows), default=1e30
            ),
        }

    topology = {
        run: _topology_summary(rows) for run, (rows, _) in loaded.items()
    }
    cache_membership = {
        label: _cache_membership_comparison(
            loaded[left_name][0], loaded[right_name][0]
        )
        for left_name, right_name, label in PAIRS
    }
    require_cache_membership = int(EXPECTED_COUNTS.get("cache_readout", 0)) > 0
    for label in GATED_PAIRS:
        membership_pass = (
            cache_membership[label]["pass"] if require_cache_membership else True
        )
        summaries[label]["cache_membership_pass"] = membership_pass
        summaries[label]["pass"] = bool(
            summaries[label]["pass"] and membership_pass
        )
    parity_pass = all(
        summaries[label]["pass"] for label in sorted(GATED_PAIRS)
    )
    if not summaries["vendored_runtime_parity"]["pass"]:
        decision = "stop_fix_vendored_runtime_before_cache_attribution"
    elif not summaries["adaptive_cache_parity"]["pass"]:
        decision = "stop_fix_adaptive_recent21_before_large_scale"
    else:
        decision = "parity_pass_proceed_to_budget_phase_screen"
    report = {
        "version": 1,
        "experiment": "v207_sf_runtime_parity",
        "parity_pass": parity_pass,
        "decision": decision,
        "thresholds": {
            "relative_l2_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "native_repeat_floor_multiplier": floor_multiplier,
        },
        "coverage": coverage,
        "comparisons": summaries,
        "cache_topology": topology,
        "cache_membership_comparisons": cache_membership,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(details[0]) if details else []
    with (output_dir / "parity_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details)
    lines = [
        "# v207 SF Runtime Parity",
        "",
        f"- Decision: `{decision}`",
        f"- Overall parity: `{str(parity_pass).lower()}`",
        "",
        "| Comparison | Pass | Failed tensors | Max relative L2 | First divergence |",
        "|---|---:|---:|---:|---|",
    ]
    for _, _, label in PAIRS:
        summary = summaries[label]
        first = summary["first_divergence"]
        first_text = "none" if first is None else (
            f"{first['event']} block={first['ar_block']} call={first['call_index']} "
            f"layer={first['layer']} tensor={first['tensor']}"
        )
        lines.append(
            f"| `{label}` | {summary['pass']} | "
            f"{summary['failed_gated_tensor_count']} | "
            f"{summary['max_relative_l2']:.6g} | {first_text} |"
        )
    lines.extend(
        [
            "",
            "The report gates on numerical trajectory events, not final VBench similarity. "
            "`cache_topology` records frame ordering diagnostics for causal explanation.",
        ]
    )
    (output_dir / "parity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--absolute-tolerance", type=float, default=5e-4)
    parser.add_argument("--native-floor-multiplier", type=float, default=5.0)
    args = parser.parse_args()
    report = analyze(
        trace_root=args.trace_root,
        output_dir=args.output_dir,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        floor_multiplier=args.native_floor_multiplier,
    )
    print(
        "[v207-parity] "
        f"decision={report['decision']} pass={int(report['parity_pass'])} "
        f"report={args.output_dir / 'parity_report.json'}"
    )


if __name__ == "__main__":
    main()
