#!/usr/bin/env python3
"""Diagnose HREM-v2 archive, admission, routing, and fusion from JSONL traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(record)
    return records


def _mean(records: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def _median(records: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.median(values) if values else None


def _finding(severity: str, code: str, message: str, action: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message, "action": action}


def _layer(record: dict[str, Any]) -> int:
    return int(record.get("layer", record.get("layer_idx", -1)))


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = Counter(str(record.get("event", "missing")) for record in records)
    configs = [record for record in records if record.get("event") == "config"]
    commits = [record for record in records if record.get("event") == "commit"]
    boundaries = [record for record in records if record.get("event") == "boundary"]
    readouts = [record for record in records if record.get("event") == "readout"]
    intra_readouts = [
        record for record in readouts
        if record.get("recall_scope") == "intra_episode"
    ]
    abstains = [record for record in records if record.get("event") == "readout_abstain"]
    abstain_reasons = Counter()
    findings: list[dict[str, str]] = []
    violations: list[str] = []

    for record in abstains:
        decision = record.get("episode_decision") or {}
        reason = record.get("reason") or decision.get("abstain_reason") or "unknown"
        abstain_reasons[str(reason)] += 1

    for record in readouts:
        current = record.get("current_episode_id")
        previous = record.get("previous_episode_id")
        allowed = record.get("allowed_episode_id")
        recall_scope = str(record.get("recall_scope", "cross_episode"))
        context = (
            f"trajectory={record.get('trajectory_id')} layer={_layer(record)} "
            f"block={record.get('block_index')}"
        )
        if recall_scope == "intra_episode":
            if not bool(record.get("allow_current_episode", False)):
                violations.append(f"{context}: intra-episode readout did not opt in to current episode")
            if allowed is None or current is None or int(allowed) != int(current):
                violations.append(
                    f"{context}: intra-episode readout allowed {allowed}, expected current {current}"
                )
            selected_episodes = record.get("selected_episode_ids") or []
            if current is not None and any(
                int(episode) != int(current) for episode in selected_episodes
            ):
                violations.append(
                    f"{context}: intra-episode readout selected a foreign episode"
                )
            recent_exclude = int(record.get("recent_exclude_frames", 0))
            selected_ages = record.get("selected_frame_ages") or []
            if any(int(age) <= recent_exclude for age in selected_ages):
                violations.append(
                    f"{context}: selected frame age violates recent exclusion {recent_exclude}"
                )
            start_frame = int(record.get("memory_start_frame", 0))
            current_frame = record.get("current_frame")
            if current_frame is not None and int(current_frame) < start_frame:
                violations.append(
                    f"{context}: intra-episode readout started at frame {current_frame} before {start_frame}"
                )
        else:
            if allowed is not None and current is not None and int(allowed) >= int(current):
                violations.append(
                    f"{context}: allowed episode {allowed} is not historical relative to {current}"
                )
            if allowed is not None and previous is not None and int(allowed) == int(previous):
                violations.append(f"{context}: immediately previous episode {previous} was admitted")
        decision = record.get("episode_decision") or {}
        winner = decision.get("winner_episode_id")
        if winner is not None and allowed is not None and int(winner) != int(allowed):
            violations.append(
                f"{context}: episode winner {winner} differs from allowed episode {allowed}"
            )

    if not records:
        findings.append(_finding(
            "ERROR", "empty_trace", "The trace contains no records.",
            "Check STRUCTURED_MEMORY_TRACE_ENABLED and STRUCTURED_MEMORY_TRACE_PATH.",
        ))
    elif not commits:
        findings.append(_finding(
            "ERROR", "no_archive_commits",
            "No clean K/V blocks were committed to the episodic archive.",
            "Inspect the clean-pass bridge and active layer range before tuning retrieval.",
        ))

    if commits:
        last_by_layer: dict[int, dict[str, Any]] = {}
        for record in commits:
            last_by_layer[_layer(record)] = record
        empty_kv = [
            layer for layer, record in last_by_layer.items()
            if float(record.get("archive_k_rms", 1.0)) <= 1e-8
            or float(record.get("archive_v_rms", 1.0)) <= 1e-8
        ]
        if empty_kv:
            findings.append(_finding(
                "ERROR", "zero_archive_signal",
                f"Archive K/V RMS is zero in layers {empty_kv}.",
                "Verify pre-RoPE K/V capture and clean-pass commit tensors.",
            ))

    if boundaries and any(not bool(record.get("archive_preserved", False)) for record in boundaries):
        findings.append(_finding(
            "ERROR", "archive_lost_at_boundary",
            "At least one scene boundary did not preserve the episodic archive.",
            "Keep native working-cache reset separate from EpisodicArchive.reset().",
        ))
    boundary_snapshots = [
        record.get("archive_state")
        for record in boundaries
        if record.get("archive_state") is not None
    ]
    if boundaries and not boundary_snapshots:
        findings.append(_finding(
            "WARNING", "boundary_archive_snapshot_missing",
            "Boundary events do not include the archive checksum snapshot.",
            "Rerun with the current bridge before diagnosing archive corruption.",
        ))
    elif any(
        not isinstance(snapshot, dict) or not (snapshot.get("archive_layers") or [])
        for snapshot in boundary_snapshots
    ):
        findings.append(_finding(
            "ERROR", "boundary_archive_snapshot_empty",
            "At least one boundary snapshot contains no active archive layer.",
            "Check the structured-memory layer mask and archive summary bridge.",
        ))

    if commits and not readouts:
        dominant = abstain_reasons.most_common(1)
        suffix = f" Dominant abstention: {dominant[0][0]}." if dominant else ""
        expected_role_rejection = (
            bool(abstain_reasons)
            and set(abstain_reasons) == {"role_evidence_spread_below_min"}
        )
        if expected_role_rejection:
            findings.append(_finding(
                "WARNING", "role_calibration_rejected_all",
                "All role-gated calls failed closed because head evidence lacked spread.",
                "Treat this cell as a negative mechanism result and retain its native output for evaluation.",
            ))
        else:
            findings.append(_finding(
                "ERROR", "no_accepted_readout",
                "The archive was populated but no memory readout reached fusion." + suffix,
                "Inspect episode thresholds first, then frame confidence and margin thresholds.",
            ))

    incomplete_intra_sidecars = [
        record for record in intra_readouts
        if not bool(record.get("selected_indices_valid", False))
        or not bool(record.get("interval_sidecar_valid", False))
        or not bool(record.get("episode_sidecar_valid", False))
        or not (record.get("selected_frame_ages") or [])
    ]
    if incomplete_intra_sidecars:
        findings.append(_finding(
            "ERROR", "intra_episode_sidecar_missing",
            f"{len(incomplete_intra_sidecars)} intra-episode readout(s) lack valid age/episode sidecars.",
            "Do not evaluate the cell until selected intervals, ages, and episode IDs are traceable.",
        ))

    delta_median = _median(readouts, "delta_to_native_rms")
    weight_mean = _mean(readouts, "effective_weight_mean")
    head_gate_mean = _mean(readouts, "head_gate_mean")
    alignment_positive = _mean(readouts, "alignment_positive_fraction")
    confidence_mean = _mean(readouts, "confidence_mean")
    intra_selected_ages = [
        float(age)
        for record in intra_readouts
        for age in (record.get("selected_frame_ages") or [])
    ]
    episode_warmup_scale = _mean(readouts, "episode_warmup_scale")
    episode_warmup_values = [
        float(record["episode_warmup_scale"])
        for record in readouts
        if record.get("episode_warmup_scale") is not None
    ]
    first_episode_block_readouts = [
        record for record in readouts if record.get("episode_block_index") == 0
    ]
    first_episode_block_gate = _mean(first_episode_block_readouts, "effective_gate")
    configured_episode_warmup = 0
    if configs:
        configured_episode_warmup = int(
            ((configs[-1].get("readout") or {}).get("episode_warmup_blocks", 0))
        )
    accepted_fraction_values = [
        float(record["accepted_head_count"]) / max(1, int(record["head_count"]))
        for record in readouts
        if record.get("accepted_head_count") is not None and record.get("head_count") is not None
    ]
    accepted_head_fraction = (
        statistics.fmean(accepted_fraction_values) if accepted_fraction_values else None
    )
    role_readouts = [
        record
        for record in readouts
        if record.get("head_routing") == "role_evidence"
        or record.get("head_role") is not None
    ]
    role_diagnostic_records = role_readouts + [
        record
        for record in abstains
        if record.get("head_routing") == "role_evidence"
        and record.get("role_calibration_valid") is not None
    ]
    role_gate_std = _mean(role_readouts, "head_gate_std")
    role_active_head_fraction = _mean(role_readouts, "head_gate_active_fraction")
    role_evidence_spread = _median(role_diagnostic_records, "role_evidence_spread")
    calibration_valid_values = [
        float(bool(record["role_calibration_valid"]))
        for record in role_diagnostic_records
        if record.get("role_calibration_valid") is not None
    ]
    role_calibration_valid_fraction = (
        statistics.fmean(calibration_valid_values)
        if calibration_valid_values else None
    )

    role_groups: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in role_readouts:
        if record.get("trajectory_id") is None or record.get("attention_call_index") is None:
            continue
        role_groups[(
            int(record["trajectory_id"]),
            _layer(record),
            int(record.get("current_start", -1)),
            str(record.get("memory_mode", "unknown")),
        )].append(record)
    temporal_gate_ranges: list[float] = []
    temporal_active_jaccards: list[float] = []
    for group in role_groups.values():
        ordered = sorted(group, key=lambda item: int(item.get("attention_call_index", 0)))
        gate_vectors = [
            (record.get("head_role") or {}).get("gate")
            for record in ordered
        ]
        gate_vectors = [vector for vector in gate_vectors if vector]
        if len(gate_vectors) < 2:
            continue
        gate_means = [statistics.fmean(float(value) for value in vector) for vector in gate_vectors]
        temporal_gate_ranges.append(max(gate_means) - min(gate_means))
        for previous, current in zip(gate_vectors, gate_vectors[1:]):
            previous_active = {index for index, value in enumerate(previous) if float(value) >= 0.5}
            current_active = {index for index, value in enumerate(current) if float(value) >= 0.5}
            union = previous_active | current_active
            temporal_active_jaccards.append(
                len(previous_active & current_active) / len(union) if union else 1.0
            )
    role_gate_temporal_range = (
        statistics.fmean(temporal_gate_ranges) if temporal_gate_ranges else None
    )
    role_active_jaccard = (
        statistics.fmean(temporal_active_jaccards)
        if temporal_active_jaccards else None
    )

    if delta_median is not None and delta_median < 1e-4:
        findings.append(_finding(
            "WARNING", "fusion_effect_negligible",
            f"Median delta/native RMS is only {delta_median:.6f}.",
            "Check alignment, confidence, and role gates before increasing the global gate.",
        ))
    if delta_median is not None and delta_median > 0.25:
        findings.append(_finding(
            "WARNING", "fusion_effect_too_large",
            f"Median delta/native RMS is {delta_median:.4f}, above the conservative range.",
            "Reduce STRUCTURED_MEMORY_GATE or tighten role/admission thresholds.",
        ))
    if role_readouts and head_gate_mean is not None and head_gate_mean < 0.05:
        findings.append(_finding(
            "WARNING", "role_gate_over_suppressed",
            f"Mean role gate is {head_gate_mean:.4f}.",
            "Lower ROLE_THRESHOLD or inspect K/V persistence and query stability per head.",
        ))
    if role_readouts and (
        (role_active_head_fraction is not None and role_active_head_fraction > 0.90)
        or (head_gate_mean is not None and head_gate_mean > 0.90)
    ):
        gate_mean_text = "missing" if head_gate_mean is None else f"{head_gate_mean:.4f}"
        active_fraction_text = (
            "missing"
            if role_active_head_fraction is None
            else f"{role_active_head_fraction:.3f}"
        )
        findings.append(_finding(
            "WARNING", "role_gate_not_selective",
            f"Mean role gate is {gate_mean_text}; active-head fraction is "
            f"{active_fraction_text}.",
            "Use absolute-threshold and relative/hybrid calibration ablations before claiming head awareness.",
        ))
    if role_gate_std is not None and role_gate_std < 0.02:
        findings.append(_finding(
            "WARNING", "role_gate_low_contrast",
            f"Mean within-call role-gate standard deviation is only {role_gate_std:.5f}.",
            "Inspect raw evidence spread; a lower gate mean alone does not establish head selectivity.",
        ))
    if role_evidence_spread is not None and role_evidence_spread < 0.01:
        findings.append(_finding(
            "WARNING", "role_evidence_not_discriminative",
            f"Median max-min role evidence spread is only {role_evidence_spread:.6f}.",
            "Treat relative ranking as unreliable or enable fail-closed minimum-spread gating.",
        ))
    if (
        role_calibration_valid_fraction is not None
        and role_calibration_valid_fraction < 0.90
    ):
        findings.append(_finding(
            "WARNING", "role_calibration_often_invalid",
            f"Only {role_calibration_valid_fraction:.1%} of role calls pass the spread check.",
            "Review per-layer evidence before lowering the minimum spread.",
        ))
    if role_gate_temporal_range is not None and role_gate_temporal_range > 0.25:
        findings.append(_finding(
            "WARNING", "role_gate_denoise_instability",
            f"Mean within-block gate-mean range is {role_gate_temporal_range:.4f}.",
            "Inspect attention_call_index traces and consider timestep-stable calibration.",
        ))
    if role_active_jaccard is not None and role_active_jaccard < 0.50:
        findings.append(_finding(
            "WARNING", "role_identity_denoise_instability",
            f"Mean active-head Jaccard across denoising calls is {role_active_jaccard:.3f}.",
            "Do not assign semantic head roles until active identities stabilize across calls.",
        ))
    if alignment_positive is not None and alignment_positive < 0.10:
        findings.append(_finding(
            "WARNING", "memory_native_conflict",
            f"Only {alignment_positive:.1%} of token-head outputs have positive alignment.",
            "Inspect positional convention and archive quality; do not increase the gate.",
        ))
    if confidence_mean is not None and confidence_mean < 0.10:
        findings.append(_finding(
            "WARNING", "weak_frame_retrieval",
            f"Mean retrieval confidence is {confidence_mean:.4f}.",
            "Inspect visual score distributions and retrieval temperature.",
        ))
    if configured_episode_warmup > 0 and readouts and not episode_warmup_values:
        findings.append(_finding(
            "ERROR", "episode_warmup_trace_missing",
            "Episode warmup is configured but accepted readouts do not record its scale.",
            "Verify the causal-model bridge and rerun before comparing transition cells.",
        ))
    elif (
        configured_episode_warmup > 0
        and episode_warmup_values
        and min(episode_warmup_values) >= 1.0
    ):
        findings.append(_finding(
            "WARNING", "episode_warmup_not_observed",
            "Episode warmup was configured, but every accepted readout used full scale.",
            "Check episode_start_frame and early-block abstentions before judging the ramp.",
        ))
    if violations:
        findings.append(_finding(
            "ERROR", "causal_invariant_violation",
            f"Detected {len(violations)} historical-selection invariant violation(s).",
            "Stop evaluation and fix episode filtering before using generated results.",
        ))
    if records and not findings:
        findings.append(_finding(
            "INFO", "diagnostics_nominal",
            "No structural failure was detected in the trace.",
            "Proceed to video metrics and causal ablations; trace health alone does not prove quality.",
        ))

    per_layer: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in readouts:
        grouped[_layer(record)].append(record)
    for layer, layer_records in sorted(grouped.items()):
        per_layer[str(layer)] = {
            "readouts": len(layer_records),
            "delta_to_native_rms_median": _median(layer_records, "delta_to_native_rms"),
            "effective_weight_mean": _mean(layer_records, "effective_weight_mean"),
            "head_gate_mean": _mean(layer_records, "head_gate_mean"),
            "head_gate_std": _mean(layer_records, "head_gate_std"),
            "head_gate_active_fraction": _mean(
                layer_records, "head_gate_active_fraction"
            ),
            "role_evidence_spread_median": _median(
                layer_records, "role_evidence_spread"
            ),
            "alignment_positive_fraction": _mean(layer_records, "alignment_positive_fraction"),
        }

    return {
        "record_count": len(records),
        "config": configs[-1] if configs else None,
        "event_counts": dict(sorted(events.items())),
        "abstain_reasons": dict(abstain_reasons.most_common()),
        "metrics": {
            "delta_to_native_rms_median": delta_median,
            "effective_weight_mean": weight_mean,
            "head_gate_mean": head_gate_mean,
            "accepted_head_fraction": accepted_head_fraction,
            "retrieval_accepted_head_fraction": accepted_head_fraction,
            "role_gate_std": role_gate_std,
            "role_active_head_fraction": role_active_head_fraction,
            "role_evidence_spread_median": role_evidence_spread,
            "role_calibration_valid_fraction": role_calibration_valid_fraction,
            "role_gate_temporal_range_mean": role_gate_temporal_range,
            "role_active_head_jaccard": role_active_jaccard,
            "alignment_positive_fraction": alignment_positive,
            "confidence_mean": confidence_mean,
            "intra_episode_readouts": len(intra_readouts),
            "intra_selected_frame_age_min": (
                min(intra_selected_ages) if intra_selected_ages else None
            ),
            "intra_selected_frame_age_median": (
                statistics.median(intra_selected_ages)
                if intra_selected_ages else None
            ),
            "intra_selected_frame_age_max": (
                max(intra_selected_ages) if intra_selected_ages else None
            ),
            "episode_warmup_blocks": configured_episode_warmup,
            "episode_warmup_scale_mean": episode_warmup_scale,
            "episode_warmup_scale_min": (
                min(episode_warmup_values) if episode_warmup_values else None
            ),
            "episode_warmup_scale_max": (
                max(episode_warmup_values) if episode_warmup_values else None
            ),
            "episode_first_block_effective_gate_mean": first_episode_block_gate,
        },
        "per_layer": per_layer,
        "violations": violations,
        "findings": findings,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("HREM-v2 diagnostic report")
    print(f"records: {report['record_count']}")
    print(f"events: {json.dumps(report['event_counts'], sort_keys=True)}")
    print(f"abstains: {json.dumps(report['abstain_reasons'], sort_keys=True)}")
    print(f"metrics: {json.dumps(report['metrics'], sort_keys=True)}")
    for finding in report["findings"]:
        print(
            f"[{finding['severity']}] {finding['code']}: {finding['message']} "
            f"Next: {finding['action']}"
        )
    for violation in report["violations"]:
        print(f"[VIOLATION] {violation}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="HREM-v2 JSONL trace")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero when an ERROR finding or invariant violation is present.",
    )
    args = parser.parse_args()
    report = analyze_records(load_records(args.trace))
    _print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    has_error = any(finding["severity"] == "ERROR" for finding in report["findings"])
    return 1 if args.strict and has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
