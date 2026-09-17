#!/usr/bin/env python3
"""Audit a v210 LPHC JSONL trace against protocol invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

ARCHIVE_LIMIT = 12
SELECTION_LIMIT = 4
_PHASES = {"e1": {0}, "e2": {0, 1}, "full": {0, 1, 2, 3}}
_MISSING = object()


def _first(row: dict, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    for value in row.values():
        if isinstance(value, dict):
            found = _first(value, names, _MISSING)
            if found is not _MISSING:
                return found
    return default


def _count(row: dict, names: Iterable[str]) -> int:
    value = _first(row, names, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _indices(row: dict, names: Iterable[str]) -> set[int]:
    value = _first(row, names, ())
    if value is None:
        return set()
    if isinstance(value, dict):
        value = value.keys()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {int(value)}
    if not isinstance(value, (list, tuple, set, dict)):
        return set()
    result = set()
    for item in value:
        if isinstance(item, dict):
            item = _first(item, ("frame", "frame_index", "index", "id"), None)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result.add(int(item))
    return result


def _event_count(row: dict, tokens: tuple[str, ...]) -> int:
    event = str(row.get("event", row.get("type", ""))).lower()
    return int(bool(event) and all(token in event for token in tokens))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            row["_audit_line"] = line_number
            rows.append(row)
    return rows


def audit_trace(
    path: Path,
    alpha: float,
    *,
    expect_second_attention: bool | None = None,
    phase: str = "full",
    tolerance: float = 1e-6,
) -> dict:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if phase not in _PHASES:
        raise ValueError("phase must be e1, e2, or full")
    expected_second = alpha > 0 if expect_second_attention is None else expect_second_attention
    rows = load_trace(path)
    errors: list[str] = []
    totals = {"lookup": 0, "random": 0, "second_attention": 0}
    maximums = {"archive_size": 0, "selected": 0, "correction_ratio": 0.0}
    attention_rows = 0
    active_lookup_rows = 0
    active_selected_rows = 0
    for row in rows:
        line = row["_audit_line"]
        # A clean-commit event may carry the just-completed noisy read snapshot
        # separately from post-commit diagnostics. Prefer that snapshot.
        sample = dict(row)
        if isinstance(row.get("read_diagnostics"), dict):
            sample.update(row["read_diagnostics"])
        if row.get("event") == "attention_call":
            attention_rows += 1
            required_groups = {
                "call kind": ("call_kind",),
                "local frames": ("local_frame_indices", "local_frame_ids", "local_frames"),
                "archive size": ("archive_frames", "archive_size", "archive_count"),
                "selected frames": ("selected_history_frames", "selected_frame_ids", "selected_frames"),
                "clean reads": ("clean_history_reads", "clean_history_read_count", "clean_read_count"),
                "correction ratio": ("correction_ratio", "max_correction_ratio", "residual_ratio"),
                "lookup count": ("lookup_count", "retrieval_calls", "provider_calls"),
                "random count": ("random_count", "random_selections", "random_draw_count"),
                "second attention count": ("second_attention_count", "second_attention_calls"),
            }
            missing = [
                label for label, names in required_groups.items()
                if _first(row, names, _MISSING) is _MISSING
            ]
            if missing:
                errors.append(f"line {line}: attention trace fields missing: {missing}")
        clean_reads = _count(sample, (
            "clean_history_read_count", "clean_history_reads", "clean_retrieval_calls",
            "clean_read_count", "read_clean_history"
        ))
        history_kind = str(_first(sample, ("history_kind", "history_source", "read_history_kind"), "")).lower()
        if clean_reads or history_kind == "clean":
            errors.append(f"line {line}: clean history read is forbidden")

        local = _indices(sample, (
            "local_frame_ids", "local_frame_indices", "local_frames", "local_indices", "window_frame_indices"
        ))
        selected = _indices(sample, (
            "selected_frame_ids", "last_selected_frame_ids", "selected_archive_indices",
            "selected_archive_frames", "selected_history_indices", "selected_history_frames",
            "selected_indices", "selected_frames"
        ))
        eligible = _indices(sample, (
            "eligible_frame_indices", "eligible_frame_ids", "last_eligible_frame_ids"
        ))
        overlap = local & selected
        if overlap:
            errors.append(f"line {line}: local/selected overlap {sorted(overlap)}")

        archive_size = _count(sample, ("archive_size", "archive_frames", "archive_count"))
        if not archive_size:
            archive = _first(sample, ("archive_indices", "archive_frame_indices"), None)
            if isinstance(archive, (list, tuple, set, dict)):
                archive_size = len(archive)
        selected_count = _count(sample, ("selected_count", "selection_count", "history_selected_count"))
        if not selected_count:
            selected_count = len(selected)
        maximums["archive_size"] = max(maximums["archive_size"], archive_size)
        maximums["selected"] = max(maximums["selected"], selected_count)
        if archive_size > ARCHIVE_LIMIT:
            errors.append(f"line {line}: archive size {archive_size} exceeds {ARCHIVE_LIMIT}")
        if selected_count > SELECTION_LIMIT:
            errors.append(f"line {line}: selected {selected_count} exceeds {SELECTION_LIMIT}")

        ratios = []
        for name in ("correction_ratio", "max_correction_ratio", "applied_correction_ratio", "residual_ratio"):
            value = _first(sample, (name,), None)
            if value is not None:
                ratios.append((name, value))
        for name, ratio_value in ratios:
            try:
                ratio = float(ratio_value)
            except (TypeError, ValueError):
                errors.append(f"line {line}: {name.replace('_', ' ')} is not numeric")
                continue
            if not math.isfinite(ratio):
                errors.append(f"line {line}: {name.replace('_', ' ')} is not finite")
            elif ratio < -tolerance or ratio > alpha + tolerance:
                errors.append(f"line {line}: {name} {ratio} exceeds alpha {alpha}")
            else:
                maximums["correction_ratio"] = max(maximums["correction_ratio"], abs(ratio))

        lookup = _count(sample, ("lookup_count", "retrieval_lookup_count", "retrieval_calls", "provider_calls", "lookups"))
        random = _count(sample, ("random_count", "random_draw_count", "random_selections", "random_draws", "used_random"))
        second = _count(sample, ("second_attention_count", "second_attention_calls", "used_second_attention"))
        lookup = max(lookup, _event_count(row, ("lookup",)))
        random = max(random, _event_count(row, ("random",)))
        second = max(second, _event_count(row, ("second", "attention")))
        if row.get("event") == "attention_call":
            totals["lookup"] += lookup
            totals["random"] += random
            totals["second_attention"] += second
            phase_index = int(row.get("phase_index", -1))
            active = (
                expected_second
                and str(row.get("call_kind")) == "noisy"
                and phase_index in _PHASES[phase]
            )
            if active and lookup:
                active_lookup_rows += 1
            if active and eligible and selected_count and selected:
                active_selected_rows += 1
            if not active and (lookup or random or second or selected_count or selected):
                errors.append(f"line {line}: unexpected second/intervention activity outside expected phase")
            if second and not expected_second:
                errors.append(f"line {line}: unexpected second attention")
            if second > 1:
                errors.append(f"line {line}: more than one second attention in a call")

    if not rows:
        errors.append("trace contains no events")
    elif not attention_rows:
        errors.append("trace contains no attention_call events")
    if expected_second:
        if totals["lookup"] == 0 or active_lookup_rows == 0:
            errors.append("expected LPHC history lookup was never observed")
        if active_selected_rows == 0:
            errors.append("expected nonempty eligible and selected LPHC history was never observed")
        if totals["second_attention"] == 0:
            errors.append("expected LPHC second attention was never observed")
    if alpha == 0.0 and any(totals.values()):
        errors.append(f"alpha0 must have zero lookup/random/second counts, got {totals}")
    return {
        "version": 1,
        "trace_path": str(path.resolve()),
        "trace_sha256": _sha256(path),
        "alpha": alpha,
        "tolerance": tolerance,
        "expect_second_attention": expected_second,
        "phase": phase,
        "row_count": len(rows),
        "totals": totals,
        "maximums": maximums,
        "errors": errors,
        "pass": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--phase", choices=tuple(_PHASES), default="full")
    parser.add_argument("--expect-second-attention", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = None if args.expect_second_attention == "auto" else args.expect_second_attention == "yes"
    report = audit_trace(
        args.trace, args.alpha, expect_second_attention=expected,
        phase=args.phase, tolerance=args.tolerance,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
