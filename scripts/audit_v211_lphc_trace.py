#!/usr/bin/env python3
"""Audit a v211 LPHC JSONL trace against protocol invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

ARCHIVE_LIMIT = 12
_ALLOWED_HISTORY_BUDGETS = {1, 4}
_PHASES = {"e1": {0}}
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


def _ordered_indices(row: dict, names: Iterable[str]) -> tuple[int, ...]:
    value = _first(row, names, ())
    if value is None:
        return ()
    if isinstance(value, dict):
        value = value.keys()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, (list, tuple, set, dict)):
        return ()
    result = []
    for item in value:
        if isinstance(item, dict):
            item = _first(item, ("frame", "frame_index", "index", "id"), None)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result.append(int(item))
    return tuple(result)


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
    phase: str = "e1",
    expected_history_budget: int = 4,
    tolerance: float = 1e-6,
    expected_blocks: int | None = None,
    expected_layers: int | None = None,
) -> dict:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if phase not in _PHASES:
        raise ValueError("v211 phase must be e1")
    if expected_history_budget not in _ALLOWED_HISTORY_BUDGETS:
        raise ValueError("expected_history_budget must be 1 or 4")
    if expected_blocks is not None and expected_blocks <= 0:
        raise ValueError("expected_blocks must be positive")
    if expected_layers is not None and expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    expected_second = alpha > 0 if expect_second_attention is None else expect_second_attention
    rows = load_trace(path)
    errors: list[str] = []
    totals = {"lookup": 0, "random": 0, "second_attention": 0}
    maximums = {"archive_size": 0, "selected": 0, "correction_ratio": 0.0}
    attention_rows = 0
    active_lookup_rows = 0
    active_selected_rows = 0
    call_matrix: dict[tuple[int, int], list[tuple[str, int]]] = {}
    headers = [row for row in rows if row.get("event") == "video_start"]
    if len(headers) != 1:
        errors.append(f"expected exactly one video_start header, got {len(headers)}")
    else:
        header = headers[0]
        if header.get("protocol") != "v211":
            errors.append("video_start protocol must be v211")
        if header.get("local_policy") != "sink1_recent20":
            errors.append("video_start local_policy must be sink1_recent20")
        if header.get("history_frames") != expected_history_budget:
            errors.append(
                "video_start history_frames does not match expected history budget"
            )
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
                "layer": ("layer_idx", "layer_id"),
                "block": ("block_id",),
                "phase": ("phase_index",),
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
            else:
                layer = int(_first(row, ("layer_idx", "layer_id")))
                block = int(row["block_id"])
                call = (str(row["call_kind"]), int(row["phase_index"]))
                call_matrix.setdefault((layer, block), []).append(call)
        clean_reads = _count(sample, (
            "clean_history_read_count", "clean_history_reads", "clean_retrieval_calls",
            "clean_read_count", "read_clean_history"
        ))
        history_kind = str(_first(sample, ("history_kind", "history_source", "read_history_kind"), "")).lower()
        if clean_reads or history_kind == "clean":
            errors.append(f"line {line}: clean history read is forbidden")

        local_names = (
            "local_frame_ids", "local_frame_indices", "local_frames", "local_indices", "window_frame_indices"
        )
        local_ordered = _ordered_indices(sample, local_names)
        local = set(local_ordered)
        if row.get("event") == "attention_call" and _first(row, ("block_id",), _MISSING) is not _MISSING:
            block_id = int(row["block_id"])
            end = (block_id + 1) * 3
            expected_local = (
                tuple(range(end))
                if end <= 21
                else (0, *range(end - 20, end))
            )
            if len(local_ordered) != len(set(local_ordered)):
                errors.append(f"line {line}: local frame IDs are not unique")
            if local_ordered != expected_local:
                errors.append(
                    f"line {line}: local topology {local_ordered} != sink1+recent20 {expected_local}"
                )
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
        selected_count_value = _first(
            sample, ("selected_count", "selection_count", "history_selected_count"), None
        )
        selected_count = len(selected) if selected_count_value is None else _count(
            sample, ("selected_count", "selection_count", "history_selected_count")
        )
        if selected_count != len(selected):
            errors.append(
                f"line {line}: selected count {selected_count} != selected IDs {len(selected)}"
            )
        maximums["archive_size"] = max(maximums["archive_size"], archive_size)
        maximums["selected"] = max(maximums["selected"], selected_count)
        if archive_size > ARCHIVE_LIMIT:
            errors.append(f"line {line}: archive size {archive_size} exceeds {ARCHIVE_LIMIT}")
        if selected_count > expected_history_budget:
            errors.append(
                f"line {line}: selected {selected_count} exceeds requested budget {expected_history_budget}"
            )
        if selected and not selected.issubset(eligible):
            errors.append(f"line {line}: selected history is not a subset of eligible history")
        if len(eligible) >= expected_history_budget and selected_count != expected_history_budget:
            errors.append(
                f"line {line}: selected {selected_count} != requested budget "
                f"{expected_history_budget} with {len(eligible)} eligible"
            )

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
        for name in ("last_scale_min", "last_scale_max", "scale_min", "scale_max"):
            value = _first(sample, (name,), None)
            if value is not None:
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    errors.append(f"line {line}: {name.replace('_', ' ')} is not finite")

        lookup = _count(sample, ("lookup_count", "retrieval_lookup_count", "retrieval_calls", "provider_calls", "lookups"))
        random = _count(sample, ("random_count", "random_draw_count", "random_selections", "random_draws", "used_random"))
        second = _count(sample, ("second_attention_count", "second_attention_calls", "used_second_attention"))
        lookup = max(lookup, _event_count(row, ("lookup",)))
        random = max(random, _event_count(row, ("random",)))
        second = max(second, _event_count(row, ("second", "attention")))
        if random:
            errors.append(f"line {line}: random retrieval is forbidden")
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
            if active and lookup != 1:
                errors.append(f"line {line}: active e1 call requires exactly one history lookup")
            if active and eligible and selected_count and selected:
                active_selected_rows += 1
            if active and eligible and not selected_count:
                errors.append(f"line {line}: active call did not select available eligible history")
            if not active and (lookup or random or second or selected_count or selected):
                errors.append(f"line {line}: unexpected second/intervention activity outside expected phase")
            if second and not expected_second:
                errors.append(f"line {line}: unexpected second attention")
            if active and selected_count and second != 1:
                errors.append(f"line {line}: active selected call requires exactly one second attention")
            if active and not selected_count and second:
                errors.append(f"line {line}: active empty-selection call cannot run second attention")
            if second > 1:
                errors.append(f"line {line}: more than one second attention in a call")

    if not rows:
        errors.append("trace contains no events")
    elif not attention_rows:
        errors.append("trace contains no attention_call events")
    expected_calls = [("noisy", index) for index in range(4)] + [("clean", 4)]
    observed_layers = sorted({layer for layer, _ in call_matrix})
    observed_blocks = sorted({block for _, block in call_matrix})
    required_layers = list(range(expected_layers)) if expected_layers is not None else observed_layers
    required_blocks = list(range(expected_blocks)) if expected_blocks is not None else observed_blocks
    if observed_layers != required_layers:
        errors.append(f"attention layer coverage mismatch: {observed_layers} != {required_layers}")
    if observed_blocks != required_blocks:
        errors.append(f"attention block coverage mismatch: {observed_blocks} != {required_blocks}")
    for layer in required_layers:
        for block in required_blocks:
            calls = call_matrix.get((layer, block), [])
            if calls != expected_calls:
                errors.append(
                    f"layer {layer} block {block}: call trajectory {calls} != {expected_calls}"
                )
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
        "expected_history_budget": expected_history_budget,
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
    parser.add_argument("--expected-blocks", type=int)
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument("--phase", choices=tuple(_PHASES), default="e1")
    parser.add_argument("--expected-history-budget", type=int, choices=tuple(sorted(_ALLOWED_HISTORY_BUDGETS)), required=True)
    parser.add_argument("--expect-second-attention", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = None if args.expect_second_attention == "auto" else args.expect_second_attention == "yes"
    report = audit_trace(
        args.trace, args.alpha, expect_second_attention=expected,
        phase=args.phase, expected_history_budget=args.expected_history_budget,
        tolerance=args.tolerance,
        expected_blocks=args.expected_blocks,
        expected_layers=args.expected_layers,
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
