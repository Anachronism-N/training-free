#!/usr/bin/env python3
"""Validate v97 per-head cache-policy traces against frozen run configs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--expected-layers", default="0,7,15,23,29")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected key=value")
            key, value = line.split("=", 1)
            result[key] = value
    return result


def load_labels(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != num_layers:
        raise ValueError(
            f"{path}: expected {num_layers} rows, found {len(rows)}"
        )
    for layer, row in enumerate(rows):
        if len(row) != num_heads:
            raise ValueError(
                f"{path}: layer {layer} expected {num_heads} heads, "
                f"found {len(row)}"
            )
    return rows


def expected_policy(
    config: dict[str, str],
    label: int,
) -> tuple[list[str], int, int]:
    if config.get("mode") == "pf_native":
        native = {
            -1: (["CyclicStrategy"], 1, 4),
            1: (["StrideStrategy"], 3, 4),
            2: (["MergeStrategy"], 3, 4),
        }
        if label not in native:
            raise ValueError(f"unexpected PF native label {label}")
        return native[label]
    ablation = config.get("pf_extended_recent_ablation", "none")
    if ablation != "none":
        if label == 3:
            sink = 1 if ablation == "wave" else 3
            recent = 5 if ablation == "veil" else 8
            return [], sink, recent
        native = {
            -1: (["CyclicStrategy"], 1, 4),
            1: (["StrideStrategy"], 3, 4),
            2: (["MergeStrategy"], 3, 4),
        }
        if label not in native:
            raise ValueError(f"unexpected PF ablation label {label}")
        return native[label]

    if label == 1:
        stable = config.get("stable_policy")
        if stable == "stride":
            return ["StrideStrategy"], 3, 4
        if stable == "hybrid":
            return ["CyclicStrategy", "StrideStrategy"], 3, 4
        raise ValueError(f"unexpected stable policy {stable!r}")
    if label == -1:
        responsive = config.get("responsive_policy")
        policies = {
            "cyclic": (["CyclicStrategy"], 1, 4),
            "merge": (["MergeStrategy"], 3, 4),
            "recent": ([], 3, 4),
        }
        if responsive not in policies:
            raise ValueError(
                f"unexpected responsive policy {responsive!r}"
            )
        return policies[responsive]
    raise ValueError(f"unexpected binary label {label}")


def load_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events = []
    failures = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                failures.append(
                    f"line {line_number}: invalid JSON: {error}"
                )
                continue
            if not isinstance(event, dict):
                failures.append(f"line {line_number}: not a JSON object")
                continue
            events.append(event)
    return events, failures


def summarize_method(
    method: str,
    trace_path: Path,
    config_path: Path,
    *,
    expected_layers: set[int],
    num_layers: int,
    num_heads: int,
) -> dict[str, Any]:
    failures: list[str] = []
    if not trace_path.is_file():
        return {
            "method": method,
            "status": "failed",
            "events": 0,
            "failures": [f"missing trace {trace_path}"],
        }
    if not config_path.is_file():
        return {
            "method": method,
            "status": "failed",
            "events": 0,
            "failures": [f"missing config {config_path}"],
        }

    try:
        config = load_env(config_path)
        label_path = Path(config["labels"])
        labels = load_labels(label_path, num_layers, num_heads)
        expected_label_hash = config.get("label_sha256")
        actual_label_hash = sha256(label_path)
        if expected_label_hash != actual_label_hash:
            failures.append(
                "head-map hash mismatch: "
                f"expected={expected_label_hash} actual={actual_label_hash}"
            )
    except (KeyError, OSError, ValueError) as error:
        return {
            "method": method,
            "status": "failed",
            "events": 0,
            "failures": [f"invalid frozen config: {error}"],
        }

    events, parse_failures = load_events(trace_path)
    failures.extend(parse_failures)
    observed_layers: set[int] = set()
    observed_keys: set[tuple[int, int, int, int, str]] = set()
    label_counts: Counter[int] = Counter()
    strategy_counts: Counter[str] = Counter()
    nonempty_by_strategy: Counter[str] = Counter()
    union_frame_counts: list[int] = []

    if not events:
        failures.append("trace is empty")
    for index, event in enumerate(events):
        if event.get("event") != "middle_selection":
            failures.append(
                f"event {index}: unexpected type {event.get('event')!r}"
            )
            continue
        try:
            layer = int(event["layer"])
            head = int(event["head"])
            seq = int(event["seq"])
            sync_t = int(event["sync_t"])
            branch = str(event["branch"])
            label = int(event["label"])
            sink_frames = int(event["sink_frames"])
            recent_frames = int(event["recent_frames"])
            strategy_rows = list(event["strategies"])
            union_ids = [int(value) for value in event["union_frame_ids"]]
            union_count = int(event["union_frame_count"])
            union_tokens = int(event["union_token_count"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"event {index}: malformed fields: {error}")
            continue

        if not (0 <= layer < num_layers and 0 <= head < num_heads):
            failures.append(
                f"event {index}: invalid layer/head ({layer}, {head})"
            )
            continue
        observed_layers.add(layer)
        key = (layer, head, seq, sync_t, branch)
        if key in observed_keys:
            failures.append(f"event {index}: duplicate trace key {key}")
        observed_keys.add(key)

        expected_label = labels[layer][head]
        if label != expected_label:
            failures.append(
                f"event {index}: label {label} != map {expected_label}"
            )
        try:
            expected_names, expected_sink, expected_recent = expected_policy(
                config, expected_label
            )
        except ValueError as error:
            failures.append(f"event {index}: {error}")
            continue

        names = [str(row.get("name")) for row in strategy_rows]
        if names != expected_names:
            failures.append(
                f"event {index}: strategies {names} != {expected_names}"
            )
        if sink_frames != expected_sink or recent_frames != expected_recent:
            failures.append(
                f"event {index}: sink/recent "
                f"{sink_frames}/{recent_frames} != "
                f"{expected_sink}/{expected_recent}"
            )
        if union_count != len(union_ids):
            failures.append(f"event {index}: union count mismatch")
        if union_ids != sorted(set(union_ids)):
            failures.append(
                f"event {index}: union frame ids are not sorted unique"
            )
        if union_tokens < 0:
            failures.append(f"event {index}: negative union token count")

        strategy_union: set[int] = set()
        for row in strategy_rows:
            name = str(row.get("name"))
            frame_ids = [int(value) for value in row.get("frame_ids", [])]
            token_count = int(row.get("token_count", -1))
            if frame_ids != sorted(frame_ids):
                failures.append(
                    f"event {index}: unsorted frames for {name}"
                )
            if token_count < 0:
                failures.append(
                    f"event {index}: negative token count for {name}"
                )
            strategy_union.update(frame_ids)
            strategy_counts[name] += 1
            if frame_ids:
                nonempty_by_strategy[name] += 1
        if strategy_union != set(union_ids):
            failures.append(
                f"event {index}: strategy union does not match readout union"
            )
        if len(expected_names) == 2 and union_count > 4:
            failures.append(
                f"event {index}: hybrid middle budget exceeded ({union_count})"
            )

        label_counts[label] += 1
        union_frame_counts.append(union_count)

    missing_layers = sorted(expected_layers - observed_layers)
    unexpected_layers = sorted(observed_layers - expected_layers)
    if missing_layers:
        failures.append(f"missing trace layers {missing_layers}")
    if unexpected_layers:
        failures.append(f"unexpected trace layers {unexpected_layers}")
    for name, count in strategy_counts.items():
        if count and nonempty_by_strategy[name] == 0:
            failures.append(f"{name} never returned a middle selection")

    return {
        "method": method,
        "status": "failed" if failures else "nominal",
        "trace": str(trace_path.resolve()),
        "config": str(config_path.resolve()),
        "head_map": str(label_path.resolve()),
        "head_map_sha256": actual_label_hash,
        "score_sha256": config.get("score_sha256"),
        "events": len(events),
        "layers": sorted(observed_layers),
        "label_events": dict(sorted(label_counts.items())),
        "strategy_events": dict(sorted(strategy_counts.items())),
        "strategy_nonempty_events": dict(
            sorted(nonempty_by_strategy.items())
        ),
        "union_frame_count": {
            "min": min(union_frame_counts) if union_frame_counts else None,
            "max": max(union_frame_counts) if union_frame_counts else None,
            "mean": (
                sum(union_frame_counts) / len(union_frame_counts)
                if union_frame_counts
                else None
            ),
        },
        "failures": failures,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v97 Policy Trace Audit",
        "",
        f"- strict pass: `{payload['strict_pass']}`",
        f"- methods: `{payload['method_count']}`",
        f"- events: `{payload['event_count']}`",
        "",
        "| Method | Status | Events | Layers | Strategies | Failures |",
        "|---|---|---:|---|---|---:|",
    ]
    for item in payload["methods"]:
        strategies = ", ".join(
            f"{name}:{count}"
            for name, count in item.get("strategy_events", {}).items()
        ) or "none"
        lines.append(
            f"| {item['method']} | {item['status']} | "
            f"{item.get('events', 0)} | "
            f"{item.get('layers', [])} | {strategies} | "
            f"{len(item.get('failures', []))} |"
        )
    failed = [
        item for item in payload["methods"] if item["status"] != "nominal"
    ]
    if failed:
        lines.extend(["", "## Failures", ""])
        for item in failed:
            for failure in item.get("failures", []):
                lines.append(f"- `{item['method']}`: {failure}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_layers = {
        int(value)
        for value in args.expected_layers.split(",")
        if value.strip()
    }
    if not expected_layers:
        raise ValueError("--expected-layers cannot be empty")
    methods = [
        summarize_method(
            method,
            args.trace_dir / f"{method}.policy.jsonl",
            args.config_dir / f"{method}.env",
            expected_layers=expected_layers,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        for method in args.methods
    ]
    payload = {
        "version": 1,
        "method": "v97_policy_trace_audit",
        "expected_layers": sorted(expected_layers),
        "method_count": len(methods),
        "event_count": sum(int(item.get("events", 0)) for item in methods),
        "strict_pass": all(item["status"] == "nominal" for item in methods),
        "methods": methods,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V97PolicyTraceAudit] "
        f"methods={len(methods)} events={payload['event_count']} "
        f"strict_pass={payload['strict_pass']}",
        flush=True,
    )
    if args.strict and not payload["strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
