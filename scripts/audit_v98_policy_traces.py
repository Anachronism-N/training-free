#!/usr/bin/env python3
"""Strictly audit sharded v98 runtime cache-policy traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = (
    "sf_native",
    "pf_native",
    "pf_explicit_parity",
    "pf_aw_hybrid_merge",
    "history_polarity_hybrid_merge",
    "history_polarity_stride_merge",
    "history_polarity_hybrid_merge_v78",
    "positive_rate_half_hybrid_merge",
)
PF_NATIVE = {
    -1: (["CyclicStrategy"], 1, 4),
    1: (["StrideStrategy"], 3, 4),
    2: (["MergeStrategy"], 3, 4),
}
HISTORY_HYBRID = {
    10: (["CyclicStrategy", "StrideStrategy"], 3, 4),
    11: (["MergeStrategy"], 3, 4),
}
HISTORY_STRIDE = {
    10: (["StrideStrategy"], 3, 4),
    11: (["MergeStrategy"], 3, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-layers", default="0,7,15,23,29")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--shards", type=int, default=4)
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
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
            f"{path}: expected {num_layers} layers, found {len(rows)}"
        )
    for layer, row in enumerate(rows):
        if len(row) != num_heads:
            raise ValueError(
                f"{path}: layer {layer} expected {num_heads} heads, "
                f"found {len(row)}"
            )
    return rows


def expected_routes(route: str) -> dict[int, tuple[list[str], int, int]]:
    if route in {"native", "pf_explicit_parity"}:
        return PF_NATIVE
    if route == "history_hybrid_merge":
        return HISTORY_HYBRID
    if route == "history_stride_merge":
        return HISTORY_STRIDE
    raise ValueError(f"unsupported PF route {route!r}")


def audit_trace(
    *,
    method: str,
    shard: int,
    config_path: Path,
    trace_path: Path,
    expected_layers: set[int],
    num_layers: int,
    num_heads: int,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        config = load_env(config_path)
        if config.get("name") != method:
            failures.append(
                f"config method {config.get('name')!r} != {method!r}"
            )
        if int(config.get("shard", -1)) != shard:
            failures.append(
                f"config shard {config.get('shard')!r} != {shard}"
            )
        label_path = Path(config["labels"])
        labels = load_labels(label_path, num_layers, num_heads)
        actual_hash = sha256(label_path)
        if actual_hash != config.get("label_sha256"):
            failures.append("head-map SHA256 does not match frozen config")
        routes = expected_routes(config["route"])
    except (KeyError, OSError, ValueError) as error:
        return {
            "method": method,
            "shard": shard,
            "status": "failed",
            "events": 0,
            "failures": [f"invalid config: {error}"],
        }

    events: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(
            trace_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(event, dict):
                failures.append(f"line {line_number}: event is not an object")
                continue
            events.append(event)
    except OSError as error:
        return {
            "method": method,
            "shard": shard,
            "status": "failed",
            "events": 0,
            "failures": [f"cannot read trace: {error}"],
        }
    if not events:
        failures.append("trace is empty")

    observed_pairs: set[tuple[int, int]] = set()
    observed_branches: set[str] = set()
    labels_seen: Counter[int] = Counter()
    strategies_seen: Counter[str] = Counter()
    max_union = 0
    for index, event in enumerate(events):
        if event.get("event") != "middle_selection":
            failures.append(
                f"event {index}: unexpected type {event.get('event')!r}"
            )
            continue
        try:
            layer = int(event["layer"])
            head = int(event["head"])
            label = int(event["label"])
            branch = str(event["branch"])
            names = [
                str(item["name"]) for item in list(event["strategies"])
            ]
            sink = int(event["sink_frames"])
            recent = int(event["recent_frames"])
            union_ids = [int(value) for value in event["union_frame_ids"]]
            union_count = int(event["union_frame_count"])
            union_tokens = int(event["union_token_count"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"event {index}: malformed: {error}")
            continue

        if layer not in expected_layers:
            failures.append(f"event {index}: unexpected traced layer {layer}")
            continue
        if not 0 <= head < num_heads:
            failures.append(f"event {index}: invalid head {head}")
            continue
        expected_label = labels[layer][head]
        if label != expected_label:
            failures.append(
                f"event {index}: label {label} != map {expected_label}"
            )
            continue
        if label not in routes:
            failures.append(
                f"event {index}: label {label} not valid for {config['route']}"
            )
            continue
        expected_names, expected_sink, expected_recent = routes[label]
        if names != expected_names:
            failures.append(
                f"event {index}: strategies {names} != {expected_names}"
            )
        if (sink, recent) != (expected_sink, expected_recent):
            failures.append(
                f"event {index}: sink/recent {sink}/{recent} != "
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
        if len(expected_names) == 2 and union_count > 4:
            failures.append(
                f"event {index}: hybrid middle budget exceeded ({union_count})"
            )

        observed_pairs.add((layer, head))
        observed_branches.add(branch)
        labels_seen[label] += 1
        strategies_seen.update(names)
        max_union = max(max_union, union_count)

    expected_pairs = {
        (layer, head)
        for layer in expected_layers
        for head in range(num_heads)
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        failures.append(
            f"missing traced layer/head pairs: {missing_pairs[:16]} "
            f"(total={len(missing_pairs)})"
        )
    if observed_branches != {"cond", "uncond"}:
        failures.append(
            f"expected cond/uncond traces, found {sorted(observed_branches)}"
        )

    is_history = config["route"].startswith("history_")
    if is_history and not set(labels_seen).issubset({10, 11}):
        failures.append(
            f"history route leaked PF labels: {sorted(labels_seen)}"
        )
    return {
        "method": method,
        "shard": shard,
        "status": "failed" if failures else "nominal",
        "config": str(config_path.resolve()),
        "trace": str(trace_path.resolve()),
        "route": config["route"],
        "head_map": str(label_path.resolve()),
        "head_map_sha256": actual_hash,
        "events": len(events),
        "observed_pairs": len(observed_pairs),
        "branches": sorted(observed_branches),
        "label_events": dict(sorted(labels_seen.items())),
        "strategy_events": dict(sorted(strategies_seen.items())),
        "max_union_frames": max_union,
        "failures": failures,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# v98 Policy Trace Audit",
        "",
        f"- strict pass: `{payload['strict_pass']}`",
        f"- audited PF shards: `{len(payload['shards'])}`",
        f"- total events: `{payload['event_count']}`",
        f"- PF parity route contract: `{payload['pf_parity_route_contract']}`",
        "",
        "| Method | Shard | Route | Status | Events | Strategies | Failures |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    for item in payload["shards"]:
        strategies = ", ".join(
            f"{name}:{count}"
            for name, count in item.get("strategy_events", {}).items()
        )
        lines.append(
            f"| {item['method']} | {item['shard']} | "
            f"{item.get('route', 'unknown')} | {item['status']} | "
            f"{item.get('events', 0)} | {strategies or 'none'} | "
            f"{len(item.get('failures', []))} |"
        )
    failures = [
        (item["method"], item["shard"], failure)
        for item in payload["shards"]
        for failure in item.get("failures", [])
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{method}.shard{shard}`: {failure}"
            for method, shard, failure in failures
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_layers = {
        int(value.strip())
        for value in args.expected_layers.split(",")
        if value.strip()
    }
    if not expected_layers:
        raise ValueError("expected layers cannot be empty")

    config_dir = args.run_root / "configs"
    trace_dir = args.run_root / "traces"
    results: list[dict[str, Any]] = []
    for method in METHODS:
        for shard in range(args.shards):
            config_path = config_dir / f"{method}.shard{shard}.env"
            if method == "sf_native":
                if not config_path.is_file():
                    results.append(
                        {
                            "method": method,
                            "shard": shard,
                            "status": "failed",
                            "events": 0,
                            "failures": [f"missing config {config_path}"],
                        }
                    )
                continue
            results.append(
                audit_trace(
                    method=method,
                    shard=shard,
                    config_path=config_path,
                    trace_path=(
                        trace_dir / f"{method}.shard{shard}.policy.jsonl"
                    ),
                    expected_layers=expected_layers,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                )
            )

    parity_contract = PF_NATIVE == expected_routes("pf_explicit_parity")
    payload = {
        "version": 1,
        "method": "v98_sharded_policy_trace_audit",
        "expected_methods": list(METHODS),
        "expected_layers": sorted(expected_layers),
        "expected_shards": args.shards,
        "pf_parity_route_contract": parity_contract,
        "event_count": sum(int(item.get("events", 0)) for item in results),
        "strict_pass": (
            parity_contract
            and len(results) == (len(METHODS) - 1) * args.shards
            and all(item["status"] == "nominal" for item in results)
        ),
        "shards": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V98PolicyTraceAudit] "
        f"shards={len(results)} events={payload['event_count']} "
        f"strict_pass={payload['strict_pass']}",
        flush=True,
    )
    if args.strict and not payload["strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
