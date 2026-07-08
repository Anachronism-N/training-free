#!/usr/bin/env python3
"""Summarize LifeCache trace JSONL into readable markdown and CSV.

Usage:
    python scripts/summarize_lifecache_trace.py \
        --trace cache_trace.jsonl \
        --out-md outputs/trace_summary.md \
        --out-csv outputs/trace_summary.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_trace(path: str) -> list[dict]:
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize(events: list[dict]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_events": len(events),
        "event_types": defaultdict(int),
        "layers_seen": set(),
        "warnings": [],
        "bank_growth": [],
        "recall_events": [],
        "compression_events": [],
        "capture_events": [],
        "rope_mode_counts": defaultdict(int),
        "q_source_counts": defaultdict(int),
    }

    for ev in events:
        event_type = ev.get("event", "unknown")
        summary["event_types"][event_type] += 1

        layer_id = ev.get("layer_id")
        if layer_id is not None and layer_id >= 0:
            summary["layers_seen"].add(layer_id)

        extra = ev.get("extra") or {}

        # Bank growth
        if "bank_total_tokens" in extra:
            summary["bank_growth"].append({
                "step": ev.get("step", 0),
                "layer": layer_id,
                "bank_total_tokens": extra["bank_total_tokens"],
                "compressed_tokens_added": extra.get("compressed_tokens_added", 0),
            })

        # Recall events
        if event_type in ("compose_active_cache", "compose_fallback"):
            summary["recall_events"].append({
                "step": ev.get("step", 0),
                "layer": layer_id,
                "event": event_type,
                "active_tokens": extra.get("active_tokens"),
                "recent_tokens": extra.get("recent_tokens"),
                "recalled_tokens": extra.get("recalled_tokens"),
                "anchor_tokens": extra.get("anchor_tokens"),
                "fallback": extra.get("fallback"),
            })

        # Compression events
        if event_type == "on_kv_evicted":
            summary["compression_events"].append({
                "step": ev.get("step", 0),
                "layer": layer_id,
                "compressed_tokens_added": extra.get("compressed_tokens_added", 0),
                "num_evicted_tokens": extra.get("num_evicted_tokens", 0),
            })

        # Capture events
        if event_type in ("begin_capture", "end_capture"):
            summary["capture_events"].append({
                "step": ev.get("step", 0),
                "event": event_type,
                "reason": extra.get("capture_reason", ""),
            })

        # Rope mode
        rope_mode = extra.get("rope_mode")
        if rope_mode:
            summary["rope_mode_counts"][rope_mode] += 1

        # Q source
        q_source = extra.get("q_source")
        if q_source:
            summary["q_source_counts"][q_source] += 1

    # Generate warnings
    _generate_warnings(events, summary)

    return summary


def _generate_warnings(events: list[dict], summary: dict) -> None:
    warnings = summary["warnings"]

    # WARN_NO_BANK_GROWTH
    max_bank = max((b["bank_total_tokens"] for b in summary["bank_growth"]), default=0)
    if max_bank == 0 and summary["event_types"].get("on_kv_evicted", 0) > 0:
        warnings.append("WARN_NO_BANK_GROWTH: bank_total_tokens is always 0 despite eviction events")

    # WARN_NO_RECALL
    total_recalled = sum(r.get("recalled_tokens", 0) or 0 for r in summary["recall_events"])
    compose_events = summary["event_types"].get("compose_active_cache", 0)
    if compose_events > 0 and total_recalled == 0:
        warnings.append("WARN_NO_RECALL: compose_active_cache events exist but recalled_tokens is always 0")

    # WARN_ALL_LAYERS_ENABLED
    if len(summary["layers_seen"]) > 6:
        warnings.append(f"WARN_ALL_LAYERS_ENABLED: {len(summary['layers_seen'])} layers seen, expected <= 6 with enable_last_n_layers")

    # WARN_WRONG_QUERY
    if summary["q_source_counts"].get("evicted_k_mean", 0) > 0:
        warnings.append(f"WARN_WRONG_QUERY: q_source=evicted_k_mean appears {summary['q_source_counts']['evicted_k_mean']} times")

    # WARN_NO_CAPTURE
    if summary["event_types"].get("begin_capture", 0) == 0:
        warnings.append("WARN_NO_CAPTURE: no begin_capture events found")

    # WARN_FAR_POST_ROPE
    post_rope_count = summary["rope_mode_counts"].get("post_rope", 0)
    if post_rope_count > 0:
        warnings.append(f"WARN_POST_ROPE_RECALL: {post_rope_count} post-RoPE recall events — may cause quality degradation at long distance")


def write_markdown(summary: dict, path: str) -> None:
    lines = []
    lines.append("# LifeCache Trace Summary")
    lines.append("")
    lines.append(f"Total events: {summary['total_events']}")
    lines.append("")

    # Warnings
    if summary["warnings"]:
        lines.append("## Warnings")
        for w in summary["warnings"]:
            lines.append(f"- **{w}**")
        lines.append("")

    # Event types
    lines.append("## Event Types")
    lines.append("| Event | Count |")
    lines.append("|---|---|")
    for event, count in sorted(summary["event_types"].items(), key=lambda x: -x[1]):
        lines.append(f"| {event} | {count} |")
    lines.append("")

    # Layers
    lines.append("## Enabled Layers")
    layers = sorted(summary["layers_seen"])
    lines.append(f"Count: {len(layers)}")
    if layers:
        lines.append(f"Range: {min(layers)}-{max(layers)}")
        lines.append(f"List: {layers}")
    lines.append("")

    # Bank growth
    if summary["bank_growth"]:
        lines.append("## Bank Growth")
        max_tokens = max(b["bank_total_tokens"] for b in summary["bank_growth"])
        total_added = sum(b["compressed_tokens_added"] for b in summary["bank_growth"])
        lines.append(f"Max bank tokens: {max_tokens}")
        lines.append(f"Total tokens added: {total_added}")
        # Show first and last few
        lines.append("| Step | Layer | Bank Tokens | Added |")
        lines.append("|---|---|---|---|")
        for b in summary["bank_growth"][:5]:
            lines.append(f"| {b['step']} | {b['layer']} | {b['bank_total_tokens']} | {b['compressed_tokens_added']} |")
        if len(summary["bank_growth"]) > 10:
            lines.append("| ... | ... | ... | ... |")
            for b in summary["bank_growth"][-5:]:
                lines.append(f"| {b['step']} | {b['layer']} | {b['bank_total_tokens']} | {b['compressed_tokens_added']} |")
        lines.append("")

    # Compression summary
    if summary["compression_events"]:
        lines.append("## Compression Summary")
        total_evicted = sum(c["num_evicted_tokens"] for c in summary["compression_events"])
        total_compressed = sum(c["compressed_tokens_added"] for c in summary["compression_events"])
        lines.append(f"Total evicted tokens: {total_evicted}")
        lines.append(f"Total compressed tokens: {total_compressed}")
        lines.append(f"Compression ratio: {total_compressed / max(total_evicted, 1):.3f}")
        lines.append("")

    # Recall summary
    if summary["recall_events"]:
        lines.append("## Recall Summary")
        total_active = sum(r.get("active_tokens", 0) or 0 for r in summary["recall_events"])
        total_recalled = sum(r.get("recalled_tokens", 0) or 0 for r in summary["recall_events"])
        fallbacks = sum(1 for r in summary["recall_events"] if r.get("fallback"))
        lines.append(f"Total recall events: {len(summary['recall_events'])}")
        lines.append(f"Total active tokens (sum): {total_active}")
        lines.append(f"Total recalled tokens (sum): {total_recalled}")
        lines.append(f"Fallback events: {fallbacks}")
        lines.append("")

    # Rope mode
    if summary["rope_mode_counts"]:
        lines.append("## Rope Mode Distribution")
        for mode, count in sorted(summary["rope_mode_counts"].items()):
            lines.append(f"- {mode}: {count}")
        lines.append("")

    # Q source
    if summary["q_source_counts"]:
        lines.append("## Q Source Distribution")
        for src, count in sorted(summary["q_source_counts"].items()):
            lines.append(f"- {src}: {count}")
        lines.append("")

    # Capture events
    if summary["capture_events"]:
        lines.append("## Capture Events")
        lines.append("| Step | Event | Reason |")
        lines.append("|---|---|---|")
        for c in summary["capture_events"][:20]:
            lines.append(f"| {c['step']} | {c['event']} | {c['reason']} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_csv(summary: dict, path: str) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Bank growth CSV
        if summary["bank_growth"]:
            writer.writerow(["# Bank Growth"])
            writer.writerow(["step", "layer", "bank_total_tokens", "compressed_tokens_added"])
            for b in summary["bank_growth"]:
                writer.writerow([b["step"], b["layer"], b["bank_total_tokens"], b["compressed_tokens_added"]])
            writer.writerow([])

        # Recall CSV
        if summary["recall_events"]:
            writer.writerow(["# Recall Events"])
            writer.writerow(["step", "layer", "event", "active_tokens", "recent_tokens", "recalled_tokens", "anchor_tokens", "fallback"])
            for r in summary["recall_events"]:
                writer.writerow([r["step"], r["layer"], r["event"], r.get("active_tokens"), r.get("recent_tokens"), r.get("recalled_tokens"), r.get("anchor_tokens"), r.get("fallback")])
            writer.writerow([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="Path to JSONL trace file")
    ap.add_argument("--out-md", default=None, help="Output markdown path")
    ap.add_argument("--out-csv", default=None, help="Output CSV path")
    args = ap.parse_args()

    events = load_trace(args.trace)
    if not events:
        print(f"WARNING: No events found in {args.trace}")
        return

    summary = summarize(events)

    if args.out_md:
        write_markdown(summary, args.out_md)
        print(f"Markdown written to {args.out_md}")

    if args.out_csv:
        write_csv(summary, args.out_csv)
        print(f"CSV written to {args.out_csv}")

    # Print warnings to stdout
    if summary["warnings"]:
        print("\nWarnings:")
        for w in summary["warnings"]:
            print(f"  {w}")

    print(f"\nTotal events: {summary['total_events']}")
    print(f"Event types: {dict(summary['event_types'])}")
    if summary["bank_growth"]:
        max_bank = max(b["bank_total_tokens"] for b in summary["bank_growth"])
        print(f"Max bank tokens: {max_bank}")


if __name__ == "__main__":
    main()
