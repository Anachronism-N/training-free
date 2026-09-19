#!/usr/bin/env python3
"""Head-preserving retrieval screen and phase-matched random control on 48 GPUs."""
from functools import partial
import json
import math
import os

import v212_lphc_protocol as base
import v213_lphc_protocol as previous
from v212_lphc_protocol import (
    AUTHORIZED_NODES, DEFAULT_CHECKPOINT, DEFAULT_PROMPT_SOURCE, DEFAULT_WAN_MODEL,
    FRAMES, frozen_json, sha256, write_frozen,
)
from v213_sf_baseline_contract import require_baseline as _require_baseline

LABEL = "v215"
EXPERIMENT = "v215_head_preserving_retrieval48"
STAGE = "screen48"
SOURCE_INDICES = tuple(sorted(set(previous.SOURCE_INDICES) | set(range(0, 128, 8))))
GATE_SOURCES = (3, 67)
SEED = 21500
SPECS = {m: dict(previous.SPECS[m]) for m in ("sf_fifo21", "fifo_correct", "fifo_random")}
for name, descriptor in (("headwise_correct", "headwise"), ("centered_correct", "headwise_centered")):
    SPECS[name] = {**SPECS["fifo_correct"], "protocol": "v215", "descriptor_mode": descriptor}
SPECS["fifo_full_a002"] = dict(previous.SPECS["fifo_full_a002"])
SPECS["fifo_full_random"] = {**SPECS["fifo_full_a002"], "retrieval_mode": "random"}
METHODS = tuple(SPECS)
CANDIDATES = ("headwise_correct", "centered_correct", "fifo_full_a002")
EFFECT_CONTROLS = ("sf_fifo21",)
PRIMARY_METRIC = "official_quality_score"
PRIMARY_HYPOTHESIS = "Head-preserving retrieval versus SF and pooled retrieval; official-formula Quality primary, DD-fixed quality and motion separately retained"
GATE_PAIRS = (("sf_fifo21", "fifo_zero"), ("sf_fifo21", "headwise_zero"), ("sf_fifo21", "centered_zero"))
CAMPAIGN = base.Campaign(LABEL, EXPERIMENT, SOURCE_INDICES, SEED, SPECS, GATE_PAIRS,
                        tuple((m, "sf_fifo21") for m in CANDIDATES), (
    ("fifo_correct", "sf_fifo21"), ("fifo_random", "sf_fifo21"),
    ("fifo_correct", "fifo_random"), ("headwise_correct", "fifo_correct"),
    ("centered_correct", "fifo_correct"), ("centered_correct", "headwise_correct"),
    ("headwise_correct", "fifo_random"), ("centered_correct", "fifo_random"),
    ("fifo_full_a002", "fifo_full_random"), ("fifo_full_a002", "fifo_correct")))
output_root = partial(base.output_root, campaign=CAMPAIGN)
spec_for = partial(base.spec_for, campaign=CAMPAIGN)
assignment = partial(base.assignment, sources=SOURCE_INDICES)
require_baseline = partial(_require_baseline, label=LABEL, sources=GATE_SOURCES)


def method_order(source):
    shift = SOURCE_INDICES.index(source) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def prepare(repo, out, prompts, checkpoint, wan, slots, *, clean=True):
    if tuple(slots) != tuple(map(str, range(8))):
        raise ValueError("v215 requires all eight GPU slots on six nodes")
    return base.prepare(repo, out, prompts, checkpoint, wan, slots, clean=clean, campaign=CAMPAIGN)


def verify(repo, out, *, runtime=True):
    data = base.verify(repo, out, runtime=runtime, campaign=CAMPAIGN)
    if data["gpu_slots"] != list(map(str, range(8))):
        raise ValueError("v215 GPU topology changed")
    return data


def validate_node(rank, num_nodes=6):
    from run_v211_worker import assert_authorized_node
    address = os.environ.get("V215_NODE_ADDRESS")
    if num_nodes != 6 or not 0 <= rank < 6 or address != AUTHORIZED_NODES[rank]:
        raise PermissionError("set V215_NODE_ADDRESS for this NODE_RANK")
    return assert_authorized_node({"authorized_nodes": list(AUTHORIZED_NODES)}, node_address=address)


def audit(path, spec, blocks, source):
    report = base.audit(path, spec, blocks, source)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mode = spec.get("descriptor_mode", "pooled")
    observed, choices, margins = 0, 0, []
    for row in rows:
        if row.get("event") in {"video_start", "attention_call"} and row.get("descriptor_mode") != mode:
            report["errors"].append("descriptor mode missing or mismatched")
        if row.get("event") != "attention_call" or row.get("call_kind") != "noisy" or row.get("phase_index") != 0:
            continue
        stats = row.get("selection_stats", {})
        if not stats:
            continue
        observed += 1
        ids, scores = stats["ranked_frame_ids"], stats["ranked_scores"]
        selected = row["selected_history_frames"]
        eligible = row["eligible_frame_indices"]
        if (stats.get("mode") != mode or len(ids) != len(scores) or len(set(ids)) != len(ids)
                or set(ids) != set(eligible) or stats.get("candidate_count") != len(ids)
                or any(not math.isfinite(s) for s in scores)
                or set(selected) != set(ids[:min(4, len(ids))])
                or any(a < b for a, b in zip(scores, scores[1:]))):
            report["errors"].append("descriptor ranking/selection mismatch")
        choices += int(stats.get("actual_choice", False))
        margin = stats.get("boundary_margin")
        if margin is not None:
            if not math.isfinite(margin) or margin < 0:
                report["errors"].append("invalid selection margin")
            else:
                margins.append(margin)
    if spec["alpha"] > 0 and spec["retrieval_mode"] == "correct" and not observed:
        report["errors"].append("selector score diagnostics were never recorded")
    report["selector"] = {"mode": mode, "scored_calls": observed, "calls_with_real_choice": choices,
                          "mean_boundary_margin": sum(margins)/len(margins) if margins else None}
    report["pass"] = not report["errors"]
    return report
