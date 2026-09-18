#!/usr/bin/env python3
"""Six nodes x eight GPUs: 96 disjoint prompts extending the v213 matrix."""
from functools import partial
import json
import os

import v212_lphc_protocol as base
import v213_lphc_protocol as previous
from v212_lphc_protocol import (
    AUTHORIZED_NODES, DEFAULT_CHECKPOINT, DEFAULT_PROMPT_SOURCE, DEFAULT_WAN_MODEL,
    FRAMES, audit, frozen_json, sha256, write_frozen,
)
from v213_sf_baseline_contract import require_baseline as _require_baseline

LABEL = "v214"
EXPERIMENT = "v214_fifo_remaining96_sixnode"
STAGE = "screen96"
SOURCE_INDICES = tuple(s for s in range(128) if s not in previous.SOURCE_INDICES)
GATE_SOURCES = (0, 64)
SEED = previous.SEED
SPECS = {name: dict(spec) for name, spec in previous.SPECS.items()}
METHODS = tuple(SPECS)
CANDIDATES = previous.CANDIDATES
GATE_PAIRS = previous.GATE_PAIRS
CAMPAIGN = base.Campaign(LABEL, EXPERIMENT, SOURCE_INDICES, SEED, SPECS,
                        GATE_PAIRS, previous.CAMPAIGN.primary, previous.CAMPAIGN.mechanism)
output_root = partial(base.output_root, campaign=CAMPAIGN)
spec_for = partial(base.spec_for, campaign=CAMPAIGN)
assignment = partial(base.assignment, sources=SOURCE_INDICES)
require_baseline = partial(_require_baseline, label=LABEL, sources=GATE_SOURCES)


def method_order(source):
    shift = SOURCE_INDICES.index(source) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def placement(slots):
    if tuple(slots) != tuple(map(str, range(8))):
        raise ValueError("v214 requires GPU_LIST=0,1,2,3,4,5,6,7 on all six nodes")
    return {
        "experiment": EXPERIMENT, "num_nodes": 6, "gpus_per_node": 8,
        "prompt_count": 96, "main_video_count": 96 * len(METHODS),
        "excluded_v213_sources": list(previous.SOURCE_INDICES),
        "same_seed_rule_as_v213": True,
        "jobs": [{"source_index": s, "node_rank": assignment(s, slots)[0],
                  "gpu": assignment(s, slots)[1], "effective_seed": SEED+s,
                  "methods": list(method_order(s))} for s in SOURCE_INDICES],
    }


def prepare(repo, out, prompts, checkpoint, wan, slots, *, clean=True):
    plan = placement(slots)
    data = base.prepare(repo, out, prompts, checkpoint, wan, slots, clean=clean, campaign=CAMPAIGN)
    frozen_json(out / "inputs/placement.json", plan)
    return data


def verify(repo, out, *, runtime=True):
    data = base.verify(repo, out, runtime=runtime, campaign=CAMPAIGN)
    plan = json.loads((out / "inputs/placement.json").read_text(encoding="utf-8"))
    if plan != placement(tuple(data["gpu_slots"])):
        raise ValueError("v214 placement drift")
    return data


def validate_node(rank, num_nodes=6):
    from run_v211_worker import assert_authorized_node
    address = os.environ.get("V214_NODE_ADDRESS")
    if num_nodes != 6 or not 0 <= rank < 6 or address != AUTHORIZED_NODES[rank]:
        raise PermissionError("set V214_NODE_ADDRESS to the frozen address for this NODE_RANK")
    return assert_authorized_node({"authorized_nodes": list(AUTHORIZED_NODES)}, node_address=address)
