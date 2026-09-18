#!/usr/bin/env python3
"""FIFO-only seed replication and phase/dose screen; no new attention operator."""
from functools import partial
import os

import v212_lphc_protocol as base
from v212_lphc_protocol import (
    AUTHORIZED_NODES, DEFAULT_CHECKPOINT, DEFAULT_PROMPT_SOURCE, DEFAULT_WAN_MODEL,
    FRAMES, GATE_SOURCES, SOURCE_INDICES, assignment, audit, frozen_json, sha256, write_frozen,
)

LABEL = "v213"
EXPERIMENT = "v213_fifo_seed_phase32"
SEED = 21300
SPECS = {name: dict(base.SPECS[name]) for name in ("sf_fifo21", "sf_fifo25", "fifo_correct", "fifo_random")}
SPECS.update({
    "fifo_e1_a010": {**SPECS["fifo_correct"], "alpha": .10},
    "fifo_e2_a002": {**SPECS["fifo_correct"], "phase": "e2"},
    "fifo_full_a002": {**SPECS["fifo_correct"], "phase": "full"},
})
METHODS = tuple(SPECS)
CANDIDATES = ("fifo_correct", "fifo_e1_a010", "fifo_e2_a002", "fifo_full_a002")
GATE_PAIRS = (("sf_fifo21", "fifo_zero"),)
CAMPAIGN = base.Campaign(
    LABEL, EXPERIMENT, SOURCE_INDICES, SEED, SPECS, GATE_PAIRS,
    (("fifo_correct", "sf_fifo21"),),
    (("fifo_correct", "fifo_random"), ("fifo_correct", "sf_fifo25"),
     ("fifo_e1_a010", "fifo_correct"), ("fifo_e2_a002", "fifo_correct"),
     ("fifo_full_a002", "fifo_correct")),
)
prepare = partial(base.prepare, campaign=CAMPAIGN)
verify = partial(base.verify, campaign=CAMPAIGN)
output_root = partial(base.output_root, campaign=CAMPAIGN)
spec_for = partial(base.spec_for, campaign=CAMPAIGN)


def method_order(source):
    shift = SOURCE_INDICES.index(source) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def validate_node(rank, num_nodes=6):
    from run_v211_worker import assert_authorized_node
    address = os.environ.get("V213_NODE_ADDRESS")
    if num_nodes != 6 or not 0 <= rank < 6 or address != AUTHORIZED_NODES[rank]:
        raise PermissionError("set V213_NODE_ADDRESS to the frozen address for this NODE_RANK")
    return assert_authorized_node({"authorized_nodes": list(AUTHORIZED_NODES)}, node_address=address)
