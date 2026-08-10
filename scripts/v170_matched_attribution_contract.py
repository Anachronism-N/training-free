#!/usr/bin/env python3
"""Frozen method and scheduling contract for v170 matched attribution."""

from __future__ import annotations

PROMPT_COUNT = 16
NUM_NODES = 2
GPUS_PER_NODE = 8
ACTIVE_LAYERS = tuple(range(10, 20))
TRACE_HEADS = (0,)

V166_A = "ours_v170_v166_a"
QUERY_A = "ours_v170_queryweighted_a"
V166_B = "ours_v170_v166_b"
QUERY_B = "ours_v170_queryweighted_b"
METHODS = (V166_A, QUERY_A, V166_B, QUERY_B)
V166_METHODS = (V166_A, V166_B)
QUERY_METHODS = (QUERY_A, QUERY_B)

BASE_POLICY = {
    V166_A: "reservoir2_multiscalemotion1",
    QUERY_A: "reservoir2_multiscalequeryweighted1",
    V166_B: "reservoir2_multiscalemotion1",
    QUERY_B: "reservoir2_multiscalequeryweighted1",
}
ANALYZER_KIND = {
    V166_A: "v166_multiscale_motion",
    QUERY_A: "v169_query_weighted",
    V166_B: "v166_multiscale_motion",
    QUERY_B: "v169_query_weighted",
}


def prompt_indices(node_rank: int, num_nodes: int = NUM_NODES) -> tuple[int, ...]:
    if num_nodes != NUM_NODES or not 0 <= node_rank < num_nodes:
        raise ValueError("v170 requires two nodes with ranks 0..1")
    per_node = PROMPT_COUNT // NUM_NODES
    start = node_rank * per_node
    return tuple(range(start, start + per_node))


def lane_methods(prompt_index: int, lane: str) -> tuple[str, str]:
    if lane not in {"a", "b"}:
        raise ValueError(f"unknown lane: {lane}")
    if lane == "a":
        natural = (V166_A, QUERY_A)
    else:
        natural = (QUERY_B, V166_B)
    return natural if prompt_index % 2 == 0 else tuple(reversed(natural))


def node_schedule(
    node_rank: int,
    num_nodes: int = NUM_NODES,
) -> tuple[dict[str, object], ...]:
    rows = []
    for local_index, prompt_index in enumerate(prompt_indices(node_rank, num_nodes)):
        for lane_offset, lane in enumerate(("a", "b")):
            gpu_slot = (2 * local_index + lane_offset) % GPUS_PER_NODE
            for order, method in enumerate(lane_methods(prompt_index, lane)):
                rows.append(
                    {
                        "node_rank": node_rank,
                        "prompt_index": prompt_index,
                        "lane": lane,
                        "gpu_slot": gpu_slot,
                        "order": order,
                        "method": method,
                    }
                )
    return tuple(rows)


def full_schedule() -> tuple[dict[str, object], ...]:
    return tuple(
        row
        for node_rank in range(NUM_NODES)
        for row in node_schedule(node_rank, NUM_NODES)
    )


def validate() -> None:
    rows = full_schedule()
    if len(rows) != PROMPT_COUNT * len(METHODS):
        raise ValueError("v170 schedule does not contain 64 tasks")
    coverage = {(str(row["method"]), int(row["prompt_index"])) for row in rows}
    expected = {
        (method, prompt_index)
        for method in METHODS
        for prompt_index in range(PROMPT_COUNT)
    }
    if coverage != expected:
        raise ValueError("v170 schedule method/prompt coverage mismatch")
    for node_rank in range(NUM_NODES):
        slots = {int(row["gpu_slot"]) for row in node_schedule(node_rank)}
        if slots != set(range(GPUS_PER_NODE)):
            raise ValueError(f"node {node_rank} does not use all eight GPU slots")


validate()
