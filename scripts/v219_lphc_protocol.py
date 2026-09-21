#!/usr/bin/env python3
"""Four-arm closure: frozen method, SF, random and head-pooled retrieval, 64 GPUs."""
import os
from pathlib import Path

import v217_lphc_protocol as parent

LABEL = "v219"
EXPERIMENT = "v219_frozen_headwise_mechanism64"
SOURCE_INDICES = parent.SOURCE_INDICES
RATIONALE = ("Frozen v216 headwise method and endpoint; new-seed SF/random replication plus "
             "pooled descriptor ablation. Added after inspecting v216, not a new method search.")


class Protocol(parent.Protocol):
    LABEL, EXPERIMENT, SOURCE_INDICES = LABEL, EXPERIMENT, SOURCE_INDICES
    METHODS = ("sf_fifo21", "ours_correct", "ours_random", "pooled_correct")
    GATE_PAIRS = (("sf_fifo21", "ours_zero"), ("sf_fifo21", "pooled_zero"))
    EFFECT_CONTROLS = ("sf_fifo21", "ours_random", "pooled_correct")
    MECHANISM = (("ours_random", "sf_fifo21"), ("pooled_correct", "sf_fifo21"),
                 ("pooled_correct", "ours_random"))

    def build_specs(self, candidate):
        if candidate != "headwise_correct":
            raise ValueError("v219 closes the frozen headwise hypothesis; do not silently change the method")
        specs = super().build_specs(candidate)
        specs["pooled_correct"] = {**specs["ours_correct"], "descriptor_mode": "pooled"}
        return specs

    def _check_inputs(self, data):
        super()._check_inputs(data)
        if data["configs"]["pooled_correct"]["sha256"] != data["configs"]["ours_correct"]["sha256"]:
            raise ValueError("pooled control inference config differs from frozen method")

    def method_order(self, source):
        position = self.SOURCE_INDICES.index(source)
        shift = (position % 8 + position // 8) % len(self.METHODS)
        return self.METHODS[shift:] + self.METHODS[:shift]


def load(out=None):
    path = out or os.environ.get("V219_OUT_ROOT")
    if not path:
        raise ValueError("set V219_OUT_ROOT to the new closure output directory")
    return Protocol(Path(path))
