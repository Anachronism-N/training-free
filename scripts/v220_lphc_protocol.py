#!/usr/bin/env python3
"""Frozen SF/LPHC 60-second extension: 64 paired prompts across 64 GPUs."""
import os
from pathlib import Path

import v216_lphc_protocol as confirmation
import v217_lphc_protocol as parent
from prepare_v210_vbench_comparison import DIMENSIONS

LABEL = "v220"
EXPERIMENT = "v220_frozen_lphc_long60_64"
SOURCE_INDICES = parent.SOURCE_INDICES
RATIONALE = ("Prespecified 60-second extension of the frozen v216 headwise method; "
             "same 64 sources and seed as v219, no horizon-specific parameter tuning. "
             "Different horizons are not independent prompt samples or guaranteed identical prefixes.")


class Protocol(parent.Protocol):
    LABEL, EXPERIMENT, SOURCE_INDICES = LABEL, EXPERIMENT, SOURCE_INDICES
    FRAMES, STAGE = 240, "long64"
    METHODS = ("sf_fifo21", "ours_correct")
    EFFECT_CONTROLS = ("sf_fifo21",)
    MECHANISM = ()

    def __init__(self, out):
        super().__init__(out)
        self.EXTRA_METRICS = tuple(DIMENSIONS)
        self.PRIMARY_HYPOTHESIS = (f"Frozen headwise LPHC minus SF FIFO21 at 60s: "
                                   f"{self.PRIMARY_WINDOW}/{self.PRIMARY_METRIC}")

    def build_specs(self, candidate):
        if candidate != "headwise_correct":
            raise ValueError("v220 extends the frozen headwise method; do not retune for 60s")
        return confirmation.Protocol.build_specs(self, candidate)

    def _check_inputs(self, data):
        confirmation.Protocol._check_inputs(self, data)

    def method_order(self, source):
        return confirmation.Protocol.method_order(self, source)


def load(out=None):
    path = out or os.environ.get("V220_OUT_ROOT")
    if not path:
        raise ValueError("set V220_OUT_ROOT to a new long-horizon output directory")
    return Protocol(Path(path))
