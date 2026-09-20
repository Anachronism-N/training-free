#!/usr/bin/env python3
"""Optional frozen-method second-seed replication with matched random history."""
import os
from pathlib import Path

import v216_lphc_protocol as confirmation

LABEL = "v217"
EXPERIMENT = "v217_frozen_method_seed_and_random64"
# Uniform ordinal subsample of the 80 non-selection prompts, independent of results.
SOURCE_INDICES = tuple(confirmation.SOURCE_INDICES[i * 80 // 64] for i in range(64))


class Protocol(confirmation.Protocol):
    LABEL, EXPERIMENT, SOURCE_INDICES = LABEL, EXPERIMENT, SOURCE_INDICES
    STAGE, SEED = "replicate64", 21600
    GATE_SOURCES = (1, 65)
    METHODS = ("sf_fifo21", "ours_correct", "ours_random")
    EFFECT_CONTROLS = ("sf_fifo21", "ours_random")
    MECHANISM = (("ours_random", "sf_fifo21"),)

    def build_specs(self, candidate):
        specs = super().build_specs(candidate)
        specs["ours_random"] = {**specs["ours_correct"], "retrieval_mode": "random"}
        return specs

    def check_scope(self):
        super().check_scope()
        path = self.out / "inputs/v216_selection.json"
        if self.sha256(path) != self.scope["parent_selection_sha256"]:
            raise ValueError("parent v216 selection changed")
        parent = confirmation.read(path)
        if (parent["experiment"] != confirmation.EXPERIMENT
                or parent["confirmation_sources"] != list(confirmation.SOURCE_INDICES)
                or parent["base_seed"] != confirmation.Protocol.SEED):
            raise ValueError("invalid parent v216 selection")
        for key in ("selected_method", "primary_metric", "primary_window", "authorized_nodes", "evidence_sha256"):
            if self.scope[key] != parent[key]:
                raise ValueError(f"v217 cannot retune the frozen v216 {key}")

    def _check_inputs(self, data):
        super()._check_inputs(data)
        if data["configs"]["ours_random"]["sha256"] != data["configs"]["ours_correct"]["sha256"]:
            raise ValueError("random control inference config differs from the frozen method")

    def method_order(self, source):
        position = self.SOURCE_INDICES.index(source)
        shift = (position % 8 + position // 8) % 3
        return self.METHODS[shift:] + self.METHODS[:shift]


def load(out=None):
    path = out or os.environ.get("V217_OUT_ROOT")
    if not path:
        raise ValueError("set V217_OUT_ROOT to the frozen replication output directory")
    return Protocol(Path(path))
