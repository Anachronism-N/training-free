#!/usr/bin/env python3
"""Frozen one-candidate confirmation on 80 prompts, eight nodes and 64 GPUs."""
from functools import partial
import ipaddress
import json
import os
from pathlib import Path

import v212_lphc_protocol as base
import v215_lphc_protocol as development
from v213_sf_baseline_contract import require_baseline

LABEL = "v216"
EXPERIMENT = "v216_frozen_candidate_confirmation80"
SOURCE_INDICES = tuple(s for s in range(128) if s not in development.SOURCE_INDICES)
CANDIDATE_CHOICES = ("fifo_correct", "headwise_correct", "centered_correct", "fifo_full_a002")
PRIMARY_CHOICES = {("official_quality_score", "full"), ("subject_consistency", "late_half"), ("imaging_quality", "full")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_nodes(values):
    if not isinstance(values, list) or len(values) != 8 or any(not isinstance(v, str) for v in values):
        raise ValueError("node file must contain eight distinct interface IP addresses in rank order")
    try:
        result = tuple(str(ipaddress.ip_address(v)) for v in values)
    except ValueError as error:
        raise ValueError("fill in all eight node interface IPs; placeholders and hostnames are not accepted") from error
    if len(set(result)) != 8 or any(ipaddress.ip_address(v).is_loopback or ipaddress.ip_address(v).is_unspecified for v in result):
        raise ValueError("eight distinct non-loopback node IPs are required")
    return result


def inference_paths(runtime):
    return {k: v for k, v in runtime.items() if k.startswith(("src/", "third_party/Self-Forcing/"))}


class Protocol:
    LABEL, EXPERIMENT, SOURCE_INDICES = LABEL, EXPERIMENT, SOURCE_INDICES
    NODE_COUNT, STAGE, FRAMES, SEED = 8, "confirm80", 120, development.SEED
    GATE_SOURCES = (1, 65)
    METHODS, CANDIDATES, EFFECT_CONTROLS = ("sf_fifo21", "ours_correct"), ("ours_correct",), ("sf_fifo21",)
    GATE_PAIRS = (("sf_fifo21", "ours_zero"),)
    MECHANISM = ()
    DEFAULT_CHECKPOINT, DEFAULT_PROMPT_SOURCE, DEFAULT_WAN_MODEL = (
        base.DEFAULT_CHECKPOINT, base.DEFAULT_PROMPT_SOURCE, base.DEFAULT_WAN_MODEL)
    sha256, frozen_json, write_frozen = staticmethod(base.sha256), staticmethod(base.frozen_json), staticmethod(base.write_frozen)

    def __init__(self, out):
        self.out = Path(out).resolve()
        path = self.out / "inputs/selection.json"
        if not path.is_file():
            raise ValueError(f"run {self.LABEL} freeze before preparing this campaign")
        self.scope = read(path)
        scope = self.scope
        if scope.get("version") != 1 or scope.get("experiment") != self.EXPERIMENT:
            raise ValueError("wrong v216 selection contract")
        if scope.get("confirmation_sources") != list(self.SOURCE_INDICES) or scope.get("base_seed") != self.SEED:
            raise ValueError("confirmation membership/seed changed")
        if set(scope.get("evidence_sha256", {})) != {"inputs.json", "comparison.json", "report.json", "summary.json"}:
            raise ValueError("selection requires the full compact development evidence")
        candidate = scope["selected_method"]
        if candidate not in CANDIDATE_CHOICES:
            raise ValueError("candidate must be one of the frozen v215 content variants")
        self.AUTHORIZED_NODES = validate_nodes(scope["authorized_nodes"])
        self.PRIMARY_METRIC, self.PRIMARY_WINDOW = scope["primary_metric"], scope["primary_window"]
        if (self.PRIMARY_METRIC, self.PRIMARY_WINDOW) not in PRIMARY_CHOICES:
            raise ValueError("unsupported primary metric/window; do not select after confirmation")
        self.EXTRA_METRICS = ("imaging_quality",) if self.PRIMARY_METRIC == "imaging_quality" else ()
        self.PRIMARY_HYPOTHESIS = f"Frozen {candidate} versus SF FIFO21: {self.PRIMARY_WINDOW}/{self.PRIMARY_METRIC}"
        self.SPECS = self.build_specs(candidate)
        self.CAMPAIGN = base.Campaign(self.LABEL, self.EXPERIMENT, self.SOURCE_INDICES, self.SEED,
                                     self.SPECS, self.GATE_PAIRS, (("ours_correct", "sf_fifo21"),), self.MECHANISM,
                                     nodes=self.AUTHORIZED_NODES, binding={"selection_sha256": self.sha256(path)})
        self.spec_for = partial(base.spec_for, campaign=self.CAMPAIGN)
        self.output_root = partial(base.output_root, campaign=self.CAMPAIGN)
        self.require_baseline = partial(require_baseline, label=self.LABEL, sources=self.GATE_SOURCES)
        self.audit = development.audit
        self.check_scope()

    def build_specs(self, candidate):
        return {"sf_fifo21": dict(development.SPECS["sf_fifo21"]),
                "ours_correct": dict(development.SPECS[candidate])}

    def check_scope(self):
        path = self.out / "inputs/selection.json"
        if self.sha256(path) != self.CAMPAIGN.binding["selection_sha256"]:
            raise ValueError("selection changed after loading")
        for name, digest in self.scope["evidence_sha256"].items():
            if name not in {"inputs.json", "comparison.json", "report.json", "summary.json"}:
                raise ValueError("unexpected selection evidence file")
            if self.sha256(self.out / "inputs/development" / name) != digest:
                raise ValueError("frozen development evidence changed")

    def assignment(self, source, slots):
        if tuple(slots) != tuple(map(str, range(8))):
            raise ValueError(f"{self.LABEL} requires GPU slots 0..7 on each of eight nodes")
        return base.assignment(source, slots, sources=self.SOURCE_INDICES, num_nodes=8)

    def method_order(self, source):
        position = self.SOURCE_INDICES.index(source)
        return self.METHODS if (position + position // 8) % 2 == 0 else self.METHODS[::-1]

    def validate_node(self, rank, num_nodes=8):
        from run_v211_worker import assert_authorized_node
        address = os.environ.get(f"{self.LABEL.upper()}_NODE_ADDRESS")
        if num_nodes != 8 or not 0 <= rank < 8 or address != self.AUTHORIZED_NODES[rank]:
            raise PermissionError(f"NODE_RANK/{self.LABEL.upper()}_NODE_ADDRESS must match the frozen eight-node list")
        return assert_authorized_node({"authorized_nodes": list(self.AUTHORIZED_NODES)}, node_address=address)

    def validate_vbench_fingerprint(self, fingerprint):
        expected = read(self.out / "inputs/development/comparison.json")["vbench_fingerprint"]
        if fingerprint != expected:
            raise ValueError("v216 evaluator differs from v215; do not pool results across evaluators")

    def validate_checkpoint(self, digest):
        repair = read(self.out / "inputs/development/report.json").get("evaluation_repair")
        if repair is not None and digest != repair["raft_checkpoint_sha256"]:
            raise ValueError("RAFT checkpoint differs from the repaired v215 evaluation")

    def _check_inputs(self, data):
        previous = read(self.out / "inputs/development/inputs.json")
        if (data["prompt_source"]["sha256"] != previous["prompt_source"]["sha256"]
                or data["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]
                or data["wan_model"]["inventory"] != previous["wan_model"]["inventory"]):
            raise ValueError("v216 prompt/model contract differs from v215")
        for local, old in (("sf_fifo21", "sf_fifo21"), ("ours_correct", self.scope["selected_method"])):
            if data["configs"][local]["sha256"] != previous["configs"][old]["sha256"]:
                raise ValueError("v216 inference config differs from selected v215 method")
        if inference_paths(data["runtime_paths"]) != inference_paths(previous["runtime_paths"]):
            raise ValueError("model/cache operator changed since v215; this is not the same frozen method")

    def prepare(self, repo, out, prompts, checkpoint, wan, slots, *, clean=True):
        self.check_scope()
        self.assignment(self.SOURCE_INDICES[0], slots)
        data = base.prepare(repo, out, prompts, checkpoint, wan, slots, clean=clean, campaign=self.CAMPAIGN)
        self._check_inputs(data)
        self.frozen_json(self.out / "inputs/placement.json", self.placement(slots))
        return data

    def placement(self, slots):
        return {"node_count": 8, "gpu_slots": list(slots), "main_video_count": len(self.METHODS)*len(self.SOURCE_INDICES),
                "jobs": [{"source_index": s, "node_rank": self.assignment(s, slots)[0],
                          "gpu": self.assignment(s, slots)[1], "methods": list(self.method_order(s)),
                          "effective_seed": self.SEED + s} for s in self.SOURCE_INDICES]}

    def verify(self, repo, out, *, runtime=True):
        self.check_scope()
        data = base.verify(repo, out, runtime=runtime, campaign=self.CAMPAIGN)
        self._check_inputs(data)
        if read(self.out / "inputs/placement.json") != self.placement(tuple(data["gpu_slots"])):
            raise ValueError("v216 placement drift")
        return data


def load(out=None):
    path = out or os.environ.get("V216_OUT_ROOT")
    if not path:
        raise ValueError("set V216_OUT_ROOT to the frozen confirmation output directory")
    return Protocol(path)
