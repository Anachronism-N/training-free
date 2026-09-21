#!/usr/bin/env python3
"""Frozen matched-history experiment; no changes to the LPHC forward path."""
from __future__ import annotations

import copy
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from prepare_v211_lphc import (
    AUTHORIZED_NODES, DEFAULT_CHECKPOINT, DEFAULT_PROMPT_SOURCE, DEFAULT_WAN_MODEL,
    file_stamp, git_commit, merged, require_clean_checkout, sha256, wan_inventory,
    verify_wan_inventory, write_frozen,
)
from audit_v210_lphc_trace import audit_trace as audit_fifo
from audit_v211_lphc_trace import audit_trace as audit_sink

EXPERIMENT = "v212_lphc_matched_history32"
SOURCE_INDICES = tuple(range(3, 128, 4))
SEED = 21200
FRAMES = 120
GATE_SOURCES = (3, 67)
METHODS = ("sf_fifo21", "sf_sink1_21", "sf_fifo25", "fifo_correct", "fifo_random",
           "sink_correct", "sink_random")
SPECS = {
    "sf_fifo21": {"window": 21, "sink": 0},
    "sf_sink1_21": {"window": 21, "sink": 1},
    "sf_fifo25": {"window": 25, "sink": 0},
}
for _policy, _sink, _protocol in (("fifo", 0, "v210"), ("sink", 1, "v211")):
    for _mode in ("correct", "random"):
        SPECS[f"{_policy}_{_mode}"] = {
            "window": 21, "sink": _sink, "lphc": True, "alpha": .02, "phase": "e1",
            "retrieval_mode": _mode, "archive_frames": 12, "history_frames": 4,
            "protocol": _protocol, "local_policy": "sink1_recent20" if _sink else "fifo21",
        }


@dataclass(frozen=True)
class Campaign:
    label: str
    experiment: str
    sources: tuple[int, ...]
    seed: int
    specs: dict
    gate_pairs: tuple[tuple[str, str], ...]
    primary: tuple[tuple[str, str], ...]
    mechanism: tuple[tuple[str, str], ...]
    nodes: tuple[str, ...] = tuple(AUTHORIZED_NODES)
    binding: dict | None = None
    frames: int = FRAMES

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.specs)


LABEL = "v212"
GATE_PAIRS = (("sf_fifo21", "fifo_zero"), ("sf_sink1_21", "sink_zero"))
CAMPAIGN = Campaign(LABEL, EXPERIMENT, SOURCE_INDICES, SEED, SPECS, GATE_PAIRS,
                    (("fifo_correct", "sf_fifo21"), ("sink_correct", "sf_sink1_21")),
                    (("fifo_correct", "fifo_random"), ("sink_correct", "sink_random")))


def frozen_json(path: Path, value: dict) -> None:
    write_frozen(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def output_root(path: Path, *, campaign: Campaign = CAMPAIGN) -> Path:
    path = path.expanduser().resolve()
    if not path.name.startswith(campaign.label + "_"):
        raise ValueError(f"use a new output directory whose basename starts with {campaign.label}_")
    return path


def runtime_hashes(repo: Path) -> dict:
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "src", "scripts", "third_party/Self-Forcing"], cwd=repo
    ).decode().split("\0")
    return {name: sha256(repo / name) for name in sorted(names)
            if name and Path(name).suffix in {".py", ".sh", ".yaml", ".json", ".cu", ".cpp", ".h"}}


def spec_for(method: str, stage: str, *, campaign: Campaign = CAMPAIGN) -> dict:
    if stage == "gate0" and method in {pair[1] for pair in campaign.gate_pairs}:
        return {**campaign.specs[method.replace("zero", "correct")], "alpha": 0.0}
    if method not in campaign.methods:
        raise ValueError(f"unknown method: {method}")
    return dict(campaign.specs[method])


def assignment(source: int, slots: tuple[str, ...], *, sources=SOURCE_INDICES, num_nodes=6) -> tuple[int, str]:
    if not slots or len(set(slots)) != len(slots) or not set(slots) <= set(map(str, range(8))):
        raise ValueError("GPU slots must be distinct members of 0..7")
    if num_nodes <= 0:
        raise ValueError("positive node count required")
    position = sources.index(source)
    return position % num_nodes, slots[(position // num_nodes) % len(slots)]


def method_order(source: int) -> tuple[str, ...]:
    shift = SOURCE_INDICES.index(source) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def validate_node(rank: int, num_nodes: int = 6) -> str:
    from run_v211_worker import assert_authorized_node
    address = os.environ.get("V212_NODE_ADDRESS")
    if num_nodes != 6 or not 0 <= rank < 6 or address != AUTHORIZED_NODES[rank]:
        raise PermissionError("set V212_NODE_ADDRESS to the frozen address for this NODE_RANK")
    return assert_authorized_node({"authorized_nodes": list(AUTHORIZED_NODES)}, node_address=address)


def prepare(repo: Path, out: Path, prompts: Path, checkpoint: Path, wan: Path,
            slots: tuple[str, ...], *, clean: bool = True, campaign: Campaign = CAMPAIGN) -> dict:
    out = output_root(out, campaign=campaign)
    assignment(campaign.sources[0], slots, sources=campaign.sources, num_nodes=len(campaign.nodes))
    if clean:
        require_clean_checkout(repo)
    lines = prompts.read_text(encoding="utf-8").splitlines()
    if len(lines) != 128 or any(not line.strip() for line in lines):
        raise ValueError("expected exactly 128 non-empty rewritten MovieGen prompts")
    old_path = out / "inputs/manifest.json"
    if old_path.exists():
        current = verify(repo, out, campaign=campaign)
        if (current["prompt_source"]["sha256"] != sha256(prompts)
                or current["checkpoint"]["path"] != str(checkpoint.resolve())
                or current["wan_model"]["weights_path"] != str(wan.resolve())
                or current["gpu_slots"] != list(slots)):
            raise ValueError("prepare arguments changed; use a new output root")
        return current
    base_path = repo / "third_party/Self-Forcing/configs"
    load = lambda name: yaml.safe_load((base_path / name).read_text(encoding="utf-8"))
    base = merged(load("default_config.yaml"), load("self_forcing_dmd.yaml"))
    base.update(use_pyramidkv=False, use_teacache=False, compile_ffn=False,
                vae_decode_mode="batch", independent_first_frame=False, few_step_cfg_enabled=False)
    configs = {}
    for method in (*campaign.methods, *(pair[1] for pair in campaign.gate_pairs)):
        spec = spec_for(method, "gate0", campaign=campaign)
        config = copy.deepcopy(base)
        config["model_kwargs"].update(local_attn_size=spec["window"], sink_size=spec["sink"])
        path = out / "inputs/configs" / f"{method}.yaml"
        digest = write_frozen(path, yaml.safe_dump(config, sort_keys=True).encode())
        configs[method] = {"path": str(path), "sha256": digest}
    items = []
    for index, source in enumerate(campaign.sources):
        text = lines[source].strip()
        path = out / "inputs/prompts" / f"source_{source:03d}.txt"
        digest = write_frozen(path, (text + "\n").encode())
        items.append({"index": index, "source_index": source, "effective_seed": campaign.seed + source,
                      "text": text, "path": str(path), "sha256": digest})
    payload = {"version": 1, "experiment": campaign.experiment, "source_commit": git_commit(repo),
               "runtime_paths": runtime_hashes(repo), "source_indices": list(campaign.sources),
               "methods": list(campaign.methods), "specs": campaign.specs, "prompt_items": items,
               "prompt_source": file_stamp(prompts), "checkpoint": file_stamp(checkpoint),
               "wan_model": {"weights_path": str(wan.resolve()), "inventory": wan_inventory(wan)},
               "configs": configs, "gpu_slots": list(slots), "authorized_nodes": list(campaign.nodes),
               "frames": campaign.frames, "base_seed": campaign.seed,
               "placement": "prompt bundle on one GPU; rotating method order; " +
                            ("six nodes" if len(campaign.nodes) == 6 else f"{len(campaign.nodes)} nodes"),
               "primary_contrasts": [list(pair) for pair in campaign.primary],
               "mechanism_contrasts": [list(pair) for pair in campaign.mechanism],
               "development_only": True, "paper_claim_ready": False}
    if campaign.binding is not None:
        payload["campaign_binding"] = campaign.binding
    frozen_json(old_path, payload)
    return payload


def verify(repo: Path, out: Path, *, runtime: bool = True, campaign: Campaign = CAMPAIGN) -> dict:
    out = output_root(out, campaign=campaign)
    data = json.loads((out / "inputs/manifest.json").read_text(encoding="utf-8"))
    if (data.get("experiment") != campaign.experiment or data.get("source_indices") != list(campaign.sources)
            or data.get("methods") != list(campaign.methods) or data.get("specs") != campaign.specs
            or data.get("authorized_nodes") != list(campaign.nodes)
            or data.get("campaign_binding") != campaign.binding
            or data.get("base_seed") != campaign.seed or data.get("frames") != campaign.frames
            or data.get("primary_contrasts") != [list(pair) for pair in campaign.primary]
            or data.get("mechanism_contrasts") != [list(pair) for pair in campaign.mechanism]):
        raise ValueError(f"{campaign.label} frozen protocol mismatch")
    assignment(campaign.sources[0], tuple(data["gpu_slots"]), sources=campaign.sources, num_nodes=len(campaign.nodes))
    if runtime and (data["source_commit"] != git_commit(repo) or data["runtime_paths"] != runtime_hashes(repo)):
        raise ValueError("source drift; use the frozen checkout, not an updated running checkout")
    for row in [data["prompt_source"], *data["configs"].values(), *data["prompt_items"]]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"input content drift: {row['path']}")
    if [(x["source_index"], x["effective_seed"]) for x in data["prompt_items"]] != [(s, campaign.seed+s) for s in campaign.sources]:
        raise ValueError("prompt seed/membership drift")
    checkpoint = Path(data["checkpoint"]["path"])
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (data["checkpoint"]["bytes"], data["checkpoint"]["mtime_ns"]):
        raise ValueError("checkpoint stamp drift")
    verify_wan_inventory(Path(data["wan_model"]["weights_path"]), data["wan_model"]["inventory"], check_hashes=False)
    return data


def audit(path: Path, spec: dict, blocks: int, source: int) -> dict:
    kwargs = {"phase": spec["phase"], "expected_blocks": blocks, "expected_layers": 30}
    if spec["sink"]:
        report = audit_sink(path, spec["alpha"], allow_random=spec["retrieval_mode"] == "random", **kwargs)
    else:
        report = audit_fifo(path, spec["alpha"], **kwargs)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    headers = [row for row in rows if row.get("event") == "video_start"]
    if len(headers) != 1 or headers[0].get("source_index") != source:
        report["errors"].append("source identity missing/mismatched in trace")
    elif any(headers[0].get(key) != value for key, value in {
        "alpha": spec["alpha"], "schedule": spec["phase"], "retrieval_mode": spec["retrieval_mode"]
    }.items()):
        report["errors"].append("trace header differs from requested intervention")
    random = report["totals"]["random"]
    expected_random = spec["retrieval_mode"] == "random" and spec["alpha"] > 0
    if (expected_random and random <= 0) or (not expected_random and random != 0):
        report["errors"].append("random history mode did not execute as requested")
    layers = {}
    for row in rows:
        if row.get("event") != "attention_call" or row.get("call_kind") != "noisy" or row.get("phase_index") != 0:
            continue
        layer = str(row.get("layer_idx", row.get("layer_id")))
        summary = layers.setdefault(layer, {"calls": 0, "nonempty_calls": 0, "ratios": [], "ages": []})
        selected = row.get("selected_history_frames", row.get("selected_frame_ids", []))
        local = row.get("local_frame_indices", row.get("local_frame_ids", []))
        summary["calls"] += 1
        summary["nonempty_calls"] += int(bool(selected))
        summary["ratios"].append(float(row.get("correction_ratio", 0)))
        if local and selected:
            summary["ages"].extend(max(local) - frame for frame in selected)
    report["exposure_by_layer"] = {layer: {
        "early_calls": value["calls"], "nonempty_calls": value["nonempty_calls"],
        "mean_correction_ratio": statistics.mean(value["ratios"]),
        "max_correction_ratio": max(value["ratios"]),
        "mean_age_latent_frames": statistics.mean(value["ages"]) if value["ages"] else None,
        "max_age_latent_frames": max(value["ages"]) if value["ages"] else None,
    } for layer, value in layers.items()}
    phase_exposure = {}
    for row in rows:
        if row.get("event") != "attention_call" or row.get("call_kind") != "noisy":
            continue
        key = f"{row.get('layer_idx', row.get('layer_id'))}:{row.get('phase_index')}"
        sample = phase_exposure.setdefault(key, {"calls": 0, "nonempty_calls": 0,
                                                "ratio_sum": 0., "max_correction_ratio": 0.})
        ratio = float(row.get("correction_ratio", 0.))
        sample["calls"] += 1
        sample["nonempty_calls"] += int(bool(row.get("selected_history_frames", row.get("selected_frame_ids", []))))
        sample["ratio_sum"] += ratio
        sample["max_correction_ratio"] = max(sample["max_correction_ratio"], ratio)
    for sample in phase_exposure.values():
        sample["mean_correction_ratio"] = sample.pop("ratio_sum") / sample["calls"]
    report["exposure_by_layer_phase"] = phase_exposure
    report["pass"] = not report["errors"]
    return report


def load_protocol(name: str, out: Path | None = None):
    import importlib
    if name in {"v216", "v217", "v219", "v220"}:
        return importlib.import_module(f"{name}_lphc_protocol").load(out)
    if name not in {"v212", "v213", "v214", "v215"}:
        raise ValueError("campaign must be v212, v213, v214 or v215")
    return importlib.import_module(f"{name}_lphc_protocol")
