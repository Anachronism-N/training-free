#!/usr/bin/env python3
"""Pinned official-SF reference and small, fail-closed baseline certificate."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from v212_lphc_protocol import sha256

UPSTREAM_URL = "https://github.com/guandeh17/Self-Forcing.git"
UPSTREAM_COMMIT = "33593df3e81fa3ec10239271dd2c100facac6de1"
SOURCES = (3, 67)
MODES = ("upstream_a", "local", "upstream_b")
COUNTS = {"input_noise": 1, "conditioning": 1, "generator_input": 50,
          "generator_output": 50, "final_latents": 1, "decoded_sample": 1}
REL_TOL = 1e-5
ABS_TOL = 5e-4


def reference_inventory(root: Path) -> dict:
    root = root.resolve()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError(f"official SF must be pinned to {UPSTREAM_COMMIT}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official SF checkout has tracked modifications")
    tracked = set(git("ls-files", "-z").split("\0"))
    for folder in ("pipeline", "utils", "wan", "demo_utils"):
        for path in (root / folder).rglob("*.py"):
            if path.relative_to(root).as_posix() not in tracked:
                raise ValueError(f"untracked Python code in official SF: {path}")
    paths = {name: sha256(root / name) for name in sorted(tracked)
             if name and Path(name).suffix in {".py", ".yaml", ".json", ".cu", ".cpp", ".h"}}
    if "pipeline/causal_inference.py" not in paths or "wan/modules/causal_model.py" not in paths:
        raise ValueError("incomplete official SF checkout")
    return {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "path": str(root), "files": paths}


def validate_config(config: dict) -> None:
    expected = {"denoising_step_list": [1000, 750, 500, 250], "warp_denoising_step": True,
                "num_frame_per_block": 3, "independent_first_frame": False,
                "context_noise": 0, "few_step_cfg_enabled": False, "use_pyramidkv": False,
                "use_teacache": False, "compile_ffn": False, "vae_decode_mode": "batch"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unexpected SF baseline setting: {key}={config.get(key)!r}")
    kwargs = config.get("model_kwargs", {})
    if kwargs != {"local_attn_size": 21, "sink_size": 0, "timestep_shift": 5.0}:
        raise ValueError(f"unexpected SF model_kwargs: {kwargs}")


def config_alignment(repo: Path, upstream: Path, frozen_config: Path) -> dict:
    config = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    validate_config(config)
    fields = ("denoising_step_list", "warp_denoising_step", "num_frame_per_block")
    references = {}
    for label, path in (("official_sf", upstream / "configs/self_forcing_dmd.yaml"),
                        ("vendored_pf_plain_sf", repo / "third_party/Pyramid-Forcing/configs/self-forcing.yaml")):
        reference = yaml.safe_load(path.read_text(encoding="utf-8"))
        if any(config[key] != reference[key] for key in fields):
            raise ValueError(f"sampling config mismatch with {label}")
        if config["model_kwargs"]["timestep_shift"] != reference["model_kwargs"]["timestep_shift"]:
            raise ValueError(f"scheduler shift mismatch with {label}")
        references[label] = {"path": str(path), "sha256": sha256(path)}
    return {"pass": True, "references": references,
            "scope": "sampling fields match; explicit FIFO21/sink0, not PF runtime/metric reproduction"}


def expected_records() -> set[tuple[str, int]]:
    return {(event, i) for event, count in COUNTS.items() for i in range(count)}


def require_baseline(out: Path, *, label="v213", sources=SOURCES) -> dict:
    path = out / "decisions/sf_upstream_gate.json"
    if not path.is_file():
        raise ValueError(f"run {label} baseline before gate0/smoke/generation")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("pass") is not True or report.get("upstream_commit") != UPSTREAM_COMMIT
            or report.get("input_manifest_sha256") != sha256(out / "inputs/manifest.json")
            or report.get("contract_sha256") != sha256(out / "baseline/contract.json")):
        raise ValueError("official SF baseline gate failed or belongs to different inputs")
    jobs = report.get("jobs", [])
    if (len(jobs) != len(sources) * len(MODES)
            or {(r["source"], r["mode"]) for r in jobs} != {(s, m) for s in sources for m in MODES}
            or len(report.get("comparisons", [])) != 2 * len(sources)
            or {(r["source"], r["kind"]) for r in report["comparisons"]}
            != {(s, k) for s in sources for k in ("official_repeat", "local_vs_official")}
            or not all(row.get("pass") is True for row in report["comparisons"])):
        raise ValueError("official SF gate coverage incomplete")
    for row in jobs:
        expected = out / "baseline/jobs" / f"source_{row['source']:03d}" / row["mode"] / "done.json"
        if Path(row["path"]).resolve() != expected.resolve() or sha256(expected) != row["sha256"]:
            raise ValueError("official SF gate receipt changed")
    return report
