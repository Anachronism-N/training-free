#!/usr/bin/env python3
"""Freeze the source-bound v210 LPHC evaluation contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

EXPERIMENT = "v210_lphc"
IMMUTABLE_V209_EXPERIMENT = "v209_sf_protocol_budget_86f10607_sixnode"
DEFAULT_PROMPT_SOURCE = Path(
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/"
    "Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
DEFAULT_CHECKPOINT = Path(
    "/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt"
)
DEFAULT_WAN_MODEL = Path(
    "/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B"
)
SOURCE_INDICES = (1, 17, 33, 49, 65, 81, 97, 113)
BASE_SEED = 21000
SCREEN_FRAMES = 120
GATE_FRAMES = 30
ARCHIVE_SIZE = 12
HISTORY_SIZE = 4
METHODS = (
    "sf_fifo21",
    "sf_sink1_21",
    "sf_fifo25",
    "lphc_e1_a002_correct",
    "lphc_e1_a005_correct",
    "lphc_e1_a010_correct",
    "lphc_e2_a010_correct",
    "lphc_full_a010_correct",
)
METHOD_SPECS = {
    "sf_fifo21": {"local_attn_size": 21, "sink_size": 0, "lphc": False},
    "sf_sink1_21": {"local_attn_size": 21, "sink_size": 1, "lphc": False},
    "sf_fifo25": {"local_attn_size": 25, "sink_size": 0, "lphc": False},
    "lphc_e1_a002_correct": {"local_attn_size": 21, "sink_size": 0, "lphc": True, "phase": "e1", "alpha": 0.02, "retrieval_mode": "correct"},
    "lphc_e1_a005_correct": {"local_attn_size": 21, "sink_size": 0, "lphc": True, "phase": "e1", "alpha": 0.05, "retrieval_mode": "correct"},
    "lphc_e1_a010_correct": {"local_attn_size": 21, "sink_size": 0, "lphc": True, "phase": "e1", "alpha": 0.10, "retrieval_mode": "correct"},
    "lphc_e2_a010_correct": {"local_attn_size": 21, "sink_size": 0, "lphc": True, "phase": "e2", "alpha": 0.10, "retrieval_mode": "correct"},
    "lphc_full_a010_correct": {"local_attn_size": 21, "sink_size": 0, "lphc": True, "phase": "full", "alpha": 0.10, "retrieval_mode": "correct"},
}
GATE0_METHODS = {
    "native": {"config": "sf_fifo21", "lphc": False, "phase": "full", "alpha": 0.0, "retrieval_mode": "correct"},
    "lphc_alpha0": {"config": "sf_fifo21", "lphc": True, "phase": "full", "alpha": 0.0, "retrieval_mode": "correct"},
}
GATE0_SOURCE_INDICES = (1, 65)
RUNTIME_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".csv", ".cu", ".cpp", ".h", ".hpp", ".sh"}
V210_RUNTIME_FILES = (
    "src/lifecycle_kv/lphc.py",
    "scripts/prepare_v210_lphc.py",
    "scripts/run_v210_worker.py",
    "scripts/run_v210_lphc.py",
    "scripts/audit_v210_lphc_trace.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v210 input differs: {path}")
    if not path.exists():
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def merged(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def short_git_commit(root: Path) -> str:
    return git_commit(root)[:8]


def default_output_root(root: Path) -> Path:
    return root / "runs" / f"v210_lphc_{short_git_commit(root)}_sixnode"


def validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if IMMUTABLE_V209_EXPERIMENT in str(resolved):
        raise ValueError(f"v209 output is immutable: {resolved}")
    return resolved


def effective_seed(source_index: int) -> int:
    if source_index not in SOURCE_INDICES:
        raise ValueError(f"source index is outside the frozen v210 suite: {source_index}")
    return BASE_SEED + source_index


def _git_inventory(root: Path) -> set[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "third_party/Self-Forcing", "src"], cwd=root
    ).decode().split("\0")
    names = {name for name in tracked if name and Path(name).suffix in RUNTIME_EXTENSIONS}
    for name in V210_RUNTIME_FILES:
        if (root / name).is_file():
            names.add(name)
    return names


def source_hashes(root: Path, names: Iterable[str] | None = None) -> dict[str, str]:
    selected = set(names) if names is not None else _git_inventory(root)
    return {name: sha256(root / name) for name in sorted(selected)}


def _checkpoint_row(checkpoint: Path, old: dict | None) -> dict:
    stat = checkpoint.stat()
    row = {"path": str(checkpoint.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if old and all(old.get(key) == value for key, value in row.items()):
        row["sha256"] = old["sha256"]
    else:
        row["sha256"] = sha256(checkpoint)
    return row


def _normalize_nodes(nodes: Iterable[str]) -> list[str]:
    result = list(dict.fromkeys(node.strip() for node in nodes if node.strip()))
    if len(result) != 6 or any(any(char.isspace() for char in node) for node in result):
        raise ValueError("require exactly six distinct authorized node hostnames")
    return result


def prepare(
    root: Path,
    source: Path,
    checkpoint: Path,
    output: Path,
    authorized_nodes: Iterable[str],
    wan_model: Path = DEFAULT_WAN_MODEL,
) -> dict:
    root = root.resolve()
    output = validate_output_root(output)
    nodes = _normalize_nodes(authorized_nodes)
    prompts = source.read_text(encoding="utf-8").splitlines()
    if len(prompts) != 128 or any(not prompt.strip() for prompt in prompts):
        raise ValueError("require the 128 non-empty Qwen MovieGen prompts")
    selected = {index: prompts[index].strip() for index in SOURCE_INDICES}
    prompt_items = []
    for index in SOURCE_INDICES:
        path = output / "prompts" / f"source_{index:03d}.txt"
        digest = write_frozen(path, (selected[index] + "\n").encode("utf-8"))
        prompt_items.append({
            "source_index": index,
            "effective_seed": effective_seed(index),
            "text": selected[index],
            "path": str(path.resolve()),
            "sha256": digest,
        })

    sf_root = root / "third_party" / "Self-Forcing"
    wan_root = sf_root / "wan"
    wan_weights = wan_model.expanduser().resolve()
    required_wan_files = (wan_root / "__init__.py", wan_root / "configs" / "wan_t2v_14B.py")
    if not wan_root.is_dir() or not all(path.is_file() for path in required_wan_files):
        raise FileNotFoundError(f"Wan runtime is absent or incomplete: {wan_root}")
    if not wan_weights.is_dir():
        raise FileNotFoundError(f"Wan model is absent: {wan_weights}")
    load = lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
    base = merged(load(sf_root / "configs/default_config.yaml"), load(sf_root / "configs/self_forcing_dmd.yaml"))
    base.update(
        use_pyramidkv=False,
        use_teacache=False,
        compile_ffn=False,
        vae_decode_mode="batch",
        independent_first_frame=False,
        few_step_cfg_enabled=False,
    )
    config_files = {}
    for method in METHODS:
        spec = METHOD_SPECS[method]
        config = copy.deepcopy(base)
        config["model_kwargs"].update(
            local_attn_size=spec["local_attn_size"], sink_size=spec["sink_size"]
        )
        config["lphc"] = {
            "enabled": bool(spec["lphc"]),
            "alpha": float(spec.get("alpha", 0.0)),
            "phase": spec.get("phase", "off"),
            "retrieval_mode": spec.get("retrieval_mode", "off"),
            "archive_frames": ARCHIVE_SIZE,
            "history_frames": HISTORY_SIZE,
        }
        path = output / "configs" / f"{method}.yaml"
        digest = write_frozen(
            path, yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
        )
        config_files[method] = {"path": str(path.resolve()), "sha256": digest}

    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    commit = git_commit(root)
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_commit": commit,
        "git_commit": commit,
        "runtime_paths": source_hashes(root),
        "prompt_source": {"path": str(source.resolve()), "sha256": sha256(source), "count": 128},
        "checkpoint": _checkpoint_row(checkpoint, old.get("checkpoint") if old else None),
        "wan_model": {
            "name": "Wan2.1-T2V-1.3B",
            "runtime_path": str(wan_root.resolve()),
            "required_files": [str(path.resolve()) for path in required_wan_files],
            "runtime_present": True,
            "weights_path": str(wan_weights.resolve()),
            "weights_present": True,
        },
        "authorized_nodes": nodes,
        "execution": {
            "node_count": 6,
            "node_rank_by_hostname": {node: rank for rank, node in enumerate(nodes)},
            "screen_assignment": "method-major jobs round-robin over node-rank/GPU slots",
        },
        "source_indices": list(SOURCE_INDICES),
        "prompt_count": len(SOURCE_INDICES),
        "prompt_items": prompt_items,
        "base_seed": BASE_SEED,
        "seed_policy": "effective_seed = 21000 + source_index",
        "screen_frames": SCREEN_FRAMES,
        "gate_frames": GATE_FRAMES,
        "methods": list(METHODS),
        "method_specs": METHOD_SPECS,
        "configs": config_files,
        "lphc_defaults": {"archive_size": ARCHIVE_SIZE, "history_size": HISTORY_SIZE},
        "gate0": {
            "source_indices": list(GATE0_SOURCE_INDICES),
            "modes": GATE0_METHODS,
            "attention": "production",
            "same_gpu_sequential": True,
        },
        "claim_boundary": "Source-bound v210 screen. LPHC claims require gate0 native equivalence and a passing smoke trace audit; production attention only.",
    }
    write_frozen(manifest_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return payload


def verify(
    path: Path,
    root: Path,
    *,
    check_runtime: bool = True,
    check_checkpoint_hash: bool = False,
) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("experiment") != EXPERIMENT
        or data.get("source_indices") != list(SOURCE_INDICES)
        or data.get("methods") != list(METHODS)
        or data.get("base_seed") != BASE_SEED
    ):
        raise ValueError("invalid v210 manifest")
    validate_output_root(path)
    if data.get("source_commit") != git_commit(root):
        raise ValueError("v210 source commit drift; use a new output root")
    source = Path(data["prompt_source"]["path"])
    if sha256(source) != data["prompt_source"]["sha256"]:
        raise ValueError("v210 source prompt drift")
    for item in data["prompt_items"]:
        prompt_path = Path(item["path"])
        if sha256(prompt_path) != item["sha256"] or item["effective_seed"] != effective_seed(item["source_index"]):
            raise ValueError("v210 frozen prompt drift")
    for row in data["configs"].values():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("v210 config drift")
    checkpoint = Path(data["checkpoint"]["path"])
    stat = checkpoint.stat()
    if stat.st_size != data["checkpoint"]["bytes"] or stat.st_mtime_ns != data["checkpoint"]["mtime_ns"]:
        raise ValueError("v210 checkpoint stamp drift")
    if check_checkpoint_hash and sha256(checkpoint) != data["checkpoint"]["sha256"]:
        raise ValueError("v210 checkpoint hash drift")
    if not all(Path(item).is_file() for item in data["wan_model"]["required_files"]):
        raise ValueError("v210 Wan runtime drift")
    if Path(data["wan_model"]["weights_path"]).is_dir() != data["wan_model"]["weights_present"]:
        raise ValueError("v210 Wan model presence drift")
    if check_runtime:
        current = source_hashes(root, data["runtime_paths"])
        if current != data["runtime_paths"]:
            raise ValueError("v210 runtime drift; use a new output root")
    _normalize_nodes(data.get("authorized_nodes", ()))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-prompts", type=Path, default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wan-model", type=Path, default=DEFAULT_WAN_MODEL)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--authorized-nodes",
        default=os.environ.get("V210_AUTHORIZED_NODES", ""),
        help="Comma-separated exact hostnames; required (or set V210_AUTHORIZED_NODES).",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_root or default_output_root(root)
    data = prepare(
        root, args.source_prompts, args.checkpoint, output,
        args.authorized_nodes.split(","), args.wan_model,
    )
    print(f"[v210-prepare] root={validate_output_root(output)} prompts={data['prompt_count']} methods={len(METHODS)}")


if __name__ == "__main__":
    main()
