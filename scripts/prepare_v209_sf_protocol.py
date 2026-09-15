#!/usr/bin/env python3
"""Freeze explicit SF sink/window controls and post-eviction canaries."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml

from prepare_v207_context_budget_phase_screen import SOURCE_INDICES, sha256
from prepare_v208_paper_confirmation import write_frozen


EXPERIMENT = "v209_sf_protocol_budget"
METHODS = ("sf_fifo21", "sf_sink1_21", "sf_sink1_13", "sf_sink1_9", "sf_sink3_9")
SPECS = {"sf_fifo21": (21, 0), "sf_sink1_21": (21, 1), "sf_sink1_13": (13, 1), "sf_sink1_9": (9, 1), "sf_sink3_9": (9, 3)}
CANARY_PROMPTS = (0, 16)
SEED = 20700
CANARY_RUNS = {
    "sf_fifo21_a": ("sf_fifo21", "sf"), "sf_fifo21_b": ("sf_fifo21", "sf"),
    "sf_sink1_21_a": ("sf_sink1_21", "sf"), "sf_sink1_21_b": ("sf_sink1_21", "sf"),
    "sf_sink1_13": ("sf_sink1_13", "sf"), "sf_sink1_9": ("sf_sink1_9", "sf"),
    "sf_sink3_9": ("sf_sink3_9", "sf"),
    "pf_fifo21": ("sf_fifo21", "pf"), "pf_sink1_21": ("sf_sink1_21", "pf"),
    "adaptive_recent9_landmark_a": ("adaptive_recent9_landmark", "pf"),
    "adaptive_recent9_landmark_b": ("adaptive_recent9_landmark", "pf"),
    "adaptive_recent9_retrieval": ("adaptive_recent9_retrieval", "pf"),
    "adaptive_recent21_retrieval": ("adaptive_recent21_retrieval", "pf"),
}


def merged(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merged(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def source_hashes(root: Path) -> dict[str, str]:
    names = subprocess.check_output([
        "git", "ls-files", "-z", "third_party/Self-Forcing", "third_party/Pyramid-Forcing",
        "src/lifecycle_kv", "scripts/*v209*",
    ], cwd=root).decode().split("\0")
    suffixes = {".py", ".yaml", ".yml", ".json", ".csv", ".cu", ".cpp", ".h", ".hpp", ".sh"}
    return {name: sha256(root / name) for name in sorted(names) if name and Path(name).suffix in suffixes}


def prepare(root: Path, source: Path, checkpoint: Path, output: Path) -> dict:
    prompts = source.read_text(encoding="utf-8").splitlines()
    if len(prompts) != 128 or any(not value.strip() for value in prompts):
        raise ValueError("require the 128 non-empty Qwen MovieGen prompts")
    prompt_path = output / "prompts.txt"
    selected = [prompts[index].strip() for index in SOURCE_INDICES]
    write_frozen(prompt_path, ("\n".join(selected) + "\n").encode())
    sf = root / "third_party/Self-Forcing"
    pf = root / "third_party/Pyramid-Forcing"
    load = lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
    base = merged(load(sf / "configs/default_config.yaml"), load(sf / "configs/self_forcing_dmd.yaml"))
    base.update(use_pyramidkv=False, use_teacache=False, compile_ffn=False, vae_decode_mode="batch", independent_first_frame=False, few_step_cfg_enabled=False)
    configs = {}
    for method, (budget, sink) in SPECS.items():
        config = copy.deepcopy(base)
        config["model_kwargs"].update(local_attn_size=budget, sink_size=sink)
        configs[method] = config
    adaptive = merged(load(pf / "configs/default_config.yaml"), load(pf / "configs/pyramid-forcing.yaml"))
    for key in ("denoising_step_list", "warp_denoising_step", "timestep_shift", "num_frame_per_block", "independent_first_frame", "negative_prompt", "mixed_precision", "few_step_cfg_enabled", "compile_ffn", "vae_decode_mode"):
        adaptive[key] = base[key]
    adaptive["model_kwargs"] = copy.deepcopy(configs["sf_sink1_21"]["model_kwargs"])
    adaptive.update(use_pyramidkv=True, use_teacache=False)
    for name in ("adaptive_recent9_landmark", "adaptive_recent9_retrieval", "adaptive_recent21_retrieval"):
        configs[name] = copy.deepcopy(adaptive)
    config_files = {}
    for name, config in configs.items():
        path = output / "configs" / f"{name}.yaml"
        digest = write_frozen(path, yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode())
        config_files[name] = {"path": str(path.resolve()), "sha256": digest}
    map_path = output / "all_heads.csv"
    write_frozen(map_path, ((",".join(["10"] * 12) + "\n") * 30).encode())
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    stat = checkpoint.stat()
    checkpoint_row = {"path": str(checkpoint.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if old is not None and all(old["checkpoint"].get(key) == value for key, value in checkpoint_row.items()):
        checkpoint_row["sha256"] = old["checkpoint"]["sha256"]
    else:
        checkpoint_row["sha256"] = sha256(checkpoint)
    payload = {
        "version": 1, "experiment": EXPERIMENT, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "runtime_paths": source_hashes(root), "checkpoint": checkpoint_row,
        "prompt_file": str(prompt_path.resolve()), "prompt_file_sha256": sha256(prompt_path),
        "source_prompt_sha256": sha256(source), "prompt_count": 32,
        "prompt_items": [{"index": i, "source_index": source_i, "text": selected[i]} for i, source_i in enumerate(SOURCE_INDICES)],
        "seed": SEED, "canary_prompts": list(CANARY_PROMPTS), "canary_frames": 30, "screen_frames": 120,
        "methods": list(METHODS), "method_specs": {key: {"budget": budget, "sink": sink} for key, (budget, sink) in SPECS.items()},
        "configs": config_files, "head_map": str(map_path.resolve()), "head_map_sha256": sha256(map_path),
        "canary_runs": {key: {"config": config, "runtime": runtime} for key, (config, runtime) in CANARY_RUNS.items()},
        "claim_boundary": "Native SF protocol/budget attribution, not a new method. Production and math-reference canaries are separate. Native screen requires native repeat/membership checks only; PF or Adaptive failures cannot establish SF budget losses.",
    }
    write_frozen(manifest_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    return payload


def verify(path: Path, root: Path, *, check_runtime: bool = True) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment") != EXPERIMENT or data.get("methods") != list(METHODS) or data.get("seed") != SEED:
        raise ValueError("invalid v209 manifest")
    if sha256(Path(data["prompt_file"])) != data["prompt_file_sha256"] or sha256(Path(data["head_map"])) != data["head_map_sha256"]:
        raise ValueError("v209 prompt/map drift")
    for row in data["configs"].values():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("v209 config drift")
    if check_runtime:
        if source_hashes(root) != data["runtime_paths"]:
            raise ValueError("v209 runtime drift; use a new output root")
        stat = Path(data["checkpoint"]["path"]).stat()
        if stat.st_size != data["checkpoint"]["bytes"] or stat.st_mtime_ns != data["checkpoint"]["mtime_ns"]:
            raise ValueError("v209 checkpoint stamp drift")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.repo_root, args.source_prompts, args.checkpoint, args.output_root)
    print(f"[v209-prepare] methods={len(METHODS)} canary_runs={len(CANARY_RUNS)} prompts={data['prompt_count']}")


if __name__ == "__main__":
    main()
