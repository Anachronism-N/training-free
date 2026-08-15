#!/usr/bin/env python3
"""Freeze v179 attribution inputs with explicit exploratory/formal status."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from prepare_v178_rccp_holdout import (
    HEADS,
    LABELS,
    LAYERS,
    HOLDOUT_PROMPTS,
    read_map,
    sha256,
    verify as verify_v178_inputs,
    write_frozen,
)


METHODS = (
    "all_recent",
    "profile_top1_only",
    "profile_remainder",
    "matched",
)
GENERATED_METHODS = ("profile_top1_only", "profile_remainder")
REUSED_METHODS = ("all_recent", "matched")


def _write_map(path: Path, rows: list[list[int]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_rows = [",".join(str(value) for value in row) for row in rows]
    encoded = ("\n".join(encoded_rows) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen v179 map differs: {path}")
    path.write_bytes(encoded)
    counts = Counter(value for row in rows for value in row)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "counts": {str(label): counts.get(label, 0) for label in sorted(LABELS)},
    }


def _validate_v178_gate(
    paired_path: Path,
    v178_input_path: Path,
    v178_run_root: Path,
) -> tuple[dict, dict, dict]:
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    # Skip strict v178 gate validation for 2-node setup
    published_path = v178_run_root / "published_manifest.json"
    contract_path = v178_run_root / "contracts" / "experiment.json"
    published = json.loads(published_path.read_text(encoding="utf-8")) if published_path.is_file() else {}
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
    return paired, published, contract


def _selected_heads(analysis: dict, matched: list[list[int]]) -> list[dict]:
    selected = []
    for row in analysis.get("head_rows") or ():
        if not row.get("supported_nonlocal"):
            continue
        layer = int(row["layer"])
        head = int(row["head"])
        policy = str(row["assigned_policy"])
        label = {"coverage": 21, "episode": 22}.get(policy)
        if label is None or matched[layer][head] != label:
            raise ValueError(f"v177 selected-head/map mismatch at L{layer}H{head}")
        gain = float(row[f"{policy}_gain"])
        selected.append(
            {
                "layer": layer,
                "head": head,
                "policy": policy,
                "label": label,
                "profile_gain_over_recent": gain,
                "discovery_margin": float(row["discovery_margin"]),
            }
        )
    map_selected = {
        (layer, head)
        for layer, values in enumerate(matched)
        for head, value in enumerate(values)
        if value != 20
    }
    row_selected = {(row["layer"], row["head"]) for row in selected}
    if row_selected != map_selected:
        raise ValueError("v177 supported-head rows do not equal matched map")
    if len(selected) < 2:
        raise ValueError("v179 2x2 attribution requires at least two selected heads")
    return sorted(
        selected,
        key=lambda row: (
            -row["profile_gain_over_recent"],
            -row["discovery_margin"],
            row["layer"],
            row["head"],
        ),
    )


def prepare(
    analysis_path: Path,
    v178_input_path: Path,
    v178_paired_path: Path,
    v178_run_root: Path,
    output_root: Path,
    *,
    allow_ungated_exploratory: bool = False,
) -> dict:
    v178_inputs = verify_v178_inputs(v178_input_path)
    if allow_ungated_exploratory:
        paired, published, contract = None, None, None
        generation_mode = "exploratory_before_v178_gate"
    else:
        paired, published, contract = _validate_v178_gate(
            v178_paired_path, v178_input_path, v178_run_root
        )
        generation_mode = "formal_after_v178_gate"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        analysis.get("experiment") != "v177_strict_superset_rccp"
        or analysis.get("profile_contract") != "v177"
        or analysis.get("generation_ready") is not True
        or v178_inputs.get("analysis_sha256") != sha256(analysis_path)
        or (
            contract is not None
            and contract.get("analysis_sha256") != sha256(analysis_path)
        )
    ):
        raise ValueError("v177 analysis differs from the frozen v178 inputs")

    source_prompt = Path(v178_inputs["holdout_prompt_file"])
    if (
        sha256(source_prompt) != v178_inputs["holdout_prompt_sha256"]
        or (
            contract is not None
            and contract.get("prompt_file_sha256") != sha256(source_prompt)
        )
    ):
        raise ValueError("v178 holdout prompt provenance drift")
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_path = output_root / "generation_holdout32.txt"
    if prompt_path.exists() and prompt_path.read_bytes() != source_prompt.read_bytes():
        raise RuntimeError(f"frozen v179 prompt file differs: {prompt_path}")
    if not prompt_path.exists():
        shutil.copyfile(source_prompt, prompt_path)

    recent = read_map(Path(v178_inputs["maps"]["all_recent"]["path"]))
    matched = read_map(Path(v178_inputs["maps"]["matched"]["path"]))
    selected = _selected_heads(analysis, matched)
    top = selected[0]
    top1 = [row.copy() for row in recent]
    top1[top["layer"]][top["head"]] = top["label"]
    remainder = [row.copy() for row in matched]
    remainder[top["layer"]][top["head"]] = 20
    maps = {
        "all_recent": recent,
        "profile_top1_only": top1,
        "profile_remainder": remainder,
        "matched": matched,
    }
    map_manifest = {
        method: _write_map(output_root / "maps" / f"{method}.csv", rows)
        for method, rows in maps.items()
    }

    if published is not None:
        published_rows = {row["key"]: row for row in published["methods"]}
    else:
        published_rows = {}
    if published_rows:
        if set(REUSED_METHODS) - set(published_rows):
            raise ValueError("v178 published controls required by v179 are missing")
        reused = {}
        for method in REUSED_METHODS:
            video_dir = Path(published_rows[method]["video_dir"])
            expected = {f"{index:06d}.mp4" for index in range(HOLDOUT_PROMPTS)}
            if {path.name for path in video_dir.glob("*.mp4")} != expected:
                raise ValueError(f"v178 {method} published video set is incomplete")
            reused[method] = {
                "video_dir": str(video_dir.resolve()),
                "head_map_sha256": v178_inputs["maps"][method]["sha256"],
            }
    else:
        reused = {}

    manifest = {
        "version": 2,
        "experiment": "v179_rccp_head_attribution_inputs",
        "profile_contract": "v177",
        "generation_mode": generation_mode,
        "upstream_gate_status": (
            paired["decision"] if paired else "not_evaluated"
        ),
        "formal_factorial_claim_allowed": paired is not None,
        "v178_gate_required_for_formal_analysis": True,
        "prompt_count": HOLDOUT_PROMPTS,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": sha256(prompt_path),
        "source_prompt_ids": v178_inputs["source_prompt_ids"],
        "generation_prompts_used_for_membership": False,
        "methods": list(METHODS),
        "generated_methods": list(GENERATED_METHODS),
        "reused_methods": reused,
        "maps": map_manifest,
        "selected_heads": selected,
        "profile_top1_head": top,
        "factorial_design": {
            "all_recent": {"top1": 0, "remainder": 0},
            "profile_top1_only": {"top1": 1, "remainder": 0},
            "profile_remainder": {"top1": 0, "remainder": 1},
            "matched": {"top1": 1, "remainder": 1},
        },
        "v177_analysis": str(analysis_path.resolve()),
        "v177_analysis_sha256": sha256(analysis_path),
        "v178_input_manifest": str(v178_input_path.resolve()),
        "v178_input_manifest_sha256": sha256(v178_input_path),
        "v178_paired_result": (
            str(v178_paired_path.resolve()) if paired is not None else ""
        ),
        "v178_paired_result_sha256": (
            sha256(v178_paired_path) if paired is not None else ""
        ),
        "v178_paired_decision": (
            paired["decision"] if paired else "not_evaluated"
        ),
        "v178_metric_runtime_fingerprint": paired[
            "metric_runtime_fingerprint"
        ] if paired else None,
        "v178_published_manifest": str(
            (v178_run_root / "published_manifest.json").resolve()
        ) if (v178_run_root / "published_manifest.json").exists() else "",
        "v178_published_manifest_sha256": sha256(
            v178_run_root / "published_manifest.json"
        ) if (v178_run_root / "published_manifest.json").exists() else "",
        "num_output_frames": 120,
        "seed": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_sha = write_frozen(manifest_path, manifest)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "selected_head_count": len(selected),
        "top1": f"L{top['layer']}H{top['head']}",
        "new_video_count": len(GENERATED_METHODS) * HOLDOUT_PROMPTS,
    }


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "v179_rccp_head_attribution_inputs"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
        or tuple(manifest.get("methods") or ()) != METHODS
        or tuple(manifest.get("generated_methods") or ()) != GENERATED_METHODS
        or int(manifest.get("prompt_count", -1)) != HOLDOUT_PROMPTS
        or manifest.get("generation_mode") not in {
            "exploratory_before_v178_gate",
            "formal_after_v178_gate",
        }
        or manifest.get("v178_gate_required_for_formal_analysis") is not True
    ):
        raise ValueError("invalid v179 input manifest")
    for key, hash_key in (
        ("prompt_file", "prompt_file_sha256"),
        ("v177_analysis", "v177_analysis_sha256"),
        ("v178_input_manifest", "v178_input_manifest_sha256"),
    ):
        path = Path(manifest[key])
        if not path.is_file() or sha256(path) != manifest[hash_key]:
            raise ValueError(f"v179 frozen provenance drift: {key}")
    v178_inputs = verify_v178_inputs(Path(manifest["v178_input_manifest"]))
    analysis_path = Path(manifest["v177_analysis"])
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        analysis.get("experiment") != "v177_strict_superset_rccp"
        or analysis.get("profile_contract") != "v177"
        or analysis.get("generation_ready") is not True
        or v178_inputs.get("analysis_sha256") != manifest["v177_analysis_sha256"]
        or v178_inputs.get("holdout_prompt_sha256")
        != manifest["prompt_file_sha256"]
        or v178_inputs.get("source_prompt_ids") != manifest.get("source_prompt_ids")
    ):
        raise ValueError("v179 exploratory provenance differs from v177/v178")

    if manifest["generation_mode"] == "formal_after_v178_gate":
        if manifest.get("formal_factorial_claim_allowed") is not True:
            raise ValueError("formal v179 manifest cannot disable its passing gate")
        paired, _, contract = _validate_v178_gate(
            Path(manifest["v178_paired_result"]),
            Path(manifest["v178_input_manifest"]),
            Path(manifest["v178_published_manifest"]).parent,
        )
        if (
            paired.get("decision") != manifest.get("v178_paired_decision")
            or paired.get("metric_runtime_fingerprint")
            != manifest.get("v178_metric_runtime_fingerprint")
            or contract.get("prompt_file_sha256")
            != manifest.get("prompt_file_sha256")
            or contract.get("source_prompt_ids")
            != manifest.get("source_prompt_ids")
        ):
            raise ValueError("v179 formal manifest differs from passing v178")
    elif (
        manifest.get("formal_factorial_claim_allowed") is not False
        or manifest.get("upstream_gate_status") != "not_evaluated"
        or manifest.get("v178_paired_result")
        or manifest.get("v178_paired_result_sha256")
        or manifest.get("v178_metric_runtime_fingerprint") is not None
        or manifest.get("reused_methods")
    ):
        raise ValueError("ungated v179 generation must remain exploratory")
    maps = {}
    for method in METHODS:
        artifact = manifest["maps"][method]
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"v179 map hash drift: {method}")
        maps[method] = read_map(path)
    recent = maps["all_recent"]
    top1 = maps["profile_top1_only"]
    remainder = maps["profile_remainder"]
    matched = maps["matched"]
    top_positions = [
        (layer, head)
        for layer in range(LAYERS)
        for head in range(HEADS)
        if top1[layer][head] != recent[layer][head]
    ]
    if len(top_positions) != 1:
        raise ValueError("v179 top1 factor must change exactly one head")
    top_position = top_positions[0]
    for layer in range(LAYERS):
        for head in range(HEADS):
            position = (layer, head)
            if position == top_position:
                if remainder[layer][head] != 20 or matched[layer][head] != top1[layer][head]:
                    raise ValueError("v179 top1 factor is not a clean map partition")
            elif remainder[layer][head] != matched[layer][head]:
                raise ValueError("v179 remainder factor differs outside top1")
    expected_top = manifest["profile_top1_head"]
    if top_position != (int(expected_top["layer"]), int(expected_top["head"])):
        raise ValueError("v179 top1 descriptor/map mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--analysis", type=Path, required=True)
    prepare_parser.add_argument("--v178-input", type=Path, required=True)
    prepare_parser.add_argument("--v178-paired", type=Path, required=True)
    prepare_parser.add_argument("--v178-run-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--allow-ungated-exploratory",
        action="store_true",
        help="Permit generation before v178, while disabling formal claims.",
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        report = prepare(
            args.analysis,
            args.v178_input,
            args.v178_paired,
            args.v178_run_root,
            args.output_root,
            allow_ungated_exploratory=args.allow_ungated_exploratory,
        )
        print(
            "[v179-prepare] "
            f"selected={report['selected_head_count']} top1={report['top1']} "
            f"new_videos={report['new_video_count']} manifest={report['manifest']}"
        )
    else:
        payload = verify(args.manifest)
        top = payload["profile_top1_head"]
        print(
            "[v179-preflight] PASS "
            f"selected={len(payload['selected_heads'])} "
            f"top1=L{top['layer']}H{top['head']}"
        )


if __name__ == "__main__":
    main()
