#!/usr/bin/env python3
"""Freeze the Qwen-rewritten diverse MovieBench-16 suite for v154."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE_PROMPT_COUNT = 128
SOURCE_SHA256 = "99468409fe54322bc383376e6037196e922cfbae47814a7a4e51740ee0571281"
PROMPT_FILENAME = "moviegen_128_qwen_v154_diverse16.txt"
MANIFEST_FILENAME = "moviegen_128_qwen_v154_diverse16.json"
SERVER_SOURCE = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/"
    "Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
SELECTIONS = (
    (0, ("human_identity", "walking", "crowd", "night_city")),
    (1, ("multi_subject", "animal", "snow", "camera_depth")),
    (4, ("3d_animation", "character_identity", "object_interaction", "fire")),
    (7, ("multi_object", "miniature_scale", "water_motion", "photorealistic")),
    (13, ("multi_person", "mobile_video", "urban", "documentary")),
    (15, ("rotating_camera", "indoor", "many_objects", "screen_content")),
    (17, ("vehicle", "fast_motion", "dust", "tracking_camera")),
    (24, ("articulated_motion", "festival", "crowd", "colorful")),
    (33, ("human_motion", "running", "cinematic", "step_printing")),
    (47, ("two_subjects", "animal_identity", "running", "neon_city")),
    (61, ("child_identity", "bicycle", "season_change", "scene_evolution")),
    (67, ("vehicle", "high_speed", "turning", "dynamic_background")),
    (75, ("object_identity", "tracking_camera", "long_motion", "abandoned_street")),
    (84, ("fpv", "scene_transition", "rapid_camera", "interior")),
    (109, ("human_identity", "anime", "ship", "camera_facing")),
    (124, ("transformation", "animal_identity", "lightning", "scene_event")),
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_payloads(source: Path) -> dict[str, bytes]:
    lines = normalized_lines(source)
    if len(lines) != SOURCE_PROMPT_COUNT:
        raise ValueError(
            f"expected {SOURCE_PROMPT_COUNT} source prompts, found {len(lines)}"
        )
    canonical_source = ("\n".join(lines) + "\n").encode("utf-8")
    if sha256(canonical_source) != SOURCE_SHA256:
        raise ValueError("Qwen MovieBench-128 source hash changed")
    selected = [lines[index] for index, _ in SELECTIONS]
    prompt_payload = ("\n".join(selected) + "\n").encode("utf-8")
    manifest = {
        "version": 1,
        "suite": "v154_qwen_moviebench_diverse16",
        "source": {
            "canonical_sha256": SOURCE_SHA256,
            "prompt_count": SOURCE_PROMPT_COUNT,
            "hash_contract": "nonempty_stripped_lines_joined_with_lf",
        },
        "selection_policy": (
            "fixed stratified coverage of identity, multi-subject interaction, "
            "motion, camera motion, style, background evolution, scene "
            "transition, and transformation; indices inherited from v116"
        ),
        "prompt_file": PROMPT_FILENAME,
        "prompt_file_sha256": sha256(prompt_payload),
        "prompt_count": len(selected),
        "items": [
            {
                "subset_index": subset_index,
                "source_index": source_index,
                "source_number": source_index + 1,
                "tags": list(tags),
                "text": lines[source_index],
            }
            for subset_index, (source_index, tags) in enumerate(SELECTIONS)
        ],
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    return {
        PROMPT_FILENAME: prompt_payload,
        MANIFEST_FILENAME: manifest_payload,
    }


def default_source(root: Path) -> Path:
    candidates = (
        os.environ.get("V154_SOURCE_PROMPTS", "").strip(),
        SERVER_SOURCE,
        str(root / "prompts" / "moviegen_128_qwen_v129.txt"),
    )
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value)
    return Path(SERVER_SOURCE)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source(root))
    parser.add_argument(
        "--output-dir", type=Path, default=root / "prompts"
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"missing Qwen MovieBench source: {args.source}")
    payloads = build_payloads(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        path = args.output_dir / filename
        if args.check:
            if not path.is_file():
                raise SystemExit(f"missing frozen v154 artifact: {path}")
            if filename.endswith(".json"):
                matches = json.loads(path.read_text(encoding="utf-8")) == json.loads(
                    payload.decode("utf-8")
                )
            else:
                existing = ("\n".join(normalized_lines(path)) + "\n").encode(
                    "utf-8"
                )
                matches = existing == payload
            if not matches:
                raise SystemExit(f"frozen v154 artifact mismatch: {path}")
        else:
            path.write_bytes(payload)
        print(f"[v154-suite] {filename} sha256={sha256(payload)}")


if __name__ == "__main__":
    main()
