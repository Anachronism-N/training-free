#!/usr/bin/env python3
"""Create a deterministic, at-most-four-video blind review for v165."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from v165_decision_contract import DIRECTION_MATCH, PRIMARY


REVIEW_COLUMNS = (
    "video",
    "prompt_index",
    "prompt",
    "identity_continuity_-2_to_2",
    "background_continuity_-2_to_2",
    "motion_amount_-2_to_2",
    "motion_naturalness_-2_to_2",
    "late_stability_-2_to_2",
    "prompt_fidelity_-2_to_2",
    "overall_preference_-2_to_2",
    "severe_failure_0_or_1",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen_json(path: Path, payload: Any) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen review artifact differs: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def link_or_validate(source: Path, target: Path) -> str:
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"missing review source video: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(f"review target points to a different source: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source)
        return "symlink"


def blind_order_key(prompt: int, method: str) -> str:
    value = f"v165-minimal-review|{prompt}|{method}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def review_rows(decision: dict[str, Any], run_root: Path) -> list[dict[str, Any]]:
    plan = decision.get("review_plan") or {}
    methods = tuple(plan.get("methods") or ())
    prompts = list(plan.get("rows") or [])
    if methods != (PRIMARY, DIRECTION_MATCH):
        raise ValueError("v165 review methods violate the frozen two-method plan")
    if not 1 <= len(prompts) <= 2 or int(plan.get("video_count", -1)) != 2 * len(
        prompts
    ):
        raise ValueError("v165 review plan must contain one or two prompt pairs")
    expanded = []
    for row in prompts:
        prompt = int(row["prompt_index"])
        prompt_text = str(row["prompt"]).strip()
        if not 0 <= prompt < 16 or not prompt_text:
            raise ValueError(f"invalid review prompt: {row}")
        for method in methods:
            source = (
                run_root
                / "published_indexed"
                / method
                / f"{prompt:06d}-0_v165.mp4"
            )
            expanded.append(
                {
                    "method": method,
                    "prompt_index": prompt,
                    "prompt": prompt_text,
                    "reasons": list(row.get("reasons") or ()),
                    "source": source.resolve(),
                }
            )
    expanded.sort(
        key=lambda row: blind_order_key(row["prompt_index"], row["method"])
    )
    if len(expanded) > 4:
        raise ValueError("v165 minimal review exceeds four videos")
    return expanded


def validate_existing_sheet(path: Path, expected: list[dict[str, str]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != REVIEW_COLUMNS:
        raise ValueError(f"existing review sheet columns differ: {path}")
    observed = [
        {
            "video": str(row["video"]),
            "prompt_index": str(row["prompt_index"]),
            "prompt": str(row["prompt"]),
        }
        for row in rows
    ]
    if observed != expected:
        raise ValueError(f"existing review sheet rows differ: {path}")


def write_review_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    expected = [
        {
            "video": row["video"],
            "prompt_index": row["prompt_index"],
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    if path.exists():
        validate_existing_sheet(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("experiment") != "v165_final_development_decision":
        raise ValueError("not a v165 final-decision report")
    rows = review_rows(decision, args.run_root.resolve())
    key_rows = []
    reviewer_rows = []
    for index, row in enumerate(rows, 1):
        video = f"V{index:03d}.mp4"
        target = args.output_root / "reviewer" / "videos" / video
        link_or_validate(row["source"], target)
        key_rows.append(
            {
                "video": video,
                "method": row["method"],
                "prompt_index": row["prompt_index"],
                "prompt": row["prompt"],
                "reasons": row["reasons"],
                "source": str(row["source"]),
                "source_sha256": sha256(row["source"]),
            }
        )
        reviewer_rows.append(
            {
                "video": video,
                "prompt_index": str(row["prompt_index"]),
                "prompt": row["prompt"],
                **{column: "" for column in REVIEW_COLUMNS[3:]},
            }
        )
    private_key = {
        "version": 1,
        "experiment": "v165_minimal_blind_review",
        "decision_sha256": sha256(args.decision),
        "rows": key_rows,
        "claim_boundary": (
            "This metric-adaptive review is engineering triage, not an "
            "unbiased paper human study."
        ),
    }
    key_path = args.output_root / "private" / "blind_key.json"
    key_sha = write_frozen_json(key_path, private_key)
    sheet_path = args.output_root / "reviewer" / "review_sheet.csv"
    write_review_sheet(sheet_path, reviewer_rows)
    manifest = {
        "version": 1,
        "experiment": "v165_minimal_blind_review",
        "video_count": len(rows),
        "prompt_count": len({row["prompt_index"] for row in rows}),
        "maximum_video_count": 4,
        "blind_key": str(key_path.resolve()),
        "blind_key_sha256": key_sha,
        "review_sheet": str(sheet_path.resolve()),
        "review_sheet_columns": list(REVIEW_COLUMNS),
        "linked_video_count": len(rows),
        "ok": len(rows) <= 4,
    }
    write_frozen_json(args.output_root / "review_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    report = prepare(args)
    print(
        "[v165-minimal-review] "
        f"videos={report['video_count']} prompts={report['prompt_count']} "
        f"output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
